import os
import csv
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score

from dataclasses import dataclass
from tqdm import tqdm

from dataloader import get_dataloaders
from model import EfficientNetB7

USE_DEVICE_ID = "0,1,2"
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Test'):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = USE_DEVICE_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = "EfficientNetB7"
    CHECKPOINT_DIR = os.path.join("/home/gssodhi/deepfake_detect/checkpoint", f"{MODEL_NAME}/{args.run}")
    METRICS_CSV = os.path.join("/home/gssodhi/deepfake_detect/train_data", f"{MODEL_NAME}/{args.run}.csv")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = EfficientNetB7(out_ftrs=2).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_bal_acc = 0.0

    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_bal_acc", "val_bal_acc"])

    for epoch in range(0, args.epochs):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(train_loader, desc=f'{epoch}/{args.epochs}'):
            # print(f"Got batch: {images.shape}")  # add this temporarily
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        scheduler.step()

        train_loss = total_loss / len(train_loader.dataset)
        train_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        val_loss, val_bal_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train BalAcc: {train_bal_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val BalAcc: {val_bal_acc:.4f}"
        )

        with open(METRICS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_bal_acc, val_bal_acc])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_bal_acc": train_bal_acc,
            "val_bal_acc": val_bal_acc,
        }

        save_checkpoint(checkpoint, os.path.join(CHECKPOINT_DIR, "last.pth"))

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            save_checkpoint(checkpoint, os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"-> New best val balanced accuracy: {best_val_bal_acc:.4f}")

    print(f"\nTraining complete. Best val balanced accuracy: {best_val_bal_acc:.4f}")
    print(f"Metrics saved to:    {METRICS_CSV}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")

@dataclass
class model_config:
    batch_size: int = 128
    num_workers:int = 8
    lr: float = 1e-5
    epochs: int = 150
    frames_per_video:int = 16
    dataset_root: str = '/home/gssodhi/deepfake_detect/dataset/FaceForensics++_C23'
    cache_dir: str = '/home/gssodhi/deepfake_detect/dataset/cache'
    resume_model_path: str = '/scratch/gssodhi/melanoma/checkpoint/chkpt_efNet'
    save_model_path: str = '/scratch/gssodhi/melanoma/checkpoint/chkpt_efNet'
    log_file_path: str = '/home/gssodhi/melanoma/baselines/data/'
    weight_decay: float = 1e-4
    run: str = 'run'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetB7 deepfake detector")
    parser.add_argument("--frames_per_video", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--run", type=str)
    cli_args = parser.parse_args()

    def get_num_workers():
        all_device = USE_DEVICE_ID.split(',')
        # Use 3 cpu cores per GPU
        return 4 * len(all_device)
    
    args = model_config(
        batch_size = cli_args.batch_size,
        num_workers = get_num_workers(),
        epochs = cli_args.epochs,
        lr = cli_args.lr,
        frames_per_video = cli_args.frames_per_video,
        run = cli_args.run,
        weight_decay=cli_args.weight_decay
    )


    train(args)
