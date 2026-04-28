"""
FaceForensics++ PyTorch DataLoader
===================================
Loads video frames from the FF++ dataset for binary deepfake detection.
Extracts faces using RetinaFace, caches cropped faces to disk for fast reloading.
Handles multi-face frames — each detected face becomes a separate training sample.

Expected folder structure:
    dataset_root/
        original/          # 1000 real videos
        DeepFakeDetection/  # 1000 deepfake videos
        Deepfakes/          # 1000 deepfake videos
        Face2Face/          # 1000 deepfake videos
        FaceShifter/        # 1000 deepfake videos
        FaceSwap/           # 1000 deepfake videos
        NeuralTextures/     # 1000 deepfake videos
        csv/                # metadata

Usage:
    train_loader, val_loader = get_dataloaders(
        dataset_root="/path/to/ff++",
        frames_per_video=16,
        batch_size=32,
    )
"""

import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ── Try importing MTCNN; fall back to Haar cascade if unavailable ──────────
try:
    from facenet_pytorch import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False
    print("facenet_pytorch not installed")
    print("For better results: pip install facenet-pytorch")


# ── Configuration ──────────────────────────────────────────────────────────

FAKE_FOLDERS = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]
REAL_FOLDERS = ["original"]

LABEL_REAL = 0
LABEL_FAKE = 1


# ── Face extraction helpers ────────────────────────────────────────────────

class FaceExtractor:
    """Detects and crops the largest face from a frame."""

    def __init__(self, face_size: int = 224, margin: int = 40):
        self.face_size = face_size
        self.margin = margin

        if _MTCNN_AVAILABLE:
            self.detector = MTCNN(
                image_size=face_size,
                margin=margin,
                keep_all=False,          # only the most prominent face
                post_process=False,       # return PIL-friendly uint8
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)

    def __call__(self, frame_bgr: np.ndarray) -> Image.Image | None:
        """
        Args:
            frame_bgr: OpenCV BGR frame (H, W, 3).
        Returns:
            PIL Image of the cropped face, or None if no face found.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        if _MTCNN_AVAILABLE:
            face = self.detector(pil_img)  # returns tensor or None
            if face is None:
                return None
            # MTCNN returns (C, H, W) float tensor; convert to PIL
            face_np = face.permute(1, 2, 0).byte().numpy()
            return Image.fromarray(face_np)
        else:
            raise ValueError("Please install all packages in requirements.txt")
        #     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        #     faces = self.detector.detectMultiScale(gray, 1.3, 5)
        #     if len(faces) == 0:
        #         return None
        #     # Take the largest face
        #     x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        #     m = self.margin
        #     x1 = max(0, x - m)
        #     y1 = max(0, y - m)
        #     x2 = min(frame_bgr.shape[1], x + w + m)
        #     y2 = min(frame_bgr.shape[0], y + h + m)
        #     crop = frame_rgb[y1:y2, x1:x2]
        #     return Image.fromarray(crop).resize(
        #         (self.face_size, self.face_size), Image.BILINEAR
        #     )


# ── Frame sampling ─────────────────────────────────────────────────────────

def sample_frames_from_video(
    video_path: str,
    num_frames: int = 16,
) -> list[np.ndarray]:
    """
    Uniformly samples `num_frames` frames from a video file.
    Returns a list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    # Pick evenly-spaced frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# ── Caching: extract faces once, save to disk ─────────────────────────────

def build_face_cache(
    dataset_root: str,
    cache_dir: str,
    frames_per_video: int = 16,
    face_size: int = 224,
):
    """
    Pre-extracts face crops from all videos and saves them as .jpg files.
    This only needs to run ONCE. Subsequent training loads from cache.

    Cache structure:
        cache_dir/
            original/video_name/frame_000.jpg
            Deepfakes/video_name/frame_000.jpg
            ...
    """
    cache_dir = Path(cache_dir)
    extractor = FaceExtractor(face_size=face_size)

    all_folders = REAL_FOLDERS + FAKE_FOLDERS
    for folder in all_folders:
        folder_path = Path(dataset_root) / folder
        if not folder_path.exists():
            print(f"Skipping {folder} (not found)")
            continue

        video_files = sorted([
            f for f in folder_path.iterdir()
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
        ])

        print(f"Processing {folder}: {len(video_files)} videos")
        for vid_path in video_files:
            out_dir = cache_dir / folder / vid_path.stem
            if out_dir.exists() and any(out_dir.iterdir()):
                continue  # already cached

            out_dir.mkdir(parents=True, exist_ok=True)
            frames = sample_frames_from_video(str(vid_path), frames_per_video)

            saved = 0
            for i, frame in enumerate(frames):
                face = extractor(frame)
                if face is not None:
                    face.save(out_dir / f"frame_{i:03d}.jpg")
                    saved += 1

            if saved == 0:
                # Clean up empty dirs
                out_dir.rmdir()

    print(f"✓ Face cache ready at {cache_dir}")


# ── Dataset ────────────────────────────────────────────────────────────────

class FaceForensicsDataset(Dataset):
    """
    Loads pre-extracted face crops from the cache directory.
    Each sample = one face crop image → label (0=real, 1=fake).
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",          # "train", "val"
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        transform=None,
        seed: int = 42,
        fake_folders: list[str] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []  # (path, label)

        fake_folders = fake_folders or FAKE_FOLDERS

        # Gather all face images grouped by video
        video_groups = {"real": [], "fake": []}

        for folder in REAL_FOLDERS:
            folder_path = self.cache_dir / folder
            if folder_path.exists():
                video_groups["real"].extend(sorted(folder_path.iterdir()))

        for folder in fake_folders:
            folder_path = self.cache_dir / folder
            if folder_path.exists():
                video_groups["fake"].extend(sorted(folder_path.iterdir()))

        # Split at the VIDEO level (not frame level) to prevent data leakage
        rng = random.Random(seed)

        for category, videos in video_groups.items():
            rng.shuffle(videos)
            n = len(videos)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            if split == "train":
                selected = videos[:train_end]
            elif split == "val":
                selected = videos[train_end:val_end]
            else:
                selected = videos[val_end:]

            label = LABEL_REAL if category == "real" else LABEL_FAKE

            for video_dir in selected:
                if not video_dir.is_dir():
                    continue
                for img_path in sorted(video_dir.glob("*.jpg")):
                    self.samples.append((str(img_path), label))

        print(f"[{split}] Loaded {len(self.samples)} face crops "
              f"(real: {sum(1 for _, l in self.samples if l == 0)}, "
              f"fake: {sum(1 for _, l in self.samples if l == 1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ── Transforms ─────────────────────────────────────────────────────────────

def get_transforms(split: str, face_size: int = 224):
    """ImageNet-normalized transforms with augmentation for training."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((face_size, face_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomGrayscale(p=0.05),
            # Simulate compression artifacts (common in real-world deepfakes)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((face_size, face_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# ── DataLoader factory ─────────────────────────────────────────────────────

def get_dataloaders(
    dataset_root: str,
    cache_dir: str = "./ff_face_cache",
    frames_per_video: int = 16,
    face_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    fake_folders: list[str] | None = None,
    balance_classes: bool = True,
):
    """
    Full pipeline: cache faces (if needed) → create train/val loaders.

    Args:
        dataset_root:    Path to the FF++ dataset root.
        cache_dir:       Where to store extracted face crops.
        frames_per_video: Frames to sample per video.
        face_size:       Face crop resolution (square).
        batch_size:      Batch size for all loaders.
        num_workers:     DataLoader workers.
        fake_folders:    Which fake subsets to include (default: all 6).
        balance_classes: Use weighted sampling to handle class imbalance
                         (6000 fake videos vs 1000 real).

    Returns:
        (train_loader, val_loader)
    """
    # Step 1: Build face cache if it doesn't exist
    if not Path(cache_dir).exists():
        raise ValueError("incorrect directory for face cache. If you are building" \
        "cache for first time then uncomment the two lines below")
        # print("Building face cache (one-time operation)...")
        # build_face_cache(dataset_root, cache_dir, frames_per_video, face_size)

    # Step 2: Create datasets
    datasets = {}
    for split in ("train", "val"):
        datasets[split] = FaceForensicsDataset(
            cache_dir=cache_dir,
            split=split,
            transform=get_transforms(split, face_size),
            fake_folders=fake_folders,
        )

    # Step 3: Handle class imbalance (6:1 fake-to-real ratio)
    train_sampler = None
    shuffle_train = True

    if balance_classes:
        labels = [label for _, label in datasets["train"].samples]
        class_counts = [labels.count(0), labels.count(1)]
        weights = [1.0 / class_counts[l] for l in labels]
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        shuffle_train = False  # sampler handles shuffling

    # Step 4: Build loaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=2,    # Batches to load in advance
        persistent_workers=True, # Keeps workers alive between epochs  
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ── Video-level inference helper ───────────────────────────────────────────

class VideoInference:
    """
    For your notary product: takes a video, samples frames,
    extracts faces, runs model, averages confidence scores.

    Usage:
        inferencer = VideoInference(model, device="cuda")
        is_fake, confidence = inferencer("suspect_video.mp4")
    """

    def __init__(self, model, device: str = "cuda", frames_per_video: int = 10):
        self.model = model.to(device).eval()
        self.device = device
        self.frames_per_video = frames_per_video
        self.extractor = FaceExtractor(face_size=224)
        self.transform = get_transforms("val", face_size=224)

    @torch.no_grad()
    def __call__(self, video_path: str) -> tuple[bool, float]:
        """
        Returns:
            (is_fake: bool, confidence: float 0-1)
            confidence = average probability of "fake" across all frames.
        """
        frames = sample_frames_from_video(video_path, self.frames_per_video)

        face_tensors = []
        for frame in frames:
            face = self.extractor(frame)
            if face is not None:
                face_tensors.append(self.transform(face))

        if len(face_tensors) == 0:
            raise ValueError(f"No faces detected in {video_path}")

        # Batch inference — single forward pass
        batch = torch.stack(face_tensors).to(self.device)
        logits = self.model(batch)
        probs = torch.sigmoid(logits).squeeze(-1)

        avg_confidence = probs.mean().item()
        is_fake = avg_confidence > 0.5

        return is_fake, avg_confidence


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to FF++ dataset root")
    parser.add_argument("--cache", default="./ff_face_cache", help="Face cache dir")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(
        dataset_root=args.root,
        cache_dir=args.cache,
        frames_per_video=args.frames,
        batch_size=args.batch,
    )

    # Sanity check
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels:      {labels[:10].tolist()}...")
    print(f"Real in batch: {(labels == 0).sum().item()}")
    print(f"Fake in batch: {(labels == 1).sum().item()}")
