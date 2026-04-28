import torch.nn as nn
from torchvision import models

def EfficientNetB7(out_ftrs = 2):
        myModel = models.efficientnet_b7(weights='IMAGENET1K_V1')

        in_ftrs = myModel.classifier[1].in_features
        myModel.classifier = nn.Linear(in_ftrs, out_ftrs, bias=True)

        return myModel



