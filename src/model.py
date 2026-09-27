"""
Model definition: MobileNetV2 backbone (pretrained on ImageNet) with a
custom classification head for waste categories.

Using transfer learning here is a deliberate choice for this project:
- Training from scratch would need far more data/time than a class
  project timeline allows.
- MobileNetV2 is small and fast enough to realistically run on a
  phone/edge device later (ties into "future scope: smart bin / mobile
  deployment").
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Build a MobileNetV2-based classifier.

    Args:
        num_classes: number of waste categories to predict.
        freeze_backbone: if True, freeze the pretrained feature extractor
            and only train the new classification head. This is faster
            and works well with small datasets (recommended for a
            first pass). Set False to fine-tune the whole network once
            you have a working baseline and want to squeeze out more
            accuracy.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the final classifier layer for our number of waste classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def get_transform_normalization():
    """Return the ImageNet mean/std used by the pretrained backbone."""
    return {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


if __name__ == "__main__":
    # quick sanity check that the model builds and runs a forward pass
    m = build_model(num_classes=6)
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print("Output shape:", out.shape)  # expect [2, 6]
