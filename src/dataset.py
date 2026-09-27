"""
Data loading utilities. Expects an ImageFolder-style layout:

    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg

torchvision.datasets.ImageFolder automatically infers class labels
from folder names and keeps them sorted alphabetically — the same
order is used at inference time (see predict.py), so don't rename
folders between training and prediction.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_transform_normalization


def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 224):
    norm = get_transform_normalization()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes  # alphabetically sorted, matches folder names
    return train_loader, val_loader, class_names
