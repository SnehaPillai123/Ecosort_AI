"""
OPTIONAL / EXPERIMENTAL — trains the CNN-LSTM comparison model
(model_cnn_lstm.py). Mirrors train.py so results are comparable
(same data pipeline, same class weighting, same optimizer settings).

Run this only after your main model (train.py) is already trained and
working — this is a bonus comparison for your report, not a
replacement. It has not been run in this sandbox (no GPU/torch here).

Usage:
    python src/train_cnn_lstm.py --data_dir data --epochs 15 \
        --out models/waste_classifier_cnn_lstm.pth
"""

import argparse
import json
import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model_cnn_lstm import build_cnn_lstm_model


def compute_class_weights(train_loader, num_classes, device):
    counts = Counter(train_loader.dataset.targets)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training the OPTIONAL CNN-LSTM comparison model (Lilhore et al., 2023 architecture)")

    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_cnn_lstm_model(num_classes=len(class_names), freeze_backbone=True).to(device)

    class_weights = compute_class_weights(train_loader, len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        start = time.time()

        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        scheduler.step()
        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "class_names": class_names},
                args.out,
            )
            print(f"  -> New best CNN-LSTM model saved (val_acc={val_acc:.4f})")

    meta_path = os.path.splitext(args.out)[0] + "_classes.json"
    with open(meta_path, "w") as f:
        json.dump(class_names, f)

    print(f"\nCNN-LSTM training complete. Best val_acc={best_val_acc:.4f}")
    print(f"Model saved to: {args.out}")
    print(
        "\nCompare this best_val_acc against your main model's (from train.py's "
        "output) for the report — same dataset, same split, same class weighting, "
        "so the comparison is fair."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/waste_classifier_cnn_lstm.pth")
    args = parser.parse_args()

    train(args)
