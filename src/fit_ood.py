"""
Fit the Mahalanobis OOD detector on top of an already-trained classifier.

Run this once, right after train.py, using the same data directory:

    python src/fit_ood.py --data_dir data --model models/waste_classifier.pth \\
        --out models/waste_classifier_ood.npz

By default `--out` is derived from `--model` (waste_classifier.pth ->
waste_classifier_ood.npz), which is also where predict.py and the
Streamlit app look for it — so the plain command above with no --out
is normally all you need.

What it does:
  1. Loads the trained model + class list from the checkpoint.
  2. Extracts 256-dim penultimate-layer features for every TRAINING
     image, using the same deterministic (no-augmentation) transform as
     validation — augmentation would blur the class clusters this
     detector fits against.
  3. Fits one mean vector per class and a single shared covariance
     matrix across all classes (Ledoit-Wolf shrinkage — far more stable
     than a raw empirical covariance when per-class sample counts are
     small relative to the 256 feature dimensions).
  4. Extracts features for the VALIDATION set and uses them to calibrate
     a distance threshold: the given percentile (default 99th) of
     in-distribution validation distances. That means ~1% of genuine,
     correctly-photographed validation images would themselves be
     (falsely) flagged as OOD at this setting — an explicit, tunable
     trade-off against how aggressively you want to catch non-waste
     inputs, not a magic number.
  5. Saves class_means / inv_covariance / threshold to the .npz file.

See src/ood_detector.py for the method and its honest limits.
"""

import argparse
import os

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model, get_transform_normalization
from ood_detector import extract_penultimate_features, MahalanobisOOD


def _deterministic_loader(split_dir: str, img_size: int, batch_size: int):
    norm = get_transform_normalization()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"]),
    ])
    dataset = datasets.ImageFolder(split_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader, dataset.classes


@torch.no_grad()
def _collect_features(model, loader, device):
    model.eval()
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        feats = extract_penultimate_features(model, images)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def fit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.model, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    train_loader, train_classes = _deterministic_loader(train_dir, args.img_size, args.batch_size)
    val_loader, _ = _deterministic_loader(val_dir, args.img_size, args.batch_size)

    if train_classes != class_names:
        raise ValueError(
            "data/train class folders don't match the checkpoint's class "
            f"order.\n  checkpoint classes: {class_names}\n"
            f"  data/train classes:  {train_classes}\n"
            "Re-check --data_dir, or retrain if the dataset changed."
        )

    print("Extracting training features...")
    train_feats, train_labels = _collect_features(model, train_loader, device)

    print("Fitting per-class means + shared covariance (Ledoit-Wolf shrinkage)...")
    num_classes = len(class_names)
    feat_dim = train_feats.shape[1]
    class_means = np.zeros((num_classes, feat_dim), dtype=np.float64)
    centered = np.zeros_like(train_feats, dtype=np.float64)
    for c in range(num_classes):
        mask = train_labels == c
        if mask.sum() == 0:
            raise ValueError(
                f"No training samples for class '{class_names[c]}' — can't fit its mean."
            )
        class_means[c] = train_feats[mask].mean(axis=0)
        centered[mask] = train_feats[mask] - class_means[c]

    cov_estimator = LedoitWolf().fit(centered)
    inv_covariance = cov_estimator.precision_

    print("Extracting validation features to calibrate the OOD threshold...")
    val_feats, _ = _collect_features(model, val_loader, device)
    detector = MahalanobisOOD(class_means, inv_covariance, threshold=np.inf)
    val_scores = detector.score(val_feats)
    threshold = float(np.percentile(val_scores, args.threshold_percentile))
    detector.threshold = threshold

    print(
        f"Validation OOD-score stats: min={val_scores.min():.1f} "
        f"median={np.median(val_scores):.1f} max={val_scores.max():.1f}"
    )
    print(
        f"Threshold (P{args.threshold_percentile:g}): {threshold:.1f} -> "
        f"~{100 - args.threshold_percentile:.1f}% of genuine validation "
        "images would be flagged as OOD at this setting."
    )

    out_path = args.out or (os.path.splitext(args.model)[0] + "_ood.npz")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    detector.save(out_path)
    print(f"Saved OOD detector to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="models/waste_classifier.pth")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Defaults to <model>_ood.npz alongside the model checkpoint.",
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--threshold_percentile", type=float, default=99.0,
        help="Percentile of in-distribution validation OOD scores used as "
             "the flagging threshold. Lower = stricter (more false OOD "
             "flags on real photos); higher = looser (misses more "
             "non-waste inputs).",
    )
    args = parser.parse_args()

    fit(args)
