"""
Evaluation script — generates the metrics you actually need for a strong
presentation: confusion matrix, per-class precision/recall/F1, and a
grid of misclassified examples (useful for the "limitations" slide).

Usage:
    python src/evaluate.py --data_dir data --model models/waste_classifier.pth --out_dir eval_results
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model, get_transform_normalization


def _build_model_for_architecture(architecture: str, num_classes: int):
    """Supports both the deployed MobileNetV2 model and the optional
    CNN-LSTM comparison model (src/model_cnn_lstm.py) — same evaluation
    script, so the metrics are directly comparable."""
    if architecture == "cnn_lstm":
        from model_cnn_lstm import build_cnn_lstm_model
        return build_cnn_lstm_model(num_classes=num_classes)
    return build_model(num_classes=num_classes)


def get_val_loader(data_dir: str, img_size: int = 224, batch_size: int = 32):
    norm = get_transform_normalization()
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"]),
    ])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return val_loader, val_dataset


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(args.model, map_location=device)
    class_names = checkpoint["class_names"]
    model = _build_model_for_architecture(args.architecture, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    val_loader, val_dataset = get_val_loader(args.data_dir)

    all_preds, all_labels, all_paths = [], [], []
    sample_paths = [p for p, _ in val_dataset.samples]

    idx = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            batch_size = images.size(0)
            all_paths.extend(sample_paths[idx: idx + batch_size])
            idx += batch_size

    # ---- Overall accuracy ----
    overall_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"\nOverall validation accuracy: {overall_acc*100:.2f}%\n")

    # ---- Classification report (per-class precision/recall/F1) ----
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    print(report)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Overall accuracy: {overall_acc*100:.2f}%\n\n")
        f.write(report)

    # ---- Confusion matrix ----
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix — Validation Set")
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # ---- Misclassified examples grid (great for the "limitations" slide) ----
    misclassified = [
        (path, class_names[true], class_names[pred])
        for path, true, pred in zip(all_paths, all_labels, all_preds)
        if true != pred
    ]
    print(f"\nMisclassified: {len(misclassified)} / {len(all_labels)} images")

    if misclassified:
        from PIL import Image
        n_show = min(9, len(misclassified))
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < n_show:
                path, true_cls, pred_cls = misclassified[i]
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"True: {true_cls}\nPred: {pred_cls}", fontsize=9, color="red")
            ax.axis("off")
        plt.tight_layout()
        mis_path = os.path.join(args.out_dir, "misclassified_examples.png")
        plt.savefig(mis_path, dpi=150)
        plt.close()
        print(f"Misclassified examples grid saved to: {mis_path}")

    print(f"\nAll evaluation artifacts saved to: {args.out_dir}/")
    return {"overall_acc": overall_acc, "num_misclassified": len(misclassified), "total": len(all_labels)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="models/waste_classifier.pth")
    parser.add_argument("--out_dir", type=str, default="eval_results")
    parser.add_argument(
        "--architecture", type=str, default="cnn", choices=["cnn", "cnn_lstm"],
        help="'cnn' for the deployed MobileNetV2 model, 'cnn_lstm' for the "
             "optional comparison model (src/model_cnn_lstm.py)",
    )
    args = parser.parse_args()
    evaluate(args)
