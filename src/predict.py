"""
Run a single-image prediction from the command line — useful as a
sanity check after training, before wiring up the Streamlit app.

Usage:
    python src/predict.py --image path/to/test.jpg --model models/waste_classifier.pth
"""

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import build_model, get_transform_normalization
from disposal_guide import get_guidance, get_icon
from ood_detector import extract_penultimate_features, MahalanobisOOD
from waste_gate import try_load_clip_gate, WASTE_PROBABILITY_THRESHOLD


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names


def load_ood_detector(model_path: str):
    """
    Load the fitted Mahalanobis OOD detector for this model, if one has
    been fit (see fit_ood.py). Returns None when it hasn't been fit yet —
    OOD scoring is an optional add-on layered on top of the classifier,
    and callers should fall back to confidence-only behavior when this
    is None rather than erroring, so the app still works before you've
    run fit_ood.py.
    """
    ood_path = os.path.splitext(model_path)[0] + "_ood.npz"
    if not os.path.exists(ood_path):
        return None
    return MahalanobisOOD.load(ood_path)


def preprocess_image(image_path: str, img_size: int = 224):
    norm = get_transform_normalization()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension


def get_top_k(probs, class_names, k: int = 3):
    """Return the top-k (class_name, probability) pairs, sorted highest first."""
    k = min(k, len(class_names))
    top_vals, top_idxs = torch.topk(probs, k)
    return [
        (class_names[idx], float(val))
        for val, idx in zip(top_vals.tolist(), top_idxs.tolist())
    ]


def predict(image_path: str, model_path: str, top_k: int = 3, clip_gate=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(model_path, device)
    ood_detector = load_ood_detector(model_path)

    pil_image = Image.open(image_path).convert("RGB")

    # Layer 1: CLIP semantic "is this even waste" gate. Runs on the raw
    # image (CLIP has its own preprocessing) and is independent of the
    # waste classifier entirely — see waste_gate.py.
    is_waste, waste_gate_prob = None, None
    if clip_gate is not None:
        is_waste, waste_gate_prob = clip_gate.is_waste(pil_image)

    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]

        # Layer 2: Mahalanobis feature-space OOD check.
        ood_score, is_ood, ood_threshold = None, None, None
        if ood_detector is not None:
            feats = extract_penultimate_features(model, input_tensor).cpu().numpy()
            ood_score = float(ood_detector.score(feats)[0])
            is_ood = bool(ood_detector.is_ood(feats)[0])
            ood_threshold = float(ood_detector.threshold)

    top_idx = int(torch.argmax(probs).item())
    predicted_class = class_names[top_idx]
    confidence = float(probs[top_idx].item())

    guidance = get_guidance(predicted_class)
    icon = get_icon(predicted_class)

    all_probs = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    top_predictions = get_top_k(probs, class_names, k=top_k)

    return {
        "predicted_class": predicted_class,
        "icon": icon,
        "confidence": confidence,
        "guidance": guidance,
        "all_probabilities": all_probs,
        "top_predictions": top_predictions,
        # None when fit_ood.py hasn't been run yet for this model.
        "ood_score": ood_score,
        "is_ood": is_ood,
        "ood_threshold": ood_threshold,
        # None when the CLIP gate isn't installed/loaded.
        "is_waste": is_waste,
        "waste_gate_probability": waste_gate_prob,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/waste_classifier.pth")
    parser.add_argument(
        "--no_clip_gate", action="store_true",
        help="Skip the CLIP semantic waste gate (faster, no model download).",
    )
    args = parser.parse_args()

    clip_gate = None if args.no_clip_gate else try_load_clip_gate()
    result = predict(args.image, args.model, clip_gate=clip_gate)

    print(f"\n{result['icon']} Predicted category: {result['predicted_class']} "
          f"({result['confidence']*100:.1f}% confidence)")

    if result["is_waste"] is None:
        print("(CLIP semantic gate not active — pass no flag / install "
              "open-clip-torch, or use --no_clip_gate to silence this.)")
    elif not result["is_waste"]:
        print(f"🚫 NOT WASTE: CLIP gives this only {result['waste_gate_probability']*100:.0f}% "
              f"waste-likelihood (below the {WASTE_PROBABILITY_THRESHOLD*100:.0f}% gate) — "
              "the category prediction below should not be trusted.")
    else:
        print(f"CLIP waste gate passed ({result['waste_gate_probability']*100:.0f}% "
              "waste-likelihood).")

    if result["ood_score"] is None:
        print("(No OOD detector fit yet for this model — run src/fit_ood.py "
              "to enable feature-space anomaly checking.)")
    elif result["is_ood"]:
        print(f"🚧 OOD FLAG: distance {result['ood_score']:.1f} exceeds "
              f"threshold {result['ood_threshold']:.1f} — this doesn't look "
              "like any known class in feature space, regardless of the "
              "softmax confidence above.")
    else:
        print(f"OOD check passed (distance {result['ood_score']:.1f} <= "
              f"threshold {result['ood_threshold']:.1f}).")

    print(f"Bin: {result['guidance']['bin']}")
    print(f"Tip: {result['guidance']['tip']}")
    print(f"Impact: {result['guidance']['impact']}")

    print("\nAI Analysis — Top 3 predictions:")
    for cls, prob in result["top_predictions"]:
        print(f"  {get_icon(cls)} {cls:12s} {prob*100:5.1f}%")

