"""
Grad-CAM (Gradient-weighted Class Activation Mapping) — visual
explainability for the CNN's predictions.

Shows *which part of the image* the model focused on to make its
decision, overlaid as a heatmap. This is a genuine explainable-AI (XAI)
technique — not a decoration — and is a strong talking point for the
"AI concept" and "advantages" sections of the presentation: it lets you
show judges *why* the model made a call, not just *what* it predicted.

Usage (standalone):
    python src/gradcam.py --image path.jpg --model models/waste_classifier.pth --out gradcam_output.jpg
"""

import argparse

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from predict import load_model, preprocess_image


def generate_gradcam(model, input_tensor, class_idx: int = None, target_layer=None):
    """
    Compute a Grad-CAM heatmap for `class_idx` (or the predicted class if
    None) with respect to `target_layer` (defaults to the last conv block
    of MobileNetV2's feature extractor — the standard choice for CNN
    Grad-CAM, since it's the last layer that still has spatial structure).

    Returns: (heatmap: np.ndarray [H, W] in [0, 1], class_idx: int)
    """
    if target_layer is None:
        target_layer = model.features[-1]

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Grad-CAM needs gradients to flow back to this layer even when the
    # backbone is frozen (requires_grad=False on its weights) — requiring
    # grad on the *input* is what makes that possible: it forces autograd
    # to build a graph through every layer downstream, frozen or not.
    input_tensor = input_tensor.clone().requires_grad_(True)

    model.zero_grad()
    output = model(input_tensor)
    if class_idx is None:
        class_idx = int(output.argmax(dim=1).item())

    score = output[0, class_idx]
    score.backward()

    h1.remove()
    h2.remove()

    acts = activations["value"][0]       # [C, H, W]
    grads = gradients["value"][0]        # [C, H, W]
    weights = grads.mean(dim=(1, 2))     # [C] — global-average-pooled gradients

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    return cam.detach().cpu().numpy(), class_idx


def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Resize `heatmap` to match `original_image` and alpha-blend a jet
    colormap version on top of it (red/yellow = high attention)."""
    img = original_image.convert("RGB")
    w, h = img.size

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), resample=Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img) / 255.0

    colored = cm.jet(heatmap_arr)[:, :, :3]  # drop alpha channel from colormap
    colored = np.uint8(colored * 255)

    original_arr = np.array(img).astype(np.float32)
    blended = (1 - alpha) * original_arr + alpha * colored.astype(np.float32)
    blended = np.uint8(np.clip(blended, 0, 255))

    return Image.fromarray(blended)


def explain_prediction(image_path: str, model_path: str):
    """Convenience wrapper: load model, run Grad-CAM, return the heatmap
    overlay image and the predicted class name."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(model_path, device)

    original = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_image(image_path).to(device)

    heatmap, class_idx = generate_gradcam(model, input_tensor)
    overlay = overlay_heatmap(original.resize((224, 224)), heatmap)

    return overlay, class_names[class_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/waste_classifier.pth")
    parser.add_argument("--out", type=str, default="gradcam_output.jpg")
    args = parser.parse_args()

    overlay, predicted_class = explain_prediction(args.image, args.model)
    overlay.save(args.out)
    print(f"Predicted: {predicted_class}")
    print(f"Grad-CAM overlay saved to: {args.out}")
