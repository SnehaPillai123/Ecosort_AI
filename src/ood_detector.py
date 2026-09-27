"""
Out-of-distribution (OOD) detection for the Smart Waste Classifier.

WHY THIS EXISTS
----------------
disposal_guide.confidence_level() gates on max-softmax confidence, and
that is a useful sanity check, but it is NOT true OOD detection: softmax
only ever answers "which of my N known classes does this look closest
to?" A closed-set classifier has no built-in notion of "none of the
above." A photo of a face, a screenshot, or a blank wall can still land
above the confidence floor purely by chance if it happens to sit near
one class's decision boundary in output space — the model was never
given the option to say "I don't know what this is at all."

THE FIX: MAHALANOBIS-DISTANCE OOD (Lee et al., NeurIPS 2018)
--------------------------------------------------------------
"A Simple Unified Framework for Detecting Out-of-Distribution Samples"
proposes scoring inputs not by the classifier's output layer, but by
their distance, in the network's penultimate FEATURE space, to each
class's fitted Gaussian:

  1. fit_ood.py extracts the 256-dim penultimate-layer embedding (the
     input to the model's final Linear layer) for every training image.
  2. It fits one mean vector per class, plus a single shared covariance
     matrix pooled across all classes (Ledoit-Wolf shrinkage, since raw
     empirical covariance is unstable with only a few hundred images
     per class relative to 256 dimensions).
  3. It calibrates a distance threshold from the validation set: the
     Nth percentile of in-distribution validation distances (default
     99th), so the false-positive rate on genuine photos is an
     explicit, tunable knob rather than a guess.
  4. At inference, a new image's Mahalanobis distance to its NEAREST
     class mean is its OOD score. Distance beyond the threshold means
     "this doesn't look like any known class in feature space" —
     independent of what the softmax layer says.

This catches cases pure confidence-thresholding misses: an image that
happens to score high softmax confidence for "plastic" but whose actual
features sit far outside the plastic training cluster (and every other
cluster) gets flagged here even though nothing in the softmax output
looked wrong.

HONEST LIMITS
-------------
This is a real, well-established OOD method — not a threshold on the
same softmax signal — but it is still not a hard guarantee:
- It assumes the training set is representative enough of each class
  that a Gaussian in feature space is a reasonable model of "normal."
- A truly adversarial or highly unusual-but-genuine photo can still
  land inside the threshold.
- Like the confidence floor, the distance threshold trades off false
  "OOD" flags on real waste photos against missed non-waste inputs;
  args.threshold_percentile in fit_ood.py controls where that line sits.
No detector of this kind gives a strict 0%/100% separation on every
possible input — but this is a genuine feature-space anomaly check,
not a rebrand of the existing confidence gate.
"""

import numpy as np
import torch
import torch.nn as nn


def extract_penultimate_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Run the MobileNetV2 forward pass but stop one layer early, returning
    the 256-dim embedding that feeds the final classification Linear —
    the feature space the Mahalanobis detector operates in.

    Mirrors torchvision's MobileNetV2.forward() (features -> avgpool ->
    flatten -> classifier) up to, but not including, model.classifier[-1]
    (the num_classes-wide Linear). Matches model.py's classifier layout:
    Sequential(Dropout, Linear(in,256), ReLU, Dropout, Linear(256,num_classes)).
    """
    feats = model.features(x)
    feats = nn.functional.adaptive_avg_pool2d(feats, 1)
    feats = torch.flatten(feats, 1)
    feats = model.classifier[0](feats)  # Dropout (no-op in eval mode)
    feats = model.classifier[1](feats)  # Linear: backbone_dim -> 256
    feats = model.classifier[2](feats)  # ReLU
    return feats  # (batch, 256)


class MahalanobisOOD:
    """Fitted per-class means + a shared inverse covariance, for scoring
    new samples' feature-space distance to the nearest known class."""

    def __init__(self, class_means: np.ndarray, inv_covariance: np.ndarray, threshold: float):
        self.class_means = class_means        # (num_classes, feat_dim)
        self.inv_covariance = inv_covariance   # (feat_dim, feat_dim)
        self.threshold = threshold

    @classmethod
    def load(cls, path: str) -> "MahalanobisOOD":
        data = np.load(path)
        return cls(data["class_means"], data["inv_covariance"], float(data["threshold"]))

    def save(self, path: str):
        np.savez(
            path,
            class_means=self.class_means,
            inv_covariance=self.inv_covariance,
            threshold=self.threshold,
        )

    def distances(self, features: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance from each row of `features` (batch,
        feat_dim) to every class mean. Returns (batch, num_classes)."""
        diffs = features[:, None, :] - self.class_means[None, :, :]        # (B, C, D)
        left = np.einsum("bcd,de->bce", diffs, self.inv_covariance)
        return np.einsum("bce,bce->bc", left, diffs)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Per-sample OOD score = distance to the nearest class mean.
        Lower = looks like a known class, higher = anomalous."""
        return self.distances(features).min(axis=1)

    def is_ood(self, features: np.ndarray) -> np.ndarray:
        return self.score(features) > self.threshold
