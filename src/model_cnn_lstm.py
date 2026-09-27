"""
OPTIONAL / EXPERIMENTAL — CNN-LSTM hybrid architecture, per Lilhore,
Simaiya, Dalal & Damasevicius (2023), "A smart waste classification
model using hybrid CNN-LSTM with transfer learning for sustainable
environment" (Multimedia Tools and Applications).

WHY THIS FILE IS SEPARATE FROM model.py, AND HONEST ABOUT ITS STATUS:
    - This has NOT been trained or tested in this environment — there's
      no GPU/torch available in this sandbox to verify it actually
      trains correctly, let alone that it beats the deployed MobileNetV2
      model. Treat this as a literature-grounded EXPERIMENT for your
      report's comparison table (several of the papers you've been
      reading do exactly this — compare 3-5 architectures), not a
      drop-in replacement.
    - Do not point app/predict.py or app/app.py at this model until
      you've (a) actually trained it, (b) confirmed its accuracy on
      YOUR dataset beats the deployed model, and (c) re-verified
      gradcam.py still produces sensible heatmaps against it (it
      *should* work unchanged, since the target layer — the last conv
      block before the LSTM — is still a normal spatial feature map,
      but "should" isn't "verified").

ARCHITECTURE NOTE: LSTMs are built for sequential data, and a single
image isn't naturally a sequence. This paper's approach (and the one
implemented here) treats each spatial location of the CNN's final
feature map as one step in a sequence, letting the LSTM model
dependencies *across the image's spatial layout* — e.g. "top-left
region relates to bottom-right region" — after the CNN has already
extracted local features. It's a legitimate technique in the
literature, but a genuinely CNN-only model already handles most of
that spatial information well; this is a research comparison to run,
not an obviously-correct upgrade.
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True,
                 lstm_hidden: int = 128, lstm_layers: int = 1):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        backbone = models.mobilenet_v2(weights=weights)

        # Keep only the convolutional feature extractor (drop the
        # classifier) — same as model.py, so this stays a fair
        # apples-to-apples comparison against the deployed model.
        self.features = backbone.features  # -> (B, 1280, 7, 7) for 224x224 input

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.feature_channels = 1280  # MobileNetV2's final channel count

        # Each of the 7x7=49 spatial locations becomes one timestep,
        # with the 1280 channels at that location as its feature vector.
        self.lstm = nn.LSTM(
            input_size=self.feature_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(lstm_hidden * 2, 256),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                          # (B, C, H, W)
        B, C, H, W = feat.shape
        seq = feat.flatten(2).permute(0, 2, 1)            # (B, H*W, C) — spatial locations as timesteps

        lstm_out, (h_n, _) = self.lstm(seq)
        # concatenate final forward + backward hidden states
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, lstm_hidden*2)

        return self.classifier(final_hidden)


def build_cnn_lstm_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    return CNNLSTMClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    # sanity check that the model builds and runs a forward pass —
    # run this yourself once torch/GPU is available; it hasn't been
    # run in this sandbox.
    m = build_cnn_lstm_model(num_classes=7)
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print("Output shape:", out.shape)  # expect [2, 7]
