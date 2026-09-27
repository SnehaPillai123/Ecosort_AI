"""
Semantic "is this actually waste?" gate, using CLIP zero-shot scoring.

WHY THIS EXISTS
----------------
ood_detector.py's Mahalanobis check asks "does this image's feature
vector, inside THIS classifier's own feature space (learned only from
the ~11 waste categories it was trained on), resemble any class I've
seen?" That's a real signal, but it's a little circular: that feature
space was never trained to represent "waste-ness" as a general concept
— it just groups images by whatever visual patterns separate plastic
from paper from metal. Something genuinely novel (a human face, a
screenshot, a landscape photo) that happens to share low-level visual
statistics with one waste class (color, texture, shape) can still land
inside that class's cluster and slip past a feature-space check.

CLIP (Radford et al., 2021) is a general-purpose vision-language model
trained on ~400M image-caption pairs from the open web. It has an
actual learned notion of "what does a photo of trash/recyclable
material look like" versus "what does a photo of a person / a
screenshot / an animal / a landscape look like" — because it saw
millions of examples of all of those during pretraining, not just this
project's waste classes. Using it zero-shot (no extra training, no
extra data collection, no fitting step) gives a much stronger, more
general "is this even waste" gate than anything built purely from the
waste classifier's own narrow features.

THE FULL PIPELINE (see predict.py / app.py)
--------------------------------------------
  1. CLIP semantic gate    - "is this even a photo of waste/trash at all?"
                              (this module)
  2. Mahalanobis OOD check - "does it match a KNOWN waste category's
                              specific feature cluster?" (ood_detector.py)
  3. Softmax confidence    - "how sure is the classifier about which
                              specific category, given it passed 1 & 2?"
                              (disposal_guide.confidence_level)
Three independent layers, each catching a different failure mode.

HOW IT WORKS
------------
Embed the input image and a small set of candidate text prompts (some
describing waste/trash/recyclables, some describing common non-waste
photo subjects) with CLIP's image/text encoders, and take a softmax
over the cosine similarities (CLIP's standard zero-shot recipe). If the
probability mass on the waste-related prompts falls below
WASTE_PROBABILITY_THRESHOLD, the image is flagged before the waste
classifier's output is trusted — independent of both the classifier's
softmax confidence and the Mahalanobis check.

HONEST LIMITS
-------------
- CLIP is powerful but not perfect: ambiguous photos (e.g. a bottle
  held in a hand, framed like a product photo) can still confuse it,
  and prompt wording meaningfully affects results — the prompts below
  are a reasonable starting set, not exhaustively tuned against a
  labeled dataset.
- Zero-shot means no waste-specific fine-tuning: a genuine but unusual
  waste item CLIP wasn't trained to associate with "trash" language
  could be (falsely) rejected here even though it IS waste.
- Like the other two layers, this trades precision for recall via a
  threshold — tune WASTE_PROBABILITY_THRESHOLD against your own
  validation photos (including some genuine "hard" waste photos and
  some deliberate non-waste photos) if you see too many false
  positives/negatives in practice.
- Adds a one-time ~350MB model download and a CLIP forward pass to
  every prediction — a real latency/footprint cost for a real
  robustness gain. This is why it's an optional layer: everything
  still works (falling back to layers 2 and 3) if open-clip-torch
  isn't installed or the model fails to load/download.
"""

import torch
import torch.nn.functional as F

WASTE_PROMPTS = [
    "a photo of a piece of trash or garbage",
    "a photo of a recyclable plastic item",
    "a photo of a paper or cardboard item to be recycled",
    "a photo of a metal can or scrap metal",
    "a photo of a glass bottle or jar",
    "a photo of organic food waste or compost",
    "a photo of an old battery or electronic waste",
    "a photo of discarded clothing or shoes",
    "a photo of litter on the ground",
]

NON_WASTE_PROMPTS = [
    "a photo of a person's face",
    "a screenshot of a computer screen or app",
    "a photo of a document or piece of text",
    "a photo of a pet or wild animal",
    "a photo of a landscape, building, or room",
    "a photo of food ready to eat on a plate",
    "a photo of a vehicle",
    "an abstract image, drawing, or meme",
]

# Probability mass on WASTE_PROMPTS (vs. NON_WASTE_PROMPTS) needed to pass
# the gate. Tune this against your own photos — see "Honest limits" above.
WASTE_PROBABILITY_THRESHOLD = 0.5


class ClipWasteGate:
    """
    Loads a CLIP model once and zero-shot-scores images against
    WASTE_PROMPTS + NON_WASTE_PROMPTS to answer "is this even waste?"
    before the waste classifier's output is trusted.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device=None):
        import open_clip  # optional dependency — imported lazily so the

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.prompts = WASTE_PROMPTS + NON_WASTE_PROMPTS
        self.num_waste_prompts = len(WASTE_PROMPTS)
        with torch.no_grad():
            tokens = self.tokenizer(self.prompts).to(self.device)
            text_features = self.model.encode_text(tokens)
            self.text_features = F.normalize(text_features, dim=-1)

    @torch.no_grad()
    def waste_probability(self, image) -> float:
        """
        image: a PIL.Image (RGB), any size — CLIP's own preprocessing
        transform handles resizing/normalization.
        Returns the probability mass CLIP assigns to the waste-related
        prompts (0-1); higher = more confidently "this looks like waste."
        """
        pixel_values = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(pixel_values)
        image_features = F.normalize(image_features, dim=-1)

        # Use the model's own learned logit-scale temperature (its
        # standard zero-shot recipe) rather than a hardcoded constant —
        # this varies slightly across CLIP checkpoints.
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.T
        probs = F.softmax(logits, dim=-1)[0]
        return float(probs[: self.num_waste_prompts].sum().item())

    def is_waste(self, image, threshold: float = WASTE_PROBABILITY_THRESHOLD):
        """Returns (is_waste: bool, waste_probability: float)."""
        p = self.waste_probability(image)
        return p >= threshold, p


def try_load_clip_gate(device=None):
    """
    Best-effort loader: returns a ready ClipWasteGate, or None if
    open-clip-torch isn't installed or the model can't be loaded/
    downloaded (e.g. no internet on first run). Callers should treat
    None as "this optional layer is unavailable" and fall back to the
    Mahalanobis OOD check + softmax confidence, not as an error.
    """
    try:
        return ClipWasteGate(device=device)
    except ImportError:
        print(
            "[waste_gate] open-clip-torch isn't installed — the semantic "
            "waste gate is disabled. Run `pip install open-clip-torch` to "
            "enable it (see requirements.txt)."
        )
        return None
    except Exception as e:  # model download / load failure, no internet, etc.
        print(f"[waste_gate] Couldn't load the CLIP semantic gate ({e}); "
              "continuing without it.")
        return None
