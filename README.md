> **📦 This is the final submission package.** Quick orientation:
> - `Smart_Waste_Classifier_Colab.ipynb` — run this top-to-bottom in Google Colab (GPU) to train the model and launch the live demo. Real results from our actual run: **95.9% validation accuracy across 10 classes** (see Section 5).
> - `Smart_Waste_Classifier_Presentation.pptx` — competition slide deck, already filled in with real accuracy numbers, confusion matrix, and misclassified-examples grid from our training run.
> - `EcoSort_AI_Report.docx` — the formal written project report.
> - `app/`, `src/` — full source code for the classifier (MobileNetV2 + Grad-CAM + OOD detection) and the Streamlit web app (community hub, analytics, impact map, route planner, etc.).
> - `requirements.txt` — install with `pip install -r requirements.txt` to run the app locally once you have a trained `models/waste_classifier.pth` (from the notebook).
>
> **Before presenting:** open the notebook in Colab, run all cells (Runtime → GPU), and launch Section 7's live demo — that gives you a real, working app to demonstrate instead of slides alone.

# 🌱 EcoSort AI — "No Green Deed Is Too Small."

EcoSort AI uses AI to turn everyday environmental actions into
measurable community impact. It classifies waste items from photos,
explains its own reasoning (Grad-CAM), and closes the loop all the way
to real-world cleanup: report → organize → before/after proof → reward.

**The golden path this project demonstrates:**
Student sees waste → AI identifies it → student takes action/reports it
→ earns Green Points → community gets involved → a cleanup is organized
→ before/after proof is uploaded → participants are rewarded → the
result appears on the Impact Dashboard.

---

## 1. Problem Identified
Manual waste segregation is slow, inconsistent, and error-prone, leading to
recyclable contamination and increased landfill waste. An automated visual
classifier can speed up and standardize sorting at the point of disposal.

## 2. AI Concept / Technique Used
- **Computer Vision** using a **Convolutional Neural Network (CNN)**
- **Transfer Learning** on a pretrained **MobileNetV2** backbone
  (fast, lightweight — good for a short project timeline and for
  eventual deployment on low-power devices like smart bins)

## 3. Input / Data Required
- Images of waste items (single item per image, various angles/lighting)
- Recommended dataset: **Kaggle Garbage Classification** (12 classes,
  consolidated to `plastic`, `paper`, `metal`, `organic`, `glass`,
  `cardboard`, `trash` + bonus `battery`/`clothes`/`shoes`) — this
  matches the competition brief's "plastic, paper, metal, organic
  waste, etc." exactly. **TrashNet** (6 classes, no `organic`) is a
  fallback if Kaggle access isn't available. Full details in
  `data/README.md`.

## 4. Working of the System
1. Image is uploaded/captured
2. Preprocessing: resize to 224x224, normalize
3. MobileNetV2 (pretrained on ImageNet) extracts visual features
4. A custom classification head predicts the waste category
5. A disposal-guidance module maps the predicted category to
   real-world disposal instructions

## 5. Output / Result
- Predicted category (e.g., "Plastic")
- Confidence score
- Disposal guidance text (which bin, any special handling notes)

## 6. Real-Life Application
- Smart recycling bins in campuses/malls/offices
- Municipal waste-sorting assistance apps
- Educational tool for schools teaching recycling habits

## 7. Advantages & Limitations
**Advantages:** fast, consistent, scalable, reduces contamination,
usable on a phone camera.
**Limitations:** accuracy depends on image quality/lighting; struggles
with mixed/overlapping items in one frame; limited to categories seen
in training data. A closed-set classifier also has no built-in notion
of "none of the above" — see the OOD detection note below for how this
project addresses (and doesn't fully solve) that.

**Out-of-distribution (OOD) detection:** early versions of this project
only gated on softmax confidence — a real check, but not true OOD
detection, since a completely unrelated image can still score high
softmax confidence for some class purely by chance. This project now
layers three independent checks, from most general to most specific:

1. **Semantic waste gate (`src/waste_gate.py`)** — CLIP, a
   general-purpose vision-language model pretrained on ~400M web
   image-caption pairs, zero-shot-scores the photo against "does this
   look like waste/trash/recyclables" vs. common non-waste subjects
   (faces, screenshots, documents, animals, landscapes, etc.). This is
   the strongest check precisely because CLIP's notion of "waste" comes
   from broad web-scale pretraining, not from this project's ~11
   classes — a photo of a face won't get grouped in with "cardboard"
   just because it shares some texture statistics. Optional dependency
   (`open-clip-torch`); the app still works without it.
2. **Feature-space OOD check (`src/ood_detector.py` + `src/fit_ood.py`,
   Lee et al., NeurIPS 2018)** — a Mahalanobis-distance check inside the
   waste classifier's own learned feature space, catching inputs that
   don't resemble any specific trained category's cluster. Run
   `python src/fit_ood.py` once after training to enable it.
3. **Softmax confidence floor (`disposal_guide.confidence_level`)** —
   the original check: how sure the classifier is about which specific
   category, given the input passed checks 1 and 2.

None of these three checks is a hard guarantee on its own — see the
"Honest limits" sections at the top of `src/waste_gate.py` and
`src/ood_detector.py` — but layering a broad, general-purpose semantic
check ahead of the narrower classifier-specific ones meaningfully
closes the gap the original confidence-only approach left open.

## 8. Future Scope
- Multi-item detection (YOLO) for real bin photos with several items
- Real-time video classification on a conveyor belt
- IoT-integrated smart bin with automatic sorting mechanism
- Mobile app deployment (TensorFlow Lite / ONNX)
- A learned/deep OOD detector (e.g. an autoencoder trained specifically
  to flag non-waste inputs) instead of the current Mahalanobis-on-frozen-
  features approach, if more labeled non-waste examples become available

---

## Project Structure
```
smart-waste-classifier/
├── data/                # dataset goes here (see data/README.md)
├── models/              # trained model weights saved here
├── uploads/issues/      # photos attached to community issue reports
├── community.db         # shared SQLite database (created on first run)
├── src/
│   ├── dataset.py       # data loading & augmentation
│   ├── model.py         # MobileNetV2 model definition
│   ├── train.py         # training script (class-weighted loss)
│   ├── predict.py       # single-image inference + top-k predictions
│   ├── evaluate.py      # confusion matrix, per-class metrics
│   ├── gradcam.py       # Grad-CAM explainability (attention heatmaps)
│   ├── ood_detector.py  # Mahalanobis feature-space OOD detection
│   ├── fit_ood.py       # fits the OOD detector after training (run once)
│   ├── waste_gate.py    # CLIP zero-shot "is this even waste" gate
│   └── disposal_guide.py # category -> bin/tip/impact/points lookup
├── app/
│   ├── app.py           # main Streamlit app (classify, batch, analytics)
│   ├── db.py            # shared SQLite persistence layer
│   └── community_hub.py # issue reports, cleanup events, marketplace, rewards
├── requirements.txt
└── README.md
```

## Features
- **Classification:** CNN (MobileNetV2 transfer learning), class-weighted
  loss to handle imbalanced categories
- **Explainable AI:** Grad-CAM heatmaps showing what the model looked at
- **Decision support:** disposal bin + tip + environmental impact + a
  "Green Points" score per category, with a low-confidence warning
- **Batch upload:** classify multiple items at once
- **Shared analytics & leaderboard:** backed by SQLite — real, not
  per-browser-session — so it reflects everyone using the same running app
- **Community Hub:** issue reporting (with photo), cleanup events with
  RSVP, an eco-swap marketplace, and a rewards store to redeem points
- **Golden path:** issue reports link directly to cleanup events; marking
  a cleanup complete requires an after-photo and auto-rewards every
  participant — the full loop shows up on the Impact Dashboard
- **Impact Map:** issues and cleanups geocoded via OpenStreetMap
  (Nominatim, free/keyless) and plotted live
- **Weather-aware planning:** Open-Meteo (free/keyless) shows rain risk
  for a cleanup's date right on the event card, to help pick a good day

## Troubleshooting

**"I turned on OOD/CLIP but nothing changed"** — check the status line
right under the app title (`✅`/`⬜` for each of the 3 layers). Common
causes:
- Feature-space OOD shows `⬜`: you must run `python src/fit_ood.py
  --data_dir data --model models/waste_classifier.pth` *after*
  training, in the same environment the app reads `models/` from. It
  must produce `models/waste_classifier_ood.npz` — the app looks for
  that exact filename next to your `.pth`.
- CLIP semantic gate shows `⬜`: run `pip install open-clip-torch`,
  then **fully restart** the Streamlit process (kill it and re-run
  `streamlit run app/app.py`, don't just refresh the browser or rerun
  one cell) — the model loads once at startup via `@st.cache_resource`.

**"My CSS/UI edits aren't showing up"** — in Colab, editing
`app/theme.py` (or any file) only takes effect after you re-run the
`%%writefile ...` cell (to actually overwrite the file on disk) *and*
restart the Streamlit + ngrok cell. A browser refresh alone reloads the
page, not the Python process serving it.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset
See `data/README.md`. Arrange images like:
```
data/train/plastic/*.jpg
data/train/paper/*.jpg
data/train/metal/*.jpg
data/train/glass/*.jpg
data/train/organic/*.jpg
data/train/trash/*.jpg
```
(the folder names become your class labels automatically)

### 3. Train the model
```bash
python src/train.py --data_dir data --epochs 15 --out models/waste_classifier.pth
```

### 4. Run a single prediction (sanity check)
```bash
python src/predict.py --image path/to/test.jpg --model models/waste_classifier.pth
```

### 5. Fit the OOD detector (recommended, run once after training)
```bash
python src/fit_ood.py --data_dir data --model models/waste_classifier.pth
```
Adds real feature-space anomaly detection on top of the classifier (see
section 7 above). Optional — everything still works without this step,
just with confidence-threshold gating only.

### 6. Launch the demo app
```bash
streamlit run app/app.py
```
On first run, if `open-clip-torch` is installed, this downloads the
CLIP model (~350MB, one-time, needs internet) to enable the semantic
waste gate. If that download fails or the package isn't installed, the
app falls back automatically to the OOD + confidence checks only.

## Recommended: Train on Google Colab (free GPU)
This sandbox has no internet/GPU. Upload this folder + dataset to Colab,
change runtime to GPU, then run the same commands above — training
15 epochs on ~2,500 images takes a few minutes on a T4 GPU.
