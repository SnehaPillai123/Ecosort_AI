# Evaluation Results

Real output from `src/evaluate.py`, from an actual training run
(15 epochs, MobileNetV2, Kaggle Garbage Classification dataset
consolidated to 10 classes) — not placeholders.

- **`classification_report.txt`** — **95.91% overall validation
  accuracy** across 10 classes, plus per-class precision/recall/F1.
  Weakest classes: `metal` (F1 0.900) and `plastic` (F1 0.908) —
  worth a mention in an honest limitations discussion.
- **`confusion_matrix.png`** — the validation-set confusion matrix
  (2,321 validation images).
- **`misclassified_examples.png`** — a 3×3 grid of the 95 misclassified
  images, most of them `cardboard` predicted as `paper` — an
  understandable confusion given how visually similar flattened
  cardboard and thick paper packaging are.
- **`gradcam_example.png`** — a bonus Grad-CAM heatmap overlay (not
  produced by `evaluate.py`; generated in the notebook's Section 6)
  showing which pixels the model attended to for one prediction.

Regenerate all of the above after retraining with:

```bash
python src/evaluate.py --data_dir data --model models/waste_classifier.pth --out_dir eval_results
```
