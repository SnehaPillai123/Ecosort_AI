# Dataset Setup

The competition brief specifies: **"Identify plastic, paper, metal,
organic waste, etc."** — so **Option A (Kaggle) is the recommended
primary choice**, since it's the only one with an organic/biological
class out of the box.

## Option A — Kaggle Garbage Classification (12 classes, recommended)
https://www.kaggle.com/datasets/mostafaabla/garbage-classification
Classes: battery, biological, brown-glass, cardboard, clothes,
green-glass, metal, paper, plastic, shoes, trash, white-glass

Consolidate down to match the brief before training:
- `biological` → `organic`
- `brown-glass` / `green-glass` / `white-glass` → `glass`
- everything else keeps its name (plastic, paper, metal, cardboard,
  trash, battery, clothes, shoes stay as bonus "etc." categories)

The Colab notebook (`Smart_Waste_Classifier_Colab.ipynb`) does this
consolidation automatically — see its Section 2.

## Option B — TrashNet (fallback, 6 classes, ~2,500 images, no login needed)
GitHub: https://github.com/garythung/trashnet
Classes: cardboard, glass, metal, paper, plastic, trash
(No `organic` class — only use this if Kaggle access isn't working.)

## Folder structure required by `src/dataset.py`
Split each dataset into `train/` and `val/` (80/20 is fine), one
sub-folder per class:

```
data/
├── train/
│   ├── plastic/
│   ├── paper/
│   ├── metal/
│   ├── glass/
│   ├── organic/     (map "biological" -> "organic" if using Kaggle set)
│   └── trash/
└── val/
    ├── plastic/
    ├── paper/
    ├── metal/
    ├── glass/
    ├── organic/
    └── trash/
```

## Quick split script (once you've downloaded the raw dataset)
```python
import os, shutil, random

random.seed(42)
SRC = "raw_dataset"          # folder with one subfolder per class
DST_TRAIN = "data/train"
DST_VAL = "data/val"
VAL_RATIO = 0.2

for cls in os.listdir(SRC):
    files = os.listdir(os.path.join(SRC, cls))
    random.shuffle(files)
    n_val = int(len(files) * VAL_RATIO)
    val_files, train_files = files[:n_val], files[n_val:]

    os.makedirs(os.path.join(DST_TRAIN, cls), exist_ok=True)
    os.makedirs(os.path.join(DST_VAL, cls), exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(SRC, cls, f), os.path.join(DST_TRAIN, cls, f))
    for f in val_files:
        shutil.copy(os.path.join(SRC, cls, f), os.path.join(DST_VAL, cls, f))

print("Done splitting dataset.")
```

Tip: if you only have limited time, TrashNet (Option A) is smaller and
faster to train on and is plenty for a strong class project.
