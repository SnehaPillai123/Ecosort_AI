Put your trained files here before deploying:
    waste_classifier.pth        (from src/train.py)
    waste_classifier_ood.npz    (from src/fit_ood.py, optional but recommended)

Both are small enough (a few MB–tens of MB) to commit directly to git —
no Git LFS needed for a MobileNetV2 head.
