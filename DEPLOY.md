# Deploying EcoSort AI — GitHub + Streamlit Community Cloud

Follow these **in order**. Steps 1–2 need Colab (GPU); steps 3+ are on your
own machine/terminal.

## 0. Prerequisite: you need a trained model committed to the repo
Streamlit Community Cloud only runs your code — it does not train anything
for you. If `models/waste_classifier.pth` isn't in the repo you push, the
app will show a clean "no model found" error instead of crashing, but it
also won't classify anything. So:

## 1. Train in Colab (if you haven't already)
Open `Smart_Waste_Classifier_Colab.ipynb` in Google Colab, Runtime → Change
runtime type → GPU, run all cells. This produces `models/waste_classifier.pth`.

## 2. Fit the OOD detector (do this — it's the difference between "guesses
   confidently on garbage input" and "correctly says I don't know")
Still in Colab (same session, model already trained):
```bash
python src/fit_ood.py --data_dir data --model models/waste_classifier.pth
```
This produces `models/waste_classifier_ood.npz`. Download both files from
Colab's file browser into your local project's `models/` folder.

## 3. Push everything to GitHub
From the project folder on your machine (with `models/waste_classifier.pth`
and `models/waste_classifier_ood.npz` now sitting in `models/`):

```bash
git init
git remote add origin https://github.com/SnehaPillai123/Ecosort_AI.git
git add .
git commit -m "EcoSort AI: fixed accuracy gating, Streamlit Cloud deploy config, restyled UI"
git branch -M main
git push -u origin main
```

If the repo already has commits (README-only, etc.), pull first to avoid a
rejected push:
```bash
git pull origin main --allow-unrelated-histories
```
then resolve any conflict Git flags and re-run the `push` line.

> Both model files together are typically well under 100MB, so plain git is
> fine — no Git LFS needed. If `git push` complains about a file over 100MB,
> that's `waste_classifier.pth`; let me know and I'll show you the Git LFS
> steps instead.

## 4. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io → **New app**
2. Pick the `Ecosort_AI` repo, branch `main`
3. **Main file path:** `app/app.py`
4. Click **Deploy**

First deploy takes a few minutes — it's installing PyTorch + CLIP and
downloading CLIP's pretrained weights (~350MB, one-time, cached after).

## 5. Confirm all 3 accuracy layers are live
Open the deployed app → **Classify** page. Right under the checkbox you'll
see a banner:
- 🟢 **green "all 3 accuracy layers active"** → you're good, this is the
  state that correctly rejects screenshots/non-waste photos and gives
  reliable confidence on real waste photos.
- 🔴 **red "degraded accuracy mode"** → see below.

## Fixing degraded accuracy mode
The banner tells you exactly which layer is missing:
- **OOD not fit** → you skipped step 2, or `waste_classifier_ood.npz` isn't
  in `models/` in the repo you pushed. Re-check step 2 and 3.
- **CLIP not available** → `open-clip-torch` failed to install or its
  weights failed to download. Check the app's deploy logs (Manage app →
  logs) for the actual pip/download error. Common cause: the free tier's
  first-boot timeout — just reboot the app once from the Streamlit Cloud
  dashboard; the CLIP weights download is cached after the first success.

## A resource note, honestly
MobileNetV2 + CLIP together use meaningfully more RAM than a typical
Streamlit Cloud free-tier app (~700MB–1GB). If the app crashes or restarts
under load rather than showing the degraded banner, that's the free tier's
1GB memory ceiling, not a code bug. Two ways out if you hit this:
1. In Streamlit Cloud → your app → Settings, request the resource bump
   Streamlit occasionally offers for student/hackathon projects, or
2. Move hosting to Hugging Face Spaces (free CPU Basic tier gives more
   headroom) — say the word and I'll give you those steps too.
