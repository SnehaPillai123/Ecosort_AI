"""
EcoSort AI — "No Green Deed Is Too Small."

Uses AI to turn everyday environmental actions into measurable community
impact. Run with:
    streamlit run app/app.py

The golden path this app is built around:
    See waste -> AI identifies it -> take action / report it -> earn
    points -> community gets involved -> cleanup gets organized ->
    before/after proof -> participants rewarded -> result shows on
    the Impact Dashboard.

UI structure: a real multi-page app (st.navigation), not cramped tabs —
Classify, Batch Upload, Analytics & Leaderboard, Community Hub, and
Impact Dashboard each get their own page with a left-sidebar nav.
Requires streamlit>=1.37 (see requirements.txt) for st.navigation/st.Page.

A nickname is requested once per session (not authenticated — see db.py
for the honest scope of what "shared" means here) so activity can be
attributed and the leaderboard means something.
"""

import hashlib
import io
import json
import os
import sys
import time

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Optional: Gemini is only used for object *localization* in the Smart
# Multi-Item Scan page (finding where items are in a messy photo). The
# actual waste classification still runs through our own MobileNetV2
# pipeline (classify(), below) — Gemini never sets the final label. The
# app degrades gracefully to the old grid-scan if this package or the
# API key isn't available (see get_gemini_client()).
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# allow importing from src/ when running `streamlit run app/app.py` from project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))

from predict import load_model, preprocess_image, get_top_k, load_ood_detector  # noqa: E402
from ood_detector import extract_penultimate_features  # noqa: E402
from waste_gate import try_load_clip_gate, WASTE_PROBABILITY_THRESHOLD  # noqa: E402
from disposal_guide import get_guidance, get_icon, confidence_level  # noqa: E402
from gradcam import generate_gradcam, overlay_heatmap  # noqa: E402
from community_hub import render_community_hub, ISSUE_TYPES, REWARDS_CATALOG  # noqa: E402
import db  # noqa: E402
import geo  # noqa: E402
import route_optimizer  # noqa: E402
import theme  # noqa: E402

st.set_page_config(page_title="EcoSort AI", page_icon="🌱", layout="wide")
theme.inject_css()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "waste_classifier.pth")


@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(MODEL_PATH, device)
    # None until you've run `python src/fit_ood.py` for this checkpoint —
    # see src/ood_detector.py. The app works fine without it (falls back
    # to confidence-only gating); fitting it adds real feature-space OOD
    # detection on top.
    ood_detector = load_ood_detector(MODEL_PATH)
    return model, class_names, device, ood_detector


@st.cache_resource
def get_clip_gate():
    """
    None if open-clip-torch isn't installed or the model couldn't be
    downloaded/loaded (e.g. no internet on first run) — see
    src/waste_gate.py. The app works fine without it, falling back to
    the Mahalanobis OOD check + softmax confidence only.
    """
    return try_load_clip_gate()


@st.cache_resource
def get_gemini_client():
    """
    None if the google-genai package isn't installed or GEMINI_API_KEY
    isn't set in Streamlit secrets — Smart Multi-Item Scan then falls
    back to the old grid-classification method instead of crashing.
    """
    if genai is None:
        return None
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


GEMINI_DETECT_PROMPT = """
You are the object-localization layer for a waste-sorting app called
EcoSort AI. Look at this photo and find every distinct physical waste
item visible in it.

For each item return only:
- a short "label" guess (e.g. "plastic bottle", "cardboard box")
- a "bounding_box" as [x_min, y_min, x_max, y_max], normalized 0-1000

Return ONLY valid JSON, no markdown fences, in this exact shape:
{"items": [{"label": "...", "bounding_box": [0, 0, 0, 0]}]}
"""


def detect_items_with_gemini(client, image_bytes: bytes, mime_type: str = "image/jpeg"):
    """
    Asks Gemini only to locate objects (bounding boxes) and offer a rough
    label for each — it does NOT decide the final waste category. Returns
    a list of {"label": str, "bounding_box": [x1,y1,x2,y2]} in 0-1000
    normalized coordinates, or raises on failure (caller handles fallback).
    """
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            GEMINI_DETECT_PROMPT,
        ],
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    items = json.loads(text).get("items", [])
    # Basic sanity filtering — a malformed box shouldn't crash the page.
    clean = []
    for it in items:
        box = it.get("bounding_box")
        if isinstance(box, list) and len(box) == 4:
            clean.append({"label": str(it.get("label", "item")), "bounding_box": box})
    return clean


def classify(image: Image.Image, model, class_names, device, ood_detector=None, clip_gate=None):
    # Layer 1: CLIP semantic "is this even waste" gate — runs on the raw
    # image, independent of the waste classifier entirely.
    is_waste, waste_gate_prob = None, None
    if clip_gate is not None:
        is_waste, waste_gate_prob = clip_gate.is_waste(image)

    tmp_path = f"/tmp/_swc_input_{time.time_ns()}.jpg"
    image.save(tmp_path)
    input_tensor = preprocess_image(tmp_path).to(device)
    os.remove(tmp_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]

        # Layer 2: Mahalanobis feature-space OOD check.
        ood_score, ood_threshold, is_ood = None, None, False
        if ood_detector is not None:
            feats = extract_penultimate_features(model, input_tensor).cpu().numpy()
            ood_score = float(ood_detector.score(feats)[0])
            ood_threshold = float(ood_detector.threshold)
            is_ood = bool(ood_detector.is_ood(feats)[0])

    top_idx = int(torch.argmax(probs).item())
    predicted_class = class_names[top_idx]
    confidence = float(probs[top_idx].item())
    guidance = get_guidance(predicted_class)
    icon = get_icon(predicted_class)
    top3 = get_top_k(probs, class_names, k=3)

    # Layer 3: softmax confidence, as before.
    conf_level = confidence_level(confidence)

    # Layers apply in order of how general/independent they are: CLIP's
    # semantic gate (broadest, entirely separate model) takes priority
    # over the Mahalanobis flag (same classifier's own feature space),
    # which in turn only overrides when softmax alone looked confident.
    if is_waste is False:
        conf_level = {
            "label": "Not recognized as waste",
            "emoji": "🚫",
            "warn": True,
            "unknown": True,
            "message": (
                f"A general-purpose vision model doesn't think this photo "
                f"shows waste, trash, or recyclables at all "
                f"(waste-likelihood {waste_gate_prob*100:.0f}%, below the "
                f"{WASTE_PROBABILITY_THRESHOLD*100:.0f}% gate) — please "
                "take a photo of an actual waste item."
            ),
        }
    elif is_ood and not conf_level.get("unknown"):
        # The softmax confidence alone looked fine, but this image's
        # feature-space distance to every known class is beyond what
        # training data for any class looked like — a genuine "this
        # probably isn't waste at all" signal, not just a low-confidence
        # guess. See src/ood_detector.py for the method.
        conf_level = {
            "label": "Out-of-distribution",
            "emoji": "🚧",
            "warn": True,
            "unknown": True,
            "message": (
                "This doesn't match any of the waste categories the model "
                "was trained on — a feature-space anomaly check flagged it "
                "as unlike anything in the training data, even though the "
                "model's raw confidence score looked high. Please try a "
                "photo of an actual waste item."
            ),
        }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "guidance": guidance,
        "icon": icon,
        "top3": top3,
        "conf_level": conf_level,
        "input_tensor": input_tensor,
        "top_idx": top_idx,
        "ood_score": ood_score,
        "ood_threshold": ood_threshold,
        "is_ood": is_ood,
        "is_waste": is_waste,
        "waste_gate_probability": waste_gate_prob,
    }


def render_result_card(image: Image.Image, result: dict, model, show_gradcam: bool):
    guidance = result["guidance"]
    conf = result["confidence"]
    conf_level = result["conf_level"]

    if conf_level.get("unknown"):
        # Hard gate: below the confidence floor, don't present a guess as
        # an answer. No disposal card, no Green Points — just an honest
        # "couldn't identify this" with a way to see the raw guesses.
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Input image", use_container_width=True)
            with col2:
                if result.get("is_waste") is False:
                    st.markdown("### 🚫 Not Recognized as Waste")
                elif result.get("is_ood"):
                    st.markdown("### 🚧 Flagged as Out-of-Distribution")
                else:
                    st.markdown("### ❓ Unable to Identify")
                st.metric("Confidence", f"{conf*100:.1f}%")
                if result.get("waste_gate_probability") is not None:
                    st.caption(
                        f"🧠 CLIP waste-likelihood: {result['waste_gate_probability']*100:.0f}%"
                    )
                if result.get("ood_score") is not None:
                    st.caption(
                        f"🔍 Feature-space OOD distance: {result['ood_score']:.1f} "
                        f"(flag above {result['ood_threshold']:.1f})"
                    )
                st.error(conf_level["message"])
        with st.expander("See the model's raw (low-confidence) guesses anyway"):
            st.caption("None of these met the confidence floor — shown for transparency only.")
            table_rows = [
                {"Category": f"{get_icon(cls)} {cls.title()}", "Probability": f"{prob*100:.1f}%"}
                for cls, prob in result["top3"]
            ]
            st.table(table_rows)
        return

    with st.container(border=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Input image", use_container_width=True)
            if show_gradcam:
                with st.spinner("🔬 Generating Grad-CAM explanation..."):
                    heatmap, _ = generate_gradcam(model, result["input_tensor"], class_idx=result["top_idx"])
                    overlay = overlay_heatmap(image, heatmap)
                st.image(overlay, caption="Where the AI is looking", use_container_width=True)

        with col2:
            st.markdown(f"### {result['icon']} {result['predicted_class'].title()}")
            st.metric("Confidence", f"{conf*100:.1f}%")
            st.caption(f"{conf_level['emoji']} {conf_level['label']}")
            if conf_level.get("warn"):
                st.warning(conf_level["message"])
            if result.get("waste_gate_probability") is not None:
                st.caption(
                    f"🧠 CLIP waste-likelihood: {result['waste_gate_probability']*100:.0f}%"
                )
            if result.get("ood_score") is not None:
                st.caption(
                    f"🔍 Feature-space OOD distance: {result['ood_score']:.1f} "
                    f"(flag above {result['ood_threshold']:.1f})"
                )

            st.markdown(f"🗑️ **Bin:** {guidance['bin']}")
            st.markdown(f"🍃 **Tip:** {guidance['tip']}")
            st.markdown(f"🌱 **Impact:** {guidance['impact']}")
            if guidance.get("fact"):
                st.caption(f"📊 {guidance['fact']}")
            st.markdown(f"🌿 **+{guidance.get('points', 1)} Green Points** for this item")

    with st.container(border=True):
        st.markdown("#### 🧠 AI Analysis — Top 3 Predictions")
        table_rows = [
            {"Category": f"{get_icon(cls)} {cls.title()}", "Probability": f"{prob*100:.1f}%"}
            for cls, prob in result["top3"]
        ]
        st.table(table_rows)


def log_classification_once(image, user_name, result, uploaded_file, camera_file, session_key="last_classify_signature"):
    """Dedup by file identity (name+size), not content hash, so Streamlit
    reruns (e.g. toggling the Grad-CAM checkbox) don't double-log — but a
    genuinely new photo still logs correctly.

    Also gates on two things before awarding points:
      - conf_level['unknown']: don't log an "unidentifiable" result as a
        real classification at all.
      - an exact image-hash duplicate: still logged (for honest activity
        counts) but with 0 points, so re-uploading the same photo can't
        farm Green Points repeatedly.

    Returns "logged", "duplicate", "unknown", or None (nothing new this rerun).
    """
    if uploaded_file is not None:
        signature = f"upload_{uploaded_file.name}_{uploaded_file.size}"
    elif camera_file is not None:
        signature = f"camera_{camera_file.name}_{camera_file.size}"
    else:
        signature = None

    if not signature or signature == st.session_state.get(session_key):
        return None
    st.session_state[session_key] = signature

    if result["conf_level"].get("unknown"):
        return "unknown"

    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    is_new = db.check_and_register_image_hash(image_hash, user_name)

    if not is_new:
        db.log_activity(
            user_name, "classify", points=0,
            category=result["predicted_class"], confidence=result["confidence"],
        )
        return "duplicate"

    db.log_activity(
        user_name, "classify",
        points=result["guidance"].get("points", 1),
        category=result["predicted_class"],
        confidence=result["confidence"],
    )
    return "logged"


# ---------------------------------------------------------------------
# Sidebar — nickname + profile card (shared across every page)
# ---------------------------------------------------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

with st.sidebar:
    st.markdown("### 🌱 EcoSort AI")
    st.caption("*No Green Deed Is Too Small.*")
    st.write("")

    name_input = st.text_input(
        "👤 Your name",
        value=st.session_state.user_name,
        placeholder="e.g. Aarav",
        help="Tracks your points on the shared leaderboard",
    )
    st.session_state.user_name = name_input.strip()

    if st.session_state.user_name:
        my_points = db.get_user_points(st.session_state.user_name)

        # progress toward the cheapest reward not yet affordable — a small
        # gamification touch that gives the points number a purpose
        next_reward = next((r for r in sorted(REWARDS_CATALOG, key=lambda r: r["cost"]) if r["cost"] > my_points), None)
        progress_label, progress_fraction = None, None
        if next_reward:
            progress_fraction = my_points / next_reward["cost"]
            remaining = next_reward["cost"] - my_points
            progress_label = f"{remaining} 🌿 to unlock {next_reward['icon']} {next_reward['name']}"
        else:
            progress_label = "🏆 You can redeem every reward in the store!"
            progress_fraction = 1.0

        theme.profile_card(st.session_state.user_name, my_points, progress_label, progress_fraction)
    else:
        st.info("👋 Enter a name to start earning Green Points.")

user_name = st.session_state.user_name or "Guest"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Train one first with:\n\n"
        "`python src/train.py --data_dir data --epochs 15 --out models/waste_classifier.pth`"
    )
    st.stop()

model, class_names, device, ood_detector = get_model()
clip_gate = get_clip_gate()


# ---------------------------------------------------------------------
# Page: Classify
# ---------------------------------------------------------------------
def page_classify():
    theme.hero(
        "🌱 EcoSort AI",
        "No Green Deed Is Too Small.",
        "Upload a photo of a waste item — the AI identifies it, checks its own "
        "confidence, shows <em>why</em> it made that call, and gives disposal + "
        "environmental guidance.",
    )

    show_gradcam = st.checkbox("🔬 Show AI explainability (Grad-CAM heatmap)", value=True)
    layers_ok = [ood_detector is not None, clip_gate is not None]
    active_layers = ["✅ Softmax confidence"]
    active_layers.append("✅ Feature-space OOD (Mahalanobis)" if ood_detector else "⬜ Feature-space OOD (not fit — run src/fit_ood.py)")
    active_layers.append("✅ Semantic waste gate (CLIP)" if clip_gate else "⬜ Semantic waste gate (open-clip-torch not available)")
    if all(layers_ok):
        st.success(" · ".join(active_layers) + "  — all 3 accuracy layers active.", icon="🛡️")
    else:
        # This is NOT a small footnote — running with fewer than 3 layers
        # is exactly what produces confident-looking wrong answers on
        # screenshots/mixed-item photos (garbage-in labeled as "Paper 52%"
        # instead of being rejected). Make it impossible to miss.
        st.error(
            "⚠️ **Running in degraded accuracy mode** — " + " · ".join(active_layers) +
            ". Predictions on non-waste or ambiguous images are LESS reliable "
            "until every layer above shows ✅. See DEPLOY.md → "
            "'Fixing degraded accuracy mode'.",
            icon="🚨",
        )
    input_mode = st.radio("Input method", ["📁 Upload Image", "📷 Use Camera"], horizontal=True, label_visibility="collapsed")

    image, uploaded_file, camera_file = None, None, None
    if input_mode == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="single_upload")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")

    if image is None:
        theme.empty_state("📸", "Ready when you are", "Upload an image or take a photo to see the AI in action.")
        return

    result = classify(image, model, class_names, device, ood_detector, clip_gate)
    render_result_card(image, result, model, show_gradcam)

    with st.expander("📊 See full probability breakdown (all categories)"):
        with torch.no_grad():
            probs = F.softmax(model(result["input_tensor"]), dim=1)[0]
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        st.bar_chart(dict(sorted(prob_dict.items(), key=lambda x: -x[1])))

    st.write("")
    with st.expander("🚨 Is this litter in a public place? Report it"):
        if not st.session_state.get("user_name"):
            st.warning("Enter a nickname in the sidebar first so this report can be credited to you.")
        else:
            report_type = st.selectbox(
                "Issue type", ISSUE_TYPES,
                index=ISSUE_TYPES.index("Illegal dumping"),
                key="quick_report_type",
            )
            report_location = st.text_input("Location (area/landmark)", key="quick_report_location")
            report_notes = st.text_area("Additional details (optional)", key="quick_report_notes")
            if st.button("🚨 Submit report using this photo", key="quick_report_submit"):
                if not report_location.strip():
                    st.error("Please enter a location.")
                else:
                    fname = f"{int(time.time())}_quickreport.jpg"
                    fpath = os.path.join(db.UPLOADS_DIR, fname)
                    image.save(fpath)
                    db.add_issue(user_name, report_type, report_location.strip(), report_notes.strip(), fpath)
                    db.log_activity(user_name, "issue", points=5, category=report_type)
                    st.success(
                        "Reported — +5 🌿 Green Points. Head to **Community Hub → "
                        "Cleanup Events** to organize a cleanup for it."
                    )

    log_status = log_classification_once(image, user_name, result, uploaded_file, camera_file)
    if log_status == "duplicate":
        st.info("📎 This exact photo already earned Green Points before — no additional points awarded this time.")


# ---------------------------------------------------------------------
# Page: Batch Upload
# ---------------------------------------------------------------------
def page_batch():
    theme.page_header("📂", "Batch Upload", "Classify several items at once")

    batch_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="batch_upload",
    )

    if not batch_files:
        theme.empty_state("🗂️", "No batch yet", "Upload two or more images to classify them together and see a summary table.")
        return

    results = []
    n_cols = 4
    cols = st.columns(n_cols)

    for i, f in enumerate(batch_files):
        img = Image.open(f).convert("RGB")
        r = classify(img, model, class_names, device, ood_detector, clip_gate)
        # Same exact-hash duplicate guard as the single Classify page — computed
        # here (read-only) so every item in a batch is subject to the same
        # anti-farming check, not just single uploads.
        r["image_hash"] = hashlib.md5(img.tobytes()).hexdigest()
        r["is_duplicate"] = (not r["conf_level"].get("unknown")) and db.image_hash_seen(r["image_hash"])
        results.append({"file": f.name, "image": img, **r})

        with cols[i % n_cols]:
            with st.container(border=True):
                st.image(img, use_container_width=True)
                if r["conf_level"].get("unknown"):
                    st.caption("❓ **Unable to identify**")
                elif r["is_duplicate"]:
                    st.caption(f"{r['icon']} **{r['predicted_class'].title()}** ({r['confidence']*100:.0f}%) · 📎 duplicate")
                else:
                    st.caption(f"{r['icon']} **{r['predicted_class'].title()}** ({r['confidence']*100:.0f}%)")

    st.write("")
    st.markdown("#### 📋 Batch Summary")
    df = pd.DataFrame([
        {
            "File": res["file"],
            "Category": "❓ Unable to identify" if res["conf_level"].get("unknown") else f"{res['icon']} {res['predicted_class'].title()}",
            "Confidence": f"{res['confidence']*100:.1f}%",
            "Bin": "—" if res["conf_level"].get("unknown") else res["guidance"]["bin"],
            "Green Points": 0 if (res["conf_level"].get("unknown") or res["is_duplicate"]) else res["guidance"].get("points", 1),
            "Duplicate?": "📎 yes" if res["is_duplicate"] else "—",
        }
        for res in results
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    total_points = sum(
        0 if (res["conf_level"].get("unknown") or res["is_duplicate"]) else res["guidance"].get("points", 1)
        for res in results
    )
    unknown_count = sum(1 for res in results if res["conf_level"].get("unknown"))
    duplicate_count = sum(1 for res in results if res["is_duplicate"])
    summary_msg = f"✅ Batch complete: {len(results)} items processed, +{total_points} 🌿 Green Points"
    extras = []
    if unknown_count:
        extras.append(f"{unknown_count} couldn't be identified")
    if duplicate_count:
        extras.append(f"{duplicate_count} already earned points before (📎 duplicate)")
    if extras:
        summary_msg += f" ({'; '.join(extras)} — no points for those)"
    st.success(summary_msg)

    batch_key = "|".join(f"{f.name}_{f.size}" for f in batch_files)
    if batch_key != st.session_state.get("last_batch_key"):
        for res in results:
            if res["conf_level"].get("unknown"):
                continue
            # check_and_register_image_hash is the one INSERT-or-detect call —
            # it will only ever return True (award points) the very first time
            # this exact photo's hash is seen across the whole app, matching
            # the is_duplicate flag computed above for display.
            is_new = db.check_and_register_image_hash(res["image_hash"], user_name)
            db.log_activity(
                user_name, "classify",
                points=res["guidance"].get("points", 1) if is_new else 0,
                category=res["predicted_class"],
                confidence=res["confidence"],
            )
        st.session_state["last_batch_key"] = batch_key


# ---------------------------------------------------------------------
# Page: Analytics & Leaderboard
# ---------------------------------------------------------------------
def page_analytics():
    theme.page_header("📊", "Analytics & Leaderboard")

    sub_mine, sub_global = st.tabs(["👤 My Activity", "🏆 Global Leaderboard"])

    with sub_mine:
        my_activity = db.get_user_activity(user_name)
        if not my_activity:
            theme.empty_state("🌱", "No activity yet", f"Nothing logged for **{user_name}** so far — try the Classify or Batch Upload pages.")
        else:
            st.info(f"🤖 **Eco Coach** *(rule-based suggestion, not a trained model)*: {db.get_eco_coach_tip(user_name)}")
            df = pd.DataFrame(my_activity)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Actions", len(df))
            c2.metric("🌿 Green Points Earned", int(df["points"].sum()))
            classify_conf = df.loc[df["kind"] == "classify", "confidence"].dropna()
            c3.metric("Avg. Classify Confidence", f"{classify_conf.mean()*100:.1f}%" if len(classify_conf) else "—")

            st.markdown("##### Activity by Type")
            st.bar_chart(df["kind"].value_counts())

            st.markdown("##### Full Activity Log")
            display_df = df[["timestamp", "kind", "category", "confidence", "points"]].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download my activity log (CSV)", data=csv,
                file_name=f"{user_name}_activity_log.csv", mime="text/csv",
            )

    with sub_global:
        stats = db.get_global_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("🍃 Items Classified", stats["total_classified"])
        c2.metric("👥 Participants", stats["total_users"])
        c3.metric("🌿 Total Green Points", stats["total_points"])

        st.markdown("##### 🏆 Leaderboard")
        leaderboard = db.get_leaderboard(limit=10)
        if leaderboard:
            lb_df = pd.DataFrame(leaderboard)
            lb_df.index = lb_df.index + 1
            lb_df = lb_df.rename(columns={"user_name": "Name", "total_points": "Green Points", "actions": "Actions"})
            st.dataframe(lb_df, use_container_width=True)
        else:
            theme.empty_state("🏆", "Leaderboard is empty", "No activity yet — be the first to earn Green Points!")

        st.markdown("##### 🕒 Recent Activity (everyone)")
        recent = db.get_recent_activity(limit=15)
        for r in recent:
            cat = f" — {r['category']}" if r.get("category") else ""
            st.caption(f"{r['timestamp']} · **{r['user_name']}** · {r['kind']}{cat} (+{r['points']} pts)")


# ---------------------------------------------------------------------
# Page: Community Hub
# ---------------------------------------------------------------------
def page_community():
    theme.page_header("🌍", "Community Hub")
    if not st.session_state.user_name:
        st.warning("Enter a nickname in the sidebar to use the Community Hub.")
    else:
        render_community_hub(user_name)


# ---------------------------------------------------------------------
# Page: Impact Dashboard
# ---------------------------------------------------------------------
def page_impact():
    theme.page_header(
        "📈", "Community Impact — Before & After",
        "This is the result of the full loop: someone spotted waste, reported "
        "it, the community organized a cleanup, and here's the proof it happened.",
    )

    stats = db.get_global_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🍃 Items Classified", stats["total_classified"])
    c2.metric("🧹 Cleanups Completed", stats["total_completed_cleanups"])
    c3.metric("👥 Participants", stats["total_users"])
    c4.metric("🌿 Total Green Points", stats["total_points"])

    est_kg = db.estimate_kg_diverted(stats["total_classified"])
    st.caption(
        f"🌍 **Estimated waste diverted: ~{est_kg} kg** "
        f"(assuming ~{db.AVG_ITEM_WEIGHT_KG} kg/item — a stated, conservative "
        f"estimate, not a measurement, since items aren't weighed)."
    )

    st.write("")
    completed = db.get_completed_events()
    if not completed:
        theme.empty_state(
            "🧹", "No cleanups completed yet",
            "Try the full loop yourself: <b>Classify</b> a photo → report it as litter → "
            "<b>Community Hub</b> → organize a cleanup → mark it complete with an "
            "after-photo. It'll appear here.",
        )
        return

    st.markdown("### ✅ Completed Cleanups")
    for ev in completed:
        with st.container(border=True):
            st.markdown(f"#### 🌿 {ev['title']}")
            st.caption(
                f"📍 {ev['location']}  ·  ✅ Completed {ev['completed_at']}  ·  "
                f"👥 {ev['participant_count']} participant(s) rewarded"
            )
            bc, ac = st.columns(2)
            with bc:
                if ev.get("before_photo_path") and os.path.exists(ev["before_photo_path"]):
                    st.image(ev["before_photo_path"], caption="Before", use_container_width=True)
                else:
                    st.caption("(no before-photo on record)")
            with ac:
                if ev.get("after_photo_path") and os.path.exists(ev["after_photo_path"]):
                    st.image(ev["after_photo_path"], caption="After", use_container_width=True)


# ---------------------------------------------------------------------
# Page: Impact Map
# ---------------------------------------------------------------------
def page_map():
    # Upgraded Impact Map: hotspot ranking, view-mode toggle, filters,
    # before/after comparison, AI recommendation, Plan Route button.
    # Uses existing db.get_map_points(), db.get_hotspots(), db.get_issues(),
    # db.get_events(), db.get_completed_events() — no new DB tables needed.
    theme.page_header(
        "🌍", "Impact Map",
        "See where waste is accumulating and where action is happening — "
        "a live geographic picture of every report and cleanup in the system.",
    )

    # ── Inline CSS ────────────────────────────────────────────────────────────
    st.markdown("""
<style>
.map-metric{background:#f0f7f0;border:1px solid #c3dfc3;border-radius:10px;padding:14px 18px;text-align:center;}
.map-metric-num{font-size:1.9rem;font-weight:800;color:#1B6B4A;}
.map-metric-lbl{font-size:0.82rem;color:#3a7a5a;margin-top:3px;}
.hs-red{border-left:4px solid #dc2626;background:#fef2f2;border-radius:8px;padding:12px 16px;margin:5px 0;}
.hs-orange{border-left:4px solid #ea580c;background:#fff7ed;border-radius:8px;padding:12px 16px;margin:5px 0;}
.hs-green{border-left:4px solid #16a34a;background:#f0fdf4;border-radius:8px;padding:12px 16px;margin:5px 0;}
.reco-card{background:#eff6ff;border:1px solid #93c5fd;border-radius:12px;padding:18px 20px;margin-top:6px;}
.reco-title{color:#1d4ed8;font-weight:700;font-size:1.05rem;}
.ba-before{background:#fef2f2;border-radius:8px;padding:14px;}
.ba-after{background:#f0fdf4;border-radius:8px;padding:14px;}
</style>""", unsafe_allow_html=True)

    # ── Fetch data ────────────────────────────────────────────────────────────
    all_points   = db.get_map_points()
    all_issues   = db.get_issues(limit=200)
    completed    = db.get_completed_events()
    hotspots     = db.get_hotspots(min_reports=2, days=30)

    # ── Filter bar ────────────────────────────────────────────────────────────
    st.markdown("---")
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
    with fc1:
        view_mode = st.selectbox(
            "Map View",
            ["Reports", "Hotspots", "Cleanups", "Impact (resolved)"],
            key="map_view_mode",
        )
    with fc2:
        status_filter = st.selectbox(
            "Status", ["All", "Submitted", "Resolved — Cleaned Up"],
            key="map_status_filter",
        )
    with fc3:
        kind_filter = st.selectbox(
            "Type", ["All", "Issue Report", "Cleanup Event"],
            key="map_kind_filter",
        )
    with fc4:
        area_filter = st.text_input(
            "Area keyword", placeholder="e.g. Park, Andheri…",
            key="map_area_filter",
        )

    # Apply filters
    filtered_points = all_points[:]
    if status_filter != "All":
        filtered_points = [p for p in filtered_points if p.get("status") == status_filter]
    if kind_filter != "All":
        filtered_points = [p for p in filtered_points if p.get("kind") == kind_filter]
    if area_filter.strip():
        kw = area_filter.strip().lower()
        filtered_points = [p for p in filtered_points if kw in p.get("location", "").lower()]

    st.markdown(f"**🔎 {len(filtered_points)} location(s) match your filters**")
    st.markdown("---")

    # ── Map + side panel ──────────────────────────────────────────────────────
    map_col, side_col = st.columns([3, 2])

    with map_col:
        st.markdown(f"##### 🗺️ {view_mode}")

        if view_mode == "Reports":
            disp = [p for p in filtered_points if p["kind"] == "Issue Report"]
        elif view_mode == "Cleanups":
            disp = [p for p in filtered_points if p["kind"] == "Cleanup Event"]
        elif view_mode == "Impact (resolved)":
            disp = [p for p in filtered_points if "Resolved" in str(p.get("status", ""))]
        else:
            # Hotspots — show all issue pins
            disp = [p for p in all_points if p["kind"] == "Issue Report"]

        if not disp and not all_points:
            theme.empty_state(
                "🗺️", "Nothing pinned yet",
                "Report an issue or organise a cleanup with a real location "
                "(e.g. a place name or address) and it will appear here. "
                "Vague locations like 'behind the shed' may not geocode — "
                "that's OK, the report still works, it just won't get a pin.",
            )
        else:
            rows = disp if disp else all_points
            map_df = pd.DataFrame(rows)

            def _color(row):
                if row.get("kind") == "Cleanup Event":
                    return [45, 106, 79]
                if "Resolved" in str(row.get("status", "")):
                    return [34, 197, 94]
                return [244, 162, 97]

            map_df["color"] = map_df.apply(_color, axis=1)
            map_df["size"]  = 140
            st.map(map_df, latitude="lat", longitude="lon", color="color", size="size")

        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown("🟠 **Issue Reports**")
        lc2.markdown("🟢 **Cleanups / Resolved**")
        lc3.caption("*(geocoded pins only)*")

    with side_col:
        # ── Summary metrics ───────────────────────────────────────────────
        n_issues   = len(all_issues)
        n_unres    = sum(1 for i in all_issues
                        if "Resolved" not in str(i.get("status", "")))
        n_cleanups = len(completed)
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="map-metric"><div class="map-metric-num">{n_issues}</div>'
            f'<div class="map-metric-lbl">📍 Reports</div></div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="map-metric"><div class="map-metric-num">{n_unres}</div>'
            f'<div class="map-metric-lbl">⚠️ Unresolved</div></div>',
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="map-metric"><div class="map-metric-num">{n_cleanups}</div>'
            f'<div class="map-metric-lbl">🧹 Done</div></div>',
            unsafe_allow_html=True,
        )

        # ── Hotspot ranking ───────────────────────────────────────────────
        st.write("")
        st.markdown("##### 📊 Area Hotspots *(last 30 days)*")
        if hotspots:
            for h in hotspots[:5]:
                cnt = h["count"]
                css   = "hs-red"    if cnt >= 5 else ("hs-orange" if cnt >= 3 else "hs-green")
                emoji = "🔴"        if cnt >= 5 else ("🟠"        if cnt >= 3 else "🟢")
                st.markdown(
                    f'<div class="{css}"><strong>{emoji} {h["location_label"]}</strong>' +
                    f'<br/><small>{cnt} reports · {h["recommended_action"]}</small></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "No hotspots yet — at least 2 reports at the same location "
                "will trigger automatic area clustering."
            )

    st.markdown("---")

    # ── Before / After ────────────────────────────────────────────────────────
    if completed:
        st.markdown("##### 📉 Before vs After — Most Recent Cleanup")
        ev  = completed[0]
        ba1, ba2 = st.columns(2)
        with ba1:
            st.markdown(
                f'<div class="ba-before"><strong>🔴 BEFORE — {ev["title"]}</strong>' +
                f'<br/>📍 {ev["location"]}</div>',
                unsafe_allow_html=True,
            )
            bpath = ev.get("before_photo_path")
            if bpath and os.path.exists(bpath):
                st.image(bpath, use_container_width=True)
            else:
                st.caption("*(no before-photo on record)*")
        with ba2:
            pcount = ev.get("participant_count", 0)
            st.markdown(
                f'<div class="ba-after"><strong>🟢 AFTER CLEANUP</strong>' +
                f'<br/>✅ Completed {ev.get("completed_at","")}' +
                f'<br/>👥 {pcount} volunteer(s)</div>',
                unsafe_allow_html=True,
            )
            apath = ev.get("after_photo_path")
            if apath and os.path.exists(apath):
                st.image(apath, use_container_width=True)
            else:
                st.caption("*(no after-photo on record)*")

        if len(completed) > 1:
            with st.expander(f"See all {len(completed)} completed cleanups"):
                for ev2 in completed[1:]:
                    st.caption(
                        f"✅ **{ev2['title']}** · {ev2['location']} · "
                        f"completed {ev2.get('completed_at','—')} · "
                        f"{ev2.get('participant_count',0)} volunteer(s)"
                    )
        st.markdown("---")

    # ── AI-style Recommendation ───────────────────────────────────────────────
    st.markdown("##### 🤖 EcoSort Recommendation *(rule-based, from stored data)*")
    if hotspots:
        top = hotspots[0]
        st.markdown(
            f'<div class="reco-card">' +
            f'<div class="reco-title">🤖 Priority Cleanup Area: {top["location_label"]}</div>' +
            f'<br/><p>Based on <strong>{top["count"]} reports</strong> in the last 30 days.' +
            f'<br/>📋 {top["recommended_action"]}</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button(
            "🗺️ Plan Cleanup Route for this area →",
            type="primary", use_container_width=True, key="map_plan_route",
        ):
            st.session_state["route_area"] = top["location_label"]
            st.success(
                f"✅ Route area set to **{top['location_label']}**. "
                "Head to **Route Planner** in the sidebar to continue!"
            )
    elif all_issues:
        st.info(
            "No clusters yet — more reports at the same location will trigger "
            "an automatic priority recommendation here."
        )
    else:
        theme.empty_state(
            "🤖", "Nothing to recommend yet",
            "Report issues from the Classify page — areas are automatically "
            "ranked by activity as soon as reports come in.",
        )

    st.markdown("---")

    # ── Full table ────────────────────────────────────────────────────────────
    st.markdown("##### 📍 All pinned locations")
    if filtered_points:
        display_df = pd.DataFrame(filtered_points)[
            ["kind", "label", "location", "status"]
        ].rename(columns={
            "kind": "Type", "label": "Title",
            "location": "Location", "status": "Status",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No locations match the current filters.")

# ---------------------------------------------------------------------
# Page: Multi-Item Scan (grid segmentation — classify several items
# in one messy photo using the existing single-item model, no new
# training required. Technique per Cheema, Hannan & Pires, 2022:
# tile the frame into a grid, classify each cell independently.)
# ---------------------------------------------------------------------
def _multiscan_grid_fallback(image, uploaded, grid_size=3):
    """Original N×N grid-tiling approach — used automatically when Gemini
    object localization isn't available (no package/API key, or a failed
    call), so the page never breaks."""
    w, h = image.size
    cell_w, cell_h = w // grid_size, h // grid_size

    results_grid = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    line_color = (244, 162, 97)

    with st.spinner(f"Classifying {grid_size * grid_size} grid cells..."):
        for row in range(grid_size):
            row_results = []
            for col in range(grid_size):
                left, top = col * cell_w, row * cell_h
                right = left + cell_w if col < grid_size - 1 else w
                bottom = top + cell_h if row < grid_size - 1 else h
                cell_img = image.crop((left, top, right, bottom))

                r = classify(cell_img, model, class_names, device, ood_detector, clip_gate)
                r["image_hash"] = hashlib.md5(cell_img.tobytes()).hexdigest()
                r["is_duplicate"] = (not r["conf_level"].get("unknown")) and db.image_hash_seen(r["image_hash"])
                row_results.append(r)

                draw.rectangle([left, top, right, bottom], outline=line_color, width=3)
                label = f"{r['icon']} {r['predicted_class'][:6]} {r['confidence']*100:.0f}%"
                draw.text((left + 6, top + 6), label, fill=(255, 255, 255))
                draw.text((left + 5, top + 5), label, fill=(27, 67, 50))
            results_grid.append(row_results)

    st.image(overlay, caption=f"{grid_size}×{grid_size} grid classification", use_container_width=True)
    return [r for row in results_grid for r in row]


def _multiscan_gemini(image, uploaded, gemini_client):
    """Gemini locates objects (bounding boxes only); each crop is then
    classified by our own MobileNetV2 pipeline — same as every other page.
    Gemini's rough label is shown as a second opinion, never as the final
    category, per the app's honesty-about-AI stance."""
    w, h = image.size
    buf = io.BytesIO()
    image.save(buf, format="JPEG")

    with st.spinner("Gemini is locating items in the photo..."):
        items = detect_items_with_gemini(gemini_client, buf.getvalue())

    if not items:
        st.warning("Gemini didn't find any distinct items — falling back to grid scan.")
        return _multiscan_grid_fallback(image, uploaded), None

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    box_color = (244, 162, 97)
    results = []

    for it in items:
        x1, y1, x2, y2 = it["bounding_box"]
        left = max(0, int(x1 / 1000 * w))
        top = max(0, int(y1 / 1000 * h))
        right = min(w, int(x2 / 1000 * w))
        bottom = min(h, int(y2 / 1000 * h))
        if right <= left or bottom <= top:
            continue
        crop = image.crop((left, top, right, bottom))

        r = classify(crop, model, class_names, device, ood_detector, clip_gate)
        r["image_hash"] = hashlib.md5(crop.tobytes()).hexdigest()
        r["is_duplicate"] = (not r["conf_level"].get("unknown")) and db.image_hash_seen(r["image_hash"])
        r["gemini_label"] = it["label"]
        results.append(r)

        draw.rectangle([left, top, right, bottom], outline=box_color, width=4)
        our_label = "unsure" if r["conf_level"].get("unknown") else r["predicted_class"]
        tag = f"{r['icon']} {our_label} {r['confidence']*100:.0f}%"
        draw.text((left + 6, max(0, top - 22)), tag, fill=(255, 255, 255))
        draw.text((left + 5, max(0, top - 23)), tag, fill=(27, 67, 50))

    st.image(overlay, caption=f"Gemini located {len(results)} item(s) — labels are our MobileNetV2 model's", use_container_width=True)
    return results, items


GEMINI_LABEL_KEYWORDS = {
    # keyword found in Gemini's free-text guess -> our model's category names
    "plastic": "plastic", "bottle": "plastic", "bag": "plastic", "jug": "plastic",
    "wrapper": "plastic", "straw": "plastic", "lid": "plastic",
    "paper": "paper", "carton": "paper", "napkin": "paper", "newspaper": "paper",
    "cardboard": "cardboard", "box": "cardboard",
    "metal": "metal", "can": "metal", "aluminum": "metal", "aluminium": "metal",
    "tin": "metal", "foil": "metal",
    "glass": "glass", "jar": "glass",
    "organic": "organic", "food": "organic", "fruit": "organic", "vegetable": "organic",
    "peel": "organic",
    "biological": "biological",
    "battery": "battery",
    "clothes": "clothes", "shirt": "clothes", "fabric": "clothes", "cloth": "clothes",
    "shoe": "shoes", "sneaker": "shoes", "sandal": "shoes",
    "trash": "trash",
}


def gemini_label_to_category(label: str):
    """Best-effort mapping of Gemini's free-text guess to one of our model's
    trained categories, so the Agreement column compares like with like
    instead of naively string-matching two different vocabularies."""
    words = label.lower().replace("-", " ").split()
    for word in words:
        if word in GEMINI_LABEL_KEYWORDS:
            return GEMINI_LABEL_KEYWORDS[word]
    return None


def page_multiscan():
    theme.page_header(
        "🧩", "Smart Multi-Item Scan",
        "Got a photo with several items — a messy bin, a cluttered table? "
        "Gemini finds where each item is in the photo, then our own trained "
        "classifier decides what each one actually is.",
    )

    gemini_client = get_gemini_client()

    with st.expander("ℹ️ How this works, and its honest limits"):
        if gemini_client:
            st.markdown(
                "- **Gemini (vision layer):** locates each distinct item in the "
                "photo and offers a rough label — it does not decide the final "
                "waste category.\n"
                "- **Our MobileNetV2 model (classifier):** every located item is "
                "cropped and run through the same trained model, CLIP waste-gate, "
                "and out-of-distribution check used everywhere else in this app — "
                "so the final label and Green Points always come from *our* model, "
                "not Gemini.\n"
                "- Gemini's guess is shown next to ours as a second opinion, not a "
                "verdict — the two can disagree, and that's shown honestly.\n"
                "- If Gemini's API is unavailable, this page automatically falls "
                "back to the original grid-tiling method below."
            )
        else:
            st.markdown(
                "- Gemini isn't configured (no `GEMINI_API_KEY` in Streamlit "
                "secrets, or the `google-genai` package isn't installed), so this "
                "falls back to splitting the photo into an N×N grid and "
                "classifying each cell independently.\n"
                "- The model was trained on single, centered items — a cell that "
                "only shows part of an item, or the background, may be "
                "misclassified.\n"
                "- An item spanning multiple cells may be counted more than once."
            )

    # Always defined, so scan_key below never hits an UnboundLocalError —
    # only shown to the user (and actually used) when Gemini isn't active.
    grid_size = 3
    if not gemini_client:
        grid_size = st.select_slider("Grid size", options=[2, 3, 4], value=3)

    uploaded = st.file_uploader("Upload a photo with multiple items", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        theme.empty_state("🧩", "No photo yet", "Upload a photo to scan it.")
        return

    image = Image.open(uploaded).convert("RGB")

    gemini_items = None
    if gemini_client:
        try:
            flat_results, gemini_items = _multiscan_gemini(image, uploaded, gemini_client)
        except Exception as e:
            st.warning(f"Gemini scan failed ({e}) — falling back to grid scan.")
            flat_results = _multiscan_grid_fallback(image, uploaded, grid_size)
    else:
        flat_results = _multiscan_grid_fallback(image, uploaded, grid_size)

    confident = [
        r for r in flat_results
        if r["confidence"] >= 0.6 and not r["conf_level"].get("unknown")
    ]
    confident_new = [r for r in confident if not r["is_duplicate"]]
    confident_dup = [r for r in confident if r["is_duplicate"]]

    st.write("")
    if gemini_items is not None and confident:
        st.markdown("#### 🧠 Final AI Decision")
        st.caption(
            "Our MobileNetV2 model makes the call on category and bin. Gemini's "
            "description is shown as a plain-language second opinion, not a "
            "competing probability — it isn't asked to output a confidence score."
        )
        for r in confident:
            guidance = r["guidance"]
            gemini_category = gemini_label_to_category(r["gemini_label"])
            if gemini_category is None:
                agree_note = f"💡 Gemini described this as \"{r['gemini_label']}\" — no clear category match to compare against."
            elif gemini_category == r["predicted_class"].lower():
                agree_note = f"✅ Gemini's description (\"{r['gemini_label']}\") matches our model's category."
            else:
                agree_note = (
                    f"⚠️ Gemini described this as \"{r['gemini_label']}\" ({gemini_category}), "
                    f"which differs from our model's call — worth a second look."
                )
            with st.container(border=True):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"## {r['icon']}")
                with col2:
                    st.markdown(f"##### {r['predicted_class'].title()}")
                    st.caption(f"🗑️ {guidance.get('bin', 'See disposal guide')}")
                    st.write(f"**Model confidence:** {r['conf_level']['emoji']} {r['conf_level']['label']}")
                    st.caption(agree_note)

    with st.expander("📋 Raw model-vs-Gemini comparison table"):
        if gemini_items is not None and confident:
            table_rows = []
            for r in confident:
                gemini_category = gemini_label_to_category(r["gemini_label"])
                if gemini_category is None:
                    agree = "❓ unclear"
                elif gemini_category == r["predicted_class"].lower():
                    agree = "✅ agree"
                else:
                    agree = "↔️ differ"
                table_rows.append({
                    "Our model's label": f"{r['icon']} {r['predicted_class'].title()}",
                    "Model confidence": r["conf_level"]["label"],
                    "Gemini's description": r["gemini_label"],
                    "Agreement": agree,
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Comparison table only applies to Gemini-powered scans.")

    st.write("")
    st.markdown("#### 📋 Detected Items Summary")
    unit = "item(s)" if gemini_items is not None else "cells"
    st.caption(f"{len(confident)} of {len(flat_results)} {unit} classified with ≥60% confidence.")

    counts = {}
    for r in confident:
        key = r["predicted_class"]
        counts[key] = counts.get(key, 0) + 1

    if counts:
        summary_df = pd.DataFrame([
            {"Category": f"{get_icon(cat)} {cat.title()}", f"{unit.capitalize()} detected": n}
            for cat, n in sorted(counts.items(), key=lambda x: -x[1])
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        total_points = sum(get_guidance(r["predicted_class"]).get("points", 1) for r in confident_new)
        summary_msg = f"✅ Scan complete — +{total_points} 🌿 Green Points for {len(confident_new)} confidently-detected items"
        if confident_dup:
            summary_msg += f" ({len(confident_dup)} matched a photo already scanned before — 📎 no points for those)"
        st.success(summary_msg)

        scan_key = f"{uploaded.name}_{uploaded.size}_{'gemini' if gemini_items is not None else grid_size}"
        if scan_key != st.session_state.get("last_multiscan_key"):
            for r in confident:
                is_new = db.check_and_register_image_hash(r["image_hash"], user_name)
                db.log_activity(
                    user_name, "classify",
                    points=get_guidance(r["predicted_class"]).get("points", 1) if is_new else 0,
                    category=r["predicted_class"], confidence=r["confidence"],
                )
            st.session_state["last_multiscan_key"] = scan_key
    else:
        st.info(f"No {unit} met the 60% confidence threshold — try a photo with clearer, more separated items.")


# ---------------------------------------------------------------------
# Page: System Usability Scale (SUS) — Brooke, 1996. The same
# validated 10-item survey used in Rahman et al., 2020 (JKSU) to
# evaluate their smart-bin system (they scored 86%). Gives a real,
# citable usability number instead of a subjective claim.
# ---------------------------------------------------------------------
SUS_QUESTIONS = [
    ("I think that I would like to use this system frequently.", True),
    ("I found the system unnecessarily complex.", False),
    ("I thought the system was easy to use.", True),
    ("I think that I would need the support of a technical person to use this system.", False),
    ("I found the various functions in this system were well integrated.", True),
    ("I thought there was too much inconsistency in this system.", False),
    ("I would imagine that most people would learn to use this system very quickly.", True),
    ("I found the system very cumbersome/awkward to use.", False),
    ("I felt very confident using the system.", True),
    ("I needed to learn a lot of things before I could get going with this system.", False),
]
SUS_SCALE = ["1 – Strongly Disagree", "2 – Disagree", "3 – Neutral", "4 – Agree", "5 – Strongly Agree"]


def page_sus():
    theme.page_header(
        "📝", "Usability Survey (SUS)",
        "The System Usability Scale — a standard, peer-reviewed 10-question "
        "instrument (Brooke, 1996) used across HCI research to score real "
        "software usability. Fill it out honestly; it takes about a minute.",
    )

    with st.form("sus_form"):
        answers = []
        for i, (question, positive) in enumerate(SUS_QUESTIONS):
            choice = st.radio(f"{i+1}. {question}", SUS_SCALE, index=2, key=f"sus_q{i}", horizontal=False)
            answers.append(int(choice.split(" ")[0]))
        submitted = st.form_submit_button("Submit survey", use_container_width=True)

    if submitted:
        if not st.session_state.get("user_name"):
            st.warning("Enter a nickname in the sidebar so your response is recorded.")
        else:
            score = db.add_sus_response(user_name, answers)
            adjective = db.sus_adjective(score)
            st.success(f"Your SUS score: **{score:.1f} / 100** — rated *{adjective}*")
            st.rerun()

    stats = db.get_sus_stats()
    if stats["count"] > 0:
        st.write("")
        st.markdown("#### 📊 Aggregate Results (all respondents)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Responses", stats["count"])
        c2.metric("Average SUS Score", f"{stats['average']:.1f}")
        c3.metric("Rating", stats["adjective"])
        st.caption(
            "Reference scale (Bangor, Kortum & Miller, 2009): "
            "85+ Excellent · 72–84 Good · 52–71 OK · 39–51 Poor · below 39 Awful."
        )

        recent = db.get_recent_sus_responses(limit=10)
        with st.expander("See individual responses"):
            for r in recent:
                st.caption(f"{r['timestamp']} · **{r['user_name']}** · {r['score']:.1f}/100")
    else:
        theme.empty_state("📝", "No responses yet", "Be the first to complete the usability survey above.")


# ---------------------------------------------------------------------
# Page: Route Planner — multi-objective (cost/distance + workload
# balance + CO2) route optimization for visiting open issues/cleanups,
# using route_optimizer.py. Pulls real geocoded locations already in
# the database (from the Community Hub reports and events).
# ---------------------------------------------------------------------
def page_routes():
    theme.page_header(
        "🚚", "Route Planner",
        "Multi-objective route optimization for volunteer teams or collection "
        "vehicles — minimizes total distance, balances workload across teams, "
        "and estimates CO2, inspired by Lu, Pu & Han (2020) and Hussain et al. (2024).",
    )

    with st.expander("ℹ️ How this works, and its honest limits"):
        st.markdown(
            "- Pulls every **open issue report** and **upcoming cleanup event** "
            "that has a geocoded location (visit them from the Community Hub or "
            "Impact Map first).\n"
            "- Splits locations across your available teams by compass-bearing "
            "sweep (a classic VRP clustering heuristic), then optimizes each "
            "team's visiting order with nearest-neighbor + 2-opt.\n"
            "- CO2 is a **distance-based estimate** using published fuel/emission "
            "constants, not a measurement — real fuel use depends on vehicle, "
            "traffic, and load.\n"
            "- This is a fast heuristic, not a mathematically-proven-optimal "
            "solver — true multi-vehicle routing is NP-hard. For a real "
            "campaign's handful of locations, that's the right tradeoff."
        )

    depot_location = st.text_input(
        "📍 Starting point (depot) — e.g. your school, a community center",
        placeholder="e.g. Central Community Hall",
    )
    num_vehicles = st.number_input("🚐 Number of teams / vehicles available", min_value=1, max_value=10, value=2)

    if st.button("🧮 Optimize routes", use_container_width=True):
        if not depot_location.strip():
            st.error("Please enter a starting point.")
        else:
            with st.spinner("Geocoding depot and locations..."):
                depot_coords = geo.geocode(depot_location.strip())

            if not depot_coords:
                st.error(
                    f"Couldn't geocode '{depot_location}' — try a more specific "
                    "place name or address."
                )
            else:
                issues = [i for i in db.get_unresolved_issues(limit=200) if i.get("lat") and i.get("lon")]
                events = [e for e in db.get_events(status="upcoming") if e.get("lat") and e.get("lon")]

                locations = [(i["lat"], i["lon"], f"📸 {i['issue_type']} @ {i['location']}") for i in issues]
                locations += [(e["lat"], e["lon"], f"🧹 {e['title']}") for e in events]

                if not locations:
                    theme.empty_state(
                        "🚚", "Nothing to route yet",
                        "No geocoded open issues or upcoming events found. Report an "
                        "issue or organize a cleanup with a real location first.",
                    )
                else:
                    result = route_optimizer.optimize_routes(depot_coords, locations, int(num_vehicles))

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("📍 Locations", result["num_locations"])
                    c2.metric("🛣️ Total Distance", f"{result['total_distance_km']} km")
                    c3.metric("🌫️ Est. CO2", f"{result['total_co2_kg']} kg")
                    c4.metric("⚖️ Workload Balance", f"±{result['workload_balance_km']} km")
                    st.caption(
                        "Workload Balance = gap between the busiest and lightest team's "
                        "distance — lower is fairer."
                    )

                    st.write("")
                    for r in result["vehicle_routes"]:
                        with st.container(border=True):
                            st.markdown(f"#### 🚐 Team {r['vehicle']} — {r['distance_km']} km, ~{r['co2_kg']} kg CO2")
                            if not r["stops"]:
                                st.caption("No stops assigned.")
                            else:
                                for i, stop in enumerate(r["stops"]):
                                    st.caption(f"{i+1}. {stop[2]}")

                    map_points = [{"lat": depot_coords[0], "lon": depot_coords[1], "label": "Depot"}]
                    for r in result["vehicle_routes"]:
                        for stop in r["stops"]:
                            map_points.append({"lat": stop[0], "lon": stop[1], "label": stop[2]})
                    st.map(pd.DataFrame(map_points), latitude="lat", longitude="lon")


# ---------------------------------------------------------------------
# Navigation — built with position="hidden" so Streamlit's automatic
# top-of-sidebar menu (which reserves its own header space above it) is
# suppressed, and the page links are instead drawn ourselves via
# st.page_link() inside the SAME `with st.sidebar:` block used for the
# brand/name/profile card above. Streamlit renders a container's content
# in the order it's written to that container across the whole script run
# — so even though this code sits far below the brand block, appending to
# st.sidebar here just continues that same sidebar in place, giving:
# brand + name + profile card *first*, nav links right after — instead of
# Streamlit's default (nav menu pinned above everything else in the
# sidebar, with its own reserved top gap).
# ---------------------------------------------------------------------
NAV_PAGES = [
    st.Page(page_classify, title="Classify", icon="🔍", default=True),
    st.Page(page_batch, title="Batch Upload", icon="📂"),
    st.Page(page_multiscan, title="Multi-Item Scan", icon="🧩"),
    st.Page(page_analytics, title="Analytics & Leaderboard", icon="📊"),
    st.Page(page_community, title="Community Hub", icon="🌍"),
    st.Page(page_impact, title="Impact Dashboard", icon="📈"),
    st.Page(page_map, title="Impact Map", icon="🗺️"),
    st.Page(page_routes, title="Route Planner", icon="🚚"),
    st.Page(page_sus, title="Usability Survey", icon="📝"),
]
pg = st.navigation(NAV_PAGES, position="hidden")

with st.sidebar:
    st.markdown('<div class="eco-nav-label">NAVIGATE</div>', unsafe_allow_html=True)
    for _nav_page in NAV_PAGES:
        st.page_link(_nav_page)

pg.run()

st.markdown("---")
st.caption(
    "🌱 EcoSort AI — No Green Deed Is Too Small. "
    "AI concept: Computer Vision · Transfer Learning (MobileNetV2) · "
    "Grad-CAM Explainability · Shared SQLite backend · "
    "Built for the AI Project Competition"
)
