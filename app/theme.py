"""
Visual system for EcoSort AI — CSS injection + small HTML component
helpers, kept separate from page logic so app.py stays readable.

PURE UI/UX LAYER: every public function here (inject_css, hero,
page_header, pill, profile_card, empty_state) keeps the exact same
name and signature as before this pass. Nothing in app.py or
community_hub.py needed to change to pick up this redesign — that's
deliberate: this file is the entire blast radius of the visual
overhaul, so existing functionality can't regress by construction.

Design approach (v2 — "loud but on-brand"):
    - Colors/fonts for native widgets (buttons, tabs, inputs) still
      come from .streamlit/config.toml first — the version-stable way
      to theme Streamlit — with this file layering richer CSS on top.
    - Typography pairs an editorial serif display face (Fraunces) for
      hero/page titles with a clean sans (Inter) for UI surfaces.
    - The palette widens from a single green into a full nature-themed
      spectrum — forest, moss, lime, sunshine amber, coral and sky —
      used the way the reference eco-marketing sites use it: colorful
      feature badges, gradient CTAs, sticker-style ribbons — while big
      blocks of text stay on plain, readable surfaces.
    - Motion is everywhere but layered by speed: slow (20s+) drifting
      background blobs and hero orbs create ambient life, medium
      hover/lift/tilt transitions give tactile feedback, and quick
      (0.4s) entrance choreography makes each page feel like it
      "arrives" instead of just appearing.
    - Glassmorphism, gradient text, glowing rings, sticker badges and
      a dot-grid backdrop are used throughout (not just the hero) so
      the whole app reads as considered brand design, not a template.
"""

import streamlit as st
import streamlit.components.v1 as st_components

PALETTE = {
    "forest": "#0F3D2E",       # deep forest — dominant dark surface
    "forest_2": "#123F2B",
    "mid": "#1B6B4A",          # primary brand green — buttons, links, accents
    "moss": "#5FAE83",         # supporting mid-tone
    "lime": "#A8E063",         # fresh accent — highlights/glow
    "mint_bg": "#EAF6EE",      # soft mint surface tint
    "mint_bg_2": "#DCEFE2",
    "cream": "#FAF9F4",        # warm off-white — page background
    "ink": "#1F2421",          # charcoal text
    "muted": "#5C6B60",
    "line": "#E1EAE3",         # hairline borders
    "amber": "#E8A34C",
    "gold": "#D9B65C",
    # v2 — wider accent spectrum for a livelier, less single-note feel
    "sunshine": "#FFB627",
    "coral": "#FF6B6B",
    "sky": "#3EC6E0",
    "grape": "#8B6FD6",
}


def scroll_to_top():
    """Force the viewport back to the top of the page.

    st.navigation(position="hidden") + our own st.page_link sidebar does a
    client-side rerun rather than a full page load, so the browser keeps
    whatever scroll offset the previous page was left at. If someone was
    scrolled halfway down "Analytics" and clicked "Impact Map", the new
    page's own title rendered off-screen above the fold — it looked like
    the title was simply missing. A tiny injected script run on every
    rerun is the standard Streamlit fix for this.
    """
    st_components.html(
        """
        <script>
            (function () {
                const doc = window.parent.document;
                doc.querySelectorAll('section.main, [data-testid="stAppViewContainer"]')
                    .forEach((el) => el.scrollTo({ top: 0, behavior: "instant" }));
                window.parent.scrollTo({ top: 0, behavior: "instant" });
            })();
        </script>
        """,
        height=0,
    )


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --eco-forest: {PALETTE['forest']};
            --eco-mid: {PALETTE['mid']};
            --eco-lime: {PALETTE['lime']};
            --eco-cream: {PALETTE['cream']};
            --eco-ink: {PALETTE['ink']};
            --eco-sunshine: {PALETTE['sunshine']};
            --eco-coral: {PALETTE['coral']};
            --eco-sky: {PALETTE['sky']};
            --eco-grape: {PALETTE['grape']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
            color: {PALETTE['ink']};
        }}
        h1, h2, h3, h4 {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: {PALETTE['forest']};
        }}
        .eco-hero h1, .eco-display {{
            font-family: 'Fraunces', serif;
            font-optical-sizing: auto;
        }}

        /* ---- Page background: dot-grid + slow drifting color blobs, ----
           ---- so every page (not just the hero) feels alive ---- */
        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle, {PALETTE['moss']}26 1.1px, transparent 1.1px);
            background-size: 22px 22px;
            background-color: {PALETTE['cream']};
            position: relative;
        }}
        [data-testid="stAppViewContainer"]::before,
        [data-testid="stAppViewContainer"]::after {{
            content: "";
            position: fixed;
            border-radius: 50%;
            filter: blur(60px);
            pointer-events: none;
            z-index: 0;
        }}
        [data-testid="stAppViewContainer"]::before {{
            width: 480px; height: 480px;
            background: radial-gradient(circle, {PALETTE['lime']}30 0%, transparent 70%);
            top: -160px; right: -140px;
            animation: ecoFloatSlow 26s ease-in-out infinite;
        }}
        [data-testid="stAppViewContainer"]::after {{
            width: 420px; height: 420px;
            background: radial-gradient(circle, {PALETTE['sunshine']}22 0%, transparent 70%);
            bottom: -140px; left: -120px;
            animation: ecoFloat 30s ease-in-out infinite;
        }}
        .block-container {{
            padding-top: 4.5rem;
            padding-bottom: 4rem;
            max-width: 1200px;
            position: relative;
            z-index: 1;
        }}
        /* Streamlit's own header toolbar (Share / star / pencil / GitHub
           icons) sits fixed at the very top of the viewport, on top of the
           page content, not above it. Page titles need enough top padding
           to clear it, plus a scroll-margin so anchor-jumps don't land a
           heading right back underneath it. */
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        h1, h2, h3 {{
            scroll-margin-top: 5rem;
        }}
        @media (max-width: 640px) {{
            .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 3.75rem; }}
            .eco-hero {{ padding: 1.6rem 1.4rem !important; }}
            .eco-hero h1 {{ font-size: 2.1rem !important; }}
            .eco-badge-row {{ gap: 0.35rem !important; }}
        }}

        /* ---- Entrance choreography — masks Streamlit's rerun flash ---- */
        .main .block-container {{ animation: ecoFadeIn 0.45s cubic-bezier(0.16, 1, 0.3, 1); }}
        @keyframes ecoFadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes ecoFloat {{
            0%, 100% {{ transform: translate(0, 0); }}
            50%      {{ transform: translate(-14px, 18px); }}
        }}
        @keyframes ecoFloatSlow {{
            0%, 100% {{ transform: translate(0, 0) scale(1); }}
            50%      {{ transform: translate(12px, -10px) scale(1.05); }}
        }}
        @keyframes ecoDrift {{
            0%   {{ transform: translate(0,0) rotate(0deg); }}
            50%  {{ transform: translate(10px,-16px) rotate(8deg); }}
            100% {{ transform: translate(0,0) rotate(0deg); }}
        }}
        @keyframes ecoPop {{
            0%   {{ opacity: 0; transform: translateY(10px) scale(0.94); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        @keyframes ecoGradientShift {{
            0%, 100% {{ background-position: 0% 50%; }}
            50%      {{ background-position: 100% 50%; }}
        }}
        @keyframes ecoSpinSlow {{
            from {{ transform: rotate(0deg); }}
            to   {{ transform: rotate(360deg); }}
        }}
        @keyframes ecoBounce {{
            0%, 100% {{ transform: translateY(0); }}
            50%      {{ transform: translateY(-8px); }}
        }}
        @keyframes ecoRingPulse {{
            0%, 100% {{ box-shadow: 0 0 0 4px {PALETTE['lime']}33, 0 4px 12px rgba(15, 61, 46, 0.3); }}
            50%      {{ box-shadow: 0 0 0 8px {PALETTE['sunshine']}2e, 0 4px 18px rgba(15, 61, 46, 0.38); }}
        }}
        @keyframes ecoDashCycle {{
            0%, 100% {{ border-color: {PALETTE['moss']}88; }}
            33%      {{ border-color: {PALETTE['sunshine']}99; }}
            66%      {{ border-color: {PALETTE['sky']}88; }}
        }}
        @keyframes ecoUnderlineGrow {{
            from {{ width: 0; }}
            to   {{ width: var(--eco-underline-w, 3.2rem); }}
        }}

        /* ---- Focus states (accessibility) ---- */
        button:focus-visible, [tabindex]:focus-visible, input:focus-visible,
        textarea:focus-visible, [role="radio"]:focus-visible {{
            outline: 2px solid {PALETTE['mid']} !important;
            outline-offset: 2px;
        }}

        /* ---- Buttons — punchier gradient + glow + light sweep ---- */
        .stButton > button {{
            border-radius: 12px;
            border: 1px solid {PALETTE['line']};
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            padding: 0.55rem 1.2rem;
            background: white;
            color: {PALETTE['forest']};
            transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
        }}
        .stButton > button:hover {{
            box-shadow: 0 8px 20px rgba(15, 61, 46, 0.16);
            transform: translateY(-2px) scale(1.015);
            border-color: {PALETTE['mid']}55;
        }}
        .stButton > button:active {{ transform: translateY(0) scale(0.99); }}
        .stButton > button[kind="primary"] {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, {PALETTE['lime']} 0%, {PALETTE['mid']} 45%, {PALETTE['forest']} 100%);
            background-size: 200% 200%;
            animation: ecoGradientShift 6s ease infinite;
            border: none;
            color: white;
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 12px 30px rgba(27, 107, 74, 0.4), 0 0 0 3px {PALETTE['lime']}30;
        }}
        .stButton > button[kind="primary"]::after {{
            content: "";
            position: absolute; top: 0; left: -60%;
            width: 40%; height: 100%;
            background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
            transform: skewX(-20deg);
            transition: left 0.55s ease;
        }}
        .stButton > button[kind="primary"]:hover::after {{ left: 130%; }}
        .stDownloadButton > button {{ border-radius: 12px; }}

        /* ---- Spinner — brand-colored instead of default ---- */
        [data-testid="stSpinner"] > div > div {{
            border-top-color: {PALETTE['mid']} !important;
            border-right-color: {PALETTE['sunshine']}88 !important;
        }}

        /* ---- Metrics — elevated, gradient numerals, hover glow ---- */
        [data-testid="stMetric"] {{
            background: white;
            border: 1px solid {PALETTE['line']};
            border-radius: 16px;
            padding: 1rem 1.1rem 0.8rem 1.1rem;
            transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
            position: relative;
            overflow: hidden;
        }}
        [data-testid="stMetric"]::before {{
            content: "";
            position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, {PALETTE['lime']}, {PALETTE['sunshine']}, {PALETTE['coral']}, {PALETTE['sky']});
            background-size: 300% 100%;
            animation: ecoGradientShift 8s ease infinite;
            opacity: 0.9;
        }}
        [data-testid="stMetric"]:hover {{
            box-shadow: 0 14px 32px rgba(15, 61, 46, 0.12);
            transform: translateY(-3px);
            border-color: {PALETTE['mid']}44;
        }}
        [data-testid="stMetricValue"] {{
            background: linear-gradient(135deg, {PALETTE['forest']}, {PALETTE['mid']} 60%, {PALETTE['moss']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            font-family: 'Inter', sans-serif;
        }}
        [data-testid="stMetricLabel"] {{ color: {PALETTE['muted']}; font-weight: 500; }}

        /* ---- Native bordered containers — depth + gradient top accent ---- */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 18px !important;
            border-color: {PALETTE['line']} !important;
            transition: box-shadow 0.22s ease, transform 0.18s ease, border-color 0.22s ease;
            position: relative;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            box-shadow: 0 16px 36px rgba(15, 61, 46, 0.13);
            transform: translateY(-3px);
            border-color: {PALETTE['mid']}44 !important;
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background: linear-gradient(90deg, {PALETTE['mid']}, {PALETTE['lime']}) !important;
        }}
        .stTabs [aria-selected="true"] {{ color: {PALETTE['mid']} !important; font-weight: 700; }}
        .stTabs [data-baseweb="tab"] {{ font-family: 'Inter', sans-serif; transition: color 0.15s ease; }}
        .stTabs [data-baseweb="tab"]:hover {{ color: {PALETTE['mid']} !important; }}

        /* ---- Alerts ---- */
        [data-testid="stAlert"] {{
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(15, 61, 46, 0.06);
        }}

        /* ---- Dataframes / tables ---- */
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid {PALETTE['line']};
        }}

        /* ---- Inputs ---- */
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {{
            border-radius: 10px !important;
            transition: box-shadow 0.15s ease, border-color 0.15s ease;
        }}
        [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {{
            box-shadow: 0 0 0 3px {PALETTE['lime']}33 !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            border-radius: 14px !important;
            background: linear-gradient(135deg, {PALETTE['mint_bg']}cc, {PALETTE['mint_bg_2']}88) !important;
            border: 1.5px dashed {PALETTE['moss']}88 !important;
            transition: border-color 0.2s ease, background 0.2s ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {PALETTE['sunshine']} !important;
        }}

        /* ---- Progress bar — rainbow gradient fill ---- */
        [data-testid="stProgress"] > div > div > div {{
            background: linear-gradient(90deg, {PALETTE['mid']}, {PALETTE['lime']}, {PALETTE['sunshine']}) !important;
            background-size: 200% 100%;
            animation: ecoGradientShift 3s ease infinite;
            border-radius: 999px;
            transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {PALETTE['mint_bg']} 0%, {PALETTE['mint_bg_2']} 100%);
            border-right: 1px solid {PALETTE['line']};
            overflow-y: auto !important;
        }}
        /* Sidebar was clipping long content with no way to reach items below the
           fold — force the inner scroll container to actually scroll, independent
           of the main page, and keep the custom scrollbar styling on it too. */
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {{
            height: 100vh !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch;
        }}
        [data-testid="stSidebar"] h3 {{ font-family: 'Fraunces', serif; }}
        [data-testid="stSidebarContent"] {{ padding-top: 1.1rem; padding-bottom: 2.5rem; }}
        [data-testid="stSidebarUserContent"] {{ padding-top: 0; }}
        .eco-nav-label {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            color: {PALETTE['muted']};
            margin: 1.1rem 0 0.3rem 0.1rem;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] {{
            border-radius: 10px;
            margin-bottom: 0.05rem;
            transition: background 0.15s ease, transform 0.15s ease;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
            background: rgba(255,255,255,0.7);
            transform: translateX(3px);
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p {{
            font-size: 0.92rem;
        }}

        /* ---- Custom scrollbar ---- */
        ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, {PALETTE['moss']}, {PALETTE['mid']});
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-thumb:hover {{ background: {PALETTE['forest']}; }}

        /* =====================================================
           HERO — full editorial banner, layered gradient mesh,
           drifting glass orbs, floating leaf particles, a corner
           sticker badge, and an animated feature-chip row — the
           "marketing site" energy from the reference designs.
        ===================================================== */
        .eco-hero {{
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(ellipse 80% 60% at 15% 0%, {PALETTE['lime']}26 0%, transparent 55%),
                radial-gradient(ellipse 60% 50% at 100% 100%, {PALETTE['sunshine']}20 0%, transparent 60%),
                linear-gradient(135deg, {PALETTE['forest']} 0%, {PALETTE['forest_2']} 45%, {PALETTE['mid']} 100%);
            background-size: 100% 100%, 100% 100%, 220% 220%;
            animation: ecoGradientShift 14s ease infinite;
            border-radius: 24px;
            padding: 3rem 2.8rem;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 20px 50px rgba(15, 61, 46, 0.25);
        }}
        .eco-hero::before, .eco-hero::after {{
            content: "";
            position: absolute;
            border-radius: 50%;
            filter: blur(1px);
            pointer-events: none;
        }}
        .eco-hero::before {{
            width: 260px; height: 260px;
            background: radial-gradient(circle, {PALETTE['lime']}45 0%, transparent 72%);
            top: -100px; right: -50px;
            animation: ecoFloatSlow 22s ease-in-out infinite;
        }}
        .eco-hero::after {{
            width: 200px; height: 200px;
            background: radial-gradient(circle, {PALETTE['sunshine']}3a 0%, transparent 72%);
            bottom: -80px; right: 140px;
            animation: ecoFloat 26s ease-in-out infinite;
        }}
        .eco-hero .particles {{
            position: absolute; inset: 0; pointer-events: none; z-index: 0;
        }}
        .eco-hero .particles span {{
            position: absolute;
            font-size: 1.4rem;
            opacity: 0.5;
            animation: ecoDrift 9s ease-in-out infinite;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.25));
        }}
        .eco-hero .particles span:nth-child(1) {{ top: 14%; left: 78%; animation-delay: 0s; font-size: 1.6rem; }}
        .eco-hero .particles span:nth-child(2) {{ top: 62%; left: 88%; animation-delay: 2.2s; font-size: 1.1rem; }}
        .eco-hero .particles span:nth-child(3) {{ top: 30%; left: 92%; animation-delay: 4.4s; font-size: 1.3rem; }}
        .eco-hero .particles span:nth-child(4) {{ top: 78%; left: 70%; animation-delay: 1.1s; font-size: 1rem; }}
        .eco-hero .sticker {{
            position: absolute;
            top: 1.4rem; right: 1.6rem;
            width: 4.6rem; height: 4.6rem;
            border-radius: 50%;
            background: linear-gradient(135deg, {PALETTE['coral']}, {PALETTE['sunshine']});
            color: white;
            display: flex; align-items: center; justify-content: center;
            text-align: center;
            font-size: 0.62rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: 0.02em;
            transform: rotate(12deg);
            box-shadow: 0 8px 20px rgba(0,0,0,0.28), 0 0 0 3px rgba(255,255,255,0.25);
            animation: ecoBounce 5s ease-in-out infinite;
            z-index: 2;
        }}
        .eco-badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0 0 1.1rem 0;
            position: relative;
            z-index: 1;
        }}
        .eco-badge-row .chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(255,255,255,0.14);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,0.28);
            border-radius: 999px;
            padding: 0.28rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: white;
            opacity: 0;
            animation: ecoPop 0.5s ease forwards;
            transition: transform 0.15s ease, background 0.15s ease;
        }}
        .eco-badge-row .chip:hover {{ transform: translateY(-2px); background: rgba(255,255,255,0.22); }}
        .eco-badge-row .chip:nth-child(1) {{ animation-delay: 0.05s; }}
        .eco-badge-row .chip:nth-child(2) {{ animation-delay: 0.15s; }}
        .eco-badge-row .chip:nth-child(3) {{ animation-delay: 0.25s; }}
        .eco-hero h1 {{
            color: white;
            font-size: 3rem;
            font-weight: 700;
            line-height: 1.06;
            margin: 0 0 0.4rem 0;
            position: relative;
            max-width: 34rem;
            background: linear-gradient(100deg, #ffffff 30%, {PALETTE['lime']} 62%, #ffffff 85%);
            background-size: 220% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: ecoGradientShift 7s ease infinite;
        }}
        .eco-hero .tagline {{
            font-family: 'Fraunces', serif;
            font-style: italic;
            font-size: 1.22rem;
            color: {PALETTE['lime']};
            margin-bottom: 0.9rem;
            position: relative;
        }}
        .eco-hero .subtext {{
            color: rgba(255,255,255,0.86);
            font-size: 1rem;
            max-width: 42rem;
            line-height: 1.6;
            position: relative;
        }}

        /* ---- Pills / tags — wider color range, subtle hover pop ---- */
        .eco-pill {{
            display: inline-block;
            background: {PALETTE['mint_bg']};
            color: {PALETTE['mid']};
            border: 1px solid {PALETTE['mid']}22;
            border-radius: 999px;
            padding: 0.2rem 0.8rem;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
            transition: transform 0.15s ease;
        }}
        .eco-pill:hover {{ transform: translateY(-1px); }}
        .eco-pill.demo {{
            background: #FFF6E5; color: #9A6B00; border-color: #E8C56A66;
        }}
        .eco-pill.live {{
            background: {PALETTE['mint_bg']}; color: {PALETTE['mid']}; border-color: {PALETTE['mid']}33;
        }}
        .eco-pill.coral {{
            background: #FFEDED; color: #C23B3B; border-color: {PALETTE['coral']}44;
        }}
        .eco-pill.sky {{
            background: #E7FAFD; color: #157A91; border-color: {PALETTE['sky']}55;
        }}
        .eco-pill.grape {{
            background: #F1ECFB; color: #6647B8; border-color: {PALETTE['grape']}44;
        }}

        /* ---- Sidebar profile card — glass-tinted, rainbow ring pulse ---- */
        .eco-profile-card {{
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 18px;
            padding: 1.2rem 1.1rem 1rem 1.1rem;
            border: 1px solid rgba(255,255,255,0.6);
            text-align: center;
            box-shadow: 0 6px 20px rgba(15, 61, 46, 0.08);
            animation: ecoPop 0.4s ease;
        }}
        .eco-avatar {{
            width: 56px; height: 56px;
            border-radius: 50%;
            background: linear-gradient(135deg, {PALETTE['mid']}, {PALETTE['forest']});
            color: white;
            display: flex; align-items: center; justify-content: center;
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 1.4rem;
            margin: 0 auto 0.55rem auto;
            box-shadow: 0 0 0 4px {PALETTE['lime']}33, 0 4px 12px rgba(15, 61, 46, 0.3);
            animation: ecoRingPulse 4s ease-in-out infinite;
        }}
        .eco-profile-card .name {{
            font-weight: 700;
            color: {PALETTE['ink']};
            margin-bottom: 0.15rem;
        }}
        .eco-profile-card .points {{
            font-family: 'Fraunces', serif;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, {PALETTE['mid']}, {PALETTE['moss']}, {PALETTE['sunshine']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
        }}
        .eco-progress-label {{
            font-size: 0.78rem;
            color: {PALETTE['muted']};
            margin-top: 0.55rem;
        }}

        /* ---- Page header — spinning-on-hover gradient badge + grown underline ---- */
        .eco-page-header {{ margin-bottom: 0.3rem; }}
        .eco-page-header .badge {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 3rem; height: 3rem;
            background: linear-gradient(135deg, {PALETTE['mint_bg']}, {PALETTE['mint_bg_2']});
            border: 1px solid {PALETTE['line']};
            border-radius: 13px;
            font-size: 1.55rem;
            margin-right: 0.8rem;
            transition: transform 0.4s ease, background 0.3s ease;
        }}
        .eco-page-header:hover .badge {{
            transform: rotate(-8deg) scale(1.08);
            background: linear-gradient(135deg, {PALETTE['lime']}55, {PALETTE['sunshine']}44);
        }}
        .eco-page-header .title-row {{ display: flex; align-items: center; }}
        .eco-page-header h2 {{
            margin: 0; font-size: 2.15rem; font-weight: 700; display: inline;
            font-family: 'Fraunces', serif; color: {PALETTE['forest']};
        }}
        .eco-page-header .underline {{
            height: 4px;
            width: 3.6rem;
            margin: 0.6rem 0 0 3.5rem;
            border-radius: 999px;
            background: linear-gradient(90deg, {PALETTE['lime']}, {PALETTE['sunshine']}, {PALETTE['coral']});
            --eco-underline-w: 3.2rem;
            animation: ecoUnderlineGrow 0.6s ease-out;
        }}

        /* ---- Empty states — bouncing icon, color-cycling dashed border ---- */
        .eco-empty {{
            text-align: center;
            padding: 3rem 1.5rem;
            background: linear-gradient(180deg, {PALETTE['mint_bg']}99, {PALETTE['mint_bg_2']}55);
            border: 1.5px dashed {PALETTE['moss']}88;
            border-radius: 18px;
            color: {PALETTE['muted']};
            animation: ecoDashCycle 6s ease-in-out infinite;
        }}
        .eco-empty .icon {{
            font-size: 2.7rem;
            margin-bottom: 0.6rem;
            display: inline-block;
            animation: ecoBounce 2.4s ease-in-out infinite;
        }}
        .eco-empty .title {{
            font-family: 'Fraunces', serif;
            font-weight: 700;
            color: {PALETTE['forest']};
            font-size: 1.18rem;
            margin-bottom: 0.35rem;
        }}

        /* =====================================================
           MASCOT — a small looping scene, fixed in the corner on
           every page: a kid carries wet waste to the wet bin, then
           dry waste to the dry bin, on repeat. Pure CSS (no JS) so
           it survives every Streamlit rerun/page-switch without
           re-registering timers. Decorative only — pointer-events
           are off throughout so it never blocks a real control, and
           it's small enough not to compete with actual content.
        ===================================================== */
        @keyframes ecoKidWalk {{
            0%   {{ transform: translateX(0); }}
            22%  {{ transform: translateX(-52px); }}
            33%  {{ transform: translateX(-52px); }}
            50%  {{ transform: translateX(0); }}
            72%  {{ transform: translateX(52px); }}
            83%  {{ transform: translateX(52px); }}
            100% {{ transform: translateX(0); }}
        }}
        @keyframes ecoItemWet {{
            0%   {{ opacity: 1; transform: translateY(0); }}
            28%  {{ opacity: 1; transform: translateY(0); }}
            33%  {{ opacity: 0; transform: translateY(6px); }}
            100% {{ opacity: 0; }}
        }}
        @keyframes ecoItemDry {{
            0%, 49% {{ opacity: 0; }}
            50%  {{ opacity: 1; transform: translateY(0); }}
            78%  {{ opacity: 1; transform: translateY(0); }}
            83%  {{ opacity: 0; transform: translateY(6px); }}
            100% {{ opacity: 0; }}
        }}
        @keyframes ecoBinDropWet {{
            0%, 30% {{ transform: scale(1); }}
            33%     {{ transform: scale(1.18) rotate(-3deg); }}
            37%     {{ transform: scale(1); }}
            100%    {{ transform: scale(1); }}
        }}
        @keyframes ecoBinDropDry {{
            0%, 80% {{ transform: scale(1); }}
            83%     {{ transform: scale(1.18) rotate(3deg); }}
            87%     {{ transform: scale(1); }}
            100%    {{ transform: scale(1); }}
        }}
        @keyframes ecoSparkleWet {{
            0%, 31% {{ opacity: 0; transform: translateY(0) scale(0.6); }}
            34%     {{ opacity: 1; transform: translateY(-10px) scale(1); }}
            40%     {{ opacity: 0; transform: translateY(-16px) scale(0.8); }}
            100%    {{ opacity: 0; }}
        }}
        @keyframes ecoSparkleDry {{
            0%, 81% {{ opacity: 0; transform: translateY(0) scale(0.6); }}
            84%     {{ opacity: 1; transform: translateY(-10px) scale(1); }}
            90%     {{ opacity: 0; transform: translateY(-16px) scale(0.8); }}
            100%    {{ opacity: 0; }}
        }}
        .eco-mascot {{
            position: fixed;
            left: 14px;
            bottom: 14px;
            width: 186px;
            height: 78px;
            z-index: 30;
            pointer-events: none;
            background: rgba(255,255,255,0.78);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border: 1px solid {PALETTE['line']};
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(15, 61, 46, 0.12);
            overflow: hidden;
        }}
        .eco-mascot .ground {{
            position: absolute;
            left: 10px; right: 10px; bottom: 20px;
            height: 2px;
            background: {PALETTE['line']};
            border-radius: 2px;
        }}
        .eco-mascot .bin {{
            position: absolute;
            bottom: 14px;
            font-size: 1.5rem;
            transform-origin: bottom center;
        }}
        .eco-mascot .bin.wet {{ left: 14px; animation: ecoBinDropWet 11s ease-in-out infinite; }}
        .eco-mascot .bin.dry {{ right: 14px; animation: ecoBinDropDry 11s ease-in-out infinite; }}
        .eco-mascot .sparkle {{
            position: absolute;
            bottom: 42px;
            font-size: 0.85rem;
        }}
        .eco-mascot .sparkle.wet {{ left: 20px; animation: ecoSparkleWet 11s ease-in-out infinite; }}
        .eco-mascot .sparkle.dry {{ right: 20px; animation: ecoSparkleDry 11s ease-in-out infinite; }}
        .eco-mascot .kid-track {{
            position: absolute;
            left: 50%; bottom: 14px;
            transform: translateX(-50%);
            animation: ecoKidWalk 11s ease-in-out infinite;
        }}
        .eco-mascot .kid {{ position: relative; font-size: 1.7rem; display: block; }}
        .eco-mascot .kid .carry {{
            position: absolute;
            top: -0.85rem; left: 50%;
            transform: translateX(-50%);
            font-size: 0.95rem;
        }}
        .eco-mascot .kid .carry.wet {{ animation: ecoItemWet 11s ease-in-out infinite; }}
        .eco-mascot .kid .carry.dry {{ animation: ecoItemDry 11s ease-in-out infinite; }}
        .eco-mascot .caption {{
            position: absolute;
            top: 4px; left: 0; right: 0;
            text-align: center;
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: {PALETTE['muted']};
        }}
        @media (max-width: 640px) {{
            .eco-mascot {{ transform: scale(0.82); transform-origin: bottom left; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .eco-mascot * {{ animation: none !important; }}
        }}

        /* =====================================================
           CELEBRATE — a one-shot CSS-only confetti burst for the
           Sorting Challenge reveal moment. No JS needed: because
           Streamlit mounts this as a brand-new DOM node each time
           the reveal renders, the "forwards, once" keyframes just
           replay naturally on every correct/trap/speed result.
        ===================================================== */
        @keyframes ecoBurst {{
            0%   {{ transform: translate(0,0) rotate(0) scale(1); opacity: 1; }}
            100% {{ transform: translate(var(--bx), var(--by)) rotate(var(--br)); opacity: 0; }}
        }}
        .eco-celebrate {{
            position: relative;
            height: 0;
            overflow: visible;
        }}
        .eco-celebrate span {{
            position: absolute;
            top: 0; left: 50%;
            font-size: 1.3rem;
            animation: ecoBurst 0.9s ease-out forwards;
        }}
        @media (prefers-reduced-motion: reduce) {{
            .eco-celebrate span {{ animation: none !important; opacity: 0; }}
        }}

        /* ---- round-progress dots (Sorting Challenge) ---- */
        .eco-progress-dots {{ display: flex; gap: 0.4rem; margin: 0.3rem 0 0.9rem; }}
        .eco-progress-dots .dot {{
            width: 26px; height: 8px; border-radius: 999px;
            background: {PALETTE['line']};
            transition: background 0.3s ease, transform 0.3s ease;
        }}
        .eco-progress-dots .dot.done {{
            background: linear-gradient(90deg, {PALETTE['mid']}, {PALETTE['lime']});
        }}
        .eco-progress-dots .dot.current {{
            background: linear-gradient(90deg, {PALETTE['sunshine']}, {PALETTE['coral']});
            transform: scaleY(1.4);
        }}
        </style>
        <div class="eco-mascot" aria-hidden="true">
            <div class="caption">sort right, every time</div>
            <span class="sparkle wet">✨</span>
            <span class="sparkle dry">✨</span>
            <span class="bin wet">🟤</span>
            <span class="bin dry">🔵</span>
            <div class="ground"></div>
            <div class="kid-track">
                <span class="kid">🧒
                    <span class="carry wet">🍌</span>
                    <span class="carry dry">📦</span>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, tagline: str, subtext: str):
    st.markdown(
        f"""
        <div class="eco-hero">
            <div class="particles">
                <span>🍃</span><span>♻️</span><span>🌿</span><span>✨</span>
            </div>
            <div class="sticker">100%<br/>AI<br/>Powered</div>
            <div class="eco-badge-row">
                <span class="chip">🤖 Instant AI ID</span>
                <span class="chip">🔍 Explainable</span>
                <span class="chip">🌍 Eco Points</span>
            </div>
            <h1>{title}</h1>
            <div class="tagline">{tagline}</div>
            <div class="subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="eco-page-header">
            <div class="title-row">
                <span class="badge">{icon}</span>
                <h2>{title}</h2>
            </div>
            <div class="underline"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)
    st.write("")


def pill(text: str, kind: str = "") -> str:
    """kind: '' (default), 'demo', 'live', 'coral', 'sky', or 'grape' —
    used for the Demo/Live data badges (and any other colorful tag) in
    the Community Hub and elsewhere."""
    cls = f"eco-pill {kind}".strip()
    return f'<span class="{cls}">{text}</span>'


def profile_card(name: str, points: int, progress_label: str = None, progress_fraction: float = None):
    """Sidebar profile card: avatar initial, name, Green Points, and an
    optional 'progress to next reward' bar for a bit of extra delight."""
    initial = (name.strip()[:1] or "?").upper()
    st.markdown(
        f"""<div class="eco-profile-card">
                <div class="eco-avatar">{initial}</div>
                <div class="name">{name}</div>
                <div class="points">🌿 {points}</div>
                <div style="font-size:0.78rem;color:{PALETTE['muted']};">Green Points</div>
            </div>""",
        unsafe_allow_html=True,
    )
    if progress_label and progress_fraction is not None:
        st.progress(min(max(progress_fraction, 0.0), 1.0))
        st.markdown(f'<div class="eco-progress-label">{progress_label}</div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""<div class="eco-empty">
                <div class="icon">{icon}</div>
                <div class="title">{title}</div>
                <div>{subtitle}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def celebration_burst(kind: str = "success"):
    """One-shot confetti burst for a result reveal moment (e.g. the
    Sorting Challenge). kind: 'success' (green/lime), 'trap' (amber/
    coral — a trap-category catch), or 'speed' (sky/grape). Call once
    per render of the reveal block; it auto-plays via CSS and never
    needs to be cleared."""
    bits_by_kind = {
        "success": ["🎉", "✨", "🌿", "♻️"],
        "trap": ["🪤", "✨", "🎯", "⭐"],
        "speed": ["⚡", "✨", "💨", "🌟"],
    }
    bits = bits_by_kind.get(kind, bits_by_kind["success"])
    spans = []
    for i in range(10):
        angle = (i / 10) * 6.283
        dist = 70 + (i % 3) * 20
        bx = round(180 * (0.5 - abs(0.5 - (i / 10))) * (1 if i % 2 == 0 else -1) + dist * 0.4, 1)
        by = round(-40 - dist, 1)
        rot = (i * 47) % 360 - 180
        emoji = bits[i % len(bits)]
        spans.append(
            f'<span style="--bx:{bx}px;--by:{by}px;--br:{rot}deg;'
            f'animation-delay:{(i % 4) * 0.03}s;">{emoji}</span>'
        )
    st.markdown(f'<div class="eco-celebrate">{"".join(spans)}</div>', unsafe_allow_html=True)


def progress_dots(current: int, total: int):
    """Round-progress indicator (1-indexed 'current' of 'total') for
    multi-round flows like the Sorting Challenge — a nicer stand-in
    for a plain 'Round 2 of 5' caption."""
    dots = []
    for i in range(1, total + 1):
        cls = "dot"
        if i < current:
            cls += " done"
        elif i == current:
            cls += " current"
        dots.append(f'<span class="{cls}"></span>')
    st.markdown(f'<div class="eco-progress-dots">{"".join(dots)}</div>', unsafe_allow_html=True)
