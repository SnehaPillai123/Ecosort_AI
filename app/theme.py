"""
Visual system for EcoSort AI — CSS injection + small HTML component
helpers, kept separate from page logic so app.py stays readable.

PURE UI/UX LAYER: every public function here (inject_css, hero,
page_header, pill, profile_card, empty_state) keeps the exact same
name and signature as before this pass. Nothing in app.py or
community_hub.py needed to change to pick up this redesign — that's
deliberate: this file is the entire blast radius of the visual
overhaul, so existing functionality can't regress by construction.

Design approach:
    - Colors/fonts for native widgets (buttons, tabs, inputs) still
      come from .streamlit/config.toml first — the version-stable way
      to theme Streamlit — with this file layering richer CSS on top.
    - Typography pairs an editorial serif display face (Fraunces) for
      the hero/page titles with a clean, modern sans (Inter) for every
      other UI surface — the serif carries the "premium product"
      feeling without hurting readability in dense dashboard views,
      where a full-serif UI would fight against scanability.
    - A warm, layered palette (deep forest, fresh moss/lime, soft mint,
      warm off-white, charcoal ink) replaces flat single-tone green so
      the app reads as a considered brand, not a template default.
    - Glass/translucent surfaces are used sparingly and deliberately —
      the hero banner and the sidebar profile card — rather than
      applied everywhere, which is what makes glassmorphism look
      premium instead of gimmicky.
    - Motion is restrained: entrance fades, hover lifts, and two slow
      (18s+) floating orbs in the hero. Nothing loops fast or fights
      for attention while someone's trying to read a result.
"""

import streamlit as st

PALETTE = {
    "forest": "#0F3D2E",      # deep forest — dominant dark surface
    "forest_2": "#123F2B",
    "mid": "#1B6B4A",         # primary brand green — buttons, links, accents
    "moss": "#5FAE83",        # supporting mid-tone
    "lime": "#A8E063",        # fresh accent — sparingly, for highlights/glow
    "mint_bg": "#EAF6EE",     # soft mint surface tint
    "mint_bg_2": "#DCEFE2",
    "cream": "#FAF9F4",       # warm off-white — page background
    "ink": "#1F2421",         # charcoal text
    "muted": "#5C6B60",
    "line": "#E1EAE3",        # hairline borders
    "amber": "#E8A34C",
    "gold": "#D9B65C",
}


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

        :root {{
            --eco-forest: {PALETTE['forest']};
            --eco-mid: {PALETTE['mid']};
            --eco-lime: {PALETTE['lime']};
            --eco-cream: {PALETTE['cream']};
            --eco-ink: {PALETTE['ink']};
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

        /* ---- Page background + layout rhythm ---- */
        [data-testid="stAppViewContainer"] {{
            background: {PALETTE['cream']};
        }}
        .block-container {{
            padding-top: 1.75rem;
            padding-bottom: 4rem;
            max-width: 1200px;
        }}
        @media (max-width: 640px) {{
            .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
            .eco-hero {{ padding: 1.6rem 1.4rem !important; }}
            .eco-hero h1 {{ font-size: 1.9rem !important; }}
        }}

        /* ---- Entrance fade — masks Streamlit's rerun flash, doesn't add to it ---- */
        .main .block-container {{ animation: ecoFadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1); }}
        @keyframes ecoFadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
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

        /* ---- Focus states (accessibility) ---- */
        button:focus-visible, [tabindex]:focus-visible, input:focus-visible,
        textarea:focus-visible, [role="radio"]:focus-visible {{
            outline: 2px solid {PALETTE['mid']} !important;
            outline-offset: 2px;
        }}

        /* ---- Buttons ---- */
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
            box-shadow: 0 8px 20px rgba(15, 61, 46, 0.14);
            transform: translateY(-1.5px);
            border-color: {PALETTE['mid']}55;
        }}
        .stButton > button:active {{ transform: translateY(0); }}
        .stButton > button[kind="primary"] {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, {PALETTE['mid']} 0%, {PALETTE['forest']} 100%);
            border: none;
            color: white;
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 10px 26px rgba(27, 107, 74, 0.35);
        }}
        /* subtle one-shot light sweep on hover — restrained, not a loop */
        .stButton > button[kind="primary"]::after {{
            content: "";
            position: absolute; top: 0; left: -60%;
            width: 40%; height: 100%;
            background: linear-gradient(120deg, transparent, rgba(255,255,255,0.28), transparent);
            transform: skewX(-20deg);
            transition: left 0.55s ease;
        }}
        .stButton > button[kind="primary"]:hover::after {{ left: 130%; }}
        .stDownloadButton > button {{ border-radius: 12px; }}

        /* ---- Spinner — brand-colored instead of default ---- */
        [data-testid="stSpinner"] > div > div {{
            border-top-color: {PALETTE['mid']} !important;
            border-right-color: {PALETTE['lime']}66 !important;
        }}

        /* ---- Metrics — elevated, icon-forward, gradient numerals ---- */
        [data-testid="stMetric"] {{
            background: white;
            border: 1px solid {PALETTE['line']};
            border-radius: 16px;
            padding: 1rem 1.1rem 0.8rem 1.1rem;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }}
        [data-testid="stMetric"]:hover {{
            box-shadow: 0 10px 28px rgba(15, 61, 46, 0.08);
            transform: translateY(-2px);
        }}
        [data-testid="stMetricValue"] {{
            background: linear-gradient(135deg, {PALETTE['forest']}, {PALETTE['mid']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
        }}
        [data-testid="stMetricLabel"] {{ color: {PALETTE['muted']}; font-weight: 500; }}

        /* ---- Native bordered containers (st.container(border=True)) — real depth, subtle top accent on hover ---- */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 18px !important;
            border-color: {PALETTE['line']} !important;
            transition: box-shadow 0.2s ease, transform 0.15s ease, border-color 0.2s ease;
            position: relative;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            box-shadow: 0 12px 32px rgba(15, 61, 46, 0.10);
            border-color: {PALETTE['mid']}44 !important;
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: {PALETTE['mid']} !important; }}
        .stTabs [aria-selected="true"] {{ color: {PALETTE['mid']} !important; font-weight: 600; }}
        .stTabs [data-baseweb="tab"] {{ font-family: 'Inter', sans-serif; }}

        /* ---- Alerts — consistent radius/elevation, semantic colors untouched for accessibility ---- */
        [data-testid="stAlert"] {{
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(15, 61, 46, 0.05);
        }}

        /* ---- Dataframes / tables — rounded, hairline border ---- */
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid {PALETTE['line']};
        }}

        /* ---- Inputs ---- */
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {{
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            border-radius: 14px !important;
            background: {PALETTE['mint_bg']}88 !important;
            border: 1.5px dashed {PALETTE['moss']}77 !important;
        }}

        /* ---- Progress bar — gradient fill, animates smoothly on change ---- */
        [data-testid="stProgress"] > div > div > div {{
            background: linear-gradient(90deg, {PALETTE['mid']}, {PALETTE['lime']}) !important;
            border-radius: 999px;
            transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {PALETTE['mint_bg']} 0%, {PALETTE['mint_bg_2']} 100%);
            border-right: 1px solid {PALETTE['line']};
        }}
        [data-testid="stSidebar"] h3 {{ font-family: 'Fraunces', serif; }}
        /* Brand/name/nav sit right at the top of the sidebar with no reserved
           header gap — since navigation runs with position="hidden" (see
           app.py), Streamlit no longer reserves its own space above this. */
        [data-testid="stSidebarContent"] {{ padding-top: 1.1rem; }}
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
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
            background: rgba(255,255,255,0.55);
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p {{
            font-size: 0.92rem;
        }}

        /* ---- Custom scrollbar ---- */
        ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: {PALETTE['moss']}99; border-radius: 8px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {PALETTE['mid']}; }}

        /* =====================================================
           HERO — full editorial banner with layered gradient mesh
           and two slow-drifting glass orbs (motion, not noise)
        ===================================================== */
        .eco-hero {{
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(ellipse 80% 60% at 15% 0%, {PALETTE['lime']}22 0%, transparent 55%),
                linear-gradient(135deg, {PALETTE['forest']} 0%, {PALETTE['forest_2']} 45%, {PALETTE['mid']} 100%);
            border-radius: 24px;
            padding: 3rem 2.8rem;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 20px 50px rgba(15, 61, 46, 0.22);
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
            background: radial-gradient(circle, {PALETTE['lime']}3d 0%, transparent 72%);
            top: -100px; right: -50px;
            animation: ecoFloatSlow 22s ease-in-out infinite;
        }}
        .eco-hero::after {{
            width: 200px; height: 200px;
            background: radial-gradient(circle, {PALETTE['gold']}30 0%, transparent 72%);
            bottom: -80px; right: 140px;
            animation: ecoFloat 26s ease-in-out infinite;
        }}
        .eco-hero h1 {{
            color: white;
            font-size: 2.9rem;
            font-weight: 600;
            line-height: 1.08;
            margin: 0 0 0.4rem 0;
            position: relative;
            max-width: 34rem;
        }}
        .eco-hero .tagline {{
            font-family: 'Fraunces', serif;
            font-style: italic;
            font-size: 1.2rem;
            color: {PALETTE['lime']};
            margin-bottom: 0.9rem;
            position: relative;
        }}
        .eco-hero .subtext {{
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
            max-width: 42rem;
            line-height: 1.6;
            position: relative;
        }}

        /* ---- Pills / tags ---- */
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
        }}
        .eco-pill.demo {{
            background: #FFF6E5; color: #9A6B00; border-color: #E8C56A66;
        }}
        .eco-pill.live {{
            background: {PALETTE['mint_bg']}; color: {PALETTE['mid']}; border-color: {PALETTE['mid']}33;
        }}

        /* ---- Sidebar profile card — glass-tinted, ring avatar ---- */
        .eco-profile-card {{
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 18px;
            padding: 1.2rem 1.1rem 1rem 1.1rem;
            border: 1px solid rgba(255,255,255,0.6);
            text-align: center;
            box-shadow: 0 6px 20px rgba(15, 61, 46, 0.08);
        }}
        .eco-avatar {{
            width: 56px; height: 56px;
            border-radius: 50%;
            background: linear-gradient(135deg, {PALETTE['mid']}, {PALETTE['forest']});
            color: white;
            display: flex; align-items: center; justify-content: center;
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-size: 1.4rem;
            margin: 0 auto 0.55rem auto;
            box-shadow: 0 0 0 4px {PALETTE['lime']}33, 0 4px 12px rgba(15, 61, 46, 0.3);
            animation: ecoRingPulse 4s ease-in-out infinite;
        }}
        @keyframes ecoRingPulse {{
            0%, 100% {{ box-shadow: 0 0 0 4px {PALETTE['lime']}33, 0 4px 12px rgba(15, 61, 46, 0.3); }}
            50%      {{ box-shadow: 0 0 0 7px {PALETTE['lime']}22, 0 4px 16px rgba(15, 61, 46, 0.35); }}
        }}
        .eco-profile-card .name {{
            font-weight: 600;
            color: {PALETTE['ink']};
            margin-bottom: 0.15rem;
        }}
        .eco-profile-card .points {{
            font-family: 'Fraunces', serif;
            font-size: 2.1rem;
            font-weight: 600;
            color: {PALETTE['mid']};
            line-height: 1.1;
        }}
        .eco-progress-label {{
            font-size: 0.78rem;
            color: {PALETTE['muted']};
            margin-top: 0.55rem;
        }}

        /* ---- Page header — icon badge + underline accent ---- */
        .eco-page-header {{ margin-bottom: 0.3rem; }}
        .eco-page-header .badge {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 2.4rem; height: 2.4rem;
            background: linear-gradient(135deg, {PALETTE['mint_bg']}, {PALETTE['mint_bg_2']});
            border: 1px solid {PALETTE['line']};
            border-radius: 12px;
            font-size: 1.25rem;
            margin-right: 0.7rem;
        }}
        .eco-page-header .title-row {{ display: flex; align-items: center; }}
        .eco-page-header h2 {{
            margin: 0; font-size: 1.6rem; display: inline;
        }}

        /* ---- Empty states — dashed, centered, inviting rather than apologetic ---- */
        .eco-empty {{
            text-align: center;
            padding: 3rem 1.5rem;
            background: linear-gradient(180deg, {PALETTE['mint_bg']}99, {PALETTE['mint_bg_2']}55);
            border: 1.5px dashed {PALETTE['moss']}88;
            border-radius: 18px;
            color: {PALETTE['muted']};
        }}
        .eco-empty .icon {{ font-size: 2.6rem; margin-bottom: 0.6rem; }}
        .eco-empty .title {{
            font-family: 'Fraunces', serif;
            font-weight: 600;
            color: {PALETTE['forest']};
            font-size: 1.15rem;
            margin-bottom: 0.35rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, tagline: str, subtext: str):
    st.markdown(
        f"""
        <div class="eco-hero">
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
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)
    st.write("")


def pill(text: str, kind: str = "") -> str:
    """kind: '' (default), 'demo', or 'live' — used for the Demo/Live
    data badges in the Community Hub."""
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
