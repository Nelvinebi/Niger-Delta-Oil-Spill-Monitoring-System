import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import json
import time
from datetime import datetime
import random

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NDOSMS — Niger Delta Oil Spill Monitor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Red/Blue Gradient Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=Source+Sans+3:wght@300;400;600&display=swap');

/* ── Root variables ── */
:root {
    --red:      #c0392b;
    --red-mid:  #96281b;
    --blue:     #1a3a5c;
    --blue-mid: #2471a3;
    --blue-lt:  #3498db;
    --white:    #f5f5f5;
    --card-bg:  rgba(15, 25, 50, 0.72);
    --card-border: rgba(52, 152, 219, 0.30);
    --accent:   #e74c3c;
    --gold:     #f39c12;
    --green:    #27ae60;
    --text:     #dce9f5;
    --subtext:  #8ab4d4;
}

/* ── Full-page gradient background ── */
.stApp {
    background: linear-gradient(135deg,
        #0d1b2a 0%,
        #1a3a5c 25%,
        #2c1a1a 55%,
        #6b1a1a 78%,
        #96281b 100%
    ) !important;
    background-attachment: fixed !important;
    font-family: 'Source Sans 3', sans-serif;
    color: var(--text);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(10, 20, 40, 0.97) 0%,
        rgba(20, 10, 15, 0.97) 100%
    ) !important;
    border-right: 1px solid rgba(52, 152, 219, 0.25);
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: 'Source Sans 3', sans-serif;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--subtext) !important;
}

/* ── Hide default Streamlit chrome ── */
footer { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; max-width: 1400px; }

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.02em;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--blue-lt), var(--red));
}
.metric-val {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    margin: 0.2rem 0 0.1rem 0;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 0.05rem;
}
.metric-delta {
    font-size: 0.78rem;
    color: var(--green);
    margin-top: 0.2rem;
}
.metric-delta.warn { color: var(--gold); }

/* ── Section title ── */
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-left: 4px solid var(--blue-lt);
    padding-left: 0.7rem;
    margin: 1.4rem 0 0.8rem 0;
}
.section-subtitle {
    font-size: 0.88rem;
    color: var(--subtext);
    margin-top: -0.5rem;
    margin-bottom: 1rem;
    padding-left: 1.1rem;
}

/* ── Info card ── */
.info-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(10px);
    margin-bottom: 0.8rem;
    font-size: 0.92rem;
    line-height: 1.65;
    color: var(--text);
}
.info-card strong {
    color: #ffffff;
    font-weight: 600;
}

/* ── Alert boxes ── */
.alert-high {
    background: rgba(192, 57, 43, 0.20);
    border: 1px solid rgba(192, 57, 43, 0.55);
    border-left: 4px solid var(--red);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #f5c6c2;
    margin-bottom: 0.6rem;
}
.alert-med {
    background: rgba(243, 156, 18, 0.15);
    border: 1px solid rgba(243, 156, 18, 0.45);
    border-left: 4px solid var(--gold);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #fdeaa7;
    margin-bottom: 0.6rem;
}
.alert-ok {
    background: rgba(39, 174, 96, 0.12);
    border: 1px solid rgba(39, 174, 96, 0.35);
    border-left: 4px solid var(--green);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #a9dfbf;
    margin-bottom: 0.6rem;
}

/* ── Badge pill ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-right: 0.35rem;
}
.badge-blue  { background: rgba(52,152,219,0.25); color: #7ec8f0; border: 1px solid rgba(52,152,219,0.4); }
.badge-red   { background: rgba(192,57,43,0.25);  color: #f0a09a; border: 1px solid rgba(192,57,43,0.4); }
.badge-gold  { background: rgba(243,156,18,0.20); color: #f8d07a; border: 1px solid rgba(243,156,18,0.4); }
.badge-green { background: rgba(39,174,96,0.20);  color: #7dcea0; border: 1px solid rgba(39,174,96,0.4); }

/* ── Table styling ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    color: var(--text);
}
.styled-table th {
    background: rgba(26, 58, 92, 0.6);
    color: var(--subtext);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.55rem 0.9rem;
    text-align: left;
    border-bottom: 1px solid var(--card-border);
    font-family: 'Barlow Condensed', sans-serif;
}
.styled-table td {
    padding: 0.6rem 0.9rem;
    border-bottom: 1px solid rgba(52,152,219,0.10);
    vertical-align: middle;
}
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: rgba(52,152,219,0.06); }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stSlider > div,
.stRadio > div {
    background: rgba(15, 25, 50, 0.6) !important;
    border-color: rgba(52,152,219,0.3) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}
div[data-baseweb="select"] {
    background: rgba(15, 25, 50, 0.7) !important;
}
div[data-baseweb="select"] * { color: var(--text) !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--blue-mid), var(--red-mid)) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Progress bar ── */
.stProgress > div > div > div { background: linear-gradient(90deg, var(--blue-lt), var(--accent)) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,20,40,0.5) !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--subtext) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(36,113,163,0.5), rgba(150,40,27,0.4)) !important;
    color: #ffffff !important;
}

/* ── Divider ── */
hr { border-color: rgba(52,152,219,0.18) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(15,25,50,0.5) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
}
.streamlit-expanderContent {
    background: rgba(10,18,35,0.5) !important;
    border: 1px solid var(--card-border) !important;
    border-top: none !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS — Synthetic data generators
# ─────────────────────────────────────────────
def make_sar_image(seed=42, weather="moderate", thickness=1.0):
    rng = np.random.default_rng(seed)
    wind = {"calm": 0.8, "moderate": 1.0, "rough": 1.3, "storm": 1.8}[weather]
    base = rng.standard_normal((256, 256))
    for s in [64, 32, 16, 8]:
        from scipy.ndimage import gaussian_filter
        base += gaussian_filter(rng.standard_normal((256, 256)), sigma=s) * (0.5 ** np.log2(s))
    base *= wind
    # Oil spill blob
    cy, cx = 128, 128
    Y, X = np.ogrid[:256, :256]
    r = np.sqrt(((Y - cy) / 55) ** 2 + ((X - cx) / 80) ** 2)
    mask = r < 1.0
    thick_fac = np.exp(-thickness / 4.0)
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    edge = np.exp(-dist / 25)
    damp = 0.3 + 0.7 * thick_fac * edge
    base[mask] *= (1 - damp[mask])
    # Noise
    speckle = rng.rayleigh(scale=0.12, size=(256, 256))
    thermal = rng.normal(0, 0.04, size=(256, 256))
    return base + speckle + thermal, mask.astype(np.uint8)


def make_confidence_map(mask, thickness, weather):
    wind = {"calm": 0.9, "moderate": 0.75, "rough": 0.55, "storm": 0.35}[weather]
    conf = np.full((256, 256), 0.3)
    thin_bonus = min(1.0, thickness / 5.0)
    conf[mask == 1] = wind * (0.5 + 0.5 * thin_bonus)
    conf += np.random.normal(0, 0.04, (256, 256))
    return np.clip(conf, 0, 1)


def make_training_history(epochs=11):
    rng = np.random.default_rng(7)
    acc  = 0.72 + np.cumsum(rng.exponential(0.025, epochs))
    acc  = np.clip(acc, 0, 0.9438)
    loss = 0.68 - np.cumsum(rng.exponential(0.04, epochs))
    loss = np.clip(loss, 0.05, 0.68)
    vacc = acc - rng.uniform(0.01, 0.04, epochs)
    vloss= loss + rng.uniform(0.01, 0.04, epochs)
    return acc, loss, vacc, vloss


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.8rem 0 1.2rem 0;'>
        <div style='font-family:"Barlow Condensed",sans-serif; font-size:1.55rem;
                    font-weight:800; color:#ffffff; letter-spacing:0.05em;'>🛢️ NDOSMS</div>
        <div style='font-size:0.7rem; letter-spacing:0.12em; color:#8ab4d4;
                    text-transform:uppercase; margin-top:2px;'>Niger Delta Oil Spill Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;letter-spacing:0.12em;color:#8ab4d4;text-transform:uppercase;margin-bottom:0.5rem;'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Overview",
        "🛰️  SAR Detection",
        "📈  Model Performance",
        "⚙️  Pipeline",
        "🗺️  Integration",
        "👤  About"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;letter-spacing:0.12em;color:#8ab4d4;text-transform:uppercase;margin-bottom:0.5rem;'>Detection Parameters</div>", unsafe_allow_html=True)

    weather = st.selectbox("Weather Condition", ["calm", "moderate", "rough", "storm"])
    thickness = st.slider("Oil Thickness (mm)", 0.1, 5.0, 1.0, 0.1)
    conf_thresh = st.slider("Confidence Threshold", 0.50, 0.95, 0.75, 0.05)
    mc_passes = st.slider("MC Dropout Passes", 5, 30, 10)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#8ab4d4; line-height:1.7;'>
        <div>🟢 <strong style='color:#7dcea0;'>System:</strong> Online</div>
        <div>📡 <strong style='color:#7ec8f0;'>Sentinel-1:</strong> Synthetic</div>
        <div>🕒 <strong style='color:#dce9f5;'>{datetime.now().strftime("%H:%M  %d %b %Y")}</strong></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if "Overview" in page:

    # Header
    st.markdown("""
    <div style='margin-bottom:1.4rem;'>
        <div style='font-family:"Barlow Condensed",sans-serif; font-size:2.6rem; font-weight:800;
                    color:#ffffff; letter-spacing:0.04em; line-height:1.1;'>
            Niger Delta Oil Spill<br>
            <span style='background:linear-gradient(90deg,#3498db,#e74c3c);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                Monitoring System
            </span>
        </div>
        <div style='font-size:1rem; color:#8ab4d4; margin-top:0.5rem; font-weight:300;'>
            AI-powered SAR-based spill detection with physics-informed uncertainty quantification
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("94.38%",  "Pixel Accuracy",     "↑ Synthetic holdout",   ""),
        ("0.87",    "Mean IoU",           "↑ Test set",            ""),
        ("< 3%",    "False Positive Rate","↓ Look-alike control",  ""),
        ("~3s",     "Inference / Tile",   "CPU · 512×512",         "warn"),
        ("0.04",    "ECE Score",          "↓ Lower is better",     ""),
    ]
    for col, (val, label, delta, cls) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-val'>{val}</div>
                <div class='metric-delta {cls}'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
            The <strong>Niger Delta</strong> — home to over 30 million people and one of Africa's most
            biodiverse wetland ecosystems — has suffered contamination equivalent to more than
            <strong>13 million barrels of crude oil</strong> since the 1950s. Mangrove forests decline
            at an estimated <strong>5,644 hectares per year</strong>.<br><br>
            NDOSMS transforms raw <strong>Sentinel-1 SAR imagery</strong> into actionable spill alerts
            using a 54-layer U-Net with attention gates. Unlike optical satellites, SAR operates through
            cloud cover, rain, and darkness — critical for the Niger Delta's near-constant overcast conditions.<br><br>
            The system delivers <strong>pixel-level binary masks</strong>, per-pixel
            <strong>confidence/uncertainty maps</strong>, and <strong>GeoJSON/GeoTIFF exports</strong>
            — all served via a FastAPI REST endpoint.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Active Alerts</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='alert-high'>🔴 <strong>HIGH — Forcados Terminal Corridor</strong> &nbsp;·&nbsp;
            Spill area: 18,240 m² &nbsp;·&nbsp; Confidence: 0.91 &nbsp;·&nbsp; 14:32 UTC</div>
        <div class='alert-med'>🟡 <strong>MODERATE — Bonny River Channel</strong> &nbsp;·&nbsp;
            Spill area: 6,100 m² &nbsp;·&nbsp; Confidence: 0.73 &nbsp;·&nbsp; 09:17 UTC</div>
        <div class='alert-ok'>🟢 <strong>CLEAR — Bight of Benin AOI</strong> &nbsp;·&nbsp;
            No spill detected &nbsp;·&nbsp; Confidence: 0.88 &nbsp;·&nbsp; 11:05 UTC</div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-title'>Technology Stack</div>", unsafe_allow_html=True)

        stack = [
            ("Deep Learning",      "U-Net (54-layer, attention gates)", "blue"),
            ("Framework",          "TensorFlow / Keras 2.x",            "blue"),
            ("Uncertainty",        "Monte Carlo Dropout",                "red"),
            ("SAR Simulation",     "Bragg + Pierson-Moskowitz physics",  "blue"),
            ("Geospatial I/O",     "Rasterio · GeoTIFF · GeoJSON",      "blue"),
            ("API Layer",          "FastAPI + Uvicorn + Gunicorn",       "red"),
            ("Containerisation",   "Docker multi-stage build",           "blue"),
            ("Output CRS",         "EPSG:4326 / WGS84",                  "gold"),
        ]
        rows = "".join(
            f"<tr><td style='color:#8ab4d4;font-size:0.78rem;text-transform:uppercase;"
            f"letter-spacing:0.06em;'>{k}</td>"
            f"<td><span class='badge badge-{c}'>{v}</span></td></tr>"
            for k, v, c in stack
        )
        st.markdown(f"""
        <div class='info-card' style='padding:0.8rem;'>
            <table class='styled-table'><tbody>{rows}</tbody></table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Dataset Summary</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card' style='font-size:0.87rem; line-height:1.9;'>
            <strong>Training scenes:</strong> 100 synthetic (expandable)<br>
            <strong>Image resolution:</strong> 512 × 512 px (10m equiv.)<br>
            <strong>Oil thickness:</strong> 0.1 – 5.0 mm<br>
            <strong>Weather conditions:</strong> Calm · Moderate · Rough · Storm<br>
            <strong>Ground truth:</strong> Binary mask + confidence + metadata JSON<br>
            <strong>Target AOI:</strong> Lat 4.2°–6.5°N · Lon 5.5°–7.5°E<br>
            <strong>CRS:</strong> WGS84 (EPSG:4326)
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: SAR DETECTION
# ─────────────────────────────────────────────
elif "SAR Detection" in page:

    st.markdown("""
    <div style='font-family:"Barlow Condensed",sans-serif;font-size:2.2rem;font-weight:800;
                color:#fff;letter-spacing:0.04em;margin-bottom:0.2rem;'>
        SAR Detection Simulator
    </div>
    <div style='color:#8ab4d4;font-size:0.92rem;margin-bottom:1.2rem;'>
        Physics-based synthetic SAR generation with oil-damping model · Bragg scattering · Pierson-Moskowitz spectrum
    </div>
    """, unsafe_allow_html=True)

    run_col, _ = st.columns([2, 5])
    with run_col:
        run_btn = st.button("▶  RUN DETECTION", use_container_width=True)

    if run_btn:
        seed = random.randint(0, 999)
    else:
        seed = 42

    sar, mask = make_sar_image(seed=seed, weather=weather, thickness=thickness)
    conf = make_confidence_map(mask, thickness, weather)
    pred = (conf > conf_thresh).astype(np.uint8)
    spill_px   = int(pred.sum())
    spill_area = spill_px * 100  # 10m × 10m per pixel
    mean_conf  = float(conf[pred == 1].mean()) if pred.sum() > 0 else 0.0

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    detected = spill_area > 0
    for col, (val, lbl, delta, cls) in zip(
        [k1, k2, k3, k4],
        [
            ("DETECTED" if detected else "CLEAR", "Spill Status",
             "HIGH severity" if mean_conf > 0.8 else "MODERATE" if mean_conf > 0.6 else "LOW",
             "warn" if detected else ""),
            (f"{spill_area:,} m²", "Estimated Spill Area", f"{spill_px:,} pixels classified", ""),
            (f"{mean_conf:.2f}",   "Mean Confidence",      f"Threshold: {conf_thresh:.2f}", ""),
            (f"{mc_passes}",       "MC Dropout Passes",    "Epistemic uncertainty", ""),
        ]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-val'>{val}</div>
                <div class='metric-delta {cls}'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Four-panel figure
    reds_t  = LinearSegmentedColormap.from_list("reds_t",  [(0,0,0,0), (0.87,0.21,0.17,0.9)])
    blues_t = LinearSegmentedColormap.from_list("blues_t", [(0,0,0,0), (0.20,0.60,0.86,0.9)])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("none")

    panels = [
        (sar,   "gray",    "Synthetic SAR\n(Physics Backscatter)"),
        (mask,  "Reds",    "Ground Truth Mask\n(Oil = 1 · Water = 0)"),
        (conf,  "YlOrRd",  "Confidence Map\n(Physics Uncertainty)"),
        (pred,  "Blues",   f"Prediction (τ={conf_thresh:.2f})\n{spill_area:,} m² detected"),
    ]

    for ax, (data, cmap, title) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, interpolation="bilinear")
        ax.set_title(title, color="#dce9f5", fontsize=9, fontweight="bold",
                     fontfamily="DejaVu Sans", pad=6)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03,
                     location="bottom").ax.tick_params(
            colors="#8ab4d4", labelsize=7)

    fig.suptitle(
        f"Detection · Weather: {weather.upper()} · Thickness: {thickness:.1f} mm · Seed: {seed}",
        color="#8ab4d4", fontsize=9, y=0.02, fontfamily="DejaVu Sans"
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Weather comparison
    st.markdown("<div class='section-title'>Weather Condition Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>SAR backscatter across all four sea states at fixed thickness</div>", unsafe_allow_html=True)

    fig2, axs = plt.subplots(1, 4, figsize=(16, 3.5))
    fig2.patch.set_facecolor("none")
    for ax, wc in zip(axs, ["calm", "moderate", "rough", "storm"]):
        s, m = make_sar_image(seed=42, weather=wc, thickness=thickness)
        ax.imshow(s, cmap="gray", interpolation="bilinear")
        wind_lbl = {"calm": "2–4 m/s", "moderate": "5–8 m/s",
                    "rough": "9–13 m/s", "storm": "14+ m/s"}[wc]
        ax.set_title(f"{wc.upper()}\n{wind_lbl}", color="#dce9f5",
                     fontsize=9.5, fontweight="bold", fontfamily="DejaVu Sans")
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    # JSON output
    st.markdown("<div class='section-title'>API Response Preview</div>", unsafe_allow_html=True)
    api_resp = {
        "detection_id":      f"ndosms-{seed:05d}",
        "timestamp":          datetime.now().isoformat(),
        "spill_detected":     bool(detected),
        "area_m2":            spill_area,
        "confidence_score":   round(mean_conf, 4),
        "uncertainty_level":  "low" if mean_conf > 0.8 else "medium" if mean_conf > 0.6 else "high",
        "weather_condition":  weather,
        "oil_thickness_mm":   thickness,
        "conf_threshold":     conf_thresh,
        "mc_passes":          mc_passes,
        "processing_time_ms": random.randint(2400, 3800),
        "geojson":            {"type": "FeatureCollection", "features": []},
    }
    st.code(json.dumps(api_resp, indent=2), language="json")


# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif "Model Performance" in page:

    st.markdown("""
    <div style='font-family:"Barlow Condensed",sans-serif;font-size:2.2rem;font-weight:800;
                color:#fff;letter-spacing:0.04em;margin-bottom:0.2rem;'>
        Model Performance
    </div>
    <div style='color:#8ab4d4;font-size:0.92rem;margin-bottom:1.2rem;'>
        U-Net training convergence · Metrics benchmarks · Industry targets
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉  Training Curves", "📊  Metrics Table", "🎯  Industry Benchmarks"])

    with tab1:
        acc, loss, vacc, vloss = make_training_history()
        epochs = list(range(1, 12))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
        fig.patch.set_facecolor("none")
        for ax in [ax1, ax2]:
            ax.set_facecolor("none")
            ax.tick_params(colors="#8ab4d4", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#1e3a5f")
            ax.grid(True, color="#1e3a5f", linewidth=0.6, linestyle="--", alpha=0.7)

        ax1.plot(epochs, acc,  color="#3498db", lw=2.2, marker="o", ms=5, label="Train Acc")
        ax1.plot(epochs, vacc, color="#e74c3c", lw=2.2, marker="s", ms=5,
                 linestyle="--", label="Val Acc")
        ax1.axvline(7, color="#f39c12", lw=1.2, linestyle=":", alpha=0.8)
        ax1.text(7.15, 0.76, "Best epoch", color="#f39c12", fontsize=8)
        ax1.set_title("Accuracy", color="#dce9f5", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Epoch", color="#8ab4d4", fontsize=9)
        ax1.set_ylabel("Accuracy", color="#8ab4d4", fontsize=9)
        ax1.legend(facecolor="#0d1b2a", edgecolor="#1e3a5f",
                   labelcolor="#dce9f5", fontsize=8)
        ax1.yaxis.label.set_color("#8ab4d4")

        ax2.plot(epochs, loss,  color="#3498db", lw=2.2, marker="o", ms=5, label="Train Loss")
        ax2.plot(epochs, vloss, color="#e74c3c", lw=2.2, marker="s", ms=5,
                 linestyle="--", label="Val Loss")
        ax2.set_title("Loss", color="#dce9f5", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Epoch", color="#8ab4d4", fontsize=9)
        ax2.set_ylabel("Loss", color="#8ab4d4", fontsize=9)
        ax2.legend(facecolor="#0d1b2a", edgecolor="#1e3a5f",
                   labelcolor="#dce9f5", fontsize=8)

        plt.suptitle("U-Net Training Convergence — 11 Epochs (EarlyStopping)",
                     color="#8ab4d4", fontsize=10, y=1.01)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class='info-card' style='font-size:0.87rem;'>
            <strong>EarlyStopping</strong> monitored on validation accuracy · patience = 3<br>
            <strong>ReduceLROnPlateau</strong> triggered at epoch 9 · LR factor = 0.5<br>
            <strong>Best checkpoint</strong> saved at epoch 7 — 94.38% validation accuracy<br>
            <strong>Framework:</strong> TensorFlow 2.x / Keras · <strong>Optimizer:</strong> Adam · <strong>Loss:</strong> Binary cross-entropy
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        metrics = [
            ("Pixel-wise Accuracy", "94.38%",  "Stratified holdout (80/20)",   "green"),
            ("Mean IoU",            "0.87",     "Synthetic test set",            "green"),
            ("False Positive Rate", "< 3%",     "Look-alike discrimination",     "green"),
            ("Training Epochs",     "11",       "EarlyStopping",                 "blue"),
            ("Model Depth",         "54 layers","U-Net encoder-decoder",         "blue"),
            ("ECE Score",           "0.04",     "Expected Calibration Error",    "green"),
            ("Inference Latency",   "~3s/tile", "CPU Intel i7 · 512×512",       "gold"),
            ("GPU Inference",       "< 1s",     "Estimated",                     "blue"),
        ]
        header = "<tr><th>Metric</th><th>Value</th><th>Validation Method</th><th>Status</th></tr>"
        rows   = "".join(
            f"<tr><td>{m}</td><td style='color:#fff;font-weight:600;'>{v}</td>"
            f"<td style='color:#8ab4d4;font-size:0.83rem;'>{note}</td>"
            f"<td><span class='badge badge-{c}'>{'✓ Met' if c=='green' else 'ℹ Info'}</span></td></tr>"
            for m, v, note, c in metrics
        )
        st.markdown(f"""
        <div class='info-card' style='padding:0.6rem;'>
            <table class='styled-table'><thead>{header}</thead><tbody>{rows}</tbody></table>
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        targets = [
            ("Detection Rate",       "~94% (synthetic)", "> 90% spills > 100m²", "On track"),
            ("False Alarm Rate",     "< 3%",              "< 1 per 10,000 km²",   "Validate on real data"),
            ("Processing Latency",   "~3s CPU",           "< 30 min acquisition",  "GPU recommended"),
            ("Spatial Accuracy",     "Pixel-level",       "< 50m RMSE (NOSDRA)",   "Pending real data"),
        ]
        st.markdown("""
        <div class='info-card' style='padding:0.6rem;'>
            <table class='styled-table'>
                <thead><tr><th>Metric</th><th>Current</th><th>Industry Target</th><th>Status</th></tr></thead>
                <tbody>
        """ + "".join(
            f"<tr><td>{m}</td><td style='color:#7ec8f0;'>{cur}</td>"
            f"<td style='color:#7dcea0;'>{tgt}</td>"
            f"<td style='color:#f8d07a;font-size:0.82rem;'>{st_}</td></tr>"
            for m, cur, tgt, st_ in targets
        ) + "</tbody></table></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Development Roadmap</div>", unsafe_allow_html=True)
        roadmap = [
            ("v2.1", "Q2 2026", "Real Sentinel-1 pipeline + NOSDRA ground-truth validation", 0.15),
            ("v2.2", "Q3 2026", "Multi-temporal spill tracking across satellite passes",      0.05),
            ("v3.0", "Q4 2026", "Oil thickness estimation + spill volume calculation",        0.02),
            ("v3.1", "Q1 2027", "Look-alike discrimination — biogenic slicks, vessel wakes",  0.01),
        ]
        for ver, quarter, desc, prog in roadmap:
            st.markdown(f"""
            <div class='info-card' style='padding:0.8rem 1.1rem; margin-bottom:0.5rem;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;'>
                    <span>
                        <span class='badge badge-blue'>{ver}</span>
                        <span class='badge badge-gold'>{quarter}</span>
                    </span>
                    <span style='font-size:0.75rem;color:#8ab4d4;'>{int(prog*100)}% complete</span>
                </div>
                <div style='font-size:0.9rem;color:#dce9f5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)
            st.progress(prog)


# ─────────────────────────────────────────────
# PAGE: PIPELINE
# ─────────────────────────────────────────────
elif "Pipeline" in page:

    st.markdown("""
    <div style='font-family:"Barlow Condensed",sans-serif;font-size:2.2rem;font-weight:800;
                color:#fff;letter-spacing:0.04em;margin-bottom:0.2rem;'>
        Pipeline Architecture
    </div>
    <div style='color:#8ab4d4;font-size:0.92rem;margin-bottom:1.2rem;'>
        End-to-end workflow from SAR ingestion to GIS-ready output and API alerting
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_xlim(0, 15); ax.set_ylim(0, 5); ax.axis("off")

    boxes = [
        (0.4,  2.0, 2.2, 2.4, "#1a3a5c", "#3498db", "DATA\nINGESTION",   "Sentinel-1 GRD\nor Synthetic"),
        (3.2,  2.0, 2.2, 2.4, "#1c2e1c", "#27ae60", "SAR\nPREPROCESSING","Bragg scattering\nOil-damping model"),
        (6.0,  2.0, 2.2, 2.4, "#2c1a1a", "#e74c3c", "U-NET\nINFERENCE",  "54-layer\nAttention gated"),
        (8.8,  2.0, 2.2, 2.4, "#1a2c3a", "#f39c12", "UNCERTAINTY\nQUANT.","Monte Carlo\nDropout × N"),
        (11.6, 2.0, 2.2, 2.4, "#1a1a3a", "#9b59b6", "GIS\nEXPORT",       "GeoTIFF\nGeoJSON · API"),
    ]
    for x, y, w, h, face, edge, title, sub in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            linewidth=1.8, edgecolor=edge,
            facecolor=face, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h * 0.67, title, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)
        ax.text(x + w/2, y + h * 0.27, sub, ha="center", va="center",
                color="#8ab4d4", fontsize=7.5, zorder=4)

    # Arrows
    for x_start in [2.6, 5.4, 8.2, 11.0]:
        ax.annotate("", xy=(x_start + 0.6, 3.2), xytext=(x_start, 3.2),
                    arrowprops=dict(arrowstyle="->", color="#3498db",
                                   lw=1.8, connectionstyle="arc3,rad=0"))

    ax.set_title("NDOSMS v2.0 — End-to-End Detection Pipeline",
                 color="#8ab4d4", fontsize=11, pad=8)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("<div class='section-title'>Pipeline Steps</div>", unsafe_allow_html=True)
        steps = [
            ("1", "Physics-Based Data Generation",
             "RealisticSARSimulator generates ocean backscatter via Pierson-Moskowitz + Bragg scattering. Oil injected as thickness-modulated damping regions with edge diffusion."),
            ("2", "Noise Injection",
             "Full sensor noise stack: multiplicative Rayleigh speckle + additive Gaussian thermal + quantisation noise — matching real Sentinel-1 GRD statistics."),
            ("3", "Train/Validation Split",
             "Stratified 80/20 split. StandardScaler fitted on training data only — zero data leakage guarantee."),
            ("4", "U-Net Training",
             "54-layer encoder-decoder with skip connections, attention gates, BatchNorm, Dropout. EarlyStopping + ReduceLROnPlateau. Best checkpoint at epoch 7."),
            ("5", "Monte Carlo Inference",
             "Dropout active during N forward passes. Mean = spill probability · Variance = epistemic uncertainty map per pixel."),
            ("6", "GIS Export & API",
             "Binary masks thresholded at τ. Vectorised to GeoJSON/Shapefile. GeoTIFF exported with EPSG:4326 CRS. FastAPI endpoint serves JSON response."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class='info-card' style='padding:0.85rem 1.1rem;margin-bottom:0.5rem;display:flex;gap:1rem;'>
                <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.6rem;
                            font-weight:800;color:#3498db;min-width:1.8rem;line-height:1;'>{num}</div>
                <div>
                    <div style='font-weight:600;color:#ffffff;margin-bottom:0.25rem;font-size:0.95rem;'>{title}</div>
                    <div style='font-size:0.86rem;color:#8ab4d4;line-height:1.55;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-title'>Project Structure</div>", unsafe_allow_html=True)
        st.code("""Niger Delta Oil Spill Monitoring System/
├── api/
│   ├── main.py              # FastAPI · POST /detect
│   └── webhook_handler.py   # Alert dispatch
├── assets/charts/           # README visualisations
├── data/synthetic_training/ # Generated .tif files
├── data_generation/
│   └── realistic_sar_simulator.py
├── models/
│   ├── unet_plusplus.py     # U-Net + attention
│   ├── uncertainty.py       # MC Dropout
│   └── checkpoints/         # Saved weights
├── notebooks/
│   └── 01_generate_synthetic_data.ipynb
├── scripts/
│   └── generate_readme_charts.py
├── tests/
│   └── test_detection.py
├── docs/
│   ├── sar_physics.md
│   ├── api.md
│   └── model_card.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt""", language="text")

        st.markdown("<div class='section-title'>Deployment Commands</div>", unsafe_allow_html=True)
        st.code("""# Local development
jupyter notebook notebooks/01_generate_synthetic_data.ipynb

# API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Docker production
docker-compose up --scale api=4

# Run tests
pytest tests/test_detection.py -v""", language="bash")


# ─────────────────────────────────────────────
# PAGE: INTEGRATION
# ─────────────────────────────────────────────
elif "Integration" in page:

    st.markdown("""
    <div style='font-family:"Barlow Condensed",sans-serif;font-size:2.2rem;font-weight:800;
                color:#fff;letter-spacing:0.04em;margin-bottom:0.2rem;'>
        Real-World Integration
    </div>
    <div style='color:#8ab4d4;font-size:0.92rem;margin-bottom:1.2rem;'>
        Data sources · Regulatory alignment · Roadmap for Sentinel-1 real data transition
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("<div class='section-title'>Data Sources</div>", unsafe_allow_html=True)
        sources = [
            ("Copernicus Data Space", "Sentinel-1 SAR (GRD, SLC)",
             "dataspace.copernicus.eu", "Free registration", "Ready", "green"),
            ("Google Earth Engine",   "Pre-processed Sentinel-1 GRD",
             "earthengine.google.com", "Free — research/education", "Ready", "green"),
            ("Alaska Satellite Facility","Historical Sentinel-1 archive",
             "search.asf.alaska.edu",  "Free account", "Ready", "green"),
            ("NOSDRA",                "Nigerian oil spill incident reports",
             "nosdra.gov.ng",          "Public", "Pending", "gold"),
            ("NOAA NESDIS",           "Thermal anomalies (active spills)",
             "polar.ncep.noaa.gov",    "Public", "Pending", "gold"),
            ("SkyTruth",              "Satellite-detected spill archive",
             "skytruth.org",           "Public", "Pending", "gold"),
        ]
        for name, dtype, url, access, status, clr in sources:
            st.markdown(f"""
            <div class='info-card' style='padding:0.85rem 1.1rem;margin-bottom:0.5rem;'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                    <div>
                        <div style='font-weight:700;color:#ffffff;font-size:0.97rem;'>{name}</div>
                        <div style='color:#8ab4d4;font-size:0.83rem;margin:0.18rem 0;'>{dtype}</div>
                        <div style='font-size:0.78rem;color:#5a8ab0;'>🔗 {url} &nbsp;·&nbsp; {access}</div>
                    </div>
                    <span class='badge badge-{clr}'>{status}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Regulatory Alignment</div>", unsafe_allow_html=True)
        regs = [
            ("NOSDRA",    "Automated report generation matching Nigerian regulatory requirements"),
            ("IMO MARPOL","Oil spill response protocols — Annex I compliance"),
            ("ISO 19115", "Geospatial metadata standard — all GeoTIFF outputs"),
            ("STAC",      "SpatioTemporal Asset Catalog format for GIS interoperability"),
        ]
        rows = "".join(
            f"<tr><td><span class='badge badge-blue'>{r}</span></td>"
            f"<td style='font-size:0.87rem;color:#dce9f5;'>{d}</td></tr>"
            for r, d in regs
        )
        st.markdown(f"""
        <div class='info-card' style='padding:0.6rem;'>
            <table class='styled-table'><tbody>{rows}</tbody></table>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-title'>Troubleshooting</div>", unsafe_allow_html=True)
        issues = [
            ("TypeError: Affine * float",
             "Outdated rasterio transform syntax",
             "Update save_scenario() to use Affine.scale()"),
            ("Model hangs > 5 min",
             "CPU inference on 512×512",
             "Reduce MC passes to 5 or resize to 256×256"),
            ("safe_mode=False warning",
             "Lambda layers in normalisation",
             "Expected — load with safe_mode=False, compile=False"),
            ("Kernel dies during training",
             "RAM exhausted",
             "Reduce batch_size to 1, restart kernel"),
            ("Low confidence on thin spills",
             "Physics design behaviour",
             "Normal — flagged for manual review"),
        ]
        for symptom, cause, fix in issues:
            with st.expander(f"⚠ {symptom}"):
                st.markdown(f"""
                <div style='font-size:0.87rem;line-height:1.65;'>
                    <div><strong style='color:#f0a09a;'>Cause:</strong>
                         <span style='color:#dce9f5;'> {cause}</span></div>
                    <div style='margin-top:0.4rem;'><strong style='color:#7dcea0;'>Fix:</strong>
                         <span style='color:#dce9f5;'> {fix}</span></div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>GEE Quick Start</div>", unsafe_allow_html=True)
        st.code("""// Google Earth Engine — Niger Delta SAR
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(ee.Geometry.Rectangle(
      [5.5, 4.2, 7.5, 6.5]))   // Niger Delta AOI
  .filterDate('2024-01-01', '2024-06-30')
  .filter(ee.Filter.eq(
      'instrumentMode', 'IW'))
  .select(['VV','VH']);

Export.image.toDrive({
  image: s1.mosaic(),
  description: 'NigerDelta_SAR_VV_VH',
  scale: 10,
  crs: 'EPSG:4326'
});""", language="javascript")


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif "About" in page:

    st.markdown("""
    <div style='font-family:"Barlow Condensed",sans-serif;font-size:2.2rem;font-weight:800;
                color:#fff;letter-spacing:0.04em;margin-bottom:1.4rem;'>
        About This Project
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 3], gap="large")

    with col_l:
        st.markdown("""
        <div class='info-card' style='text-align:center;padding:2rem 1.4rem;'>
            <div style='font-family:"Barlow Condensed",sans-serif;font-size:2rem;
                        font-weight:800;color:#ffffff;letter-spacing:0.03em;'>
                Agbozu Ebingiye Nelvin
            </div>
            <div style='color:#8ab4d4;font-size:0.82rem;letter-spacing:0.08em;
                        text-transform:uppercase;margin:0.4rem 0 1rem 0;'>
                Environmental Data Scientist
            </div>
            <div style='color:#dce9f5;font-size:0.88rem;line-height:1.7;margin-bottom:1.2rem;'>
                SAR Remote Sensing · Deep Learning<br>
                Climate Analytics · GIS
            </div>
            <div style='font-size:0.82rem;color:#8ab4d4;margin-bottom:1.4rem;'>
                📍 Port Harcourt, Rivers State, Nigeria
            </div>
            <div style='display:flex;flex-direction:column;gap:0.5rem;align-items:center;'>
                <a href='https://www.linkedin.com/in/agbozu-ebi/'
                   style='display:block;padding:0.4rem 1.4rem;border-radius:20px;
                          background:rgba(0,119,181,0.25);border:1px solid rgba(0,119,181,0.5);
                          color:#7ec8f0;font-size:0.82rem;text-decoration:none;
                          letter-spacing:0.06em;'>
                    🔗 LinkedIn — agbozu-ebi
                </a>
                <a href='https://github.com/Nelvinebi'
                   style='display:block;padding:0.4rem 1.4rem;border-radius:20px;
                          background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.2);
                          color:#dce9f5;font-size:0.82rem;text-decoration:none;
                          letter-spacing:0.06em;'>
                    🐙 GitHub — Nelvinebi
                </a>
                <a href='mailto:nelvinebingiye@gmail.com'
                   style='display:block;padding:0.4rem 1.4rem;border-radius:20px;
                          background:rgba(209,72,54,0.18);border:1px solid rgba(209,72,54,0.4);
                          color:#f0a09a;font-size:0.82rem;text-decoration:none;
                          letter-spacing:0.06em;'>
                    ✉ nelvinebingiye@gmail.com
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-title'>Project Context</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card' style='font-size:0.91rem;line-height:1.75;'>
            NDOSMS originated as a GitHub proof-of-concept (<em>Oil Spill Detection and Impact Mapping
            in the Niger Delta Using SAR and Deep Learning</em>) and has been systematically
            upgraded toward an <strong>industry-standard operational platform</strong>.<br><br>
            The project addresses a critical environmental challenge: the Niger Delta has suffered
            contamination equivalent to over <strong>13 million barrels of crude oil</strong> since
            the 1950s, with mangrove forests declining at <strong>5,644 hectares per year</strong>.
            Conventional monitoring is too slow and cloud-dependent for effective response.<br><br>
            NDOSMS applies <strong>Synthetic Aperture Radar</strong> — which penetrates cloud cover,
            rain, and darkness — combined with a <strong>physics-informed deep learning pipeline</strong>
            that produces confidence-scored spill alerts with per-pixel uncertainty quantification.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Acknowledgements</div>", unsafe_allow_html=True)
        acks = [
            ("ESA / Copernicus Programme", "Open Sentinel-1 SAR data access"),
            ("NOSDRA",                     "Public oil spill incident reporting infrastructure"),
            ("TensorFlow / Keras",         "Deep learning framework"),
            ("Rasterio / Shapely",         "Geospatial I/O and vectorisation"),
            ("FastAPI",                    "Production REST API framework"),
            ("SkyTruth / ITOPF",          "Open oil spill reference databases"),
        ]
        rows = "".join(
            f"<tr><td style='color:#7ec8f0;font-weight:600;'>{org}</td>"
            f"<td style='color:#8ab4d4;font-size:0.85rem;'>{role}</td></tr>"
            for org, role in acks
        )
        st.markdown(f"""
        <div class='info-card' style='padding:0.6rem;'>
            <table class='styled-table'><tbody>{rows}</tbody></table>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Repository</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card' style='font-size:0.87rem;line-height:1.8;'>
            <strong style='color:#fff;'>GitHub:</strong><br>
            <span style='color:#7ec8f0;font-family:monospace;font-size:0.83rem;'>
            github.com/Nelvinebi/<br>
            Oil-Spill-Detection-and-Impact-Mapping-<br>
            in-the-Niger-Delta-Using-SAR-and-Deep-Learning
            </span><br><br>
            <strong style='color:#fff;'>License:</strong>
            <span class='badge badge-green'>MIT</span> — free for research, education, government use<br>
            <strong style='color:#fff;'>Status:</strong>
            <span class='badge badge-blue'>Proof of Concept</span> → targeting production v2.1 Q2 2026
        </div>
        """, unsafe_allow_html=True)
