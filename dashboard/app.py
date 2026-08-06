"""
SIADS 699 — Sleep & Lifestyle in the All of Us Cohort
Streamlit Dashboard  |  Team Sleep Deprived  |  2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from html import escape

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Sleep & Lifestyle — All of Us",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG    = '#07131F'
WHITE = '#0F1B2A'
DGRAY = '#F8FAFC'
MGRAY = '#C7D4E5'
LGRAY = '#2B4056'
VGRAY = '#16283A'
RED   = '#F26D5B'
BLUE  = '#4FA3D9'
GREEN = '#5CC8A1'
AMBER = '#E9B44C'
PURPLE = '#9B8AE6'

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(92,200,161,0.15), transparent 34rem),
        radial-gradient(circle at left 18rem, rgba(79,163,217,0.10), transparent 30rem),
        linear-gradient(180deg, #07131F 0%, #0B1725 48%, #101827 100%) !important;
}
[data-testid="stHeader"] {
    background: rgba(7,19,31,0.86) !important;
}
.block-container {
    max-width: 1240px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, li, label, span, div {
    letter-spacing: 0 !important;
}
h1 {
    color: #F8FAFC !important;
    font-weight: 750 !important;
    line-height: 1.08 !important;
}
p, li, label {
    color: #DDE6F3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #07131F 0%, #0F2335 100%) !important;
}
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
[data-testid="stSidebar"] .stRadio label { color: #e8eaf6 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 8px;
    padding: 6px 8px;
    margin: 2px 0;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.08);
}
/* Targets the st.container(key="topnav") below. A raw <div> written around
   st.radio() does not wrap it — Streamlit closes the markdown block first, so
   the div renders empty and the radio lands outside it, unstyled. */
.st-key-topnav {
    position: sticky;
    top: 0;
    z-index: 5;
    background: rgba(7,19,31,0.94);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 8px;
    padding: 8px 10px;
    margin: -8px 0 18px;
    backdrop-filter: blur(10px);
}
.st-key-topnav [role="radiogroup"] {
    gap: 6px;
}
.st-key-topnav [role="radiogroup"] label {
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 8px;
    padding: 7px 10px;
    background: rgba(15,27,42,0.72);
}

/* Metric cards */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, minmax(142px, 1fr));
    gap: 14px;
    margin: 16px 0;
}
.kpi-card {
    background: rgba(15,27,42,0.88); border-radius: 8px;
    padding: 22px 18px; text-align: center;
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
    border: 1px solid rgba(148,163,184,0.18);
    border-top: 4px solid var(--accent, #3B7EC8);
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 18px 38px rgba(59,126,200,0.20); }
.kpi-card.primary {
    grid-column: span 2;
    text-align: left;
    background: linear-gradient(135deg, rgba(79,163,217,0.22), rgba(92,200,161,0.13));
}
.kpi-val   { font-size: clamp(1.45rem, 2.6vw, 2rem); font-weight: 750; color: var(--accent, #3B7EC8); margin: 0; line-height: 1; }
.kpi-label { font-size: 0.78rem; font-weight: 600; color: #CBD5E1; margin: 6px 0 2px;
             letter-spacing: 0; }
.kpi-sub   { font-size: 0.75rem; color: #94A3B8; margin: 0; }

/* Insight chips */
.insight-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(170px, 1fr));
    gap: 12px;
    margin: 16px 0 8px;
}
.insight {
    background: rgba(15,27,42,0.86);
    border: 1px solid rgba(148,163,184,0.18);
    border-left: 5px solid var(--accent, #3B7EC8);
    border-radius: 8px;
    padding: 14px 16px;
    min-height: 92px;
}
.insight-val {
    color: var(--accent, #3B7EC8);
    font-size: 1.45rem;
    font-weight: 750;
    line-height: 1;
    margin: 0 0 7px;
}
.insight-label {
    color: #F8FAFC;
    font-size: 0.9rem;
    font-weight: 650;
    margin: 0 0 4px;
}
.insight-sub {
    color: #CBD5E1;
    font-size: 0.78rem;
    margin: 0;
    line-height: 1.35;
}

/* Callout boxes */
.box {
    border-radius: 8px; padding: 14px 18px; margin: 10px 0;
    font-size: 0.92rem; line-height: 1.55; color: #F8FAFC;
    border: 1px solid rgba(148,163,184,0.18);
}
.box-blue  { background: rgba(79,163,217,0.15); border-left: 4px solid #4FA3D9; }
.box-red   { background: rgba(242,109,91,0.15); border-left: 4px solid #F26D5B; }
.box-green { background: rgba(92,200,161,0.14); border-left: 4px solid #5CC8A1; }
.box b     { color: #FFFFFF; }

/* Cluster cards */
.cl-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(180px, 1fr));
    gap: 14px;
    margin: 16px 0;
}
.cl-card {
    background: rgba(15,27,42,0.9); border-radius: 8px; padding: 18px 16px;
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
    transition: transform 0.15s, box-shadow 0.15s;
}
.cl-card:hover { transform: translateY(-2px); box-shadow: 0 18px 38px rgba(0,0,0,0.26); }
.cl-icon  { font-size: 1.6rem; margin-bottom: 6px; }
.cl-name  { font-weight: 700; font-size: 0.88rem; margin: 6px 0 4px; }
.cl-n     { font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin: 0; }
.cl-pct   { font-size: 0.78rem; color: #94A3B8; margin: 0 0 10px; }
.cl-divider { border: none; border-top: 1px solid rgba(148,163,184,0.18); margin: 10px 0; }
.cl-stat  { font-size: 0.82rem; color: #CBD5E1; margin: 3px 0; }

/* Section header */
.section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    color: #94A3B8; margin: 24px 0 8px;
}

/* Clinical summary */
.clinical-hero {
    display: grid;
    grid-template-columns: minmax(260px, 1.15fr) minmax(260px, 0.85fr);
    gap: 16px;
    margin: 18px 0 20px;
}
.clinical-panel {
    background: linear-gradient(135deg, rgba(15,27,42,0.96), rgba(14,43,55,0.9));
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 8px;
    padding: 22px;
    box-shadow: 0 18px 44px rgba(0,0,0,0.28);
}
.clinical-label {
    color: #8FB7CF;
    font-size: 0.76rem;
    font-weight: 700;
    margin: 0 0 8px;
}
.clinical-score {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 8px;
}
.clinical-score strong {
    color: #F8FAFC;
    font-size: clamp(2.2rem, 4.5vw, 4rem);
    line-height: 1;
}
.clinical-score span {
    color: #9BE7C9;
    font-weight: 700;
    font-size: 1rem;
}
.clinical-copy {
    color: #C7D4E5;
    max-width: 62ch;
    margin: 0 0 16px;
    line-height: 1.5;
}
.status-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(110px, 1fr));
    gap: 10px;
}
.status-pill {
    border-radius: 8px;
    padding: 12px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(148,163,184,0.16);
}
.status-pill b {
    display: block;
    color: var(--accent);
    font-size: 1.05rem;
    margin-bottom: 3px;
}
.status-pill span {
    color: #CBD5E1;
    font-size: 0.78rem;
}
.reference-card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
/* A 4–9 hour axis. The colour stops are the 6h and 7h marks, so the band a
   value sits in is the band it actually falls in. */
.range-track {
    position: relative;
    height: 14px;
    border-radius: 999px;
    background: linear-gradient(90deg, #F26D5B 0 40%, #E9B44C 40% 60%, #5CC8A1 60% 100%);
    margin: 30px 0 10px;
}
.range-marker {
    position: absolute;
    top: -6px;
    left: var(--pos, 55%);
    width: 4px;
    height: 26px;
    border-radius: 999px;
    background: #F8FAFC;
    box-shadow: 0 0 0 4px rgba(248,250,252,0.14);
}
.range-marker b {
    position: absolute;
    left: 50%;
    bottom: 30px;
    transform: translateX(-50%);
    white-space: nowrap;
    color: #F8FAFC;
    font-size: 0.76rem;
}
.range-labels {
    position: relative;
    height: 16px;
    color: #94A3B8;
    font-size: 0.75rem;
}
.range-labels span {
    position: absolute;
    transform: translateX(-50%);
}
.phenotype-bars {
    display: grid;
    gap: 10px;
}
.phenotype-bar {
    display: grid;
    grid-template-columns: minmax(110px, 1fr) 3fr 42px;
    gap: 10px;
    align-items: center;
    color: #CBD5E1;
    font-size: 0.8rem;
}
.phenotype-fill {
    height: 9px;
    border-radius: 999px;
    background: rgba(148,163,184,0.18);
    overflow: hidden;
}
.phenotype-fill span {
    display: block;
    height: 100%;
    width: var(--w);
    border-radius: inherit;
    background: var(--accent);
}

/* Compact tables rendered as responsive cards */
.table-card {
    background: rgba(16,24,39,0.92);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 8px;
    overflow-x: auto;
    box-shadow: 0 1px 5px rgba(0,0,0,0.04);
}
.mini-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 360px;
}
.mini-table th {
    background: rgba(15,23,42,0.92);
    color: #CBD5E1;
    font-size: 0.73rem;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(148,163,184,0.18);
}
.mini-table td {
    color: #F8FAFC;
    font-size: 0.86rem;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(148,163,184,0.12);
    white-space: nowrap;
}
.mini-table tr:last-child td {
    border-bottom: 0;
}
.rank-pill {
    display: inline-block;
    min-width: 28px;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(59,126,200,0.22);
    color: #3B7EC8;
    font-weight: 700;
    text-align: center;
}

.read-note {
    background: rgba(79,163,217,0.10);
    border: 1px solid rgba(79,163,217,0.24);
    border-left: 4px solid #4FA3D9;
    border-radius: 8px;
    padding: 11px 14px;
    margin: 10px 0 16px;
    color: #DDE6F3;
    font-size: 0.92rem;
    line-height: 1.5;
}
.read-note b {
    color: #FFFFFF;
}
.takeaway {
    background: rgba(92,200,161,0.11);
    border: 1px solid rgba(92,200,161,0.26);
    border-left: 5px solid #5CC8A1;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 12px 0 18px;
    color: #EAF7F1;
    font-size: 1rem;
    line-height: 1.5;
}
.takeaway b {
    color: #FFFFFF;
}
.profile-table {
    display: grid;
    gap: 8px;
    margin: 12px 0 18px;
}
.profile-row {
    display: grid;
    grid-template-columns: 190px repeat(6, minmax(110px, 1fr));
    gap: 8px;
    align-items: stretch;
}
.profile-head,
.profile-name,
.profile-cell {
    border-radius: 8px;
    border: 1px solid rgba(148,163,184,0.16);
    padding: 10px 11px;
}
.profile-head {
    color: #94A3B8;
    font-size: 0.78rem;
    font-weight: 700;
    background: rgba(15,23,42,0.72);
}
.profile-name {
    color: #F8FAFC;
    font-weight: 700;
    background: rgba(15,27,42,0.88);
}
.profile-cell {
    background: rgba(15,27,42,0.78);
}
.profile-cell b {
    display: block;
    color: #F8FAFC;
    font-size: 0.9rem;
}
.profile-cell span {
    display: block;
    color: #CBD5E1;
    font-size: 0.75rem;
    margin-top: 3px;
}
.profile-cell.high { border-left: 5px solid #E9B44C; }
.profile-cell.low { border-left: 5px solid #4FA3D9; }
.profile-cell.mid { border-left: 5px solid #64748B; }

@media (max-width: 1100px) {
    .kpi-row { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
    .insight-grid { grid-template-columns: repeat(2, minmax(170px, 1fr)); }
    .cl-grid { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
    .clinical-hero { grid-template-columns: 1fr; }
    .profile-row { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1.25rem;
    }
    h1 { font-size: 2rem !important; }
    .kpi-row, .insight-grid, .cl-grid {
        grid-template-columns: 1fr;
    }
    .kpi-card.primary { grid-column: span 1; }
    .kpi-card, .insight, .cl-card {
        padding: 16px 14px;
    }
    .kpi-card {
        text-align: left;
    }
    .status-row { grid-template-columns: 1fr; }
    .phenotype-bar { grid-template-columns: 1fr; gap: 5px; }
    .clinical-panel { padding: 16px; }
    .clinical-copy { font-size: 0.92rem; }
    .top-nav {
        margin-top: -4px;
        overflow-x: auto;
    }
    .top-nav [role="radiogroup"] {
        flex-wrap: nowrap;
        min-width: max-content;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(text, kind='blue'):
    st.markdown(f'<div class="box box-{kind}">{text}</div>', unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

def read_note(text):
    st.markdown(f'<div class="read-note">{text}</div>', unsafe_allow_html=True)

def takeaway(text):
    st.markdown(f'<div class="takeaway">{text}</div>', unsafe_allow_html=True)

def insight_cards(cards):
    html = '<div class="insight-grid">'
    for card in cards:
        html += f"""
        <div class="insight" style="--accent:{card['color']}">
            <p class="insight-val">{card['value']}</p>
            <p class="insight-label">{card['label']}</p>
            <p class="insight-sub">{card['sub']}</p>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def mini_table(df, formats=None, rank=False):
    formats = formats or {}
    headers = ''.join(f'<th>{escape(str(c))}</th>' for c in df.columns)
    rows = []
    for ridx, (_, row) in enumerate(df.iterrows(), start=1):
        cells = []
        for c in df.columns:
            val = row[c]
            if c in formats:
                val = formats[c](val)
            elif isinstance(val, float):
                val = f"{val:.3f}"
            cell = f'<span class="rank-pill">{ridx}</span>' if rank and c == df.columns[0] else escape(str(val))
            if rank and c == df.columns[0]:
                cell += f" {escape(str(row[c]))}"
            cells.append(f'<td>{cell}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f'<div class="table-card"><table class="mini-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )

def profile_table(raw, z, features, labels):
    header = '<div class="profile-row"><div class="profile-head">Group</div>'
    header += ''.join(f'<div class="profile-head">{escape(label)}</div>' for label in labels)
    header += '</div>'
    rows = [header]
    for group in z.index:
        row_html = f'<div class="profile-row"><div class="profile-name">{escape(str(group))}</div>'
        for feat, label in zip(features, labels):
            val = raw.loc[group, feat]
            score = z.loc[group, feat]
            if score > 0.75:
                status, cls = "Above avg", "high"
            elif score < -0.75:
                status, cls = "Below avg", "low"
            else:
                status, cls = "Typical", "mid"
            if feat == 'pct_short':
                shown = f"{val:.0%}"
            elif feat == 'mean_steps':
                shown = f"{val:,.0f}"
            else:
                shown = f"{val:.1f}"
            row_html += f'<div class="profile-cell {cls}"><b>{escape(shown)}</b><span>{status}</span></div>'
        row_html += '</div>'
        rows.append(row_html)
    st.markdown(f'<div class="profile-table">{"".join(rows)}</div>', unsafe_allow_html=True)

def chart_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(LGRAY)
    ax.spines['bottom'].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=9.5)
    ax.yaxis.grid(True, color=LGRAY, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax

# ── Data ──────────────────────────────────────────────────────────────────────
DATA = os.path.dirname(os.path.abspath(__file__)) + "/"

# Regenerated from the analysis rather than transcribed. Every file is an aggregate —
# model scores, subgroup R², cluster means, feature importances — with no participant rows.
@st.cache_data
def load():
    cv      = pd.read_csv(DATA + "cv_results.csv")
    fair    = pd.read_csv(DATA + "fairness.csv")
    imp_d   = pd.read_csv(DATA + "importances_duration.csv")
    imp_c   = pd.read_csv(DATA + "importances_consistency.csv")
    coef_d  = pd.read_csv(DATA + "coefficients_duration.csv")
    coef_c  = pd.read_csv(DATA + "coefficients_consistency.csv")
    cohort  = pd.read_csv(DATA + "cohort_summary.csv").set_index("metric")["value"]

    # The profile table keeps the pipeline's own column names; the shorter ones
    # below are what this page's layout code expects.
    cl = pd.read_csv(DATA + "cluster_profiles.csv").rename(columns={
        "phenotype":        "cluster_label",
        "mean_sleep_hrs":   "mean_sleep",
        "std_sleep_hrs":    "mean_sd",
        "pct_short_sleep":  "pct_short",
        "mean_daily_steps": "mean_steps",
        "age":              "mean_age",
        "bmi":              "mean_bmi",
    })
    return cv, fair, imp_d, imp_c, coef_d, coef_c, cl, cohort

cv, fair, imp_d, imp_c, coef_d, coef_c, cl, cohort = load()

N_PARTICIPANTS = int(cohort["n_participants"])

# The nine numeric features plus the one-hot dummies, named for a general reader.
# Mirrors src/viz.LABELS so the dashboard and the report figures agree.
FEAT = {
    'log_steps':'Daily steps (log)', 'steps_cv':'Activity irregularity',
    'n_valid_nights':'Nights tracked', 'age':'Age', 'bmi':'BMI',
    'employed':'Employed', 'education_num':'Education', 'health_num':'Self-rated health',
    'income_num':'Income',
    'gender_Male':'Male', 'gender_Other':'Gender: other',
    'race_ethnicity_White':'Race/Ethnicity: White',
    'race_ethnicity_Black or African American':'Race/Ethnicity: Black',
    'race_ethnicity_Hispanic or Latino':'Race/Ethnicity: Hispanic',
    'race_ethnicity_More than one population':'Multiracial',
    'race_ethnicity_Other':'Race/Ethnicity: other',
    'race_ethnicity_Unknown':'Race/Ethnicity: unknown',
}

def clean_imp(df, col='importance'):
    """A feature-indexed Series with reader-facing names, largest first."""
    s = df.set_index('feature')[col]
    s.index = [FEAT.get(i, i) for i in s.index]
    return s.sort_values(ascending=False)

PAGES = ["Overview", "Models", "Sleep Phenotypes", "Fairness", "Feature Importance"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Sleep & Lifestyle")
    st.markdown("**All of Us Research Program**")
    st.markdown("SIADS 699 &nbsp;·&nbsp; Team Sleep Deprived &nbsp;·&nbsp; 2026", unsafe_allow_html=True)
    st.caption("Sophia Boettcher · Auston Balwinski\nHunter Belous · Jared Fox")
    st.divider()
    st.markdown(f"**Cohort:** {N_PARTICIPANTS:,} participants  \n**Source:** All of Us CDR v9  \n**Device:** Fitbit")
    st.divider()
    st.caption("Aggregate statistics only.\nNo individual-level data exported from Workbench.")

with st.container(key="topnav"):
    page = st.radio("Choose a dashboard section", PAGES, horizontal=True,
                    label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Sleep & Lifestyle in the All of Us Cohort")
    st.markdown(f"*Cohort-level sleep health, lifestyle signals, and model equity in {N_PARTICIPANTS:,} Fitbit wearers.*")

    best_cons = cv[cv.target == "consistency"].sort_values("r2_mean", ascending=False).iloc[0]
    shares = (cl.set_index("cluster_label")["N"] / cl["N"].sum() * 100)

    # Derived rather than typed in, so the headline gaps cannot drift from fairness.csv.
    age_dur = fair[(fair.target == "duration") & (fair.stratum == "age_band")].set_index("subgroup")["r2"]
    race_cons = fair[(fair.target == "consistency") & (fair.stratum == "race_ethnicity")].set_index("subgroup")["r2"]
    age_gap = (age_dur["18-40"] - age_dur["81+"]) / age_dur["18-40"] * 100
    race_gaps = sorted((race_cons["White"] - race_cons[g]) / race_cons["White"] * 100
                       for g in ("Black or African American", "Asian"))

    # Where the cohort mean falls on the 4–9h track the reference bar draws.
    TRACK_LO, TRACK_HI = 4.0, 9.0
    mean_pos = (cohort["mean_sleep_hrs"] - TRACK_LO) / (TRACK_HI - TRACK_LO) * 100

    st.markdown(f"""
    <div class="clinical-hero">
      <div class="clinical-panel">
        <p class="clinical-label">Cohort sleep health signal</p>
        <div class="clinical-score"><strong>{cohort['mean_sleep_hrs']:.2f}</strong><span>hours/night average</span></div>
        <p class="clinical-copy">
          Average sleep duration sits just below the 7-hour reference threshold. Short sleep and night-to-night
          variability identify a meaningful risk segment for follow-up.
        </p>
        <div class="status-row">
          <div class="status-pill" style="--accent:#E9B44C"><b>Below reference</b><span>Mean duration under 7h</span></div>
          <div class="status-pill" style="--accent:#F26D5B"><b>{cohort['pct_short_nights']:.0f}% burden</b><span>Nights under 6 hours</span></div>
          <div class="status-pill" style="--accent:#5CC8A1"><b>Consistency signal</b><span>More predictable than duration</span></div>
        </div>
      </div>
      <div class="clinical-panel reference-card">
        <div>
          <p class="clinical-label">Sleep duration reference</p>
          <div class="range-track">
            <span class="range-marker" style="--pos:{mean_pos:.1f}%"><b>cohort mean {cohort['mean_sleep_hrs']:.2f}h</b></span>
          </div>
          <div class="range-labels">
            <span style="left:0%">4h</span><span style="left:40%">6h</span>
            <span style="left:60%">7h guideline</span><span style="left:100%">9h</span>
          </div>
        </div>
        <div>
          <p class="clinical-label" style="margin-top:20px;">Phenotype distribution</p>
          <div class="phenotype-bars">
            <div class="phenotype-bar"><span>Good sleepers</span><div class="phenotype-fill"><span style="--w:{shares['Consistent Good Sleepers']:.0f}%; --accent:#5CC8A1"></span></div><span>{shares['Consistent Good Sleepers']:.0f}%</span></div>
            <div class="phenotype-bar"><span>Short regular</span><div class="phenotype-fill"><span style="--w:{shares['Short but Regular']:.0f}%; --accent:#E9B44C"></span></div><span>{shares['Short but Regular']:.0f}%</span></div>
            <div class="phenotype-bar"><span>Short variable</span><div class="phenotype-fill"><span style="--w:{shares['Chronic Short & Variable']:.0f}%; --accent:#F26D5B"></span></div><span>{shares['Chronic Short & Variable']:.0f}%</span></div>
            <div class="phenotype-bar"><span>Variable long</span><div class="phenotype-fill"><span style="--w:{shares['Variable Long Sleepers']:.0f}%; --accent:#9B8AE6"></span></div><span>{shares['Variable Long Sleepers']:.0f}%</span></div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card primary" style="--accent:#5CC8A1"><p class="kpi-val">R²={best_cons['r2_mean']:.3f}</p><p class="kpi-label">Best model signal</p><p class="kpi-sub">Consistency model, {best_cons['Model']}</p></div>
      <div class="kpi-card" style="--accent:#4FA3D9"><p class="kpi-val">{N_PARTICIPANTS:,}</p><p class="kpi-label">Participants</p><p class="kpi-sub">Fitbit wearers, AoU CDR v9</p></div>
      <div class="kpi-card" style="--accent:#F26D5B"><p class="kpi-val">{age_gap:.0f}%</p><p class="kpi-label">Largest equity gap</p><p class="kpi-sub">Duration R², age 81+ vs 18–40</p></div>
      <div class="kpi-card" style="--accent:#9B8AE6"><p class="kpi-val">4</p><p class="kpi-label">Phenotypes</p><p class="kpi-sub">Distinct sleep profiles</p></div>
    </div>""", unsafe_allow_html=True)

    dur = cv[cv.target == "duration"].set_index("Model")["r2_mean"]
    cons = cv[cv.target == "consistency"].set_index("Model")["r2_mean"]
    boost_lift = (cons["HistGBM"] - cons["Ridge"]) / cons["Ridge"] * 100
    top_cons = clean_imp(imp_c)

    insight_cards([
        {"value": f"+{boost_lift:.0f}%", "label": "Boosting lift", "sub": "HistGBM vs. Ridge on consistency (nothing on duration)", "color": GREEN},
        {"value": f"{top_cons.iloc[0] / top_cons.iloc[1]:.1f}×", "label": "Dominant signal", "sub": f"{top_cons.index[0]} vs. the next feature, consistency", "color": BLUE},
        {"value": f"{race_gaps[0]:.0f}–{race_gaps[1]:.0f}%", "label": "Racial gap", "sub": "Lower consistency R² for Black and Asian participants", "color": RED},
        {"value": f"{shares.max():.0f}%", "label": "Largest phenotype", "sub": "Consistent Good Sleepers", "color": PURPLE},
    ])

    st.markdown(" ")
    col1, col2 = st.columns(2)

    with col1:
        section("Key Findings")
        box(f"<b>Model class matters for one outcome, not the other.</b> Boosting reaches R²={cons['HistGBM']:.3f} on consistency, {boost_lift:.0f}% above Ridge — but ties Ridge on duration ({dur['HistGBM']:.3f} vs {dur['Ridge']:.3f}). What limits duration prediction is missing information, not model form.", "blue")
        box(f"<b>{top_cons.index[0]} is the strongest consistency signal</b> — about {top_cons.iloc[0] / top_cons.iloc[1]:.0f}× the next feature. The activity–sleep link is curved, not age-dependent, and no interaction term earned a place in the model.", "blue")
        box(f"<b>Four sleep groups identified:</b> {shares['Consistent Good Sleepers']:.0f}% Consistent Good Sleepers, {shares['Short but Regular']:.0f}% Short but Regular, {shares['Chronic Short & Variable']:.0f}% Chronic Short & Variable, {shares['Variable Long Sleepers']:.0f}% Variable Long Sleepers.", "green")
        box(f"<b>Two fairness gaps, on different outcomes.</b> Consistency accuracy is {race_gaps[0]:.0f}–{race_gaps[1]:.0f}% lower for Black and Asian participants than for White participants; duration accuracy falls {age_gap:.0f}% from the youngest age band to the oldest. The two axes pick out largely different people, so no single number captures both.", "red")
        box("<b>Sleep consistency is more predictable than duration</b> across every model.", "blue")

    with col2:
        section("Research Questions")
        # The same three the report asks, in the same order — the Feature
        # Importance page answers the second half of RQ1 rather than its own RQ.
        st.markdown("""
**RQ1 · Prediction** To what extent do sociodemographic, behavioral, and health
characteristics predict sleep duration and consistency — and which of them carry the signal?

**RQ2 · Phenotyping** What distinct sleep behavioral phenotypes emerge from unsupervised
clustering of the wearable cohort?

**RQ3 · Fairness** How equitably do the models perform across racial/ethnic and age groups?
        """)
        st.divider()
        section("Study Design")
        st.markdown(f"""
| | |
|---|---|
| **Data** | AoU CDR v9 Fitbit sleep + activity |
| **Inclusion** | ≥30 valid nights, 4–12 hrs range |
| **Features** | 9 numeric + 8 dummies, after leakage and collinearity pruning |
| **Outcomes** | Mean nightly hours; within-person SD of those hours |
| **Validation** | 5-fold participant-level CV, one matrix for every model |
| **Clustering** | KMeans k=4 on four sleep-behaviour columns |
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Models" in page:
    st.title("Which Model Works Best?")
    st.markdown("Four models, one identical set of inputs — so any difference below belongs to the model, not the data.")
    st.divider()

    ctl1, ctl2 = st.columns(2)
    target = ctl1.radio("What should the model predict?", ["Sleep Duration", "Sleep Consistency"], horizontal=True)
    model_metric = ctl2.radio("How should accuracy be shown?", ["Model fit", "Prediction error"], horizontal=True)
    tkey = "duration" if "Duration" in target else "consistency"

    scores = cv[cv.target == tkey][['Model', 'r2_mean', 'mae_mean']].copy()
    ranked = scores.sort_values('r2_mean', ascending=False)
    best_r2 = ranked.iloc[0]
    by_model = scores.set_index('Model')['r2_mean']

    if tkey == "consistency":
        lift = (by_model['HistGBM'] - by_model['Ridge']) / by_model['Ridge'] * 100
        takeaway(f"<b>Bottom line:</b> {best_r2['Model']} is strongest here, about {lift:.0f}% better than the linear model — there is real nonlinear structure in what makes sleep regular.")
    else:
        takeaway("<b>Bottom line:</b> the linear model and the boosted tree tie on sleep duration. When a flexible model finds nothing extra, the limit is the information available rather than the method.")

    metric_col = 'r2_mean' if model_metric == "Model fit" else 'mae_mean'
    # Both targets are in hours, but they are hours of different things — average
    # nightly sleep vs. the night-to-night SD of it — so the axis has to say which.
    err_unit = "hours of nightly sleep" if tkey == "duration" else "hours of night-to-night SD"
    metric_label = ('Model fit score (higher is better)' if metric_col == 'r2_mean'
                    else f'Average prediction error, {err_unit} (lower is better)')

    # Ascending for R² so the strongest model lands at the top of a horizontal bar chart.
    plot_df = scores.sort_values(metric_col, ascending=(metric_col == 'r2_mean'))
    best_model = by_model.idxmax()
    colors = [GREEN if m == best_model else LGRAY if m == "Baseline (mean)" else AMBER
              for m in plot_df['Model']]

    y = np.arange(len(plot_df))
    fig, ax = chart_fig(10, 5.4)
    ax.barh(y, plot_df[metric_col].clip(lower=0), color=colors, alpha=0.9, edgecolor=WHITE, height=0.6)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        # The baseline's R² is -0.0001 by construction; printing "-0.000" invites
        # the reader to wonder what the minus sign means.
        shown = max(row[metric_col], 0.0)
        ax.text(shown + plot_df[metric_col].max() * 0.015, i, f"{shown:.3f}",
                va='center', fontsize=9.5, color=DGRAY,
                fontweight='700' if row['Model'] == best_model else '500')
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Model'], fontsize=10, color=DGRAY)
    ax.set_xlabel(metric_label, fontsize=10, color=MGRAY)
    ax.set_title(f'Model comparison on one feature set — {target}', fontsize=12,
                 fontweight='bold', color=DGRAY, pad=12)
    ax.xaxis.grid(True, color=LGRAY, lw=0.8)
    ax.yaxis.grid(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, plot_df[metric_col].max() * 1.22)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, width='stretch'); plt.close()
    read_note("<b>How to read this:</b> each bar is one model, scored on exactly the same inputs. The grey bar predicts the cohort average for everyone — it is the floor any real model has to beat.")

    section("Results")
    detail1, detail2 = st.columns([1, 1])
    with detail1:
        d = ranked.copy()
        d['R²'] = d['r2_mean'].clip(lower=0).round(3)
        d['MAE (hrs)'] = d['mae_mean'].round(3)
        mini_table(d[['Model', 'R²', 'MAE (hrs)']], rank=True)
    with detail2:
        box(f"Best: <b>{best_r2['Model']}</b><br>R² = {best_r2['r2_mean']:.3f} &nbsp;|&nbsp; typical miss {best_r2['mae_mean']*60:.0f} min of {err_unit.split(' of ')[-1]}", "green")
        box("Model fit is modest because sleep is shaped by many things not measured here, including stress, medications, work schedules, and health conditions.", "blue")

    st.divider()
    section("Boosting vs. a Linear Model")
    c1, c2 = st.columns(2)
    for col, (tgt, lbl) in zip([c1, c2], [("duration", "Duration"), ("consistency", "Consistency")]):
        s = cv[cv.target == tgt].set_index('Model')['r2_mean']
        col.metric(f"{lbl} R²", f"{s['HistGBM']:.3f}",
                   f"{(s['HistGBM'] - s['Ridge']) / s['Ridge'] * 100:+.1f}% vs Ridge")

# ═══════════════════════════════════════════════════════════════════════════════
# SLEEP PHENOTYPES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Phenotypes" in page:
    st.title("Sleep Groups in the Cohort")
    st.markdown("Participants group into four sleep patterns based on how much they sleep, how variable it is, and how often their nights run short or long.")
    st.divider()

    CC = {'Consistent Good Sleepers': GREEN, 'Short but Regular': BLUE,
          'Chronic Short & Variable': RED,   'Variable Long Sleepers': AMBER}
    CI = {'Consistent Good Sleepers':'✅','Short but Regular':'⚠️',
          'Chronic Short & Variable':'❌','Variable Long Sleepers':'🔄'}

    total = cl['N'].sum()
    ctl1, ctl2, ctl3 = st.columns(3)
    # Cohort share first: it is the ordering the cards, the report table and the
    # deck all use, so the page opens on the same reading as everything else.
    sort_choice = ctl1.selectbox("How should groups be sorted?", ["Cohort share", "Sleep duration", "Variability", "Short-night rate"])
    x_choice = ctl2.selectbox("What should define the horizontal position?", ["Sleep duration", "Daily steps", "Age", "BMI"])
    y_choice = ctl3.selectbox("What should define the vertical position?", ["Variability", "Short-night rate", "Daily steps", "BMI"])

    sort_map = {
        "Sleep duration": ("mean_sleep", False),
        "Cohort share": ("N", False),
        "Variability": ("mean_sd", False),
        "Short-night rate": ("pct_short", False),
    }
    x_map = {"Sleep duration": "mean_sleep", "Daily steps": "mean_steps", "Age": "mean_age", "BMI": "mean_bmi"}
    y_map = {"Variability": "mean_sd", "Short-night rate": "pct_short", "Daily steps": "mean_steps", "BMI": "mean_bmi"}
    sort_col, ascending = sort_map[sort_choice]
    short_share = (cl.set_index("cluster_label")["N"]
                     .reindex(["Short but Regular", "Chronic Short & Variable"]).sum() / total * 100)
    takeaway(f"<b>Bottom line:</b> the largest group sleeps consistently and adequately, but {short_share:.0f}% of the cohort — just over half — sits in one of the two short-sleep groups.")
    cards_html = '<div class="cl-grid">'
    for _, row in cl.sort_values(sort_col, ascending=ascending).iterrows():
        nm = row['cluster_label']; c = CC.get(nm, BLUE); ic = CI.get(nm, '•')
        cards_html += f"""
        <div class="cl-card" style="border-top: 5px solid {c}">
            <div class="cl-icon">{ic}</div>
            <div class="cl-name" style="color:{c}">{nm}</div>
            <p class="cl-n">{row['N']:,}</p>
            <p class="cl-pct">{row['N']/total*100:.0f}% of cohort</p>
            <hr class="cl-divider">
            <p class="cl-stat">Sleep: {row['mean_sleep']:.2f} hrs avg</p>
            <p class="cl-stat">Variability: SD {row['mean_sd']:.2f} hrs</p>
            <p class="cl-stat">Short nights: {row['pct_short']:.0%}</p>
            <p class="cl-stat">Activity: {row['mean_steps']:,.0f} steps/day</p>
            <p class="cl-stat">Profile: age {row['mean_age']:.0f} · BMI {row['mean_bmi']:.1f}</p>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    section("Group Map")
    fig, ax = chart_fig(9, 6)
    for _, row in cl.iterrows():
        nm = row['cluster_label']
        c = CC.get(nm, BLUE)
        size = 240 + (row['N'] / total) * 2300
        xval = row[x_map[x_choice]]
        yval = row[y_map[y_choice]]
        ax.scatter(xval, yval, s=size, color=c, alpha=0.82,
                   edgecolor=WHITE, linewidth=2.2, zorder=3)
        ax.text(xval, yval, f"{row['N']/total:.0%}",
                ha='center', va='center', fontsize=11, color='white', fontweight='750', zorder=4)
        ax.annotate(nm, (xval, yval),
                    xytext=(10, 6), textcoords='offset points', fontsize=9.5,
                    color=DGRAY, fontweight='650')
    if x_choice == "Sleep duration":
        ax.axvline(7, color=AMBER, ls='--', lw=1.6, alpha=0.8)
        ax.text(7.03, ax.get_ylim()[0] + 0.04, '7h reference', color=AMBER, fontsize=8.8, fontweight='650')
    ax.set_xlabel(x_choice, fontsize=10, color=MGRAY)
    ax.set_ylabel(y_choice, fontsize=10, color=MGRAY)
    ax.set_title('How the sleep groups differ', fontsize=12, fontweight='bold', color=DGRAY, pad=10)
    ax.xaxis.grid(True, color=LGRAY, lw=0.8)
    plt.tight_layout()
    st.pyplot(fig, width='stretch'); plt.close()
    read_note("<b>How to read this:</b> each bubble is one sleep group. Larger bubbles represent more people. Moving right or up means the group has more of the selected trait.")

    section("Profile Comparison")
    fc  = ['mean_sleep','mean_sd','pct_short','mean_steps','mean_bmi','mean_age']
    fl  = ['Sleep Duration','Variability','Short Night %','Daily Steps','BMI','Age']
    z   = cl.set_index('cluster_label')[fc].copy()
    zn  = (z - z.mean()) / z.std()
    ORD = ['Consistent Good Sleepers','Short but Regular','Chronic Short & Variable','Variable Long Sleepers']
    zn  = zn.reindex([r for r in ORD if r in zn.index])

    profile_table(z, zn, fc, fl)
    read_note("<b>How to read this:</b> each cell shows the group's actual value, plus how it compares with the "
              "<b>average of the four groups</b> — not with the cohort average, which weights by group size. "
              "The wording replaces a colour scale that would need its own legend.")

    prof = cl.set_index('cluster_label')
    shr = prof['N'] / prof['N'].sum() * 100
    box(f"⚠️ <b>Short but Regular</b> ({shr['Short but Regular']:.0f}%): {prof.loc['Short but Regular','pct_short']:.0%} of nights fall under 6 hours — chronic short sleep on a consistent schedule. High steps suggest trading sleep for activity, and a variability-based screen would miss this group entirely.", "red")
    box(f"💡 <b>Variable Long Sleepers</b> ({shr['Variable Long Sleepers']:.0f}%): highest sleep duration but the lowest step count ({prof.loc['Variable Long Sleepers','mean_steps']:,.0f}/day) and high night-to-night variability. Possibly shift workers, retirees, or those with irregular schedules.", "blue")
    box("📌 <b>The groups are cuts through a continuous distribution</b>, not natural kinds. KMeans imposes equal-sized, spherical groups, and a participant near a boundary is not meaningfully in either one.", "blue")

# ═══════════════════════════════════════════════════════════════════════════════
# FAIRNESS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Fairness" in page:
    st.title("Does the Model Work Equally Well?")
    st.markdown("Check whether model accuracy is similar across demographic groups.")
    st.divider()

    ctl1, ctl2, ctl3 = st.columns(3)
    target = ctl1.radio("Which outcome should be checked?", ["Sleep Duration", "Sleep Consistency"], horizontal=True)
    stratum = ctl2.radio("Which grouping?", ["Race / ethnicity", "Age band"], horizontal=True)
    fairness_view = ctl3.radio("How should differences be shown?", ["Deviation from overall", "Raw model fit"], horizontal=True)

    tkey = "duration" if "Duration" in target else "consistency"
    skey = "race_ethnicity" if "Race" in stratum else "age_band"

    # `Unknown` and `Other` are residual buckets rather than populations, so neither
    # supports a statement about a group of people.
    sub = fair[(fair.target == tkey) & (fair.stratum == skey)
               & (~fair.subgroup.isin(["Unknown", "Other"]))].copy()
    overall = sub['overall_r2'].iloc[0]
    sub['delta'] = sub['r2'] - overall
    sub = sub.sort_values('delta' if fairness_view == "Deviation from overall" else 'r2', ascending=False)

    best, worst = sub.iloc[0], sub.iloc[-1]
    gap = (best['r2'] - worst['r2']) / best['r2'] * 100
    takeaway(f"<b>Bottom line:</b> model fit is not even across groups. On {target.lower()}, the model explains about {gap:.0f}% less for <b>{worst['subgroup']}</b> (R²={worst['r2']:.3f}) than for <b>{best['subgroup']}</b> (R²={best['r2']:.3f}).")

    colors = [RED if d < -0.005 else GREEN if d > 0.005 else LGRAY for d in sub['delta']]
    fig, ax = chart_fig(10, 6.0)
    plot_values = sub['delta'] if fairness_view == "Deviation from overall" else sub['r2']
    bars = ax.barh(sub['subgroup'], plot_values, color=colors, alpha=0.9, edgecolor=WHITE, height=0.58)
    if fairness_view == "Deviation from overall":
        ax.axvline(0, color=DGRAY, lw=1.8, alpha=0.65)
    else:
        ax.axvline(overall, color=AMBER, lw=1.8, ls='--', alpha=0.85)
    # One short number per bar, and it is always the quantity the bar length
    # encodes. The longer "-0.041 fit 0.063" form ran off the left of the axis
    # and landed on top of the subgroup tick labels; the table below carries both.
    for bar, val, delta, r2 in zip(bars, plot_values, sub['delta'], sub['r2']):
        ha = 'left' if val >= 0 else 'right'
        offset = 0.002 if val >= 0 else -0.002
        label = f"{delta:+.3f}" if fairness_view == "Deviation from overall" else f"{r2:.3f}"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, label,
                va='center', ha=ha, fontsize=9.3, color=DGRAY)
    ax.set_xlabel('Difference from overall model performance' if fairness_view == "Deviation from overall" else "Model fit score",
                  fontsize=10, color=MGRAY)
    ax.set_title(f'Where does model performance differ? — {target} by {stratum.lower()}',
                 fontsize=12, fontweight='bold', color=DGRAY, pad=10)
    ax.xaxis.grid(True, color=LGRAY, lw=0.8); ax.yaxis.grid(False)
    ax.legend(handles=[mpatches.Patch(color=RED, label='Below overall (gap)'),
                       mpatches.Patch(color=GREEN, label='Above overall')],
              fontsize=9, loc='lower right', framealpha=0.85)
    if fairness_view == "Deviation from overall":
        lim = max(abs(sub['delta'].min()), abs(sub['delta'].max())) * 1.5
        ax.set_xlim(-lim, lim)
    else:
        ax.set_xlim(0, sub['r2'].max() * 1.25)
    plt.tight_layout()
    st.pyplot(fig, width='stretch'); plt.close()
    read_note("<b>How to read this:</b> groups to the right are predicted more accurately than the cohort overall, groups to the left less accurately. This says nothing about whether a group sleeps better or worse — only about how well the model does for them. Compare groups with each other rather than with the overall line.")

    section("Results by Subgroup")
    detail1, detail2 = st.columns([1, 1])
    with detail1:
        d = sub[['subgroup', 'n', 'r2']].copy()
        d['R²'] = d['r2'].round(3)
        d['vs overall'] = d['r2'].apply(lambda x: f"{x - overall:+.3f}")
        d = d.rename(columns={'subgroup': 'Subgroup', 'n': 'Participants'})
        mini_table(d[['Subgroup', 'Participants', 'R²', 'vs overall']],
                   formats={'Participants': lambda x: f"{int(x):,}"})

    with detail2:
        race = fair[(fair.target == "consistency") & (fair.stratum == "race_ethnicity")].set_index("subgroup")["r2"]
        age = fair[(fair.target == "duration") & (fair.stratum == "age_band")].set_index("subgroup")["r2"]
        race_dur = fair[(fair.target == "duration") & (fair.stratum == "race_ethnicity")].set_index("subgroup")["r2"]
        box(f"⚠️ <b>Race, on consistency:</b> the model explains {(race['White']-race['Black or African American'])/race['White']*100:.0f}% less for Black participants (R²={race['Black or African American']:.3f}) and {(race['White']-race['Asian'])/race['White']*100:.0f}% less for Asian participants (R²={race['Asian']:.3f}) than for White participants (R²={race['White']:.3f}).", "red")
        box(f"⚠️ <b>Age, on duration:</b> the largest gap in the analysis. Fit falls {(age['18-40']-age['81+'])/age['18-40']*100:.0f}% from ages 18–40 (R²={age['18-40']:.3f}) to 81+ (R²={age['81+']:.3f}), declining at every band in between.", "red")
        box(f"📌 <b>On duration the racial ordering does not favour White participants</b> — they sit level with Black participants (both R²={race_dur['White']:.3f}) and below Hispanic or Latino participants (R²={race_dur['Hispanic or Latino']:.3f}), with Asian participants lowest (R²={race_dur['Asian']:.3f}). So the race gap and the age gap sit on different outcomes and pick out largely different people, and no single number captures both.", "blue")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Importance" in page:
    st.title("Which Factors Mattered Most?")
    st.markdown("See which inputs the model relied on most, measured by how much accuracy it loses when each one is scrambled.")
    st.divider()

    ctl1, ctl2, ctl3 = st.columns([1.2, 1.2, 0.8])
    target = ctl1.radio("Which outcome should be explained?", ["Sleep Duration", "Sleep Consistency"], horizontal=True)
    imp = clean_imp(imp_d if "Duration" in target else imp_c)
    imp = imp[imp > 0.005]
    # Only nine features clear the threshold on either target, so a slider that
    # ran to 18 spent half its travel doing nothing.
    top_n = (ctl2.slider("How many factors should be shown?", 5, len(imp), len(imp))
             if len(imp) > 5 else len(imp))
    show_table = ctl3.toggle("Show full table", value=True)
    imp_chart = imp.head(top_n)
    tlbl = "Sleep Duration" if "Duration" in target else "Sleep Consistency"
    takeaway(f"<b>Bottom line:</b> {imp.index[0]} is the strongest prediction signal for this outcome. These factors help prediction; they do not prove cause.")

    simp = imp_chart.sort_values()
    colors = []
    for i in simp.index:
        if i == simp.index[-1]:
            colors.append(RED)
        elif i in list(simp.index[-3:-1]):
            colors.append(AMBER)
        else:
            colors.append(BLUE)
    fig, ax = chart_fig(9, max(6.6, len(simp)*0.58))
    bars = ax.barh(simp.index, simp.values, color=colors, alpha=0.88, edgecolor=WHITE, height=0.58)
    # The leader is called out on its own value label. A separate annotation
    # pinned to the right edge of the axis landed on top of that label.
    top = simp.max()
    for bar, val in zip(bars, simp.values):
        leader = val == top
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}   ← top driver' if leader else f'{val:.3f}',
                va='center', fontsize=9, color=RED if leader else DGRAY,
                fontweight='700' if leader else '400')
    ax.set_xlabel('Relative influence in the model', fontsize=10, color=MGRAY)
    ax.set_title(f'Which inputs influenced the model most? — {tlbl}', fontsize=12, fontweight='bold', color=DGRAY, pad=10)
    ax.xaxis.grid(True, color=LGRAY, lw=0.8); ax.yaxis.grid(False)
    ax.tick_params(left=False)
    for sp in ['left']: ax.spines[sp].set_visible(False)
    ax.set_xlim(0, top * 1.45)      # room for the leader's longer label
    plt.tight_layout()
    st.pyplot(fig, width='stretch'); plt.close()
    read_note("<b>How to read this:</b> longer bars mean the model lost more accuracy when that input was scrambled. These are prediction signals, not proof that the factor directly causes sleep duration or consistency.")

    section("Top 5 Features")
    detail1, detail2 = st.columns([1, 1])
    with detail1:
        t5 = imp.head(5).reset_index(); t5.columns = ['Feature','Importance']
        t5['Importance'] = t5['Importance'].round(3)
        mini_table(t5, rank=True)

    with detail2:
        top3_share = imp.head(3).sum() / imp.sum() * 100
        insight_cards([
            {"value": f"{top3_share:.0f}%", "label": "Top-3 concentration", "sub": "Share of displayed importance in the first three signals", "color": RED if top3_share > 50 else BLUE},
            {"value": f"{imp.iloc[0]/imp.iloc[1]:.1f}×", "label": "Lead feature ratio", "sub": "Top feature compared with the second-ranked feature", "color": AMBER},
        ])

        if "Duration" in target:
            box(f"🔑 No single factor dominates. <b>{imp.index[0]}</b> ({imp.iloc[0]:.3f}) leads by a hair, with gender, race/ethnicity, age and BMI close behind — sleep duration is spread thinly across many weak signals, which is why it is the harder outcome to predict.", "blue")
            box("📌 <b>Nights tracked</b> barely registers. It is kept in the model as a control for how long someone wore the device, not as a driver of sleep.", "blue")
        else:
            box(f"🔑 <b>{imp.index[0]}</b> ({imp.iloc[0]:.3f}) is far ahead of everything else — about {imp.iloc[0]/imp.iloc[1]:.0f}× the next feature. People with erratic daily activity have erratic sleep.", "blue")
            box("📌 The activity–sleep relationship is <b>curved, not age-dependent</b>. No interaction term earned a place in the model.", "blue")
            # Looked up by name rather than by rank, so a reordering of the
            # importances cannot silently point this claim at another feature.
            health = "Self-rated health"
            if health in imp.index:
                health_dur = clean_imp(imp_d).get(health, 0.0)
                box(f"📌 <b>{health}</b> ({imp[health]:.3f}) matters far more here than for duration ({health_dur:.3f}) — perceived wellbeing tracks sleep regularity, not sleep length.", "green")

    if show_table:
        st.divider()
        section("Full Table")
        ft = imp.reset_index(); ft.columns = ['Feature','Importance']
        ft['Importance'] = ft['Importance'].round(4)
        mini_table(ft)
