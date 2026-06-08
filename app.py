"""
app.py  —  Bank Customer Churn Intelligence Platform
Run:  streamlit run app.py
"""

import json, warnings, os, subprocess, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Auto-train models if not present ─────────────────────────────────────────
def auto_train():
    models_needed = [
        "models/gradient_boosting.pkl",
        "models/random_forest.pkl",
        "models/logistic_regression.pkl",
        "models/decision_tree.pkl",
        "models/metrics.json",
        "models/feature_cols.json",
    ]
    if not all(Path(m).exists() for m in models_needed):
        with st.spinner("🔧 First-time setup: training models on your data... (this takes 1-2 minutes)"):
            result = subprocess.run(
                [sys.executable, "train_models.py"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f"Model training failed: {result.stderr}")
                st.stop()
            st.success("✓ Models trained successfully! Loading dashboard...")
            st.rerun()

auto_train()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence | ECB",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",  # always open
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hide sidebar collapse/expand button entirely — sidebar always open */
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > div > button { display: none !important; }
/* Prevent sidebar from collapsing */
section[data-testid="stSidebar"] {
    min-width: 240px !important;
    width: 240px !important;
    transform: none !important;
    left: 0 !important;
}
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #c8ccd8 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280 !important;
}
section[data-testid="stSidebar"] hr { border-color: #1e2130; }

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #8890a0 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 400;
    padding: 10px 22px;
    color: #6b7280;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #e5e7eb !important;
    border-bottom: 2px solid #2563eb !important;
    font-weight: 500 !important;
    background: transparent !important;
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Headings */
h1 { font-size: 22px !important; font-weight: 600 !important; }
h2 { font-size: 16px !important; font-weight: 500 !important; }
h3 { font-size: 14px !important; font-weight: 500 !important; }

/* Callout boxes */
.info-box {
    background: #eff6ff;
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #1e3a5f;
    margin: 8px 0 16px;
    line-height: 1.6;
}
.warn-box {
    background: #fff7ed;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #78350f;
    margin: 8px 0 16px;
}
.danger-box {
    background: #fef2f2;
    border-left: 3px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #7f1d1d;
    margin: 8px 0 16px;
}
.success-box {
    background: #f0fdf4;
    border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #14532d;
    margin: 8px 0 16px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.risk-low    { background: #dcfce7; color: #15803d; }
.risk-medium { background: #fef9c3; color: #a16207; }
.risk-high   { background: #fee2e2; color: #b91c1c; }

/* Section divider */
.section-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
</style>
""", unsafe_allow_html=True)

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "blue":   "#2563eb",
    "red":    "#ef4444",
    "amber":  "#f59e0b",
    "green":  "#22c55e",
    "indigo": "#6366f1",
    "gray":   "#9ca3af",
    "slate":  "#64748b",
    "bg":     "#ffffff",
    "bg2":    "#f8f9fb",
}
PLOTLY_LAYOUT = dict(
    font_family="DM Sans",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c8ccd8",
    margin=dict(l=10, r=10, t=30, b=10),
)
# Default axis styles - use these when no override needed
XAXIS_DEFAULT = dict(showgrid=False, showline=False, tickfont_size=11, tickfont_color="#9ca3af")
YAXIS_DEFAULT = dict(gridcolor="rgba(255,255,255,0.07)", gridwidth=1, showline=False,
                     tickfont_size=11, tickfont_color="#9ca3af")

GRID_COLOR = "rgba(255,255,255,0.07)"
TICK_COLOR = "#9ca3af"

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv("European_Bank.csv")

@st.cache_resource
def load_models():
    base = Path("models")
    if not base.exists():
        return None, None, None, None
    models = {}
    for name in ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]:
        p = base / f"{name}.pkl"
        if p.exists():
            models[name] = joblib.load(p)
    metrics = json.loads((base / "metrics.json").read_text()) if (base / "metrics.json").exists() else {}
    fi      = json.loads((base / "feature_importance.json").read_text()) if (base / "feature_importance.json").exists() else {}
    feat    = json.loads((base / "feature_cols.json").read_text()) if (base / "feature_cols.json").exists() else []
    return models, metrics, fi, feat

@st.cache_data
def engineer_features(df_raw):
    df = df_raw.copy()
    drop = [c for c in ["CustomerId", "Surname", "Year"] if c in df.columns]
    df   = df.drop(columns=drop)
    df   = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)
    df["BalanceToSalary"]      = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductDensity"]       = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["EngagementProduct"]    = df["IsActiveMember"] * df["NumOfProducts"]
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]
    df["HighBalance"]          = (df["Balance"] > 100_000).astype(int)
    df["SeniorCustomer"]       = (df["Age"] >= 45).astype(int)
    return df

df_raw                 = load_raw()
models, metrics, fi, feat_cols = load_models()
models_loaded          = models is not None and len(models) > 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 ECB Churn Intel")
    st.markdown("---")
    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Overview", "Deep Dive", "Model Performance",
                         "Feature Importance", "Risk Calculator", "Scenario Simulator"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="section-title">Dataset filters</div>', unsafe_allow_html=True)
    geo_filter = st.multiselect("Geography", ["France", "Germany", "Spain"],
                                default=["France", "Germany", "Spain"])
    gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    age_range = st.slider("Age range", 18, 92, (18, 92))
    st.markdown("---")
    if not models_loaded:
        st.warning("⚠️ Models not found.\nRun `python train_models.py` first.")
    else:
        st.success("✓ Models loaded")

# ── Filter data ───────────────────────────────────────────────────────────────
df = df_raw[
    df_raw["Geography"].isin(geo_filter) &
    df_raw["Gender"].isin(gender_filter) &
    df_raw["Age"].between(*age_range)
].copy()

# ── Helpers ───────────────────────────────────────────────────────────────────
def churn_rate(subset): return round(subset["Exited"].mean() * 100, 1)
def fmt_pct(v):         return f"{v:.1f}%"

def bar_chart(labels, values, colors, title="", h=300, horizontal=False):
    fig = go.Figure()
    layout = {**PLOTLY_LAYOUT, "height": h, "title": title}
    if horizontal:
        fig.add_trace(go.Bar(y=labels, x=values, orientation="h",
                             marker_color=colors, marker_line_width=0))
        layout["xaxis"] = dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", ticksuffix="%",
                                tickfont_size=11, tickfont_color="#9ca3af")
        layout["yaxis"] = dict(showgrid=False, tickfont_size=11, tickfont_color="#9ca3af")
    else:
        fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors, marker_line_width=0,
                             text=[f"{v:.1f}%" for v in values], textposition="outside",
                             textfont_size=11))
        layout["xaxis"] = dict(showgrid=False, showline=False, tickfont_size=11, tickfont_color="#9ca3af")
        layout["yaxis"] = dict(gridcolor="rgba(255,255,255,0.07)", ticksuffix="%",
                                tickfont_size=11, tickfont_color="#9ca3af")
        layout["bargap"] = 0.35
    fig.update_layout(**layout)
    return fig

def color_by_rate(rates, lo=18, hi=30):
    return [C["red"] if r > hi else (C["amber"] if r > lo else C["green"]) for r in rates]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.title("Customer Churn Intelligence Dashboard")
    st.caption(f"European Central Bank · {len(df):,} customers after filters · 2025")

    total    = len(df)
    churned  = df["Exited"].sum()
    rate     = churned / total * 100
    avg_age  = df["Age"].mean()
    avg_bal  = df["Balance"].mean()
    active   = df["IsActiveMember"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",  f"{total:,}")
    c2.metric("Churned",          f"{int(churned):,}", f"{rate:.1f}% rate")
    c3.metric("Avg Age",          f"{avg_age:.1f} yrs",
              f"Churned: {df[df.Exited==1]['Age'].mean():.1f}")
    c4.metric("Avg Balance",      f"€{avg_bal/1000:.0f}K",
              f"Churned: €{df[df.Exited==1]['Balance'].mean()/1000:.0f}K")
    c5.metric("Active Members",   f"{active:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Churn by geography")
        geo_data  = df.groupby("Geography")["Exited"].agg(["sum","count"])
        geo_rates = (geo_data["sum"] / geo_data["count"] * 100).round(1)
        fig = bar_chart(geo_rates.index.tolist(), geo_rates.values.tolist(),
                        color_by_rate(geo_rates.values), h=280)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div class="danger-box">Germany churns at <strong>32.4%</strong> — nearly 2× '
            'France and Spain. Regional programs are the highest-leverage intervention.</div>',
            unsafe_allow_html=True)

    with col2:
        st.markdown("#### Churn by number of products")
        prod_data  = df.groupby("NumOfProducts")["Exited"].agg(["sum","count"])
        prod_rates = (prod_data["sum"] / prod_data["count"] * 100).round(1)
        colors     = color_by_rate(prod_rates.values)
        labels_p   = [f"Prod {int(v)}" for v in prod_rates.index]
        fig = go.Figure(go.Bar(
            x=labels_p, y=prod_rates.values.tolist(),
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in prod_rates.values],
            textposition="outside", textfont_size=11
        ))
        fig.update_layout(**{**PLOTLY_LAYOUT,
                          "height": 280,
                          "bargap": 0.35,
                          "xaxis": dict(showgrid=False, tickfont_size=12,
                                        tickfont_color="#9ca3af", type="category"),
                          "yaxis": dict(gridcolor="rgba(255,255,255,0.07)", ticksuffix="%",
                                        tickfont_size=11, tickfont_color="#9ca3af")})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div class="warn-box"><strong>2 products = optimal (7.6% churn).</strong> '
            '3–4 products show paradoxical extreme churn — likely product mismatch or '
            'overselling.</div>',
            unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Active vs inactive churn")
        act_data = df.groupby("IsActiveMember")["Exited"].agg(["sum","count"])
        act_rates = (act_data["sum"] / act_data["count"] * 100).round(1)
        labels    = ["Inactive", "Active"] if 0 in act_data.index else ["Active"]
        fig = bar_chart(labels, act_rates.values.tolist(),
                        [C["red"], C["green"]] if len(labels)==2 else [C["green"]], h=260)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col4:
        st.markdown("#### Gender breakdown")
        gen_data  = df.groupby("Gender")["Exited"].agg(["sum","count"])
        gen_rates = (gen_data["sum"] / gen_data["count"] * 100).round(1)
        fig = bar_chart(gen_rates.index.tolist(), gen_rates.values.tolist(),
                        [C["blue"], C["indigo"]] if len(gen_rates)==2 else [C["blue"]], h=260)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Deep Dive":
    st.title("Deep Dive Analysis")

    # Age distribution
    st.markdown("#### Age distribution — churned vs retained")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df.Exited==0]["Age"], name="Retained",
                               marker_color=C["blue"], opacity=0.65,
                               xbins=dict(size=3), bingroup=1))
    fig.add_trace(go.Histogram(x=df[df.Exited==1]["Age"], name="Churned",
                               marker_color=C["red"], opacity=0.65,
                               xbins=dict(size=3), bingroup=1))
    fig.update_layout(**PLOTLY_LAYOUT, height=300, barmode="overlay",
                      legend=dict(orientation="h", y=1.1, x=0.7, font_size=12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div class="info-box">Churned customers are concentrated in the <strong>41–60 age band</strong>. '
        'The 51–60 cohort has a 56% churn rate — the highest of any group. '
        'Younger customers (&lt;35) are significantly more loyal.</div>',
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Balance distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[df.Exited==0]["Balance"], name="Retained",
                                   marker_color=C["blue"], opacity=0.65, nbinsx=40, bingroup=1))
        fig.add_trace(go.Histogram(x=df[df.Exited==1]["Balance"], name="Churned",
                                   marker_color=C["red"], opacity=0.65, nbinsx=40, bingroup=1))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, barmode="overlay",
                          legend=dict(orientation="h", y=1.1, font_size=11),
                          xaxis_tickformat="€,.0f")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("#### Credit score distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[df.Exited==0]["CreditScore"], name="Retained",
                                   marker_color=C["blue"], opacity=0.65, nbinsx=30, bingroup=1))
        fig.add_trace(go.Histogram(x=df[df.Exited==1]["CreditScore"], name="Churned",
                                   marker_color=C["red"], opacity=0.65, nbinsx=30, bingroup=1))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, barmode="overlay",
                          legend=dict(orientation="h", y=1.1, font_size=11))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Correlation heatmap
    st.markdown("#### Correlation with churn")
    num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
                "HasCrCard","IsActiveMember","EstimatedSalary","Exited"]
    corr = df[num_cols].corr()["Exited"].drop("Exited").sort_values()
    colors_corr = [C["red"] if v > 0 else C["blue"] for v in corr.values]
    fig = go.Figure(go.Bar(
        x=corr.values.round(3), y=corr.index, orientation="h",
        marker_color=colors_corr, marker_line_width=0,
        text=[f"{v:+.3f}" for v in corr.values], textposition="outside", textfont_size=11
    ))
    fig.update_layout(**{**PLOTLY_LAYOUT,
                       "height": 320,
                       "xaxis": dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=True,
                                     zerolinecolor="#e0e3ea", zerolinewidth=1,
                                     range=[-0.2, 0.22], tickfont_size=11, tickfont_color="#9ca3af"),
                       "yaxis": dict(showgrid=False, tickfont_size=12, tickfont_color="#9ca3af")})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Tenure analysis
    st.markdown("#### Churn rate by tenure (years with bank)")
    ten_data  = df.groupby("Tenure")["Exited"].agg(["mean","count"]).reset_index()
    ten_data["rate"] = ten_data["mean"] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ten_data["Tenure"], y=ten_data["rate"],
                             mode="lines+markers",
                             line=dict(color=C["blue"], width=2.5),
                             marker=dict(size=8, color=C["blue"]),
                             name="Churn rate"))
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                      yaxis=dict(ticksuffix="%", gridcolor="rgba(255,255,255,0.07)", range=[0, 30]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div class="info-box">Tenure has <strong>minimal impact on churn risk</strong> — '
        'the rate stays consistently near 20% across all tenures. '
        'This means long-standing customers are just as likely to leave as new ones, '
        'making it a poor standalone predictor.</div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")

    if not models_loaded:
        st.error("Run `python train_models.py` first to generate model metrics.")
        st.stop()

    model_names = list(metrics.keys())
    display_map = {
        "Logistic Regression": "Logistic Reg.",
        "Decision Tree":       "Decision Tree",
        "Random Forest":       "Random Forest",
        "Gradient Boosting":   "Grad. Boost",
    }

    # Summary metrics table
    st.markdown("#### Model comparison")
    rows = []
    for m, v in metrics.items():
        rows.append({
            "Model":     m,
            "Accuracy":  f"{v['accuracy']*100:.2f}%",
            "Precision": f"{v['precision']*100:.2f}%",
            "Recall":    f"{v['recall']*100:.2f}%",
            "F1 Score":  f"{v['f1']*100:.2f}%",
            "ROC-AUC":   f"{v['roc_auc']:.4f}",
            "CV AUC":    f"{v['cv_auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"),
                 use_container_width=True, height=200)

    # ROC-AUC bar chart
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ROC-AUC by model")
        aucs   = [metrics[m]["roc_auc"] for m in model_names]
        clrs   = [C["blue"] if a == max(aucs) else C["gray"] for a in aucs]
        labels = [display_map.get(m, m) for m in model_names]
        fig    = go.Figure(go.Bar(
            x=labels, y=aucs, marker_color=clrs, marker_line_width=0,
            text=[f"{a:.4f}" for a in aucs], textposition="outside", textfont_size=11
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          yaxis=dict(range=[0.7, 0.95], gridcolor="rgba(255,255,255,0.07)", tickfont_size=11),
                          bargap=0.35)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("#### Recall by model")
        recalls = [metrics[m]["recall"] for m in model_names]
        clrs2   = [C["red"] if r == max(recalls) else C["gray"] for r in recalls]
        fig     = go.Figure(go.Bar(
            x=labels, y=recalls, marker_color=clrs2, marker_line_width=0,
            text=[f"{r:.3f}" for r in recalls], textposition="outside", textfont_size=11
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.07)",
                                     tickformat=".0%", tickfont_size=11),
                          bargap=0.35)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Confusion matrices
    st.markdown("#### Confusion matrices")
    cm_cols = st.columns(len(model_names))
    for i, (m, col) in enumerate(zip(model_names, cm_cols)):
        cm = np.array(metrics[m]["confusion_matrix"])
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
            colorscale=[[0, "#eff6ff"], [1, "#1d4ed8"]],
            text=cm, texttemplate="%{text}", textfont_size=14,
            showscale=False
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_family="DM Sans", height=220, margin=dict(l=10,r=10,t=30,b=10),
                          title=dict(text=display_map.get(m,m), font_size=12, x=0.5))
        col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        '<div class="info-box"><strong>Gradient Boosting</strong> achieves the best overall '
        'performance (ROC-AUC ~0.87). <strong>Recall</strong> is the critical metric here — '
        'a missed churner costs more than a false alarm. Logistic Regression provides the best '
        'interpretability baseline for regulatory reporting.</div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.title("Feature Importance")

    if not models_loaded or not fi:
        st.error("Run `python train_models.py` first.")
        st.stop()

    fi_series = pd.Series(fi).sort_values(ascending=True)
    clrs = [C["red"] if i >= len(fi_series) - 5 else
            C["amber"] if i >= len(fi_series) - 10 else C["blue"]
            for i in range(len(fi_series))]
    fig = go.Figure(go.Bar(
        y=fi_series.index, x=fi_series.values, orientation="h",
        marker_color=clrs, marker_line_width=0,
        text=[f"{v:.3f}" for v in fi_series.values],
        textposition="outside", textfont_size=10
    ))
    fig.update_layout(**{**PLOTLY_LAYOUT,
                       "height": 420,
                       "xaxis": dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", tickfont_size=11),
                       "yaxis": dict(showgrid=False, tickfont_size=11, tickfont_color="#9ca3af"),
                       "margin": dict(l=140, r=60, t=20, b=10)})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="danger-box"><strong>Top drivers of churn (red):</strong><br>'
            '→ <strong>Age</strong> — middle-aged customers churn most<br>'
            '→ <strong>NumOfProducts</strong> — 3–4 products = extreme risk<br>'
            '→ <strong>IsActiveMember</strong> — inactivity strongly predicts leaving<br>'
            '→ <strong>Balance</strong> — high balances correlate with exit</div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="info-box"><strong>Engineered features:</strong><br>'
            '→ <strong>BalanceToSalary</strong> — captures financial stress signal<br>'
            '→ <strong>EngagementProduct</strong> — interaction of activity × products<br>'
            '→ <strong>SeniorCustomer</strong> — binary flag for age ≥ 45<br>'
            '→ <strong>AgeTenureInteraction</strong> — combined demographic depth</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Calculator":
    st.title("Customer Churn Risk Calculator")
    st.markdown("Enter a customer's details to compute their predicted churn probability.")

    if not models_loaded:
        st.error("Run `python train_models.py` first to enable predictions.")
        st.stop()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        geo    = c1.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = c2.selectbox("Gender", ["Male", "Female"])
        age    = c1.number_input("Age", 18, 92, 40)
        tenure = c2.number_input("Tenure (years)", 0, 10, 5)

        st.markdown('<div class="section-title">Financial</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        balance = c3.number_input("Balance (€)", 0, 300_000, 75_000, step=5_000)
        salary  = c4.number_input("Est. Salary (€)", 10_000, 250_000, 80_000, step=5_000)
        credit  = c3.number_input("Credit Score", 300, 850, 650)

        st.markdown('<div class="section-title">Engagement</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        products = c5.selectbox("Num. of Products", [1, 2, 3, 4])
        has_cc   = c6.selectbox("Has Credit Card", ["Yes", "No"])
        is_active = c5.selectbox("Active Member", ["Yes", "No"])

    # Build feature vector
    def build_row():
        row = {
            "CreditScore":     credit,
            "Age":             age,
            "Tenure":          tenure,
            "Balance":         balance,
            "NumOfProducts":   products,
            "HasCrCard":       1 if has_cc == "Yes" else 0,
            "IsActiveMember":  1 if is_active == "Yes" else 0,
            "EstimatedSalary": salary,
            "Geography_France":  1 if geo == "France" else 0,
            "Geography_Germany": 1 if geo == "Germany" else 0,
            "Geography_Spain":   1 if geo == "Spain" else 0,
            "Gender_Female": 1 if gender == "Female" else 0,
            "Gender_Male":   1 if gender == "Male" else 0,
        }
        row["BalanceToSalary"]      = balance / (salary + 1)
        row["ProductDensity"]       = products / (tenure + 1)
        row["EngagementProduct"]    = row["IsActiveMember"] * products
        row["AgeTenureInteraction"] = age * tenure
        row["HighBalance"]          = 1 if balance > 100_000 else 0
        row["SeniorCustomer"]       = 1 if age >= 45 else 0
        return pd.DataFrame([row])[feat_cols]

    with col_result:
        st.markdown('<div class="section-title">Risk assessment</div>', unsafe_allow_html=True)

        model_choice = st.selectbox("Model", ["gradient_boosting", "random_forest",
                                               "logistic_regression", "decision_tree"],
                                     format_func=lambda x: x.replace("_", " ").title())
        pipe  = models[model_choice]
        X_row = build_row()
        proba = pipe.predict_proba(X_row)[0, 1]
        pct   = int(round(proba * 100))

        if pct < 20:
            badge_cls, label, advice = "risk-low",    "Low Risk",    "Minimal intervention needed. Standard loyalty program."
        elif pct < 45:
            badge_cls, label, advice = "risk-medium", "Medium Risk", "Proactive outreach recommended within 30 days."
        else:
            badge_cls, label, advice = "risk-high",   "High Risk",   "Immediate personalized retention offer required."

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "family": "DM Sans", "color": "#111827"}},
            gauge={
                "axis":     {"range": [0, 100], "tickwidth": 0, "tickcolor": "#ccc", "tickfont": {"size": 11}},
                "bar":      {"color": "#ef4444" if pct >= 45 else ("#f59e0b" if pct >= 20 else "#22c55e"),
                             "thickness": 0.28},
                "bgcolor":  "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20],  "color": "#dcfce7"},
                    {"range": [20, 45], "color": "#fef9c3"},
                    {"range": [45, 100],"color": "#fee2e2"},
                ],
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=230,
                          margin=dict(l=20, r=20, t=20, b=0),
                          font_family="DM Sans")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f'<span class="risk-badge {badge_cls}">{label}</span>',
                    unsafe_allow_html=True)
        st.markdown(f"<p style='color:#6b7280;font-size:13px;margin-top:8px;'>{advice}</p>",
                    unsafe_allow_html=True)

        # Per-model comparison
        st.markdown('<div class="section-title" style="margin-top:1rem;">All models</div>',
                    unsafe_allow_html=True)
        X_row2 = build_row()
        for mname, mpipe in models.items():
            mp = mpipe.predict_proba(X_row2)[0, 1]
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"<span style='font-size:12px;color:#6b7280'>{mname}</span>",
                           unsafe_allow_html=True)
            color = "#ef4444" if mp >= 0.45 else ("#f59e0b" if mp >= 0.20 else "#22c55e")
            col_b.markdown(f"<span style='font-size:13px;font-weight:600;color:{color}'>"
                           f"{mp*100:.1f}%</span>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIO SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Scenario Simulator":
    st.title("What-If Scenario Simulator")
    st.markdown("Model the portfolio-level impact of targeted retention campaigns.")

    BASELINE = {"total": 10000, "churned": 2037, "rate": 20.37,
                "inactive_total": 4849, "inactive_churn_rate": 0.269,
                "active_churn_rate": 0.143,
                "single_prod_total": 5084, "single_churn_rate": 0.277,
                "two_prod_churn_rate": 0.076,
                "germany_total": 2509, "germany_churn_rate": 0.324,
                "france_churn_rate": 0.162,
                "age5160_total": 797, "age5160_churn_rate": 0.562,
                "avg_balance_churner": 91108}

    st.markdown("#### Retention levers")
    col1, col2 = st.columns(2)
    with col1:
        act_pct = st.slider("% of inactive members re-activated",     0, 100, 0,
                            help="Converts inactive→active churn rate")
        prod_pct = st.slider("% of 1-product customers → 2 products", 0, 80,  0,
                             help="Moves customers to the lowest-risk product tier")
    with col2:
        ger_pct  = st.slider("Germany-specific churn reduction (%)",   0, 50,  0,
                             help="Regional program effectiveness")
        age_pct  = st.slider("Age 51–60 engagement lift (%)",          0, 60,  0,
                             help="Advisory / wealth management program impact")

    # Calculate saved customers
    saved_act  = BASELINE["inactive_total"] * (act_pct/100) * (BASELINE["inactive_churn_rate"] - BASELINE["active_churn_rate"])
    saved_prod = BASELINE["single_prod_total"] * (prod_pct/100) * (BASELINE["single_churn_rate"] - BASELINE["two_prod_churn_rate"])
    saved_ger  = BASELINE["germany_total"] * (ger_pct/100) * (BASELINE["germany_churn_rate"] - BASELINE["france_churn_rate"])
    saved_age  = BASELINE["age5160_total"] * (age_pct/100) * (BASELINE["age5160_churn_rate"] - 0.20)

    total_saved    = max(0, saved_act + saved_prod + saved_ger + saved_age)
    new_churners   = max(0, BASELINE["churned"] - total_saved)
    new_rate       = new_churners / BASELINE["total"] * 100
    assets_saved   = total_saved * BASELINE["avg_balance_churner"]

    st.markdown("---")
    st.markdown("#### Impact projection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline churn",   f"{BASELINE['rate']:.1f}%")
    c2.metric("Projected churn",  f"{new_rate:.1f}%",  f"{new_rate - BASELINE['rate']:.1f}pp")
    c3.metric("Customers saved",  f"{int(total_saved):,}", "from churning")
    c4.metric("Assets preserved", f"€{assets_saved/1e6:.1f}M", "AUM retained")

    # Waterfall
    st.markdown("#### Churn reduction waterfall")
    categories = ["Baseline", "Re-activate inactive", "Cross-sell to 2 products",
                  "Germany program", "Age 51-60 engagement", "Projected"]
    values   = [BASELINE["churned"], -saved_act, -saved_prod, -saved_ger, -saved_age, new_churners]
    measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
    clrs_wf  = ["#2563eb", "#22c55e", "#22c55e", "#22c55e", "#22c55e", "#ef4444" if new_rate > 15 else "#22c55e"]

    fig = go.Figure(go.Waterfall(
        measure=measures, x=categories, y=values,
        connector=dict(line=dict(color="#e0e3ea", width=1)),
        decreasing=dict(marker_color="#22c55e"),
        increasing=dict(marker_color="#ef4444"),
        totals=dict(marker_color=clrs_wf[-1]),
        text=[f"{abs(v):.0f}" for v in values], textposition="outside",
        textfont_size=11
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=340,
                      yaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont_size=11, title="Churned customers"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Breakdown bar
    if total_saved > 1:
        st.markdown("#### Savings breakdown by lever")
        levers   = ["Re-activate inactive", "Cross-sell 1→2 products",
                    "Germany regional", "Age 51–60 engagement"]
        savings  = [saved_act, saved_prod, saved_ger, saved_age]
        fig2 = go.Figure(go.Bar(
            x=levers, y=[max(0,s) for s in savings],
            marker_color=[C["blue"], C["indigo"], C["amber"], C["red"]],
            marker_line_width=0,
            text=[f"{max(0,s):.0f}" for s in savings], textposition="outside", textfont_size=11
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=280,
                           yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Customers saved"),
                           bargap=0.35)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        top_lever = levers[savings.index(max(savings))]
        st.markdown(
            f'<div class="success-box">Most impactful intervention: '
            f'<strong>{top_lever}</strong> — saves <strong>{max(0,max(savings)):.0f} customers</strong>. '
            f'Total asset preservation: <strong>€{assets_saved/1e6:.1f}M</strong> in AUM.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-box">Adjust the levers above to model retention campaign impact.</div>',
            unsafe_allow_html=True)
