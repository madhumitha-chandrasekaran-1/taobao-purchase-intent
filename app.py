from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Attention Depth Score", layout="wide", page_icon="📊")

APP_DIR = Path(__file__).parent

# while it's empty, the sidebar shows the local README.md instead.
README_URL = ""

V1_COLOR = "#eb6834"   # V1 baseline
V2_COLOR = "#2a78d6"   # V2 attention score (also the single-series color)
FUNNEL_COLORS = ["#86b6ef", "#5598e7", "#256abf", "#184f95"]  # light -> dark blue (ordinal)

RESULTS = {
    "events_after_cleaning": 839_613,
    # Engagement funnel after cleaning (event counts)
    "funnel": {"Page view (pv)": 748_690, "Favorite (fav)": 24_465,
               "Add to cart": 47_854, "Purchase (buy)": 18_604},
    # Test-set metrics (128,813 pairs, 3,568 buyers)
    "v1": {"roc_auc": 0.6781, "pr_auc": 0.0933, "precision": 0.125, "recall": 0.356, "f1": 0.185},
    "v2": {"roc_auc": 0.9122, "pr_auc": 0.6127, "precision": 0.172, "recall": 0.790, "f1": 0.282},
    # Confusion matrices at the 0.5 threshold
    "cm_v1": {"TN": 116_315, "FP": 8_930, "FN": 2_297, "TP": 1_271},
    "cm_v2": {"TN": 111_677, "FP": 13_568, "FN": 750, "TP": 2_818},
    # Train vs test (overfitting check); CV = 3-fold GridSearchCV score
    "train_test": {
        "V1 Logistic Regression": {"train_roc": 0.6850, "test_roc": 0.6781,
                                   "train_pr": 0.0829, "test_pr": 0.0933},
        "V2 XGBoost (tuned)":     {"train_roc": 0.9273, "test_roc": 0.9122,
                                   "train_pr": 0.6278, "test_pr": 0.6127},
    },
    "v2_cv_pr_auc": 0.6024,
    "v2_best_params": "learning_rate=0.05, max_depth=6, n_estimators=300, subsample=1.0",
    # Cutoff chosen on training-set CV predictions, then applied once to test
    "threshold": {"cutoff": 0.92, "precision": 0.790, "recall": 0.466, "f1": 0.586},
    # Model x feature-set comparison (3-fold CV PR-AUC on the training set)
    "model_vs_features": {
        "Logistic Regression": {"V1 features (2)": 0.0830, "V2 features (10)": 0.2201},
        "XGBoost":             {"V1 features (2)": 0.1424, "V2 features (10)": 0.6018},
    },
    # Ablation: tuned XGBoost with feature groups removed (3-fold CV PR-AUC, training set)
    "ablation": {
        "All 10 features": 0.6018,
        "Without session_count": 0.5809,
        "Without depth_score": 0.5989,
        "Without session_count & depth_score": 0.5785,
        "Without the 4 new features": 0.5999,
    },
    # Tuned V2 feature importance (10 features)
    "importance": {
        "session_count": 0.6088, "depth_score": 0.2227, "view_count": 0.0732,
        "funnel_steps": 0.0445, "category_avg_depth": 0.0186, "has_cart": 0.0095,
        "has_fav": 0.0075, "days_since_last": 0.0073, "engagement_span_days": 0.0048,
        "fav_count": 0.0034,
    },
}

# Data loading (cached so the 644K-row CSV is read only once)
@st.cache_data
def load_model_ready() -> pd.DataFrame:
    """Load the feature-engineered user-item table"""
    # The deployed repo ships the gzipped copy (~11 MB vs ~50 MB); pandas reads it directly
    names = ["model_ready.csv", "model_ready.csv.gz"]
    for path in [APP_DIR / "data" / n for n in names] + [APP_DIR / n for n in names]:
        if path.exists():
            return pd.read_csv(path)
    st.error("Could not find model_ready.csv — run userbehavior.ipynb first.")
    st.stop()


@st.cache_data
def load_readme() -> str:
    """Read README.md for the sidebar (empty string if it's missing)."""
    path = APP_DIR / "README.md"
    return path.read_text() if path.exists() else ""


@st.cache_data
def rate_by_bucket(df: pd.DataFrame, col: str, cap: int) -> pd.DataFrame:
    """Purchase rate and pair count for each value of `col`, with values >= cap grouped as 'cap+'."""
    buckets = df[col].clip(upper=cap).astype(int)
    out = df.groupby(buckets)["purchased"].agg(rate="mean", pairs="size").reset_index()
    out["label"] = out[col].astype(str)
    out.loc[out[col] == cap, "label"] = f"{cap}+"
    return out


df = load_model_ready()

def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Shared chart styling: quiet grid, no plotly toolbar clutter."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False)
    return fig


def rate_bar_chart(data: pd.DataFrame, title: str, x_title: str) -> go.Figure:
    """Single-series bar chart of purchase rate, with hover showing the pair count."""
    fig = go.Figure(go.Bar(
        x=data["label"], y=data["rate"], marker_color=V2_COLOR,
        marker_line_width=0, customdata=data["pairs"],
        text=[f"{r:.1%}" for r in data["rate"]], textposition="outside",
        hovertemplate=f"{x_title}: %{{x}}<br>Purchase rate: %{{y:.2%}}"
                      "<br>Pairs: %{customdata:,}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="Purchase rate",
                      yaxis_tickformat=".0%", bargap=0.3)
    fig.update_yaxes(range=[0, data["rate"].max() * 1.2])
    return style_fig(fig)


def confusion_fig(cm: dict, title: str) -> go.Figure:
    """2x2 confusion-matrix heatmap with counts written in each cell."""
    z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    labels = [[f"TN<br>{cm['TN']:,}", f"FP<br>{cm['FP']:,}"],
              [f"FN<br>{cm['FN']:,}", f"TP<br>{cm['TP']:,}"]]
    fig = go.Figure(go.Heatmap(
        z=np.log10(z),  # log scale so the small cells (TP, FN) aren't washed out
        x=["Predicted: No purchase", "Predicted: Purchase"],
        y=["Actual: No purchase", "Actual: Purchase"],
        text=labels, texttemplate="%{text}", textfont_size=16,
        colorscale=[[0, "#cde2fb"], [1, "#184f95"]], showscale=False,
        customdata=z, hovertemplate="%{y}<br>%{x}<br>Count: %{customdata:,}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)
    return style_fig(fig, height=340)


def approx_roc(auc: float, n: int = 200):
    """
    Approximate ROC curve from an AUC value using the binormal model:
    TPR = Phi(a + Phi^-1(FPR)), with a = sqrt(2) * Phi^-1(AUC).
    The real curve needs the model's predictions, which this app doesn't load.
    """
    fpr = np.linspace(0.0005, 0.9995, n)
    a = np.sqrt(2) * norm.ppf(auc)
    tpr = norm.cdf(a + norm.ppf(fpr))
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]

# Sidebar
with st.sidebar:
    st.title("📊 Attention Depth Score")
    st.caption("Purchase-intent prediction from engagement depth")
    st.badge("Model V2 · XGBoost · 10 features", color="blue")

    st.subheader("Dataset")
    st.markdown(
        "- **Taobao UserBehavior**, released by Alibaba on "
        "[Tianchi](https://tianchi.aliyun.com/dataset/649) "
        "([Kaggle mirror](https://www.kaggle.com/datasets/marwa80/userbehavior))\n"
        "- Sampled from 100M events: 10,000 users, 839,613 events after cleaning\n"
        "- 2017-11-25 to 2017-12-03 (Beijing time)\n"
        f"- **{len(df):,}** user–item pairs\n"
        f"- **{df['purchased'].mean():.2%}** purchase rate"
    )

    st.subheader("Documentation")
    if README_URL:
        st.link_button("Open README", README_URL)
    else:
        readme = load_readme()
        if readme:
            with st.expander("📄 README.md"):
                st.markdown(readme)
        else:
            st.caption("README.md not found next to app.py.")


tab_eda, tab_features, tab_results, tab_insights = st.tabs(
    ["🔍 EDA", "🧩 Features", "📈 Model Results", "💡 Insights"]
)

# Tab 1 — EDA
with tab_eda:
    st.header("Exploratory Data Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pairs", f"{len(df) / 1000:,.0f}K", help=f"{len(df):,} user–item pairs")
    c2.metric("Purchase rate", f"{df['purchased'].mean():.2%}")
    c3.metric("Total users", f"{df['user_id'].nunique():,}")
    c4.metric("Events", f"{RESULTS['events_after_cleaning'] / 1000:,.0f}K",
              help=f"{RESULTS['events_after_cleaning']:,} events after cleaning")

    # Engagement funnel — horizontal bars, page views on top
    funnel = RESULTS["funnel"]
    pv = funnel["Page view (pv)"]
    stages = list(funnel.keys())
    fig = go.Figure(go.Bar(
        y=stages, x=list(funnel.values()), orientation="h",
        marker_color=FUNNEL_COLORS, marker_line_width=0,
        text=[f"{v:,}  ({v / pv:.2%} of views)" for v in funnel.values()],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,} events<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, pv * 1.35], title="Event count")
    fig.update_layout(title="Engagement funnel: most attention is shallow")
    st.plotly_chart(style_fig(fig, height=320), width="stretch")

    left, right = st.columns(2)
    with left:
        # Depth 0 = pairs bought with no view/fav/cart event: 100% purchase rate
        #by construction, so it's hidden by default to keep the scale readable
        include_zero = st.checkbox("Include depth 0 (buy-only pairs, always 100%)", value=False)
        depth = rate_by_bucket(df, "depth_score", cap=8)
        if not include_zero:
            depth = depth[depth["depth_score"] > 0]
        st.plotly_chart(rate_bar_chart(depth, "Purchase rate by depth score", "Depth score"),
                        width="stretch")
    with right:
        st.write("") 
        st.write("")
        sessions = rate_by_bucket(df, "session_count", cap=6)
        st.plotly_chart(rate_bar_chart(sessions, "Purchase rate by session count",
                                       "Distinct days with the item"),
                        width="stretch")

    st.caption("Hover a bar to see how many pairs are in each bucket. "
               "Purchase rate climbs steadily with both depth and repeat visits.")

# Tab 2 — Features
with tab_features:
    st.header("Feature Engineering")

    features = pd.DataFrame([
        ("has_view", "Did the user view the item at all (binary)", "V1"),
        ("category_avg_depth", "Average depth score of the item's category (control)", "V1 + V2"),
        ("view_count", "Number of page views", "V2"),
        ("has_fav", "Favorited at least once (binary)", "V2"),
        ("fav_count", "Number of times favorited", "V2"),
        ("has_cart", "Added to cart at least once (binary)", "V2"),
        ("depth_score", "Weighted engagement: pv×1 + fav×2 + cart×3", "V2"),
        ("session_count", "Distinct days the user came back to the item", "V2"),
        ("days_since_last", "Days from last interaction to end of dataset (recency)", "V2"),
        ("engagement_span_days", "Days between first and last interaction", "V2"),
        ("funnel_steps", "Distinct pre-purchase stages reached (pv / fav / cart)", "V2"),
    ], columns=["Feature", "What it captures", "Used by"])
    st.dataframe(features, hide_index=True, width="stretch")

    #Feature importance, largest at the top
    imp = pd.Series(RESULTS["importance"]).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=V2_COLOR, marker_line_width=0,
        text=[f"{v:.3f}" for v in imp.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, imp.max() * 1.15], title="Importance (XGBoost gain)")
    fig.update_layout(title="V2 feature importance (tuned model, 10 features)")
    st.plotly_chart(style_fig(fig, height=420), width="stretch")
    st.caption("Importance shows what the model *used*, not what it *needs*. Removing "
               "session_count costs only 0.021 CV PR-AUC, because other features overlap "
               "with it (see the ablation in Model Results). The 4 features added in round 2 "
               "add nothing measurable (CV PR-AUC 0.600 without them vs 0.602 with them).")

    #User A vs User B — the same 5 views, spread differently over time
    st.subheader("Why session_count matters: User A vs User B")
    five_views = df[df["view_count"] == 5]
    a_pairs = five_views[five_views["session_count"] == 1]
    b_pairs = five_views[five_views["session_count"] >= 3]

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("#### 👤 User A — one burst")
            st.markdown("Views an item **5 times in a single day**.\n\n"
                        "Looks like quick comparison shopping.")
            st.metric("view_count / session_count", "5 / 1")
            st.metric("Actual purchase rate", f"{a_pairs['purchased'].mean():.1%}",
                      help=f"{len(a_pairs):,} pairs with 5 views on 1 day")
    with col_b:
        with st.container(border=True):
            st.markdown("#### 👤 User B — keeps coming back")
            st.markdown("Views an item **5 times across 3+ different days**.\n\n"
                        "Keeps thinking about it.")
            st.metric("view_count / session_count", "5 / 3+")
            st.metric("Actual purchase rate", f"{b_pairs['purchased'].mean():.1%}",
                      help=f"{len(b_pairs):,} pairs with 5 views on 3+ days")

    lift = b_pairs["purchased"].mean() / a_pairs["purchased"].mean()
    st.info(f"Same view count, but User B is **{lift:.1f}× more likely to buy**. "
            "session_count is what separates them, and it's the model's most-used feature.")

# Tab 3 — Model Results
with tab_results:
    st.header("Model Results")
    v1, v2 = RESULTS["v1"], RESULTS["v2"]
    cm1, cm2 = RESULTS["cm_v1"], RESULTS["cm_v2"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("V2 ROC-AUC", f"{v2['roc_auc']:.3f}", f"+{v2['roc_auc'] - v1['roc_auc']:.3f} vs V1")
    m2.metric("V2 PR-AUC", f"{v2['pr_auc']:.3f}", f"+{v2['pr_auc'] - v1['pr_auc']:.3f} vs V1")
    m3.metric("Recall (buyers)", f"{v2['recall']:.0%}", f"+{(v2['recall'] - v1['recall']) * 100:.0f} pts vs V1")
    m4.metric("Buyers caught", f"{cm2['TP']:,}", f"+{cm2['TP'] - cm1['TP']:,} vs V1",
              help=f"Out of {cm2['TP'] + cm2['FN']:,} buyers in the test set, at threshold 0.5")

    # V1 vs V2 comparison table with lift
    rows = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Precision (buy)", "precision"),
            ("Recall (buy)", "recall"), ("F1 (buy)", "f1")]
    comparison = pd.DataFrame({
        "Metric": [name for name, _ in rows],
        "V1 Baseline": [v1[k] for _, k in rows],
        "V2 Attention Score": [v2[k] for _, k in rows],
        "Lift": [f"{(v2[k] - v1[k]) / v1[k]:+.1%}" for _, k in rows],
    })
    st.subheader("V1 vs V2")
    st.dataframe(comparison.style.format({"V1 Baseline": "{:.4f}", "V2 Attention Score": "{:.4f}"}),
                 hide_index=True, width="stretch")
    st.caption("Test set: 128,813 pairs, 3,568 buyers. Precision / recall / F1 at threshold 0.5. "
               "A random model would score PR-AUC ≈ 0.028.")

    # Confusion matrices side by side
    left, right = st.columns(2)
    left.plotly_chart(confusion_fig(cm1, "V1 Baseline (threshold 0.5)"), width="stretch")
    right.plotly_chart(confusion_fig(cm2, "V2 Attention Score (threshold 0.5)"), width="stretch")

    # Approximate ROC curves
    fig = go.Figure()
    for name, auc, color in [("V1 Baseline", v1["roc_auc"], V1_COLOR),
                             ("V2 Attention Score", v2["roc_auc"], V2_COLOR)]:
        fpr, tpr = approx_roc(auc)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
                                 line=dict(color=color, width=2),
                                 hovertemplate="FPR %{x:.2f} → TPR %{y:.2f}<extra>" + name + "</extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC=0.5)",
                             line=dict(color="gray", width=1, dash="dash"), hoverinfo="skip"))
    fig.update_layout(title="ROC curve: V1 vs V2", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", hovermode="x unified",
                      legend=dict(x=0.55, y=0.08))
    fig.update_xaxes(range=[0, 1], showgrid=True)
    fig.update_yaxes(range=[0, 1.02])
    st.plotly_chart(style_fig(fig, height=460), width="stretch")
    st.caption("Curves are approximated from the AUC values with a binormal model — "
               "the shape is illustrative; the AUCs are the real test-set values. "
               "The notebook has the exact curves.")

    # Overfitting check
    st.subheader("Overfitting check: train vs test")
    tt = pd.DataFrame([
        {"Model": name, "Train ROC-AUC": m["train_roc"], "Test ROC-AUC": m["test_roc"],
         "ROC gap": m["train_roc"] - m["test_roc"], "Train PR-AUC": m["train_pr"],
         "Test PR-AUC": m["test_pr"], "PR gap": m["train_pr"] - m["test_pr"]}
        for name, m in RESULTS["train_test"].items()
    ])
    st.dataframe(tt.style.format({c: "{:+.4f}" if "gap" in c else "{:.4f}" for c in tt.columns if c != "Model"}),
                 hide_index=True, width="stretch")
    st.caption(f"V2's train–test gap is small (PR-AUC 0.015) and its 3-fold CV PR-AUC "
               f"({RESULTS['v2_cv_pr_auc']:.4f}) is close to the test score, so the model isn't "
               "memorising training pairs. This does **not** rule out leakage: the split is random, "
               "so train and test share the same users and days. "
               f"Best params: {RESULTS['v2_best_params']}.")

    # Threshold chosen without touching the test set
    st.subheader("Choosing a cutoff")
    th = RESULTS["threshold"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Cutoff (chosen on validation)", f"{th['cutoff']:.2f}")
    t2.metric("Test precision", f"{th['precision']:.0%}")
    t3.metric("Test recall", f"{th['recall']:.0%}")
    t4.metric("Test F1", f"{th['f1']:.3f}", f"{th['f1'] - v2['f1']:+.3f} vs cutoff 0.5")
    st.caption("The cutoff maximises F1 on out-of-fold predictions from the training set, "
               "then is applied once to the untouched test set. Class weighting pushes V2's "
               "scores up, which is why the useful cutoff sits far above 0.5.")

    # Model vs features: V1 -> V2 changes both at once, so separate them
    st.subheader("What drives the gain: model or features?")
    mvf = pd.DataFrame(RESULTS["model_vs_features"]).T
    st.dataframe(mvf.style.format("{:.4f}"), width="stretch")
    st.caption("3-fold CV PR-AUC on the training set. Engagement features help logistic "
               "regression (0.083 → 0.220) and XGBoost helps the view-only features "
               "(0.083 → 0.142), but the big jump to 0.602 needs **both**: a nonlinear model "
               "that can combine the engagement features.")

    # Ablation: what the model needs, not just what it uses
    st.subheader("Ablation: removing feature groups")
    abl = pd.Series(RESULTS["ablation"], name="CV PR-AUC").to_frame()
    abl["Change vs all"] = abl["CV PR-AUC"] - abl.loc["All 10 features", "CV PR-AUC"]
    st.dataframe(abl.style.format({"CV PR-AUC": "{:.4f}", "Change vs all": "{:+.4f}"}), width="stretch")
    st.caption("Removing session_count and depth_score together costs only 0.023. "
               "The engagement features overlap heavily, so no single feature is essential.")

# Tab 4 — Insights
with tab_insights:
    st.header("Insights")

    st.success(
        "**Core finding:** *how deeply and how often* a user engages with an item predicts "
        "a purchase far better than whether they viewed it. Moving from V1 (logistic "
        "regression, binary view signal) to V2 (XGBoost, engagement features) raises PR-AUC "
        f"from {RESULTS['v1']['pr_auc']:.3f} to {RESULTS['v2']['pr_auc']:.3f}. That gain needs "
        "both the richer features and a model that can combine them. Returning to an item "
        "on several days is the model's most-used signal."
    )
    st.warning("These scores are optimistic: the features use each pair's full history, "
               "including the purchase day (see Limitations).")

    st.subheader("Ad-tech proxy mapping")
    mapping = pd.DataFrame([
        ("pv (page view)", "Impression", "Someone was exposed to the item"),
        ("depth_score", "Attention Unit", "How much attention the exposure actually got"),
        ("session_count", "Cross-session reach", "Coming back to the same item on different days"),
        ("V1 model", "CPM buying", "Pay for every impression, treat all views alike"),
        ("V2 model", "Attention-weighted buying", "Value exposures by depth and persistence"),
    ], columns=["This project", "Ad-tech equivalent", "Meaning"])
    st.dataframe(mapping, hide_index=True, width="stretch")

    st.subheader("Limitations")
    st.markdown(
        "- **Target leakage:** features use each pair's full history, including the purchase "
        "day and any views after buying, so scores are optimistic. A stricter version would "
        "build features only from events before a cutoff date.\n"
        "- **Random split, not time-based:** the same users and days appear in train and test, "
        "so this doesn't measure forecasting future purchases.\n"
        "- **One sample of typical users:** results come from a single 10,000-user sample, "
        "with the 471 heaviest users removed as outliers.\n"
        "- **Uncalibrated probabilities:** class weighting inflates V2's scores, so the "
        "useful cutoff (0.92, chosen on validation data) is far from 0.5. The threshold "
        "has to be chosen deliberately."
    )

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:center; font-style:italic;'>"
        "“The signal isn’t presence — it’s persistence.”</h2>",
        unsafe_allow_html=True,
    )
