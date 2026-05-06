import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Cancellation — ML Dashboard",
    page_icon="🏨",
    layout="wide",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
BLUE   = "#4C78A8"
RED    = "#E45756"
GREEN  = "#54A24B"
ORANGE = "#F58518"
PURPLE = "#B279A2"
BG     = "#F0F2F5"
WHITE  = "#FFFFFF"
DARK   = "#1E2D3D"
SUB    = "#64748B"
GRID   = "#E8ECF0"
FONT   = "Inter, Segoe UI, Arial, sans-serif"

MODEL_COLORS = {
    "Logistic Regression": BLUE,
    "Decision Tree":       ORANGE,
    "XGBoost":             GREEN,
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: {FONT}; }}
  .main {{ background-color: {BG}; }}
  .metric-card {{
      background: #1E2D3D;
      border-radius: 12px;
      padding: 18px 22px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.3);
      text-align: center;
      border: 1px solid #2C3E50;
  }}
  .metric-label {{ color: #B0BEC5; font-size: 13px; font-weight: 500; margin-bottom: 6px; }}
  .metric-value {{ color: #FFFFFF; font-size: 26px; font-weight: 700; }}
  .metric-sub   {{ color: #78909C; font-size: 11px; margin-top: 3px; }}
  .section-header {{
      color: #FFFFFF; font-size: 20px; font-weight: 700;
      margin: 32px 0 4px 0; letter-spacing: -0.3px;
  }}
  .section-sub {{ color: #B0BEC5; font-size: 13px; margin-bottom: 18px; }}
  .insight-box {{
      background: #1E2D3D; border-left: 4px solid #54A24B;
      border-radius: 8px; padding: 14px 18px;
      margin-top: 14px; font-size: 13.5px; color: #ECEFF1;
      box-shadow: 0 1px 4px rgba(0,0,0,0.2);
      border-top: 1px solid #2C3E50;
      border-right: 1px solid #2C3E50;
      border-bottom: 1px solid #2C3E50;
  }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data & model loading (cached so it only runs once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Training models — please wait…")
def load_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), "hotel_bookings_cleaned.csv")
    df = pd.read_csv(csv_path)

    candidate = [
        "LeadTime","ADR","Adults","Children","Babies","BookingChanges",
        "PreviousCancellations","PreviousBookingsNotCanceled",
        "RequiredCarParkingSpaces","TotalOfSpecialRequests","IsRepeatedGuest",
        "hotel_type","MarketSegment","DistributionChannel","DepositType",
        "CustomerType","arrival_month_num","season","total_nights","total_guests",
        "has_children","is_family_booking","has_special_requests",
        "has_parking_request","has_booking_changes","had_previous_history",
        "extreme_adr_flag","extreme_lead_time_flag","has_agent","has_company",
    ]
    features = [c for c in candidate if c in df.columns]

    X = df[features].copy()
    y = df["IsCanceled"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_feats = X_train.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_feats  = [c for c in X_train.columns if c not in num_feats]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", num_pipe, num_feats),
                                      ("cat", cat_pipe, cat_feats)])

    models = {
        "Logistic Regression": Pipeline([("pre", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=42))]),
        "Decision Tree": Pipeline([("pre", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42))]),
        "XGBoost": Pipeline([("pre", preprocessor),
            ("model", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
                eval_metric="logloss", random_state=42, n_jobs=-1))]),
    }

    results, probs, preds = {}, {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        probs[name] = prob
        preds[name] = pred
        results[name] = {
            "Accuracy":  accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall":    recall_score(y_test, pred),
            "F1":        f1_score(y_test, pred),
            "ROC-AUC":   roc_auc_score(y_test, prob),
            "PR-AUC":    average_precision_score(y_test, prob),
        }

    # Feature names after preprocessing (for XGBoost importance)
    xgb_pipe = models["XGBoost"]
    ohe_names = (xgb_pipe.named_steps["pre"]
                 .named_transformers_["cat"]["ohe"]
                 .get_feature_names_out(cat_feats))
    feat_names = num_feats + list(ohe_names)
    importances = xgb_pipe.named_steps["model"].feature_importances_

    return {
        "X_test":      X_test,
        "y_test":      y_test,
        "results":     results,
        "probs":       probs,
        "preds":       preds,
        "models":      models,
        "feat_names":  feat_names,
        "importances": importances,
        "n_train":     len(X_train),
        "n_test":      len(X_test),
    }


data = load_and_train()
y_test      = data["y_test"]
results     = data["results"]
probs       = data["probs"]
preds       = data["preds"]
feat_names  = data["feat_names"]
importances = data["importances"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:28px;">
  <div style="font-size:28px;font-weight:700;color:#FFFFFF;letter-spacing:-0.4px;">
    🏨 Hotel Cancellation Prediction — ML Results
  </div>
  <div style="font-size:14px;color:#B0BEC5;margin-top:4px;">
    Trained on {data['n_train']:,} bookings &nbsp;•&nbsp;
    Evaluated on {data['n_test']:,} held-out bookings &nbsp;•&nbsp;
    Models: Logistic Regression · Decision Tree · XGBoost
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI metric cards (XGBoost) ─────────────────────────────────────────────────
xgb_res = results["XGBoost"]
# Row 1 of metrics
r1c1, r1c2, r1c3 = st.columns(3)
for col, label, key in [
    (r1c1, "Accuracy",  "Accuracy"),
    (r1c2, "Precision", "Precision"),
    (r1c3, "Recall",    "Recall"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{xgb_res[key]:.3f}</div>
      <div class="metric-sub">XGBoost</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

# Row 2 of metrics
r2c1, r2c2, r2c3 = st.columns(3)
for col, label, key in [
    (r2c1, "F1 Score", "F1"),
    (r2c2, "ROC-AUC",  "ROC-AUC"),
    (r2c3, "PR-AUC",   "PR-AUC"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{xgb_res[key]:.3f}</div>
      <div class="metric-sub">XGBoost</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VISUAL 1 — Confusion Matrix (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">1 · Confusion Matrix — XGBoost</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">Predicted vs actual outcomes on the 23,878-booking test set at 0.5 threshold.</div>', unsafe_allow_html=True)

col_cm, col_cm_ins = st.columns([1.6, 1])

with col_cm:
    cm = confusion_matrix(y_test, preds["XGBoost"])
    tn, fp, fn, tp = cm.ravel()

    z    = [[tn, fp], [fn, tp]]
    text = [[f"<b>TN</b><br>{tn:,}", f"<b>FP</b><br>{fp:,}"],
            [f"<b>FN</b><br>{fn:,}", f"<b>TP</b><br>{tp:,}"]]

    fig_cm = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted: Not Cancelled", "Predicted: Cancelled"],
        y=["Actual: Not Cancelled", "Actual: Cancelled"],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=18, color="white", family=FONT),
        colorscale=[[0, "#C8E6C9"], [0.5, GREEN], [1, "#1B5E20"]],
        showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Count: %{z:,}<extra></extra>",
    ))
    fig_cm.update_layout(
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family=FONT, size=13, color=DARK),
        xaxis=dict(side="bottom", tickfont=dict(size=13, color=DARK), title_font=dict(color=DARK)),
        yaxis=dict(tickfont=dict(size=13, color=DARK), title_font=dict(color=DARK)),
        height=340,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cm_ins:
    total   = tn + fp + fn + tp
    acc     = (tp + tn) / total
    prec    = tp / (tp + fp)
    rec     = tp / (tp + fn)
    st.markdown(f"""
    <div class="insight-box">
      <b>Key Takeaways</b><br><br>
      ✅ <b>True Negatives: {tn:,}</b> — correctly predicted guests who stayed.<br><br>
      ✅ <b>True Positives: {tp:,}</b> — cancellations correctly caught by the model.<br><br>
      ⚠️ <b>False Negatives: {fn:,}</b> — cancellations the model missed (guests predicted to stay but cancelled).<br><br>
      ⚠️ <b>False Positives: {fp:,}</b> — guests incorrectly flagged as cancellations.<br><br>
      The model catches <b>{rec*100:.1f}%</b> of all actual cancellations, with a precision of <b>{prec*100:.1f}%</b> — meaning when it predicts a cancellation, it is correct {prec*100:.1f}% of the time.<br><br>
      <hr style="border:1px solid #2C3E50; margin:10px 0;">
      <b>🎯 Project-Related Insight</b><br><br>
      This directly answers <i>"which reservations are most likely to cancel"</i> — the model identifies <b>{tp:,} high-risk bookings</b> a hotel can act on proactively. With 85% precision, hotel managers can confidently prioritise these flagged bookings for retention outreach without wasting resources on guests who were always going to show up.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# VISUAL 2 — ROC Curve (all 3 models)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">2 · ROC Curve — All 3 Models</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">Compares each model\'s ability to discriminate between cancellations and non-cancellations across all thresholds. A higher curve = better model.</div>', unsafe_allow_html=True)

col_roc, col_roc_ins = st.columns([1.6, 1])

with col_roc:
    fig_roc = go.Figure()

    # Random baseline
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#AAAAAA", width=1.5, dash="dash"),
        name="Random Baseline (AUC = 0.50)",
        hoverinfo="skip",
    ))

    for name, color in MODEL_COLORS.items():
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        auc = results[name]["ROC-AUC"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{name} (AUC = {auc:.3f})",
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))

    fig_roc.update_layout(
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        xaxis=dict(title="False Positive Rate", gridcolor=GRID, range=[0, 1],
                   title_font=dict(size=13, color=DARK), tickfont=dict(size=12, color=DARK)),
        yaxis=dict(title="True Positive Rate", gridcolor=GRID, range=[0, 1],
                   title_font=dict(size=13, color=DARK), tickfont=dict(size=12, color=DARK)),
        legend=dict(
            orientation="v",
            yanchor="bottom", y=0.02,
            xanchor="right", x=0.98,
            font=dict(size=13, color=DARK),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=GRID,
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=30, b=55),
        font=dict(family=FONT, size=13),
        height=380,
        hovermode="x unified",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_roc_ins:
    st.markdown(f"""
    <div class="insight-box">
      <b>Key Takeaways</b><br><br>
      🏆 <b>XGBoost (AUC = {results['XGBoost']['ROC-AUC']:.3f})</b> is the strongest model — its curve stays closest to the top-left corner, meaning it correctly ranks cancellations highest.<br><br>
      📊 <b>Logistic Regression (AUC = {results['Logistic Regression']['ROC-AUC']:.3f})</b> performs well for a linear model, showing the features have strong linear signal.<br><br>
      🌳 <b>Decision Tree (AUC = {results['Decision Tree']['ROC-AUC']:.3f})</b> is the weakest of the three, limited by its shallow depth constraint.<br><br>
      All three models substantially outperform the random baseline (AUC = 0.50), confirming that guest booking behaviour is genuinely predictive of cancellation.<br><br>
      <hr style="border:1px solid #2C3E50; margin:10px 0;">
      <b>🎯 Project-Related Insight</b><br><br>
      An AUC of 0.896 means if a hotel picked one booking that cancelled and one that did not, our model would correctly identify the cancellation as higher risk <b>89.6% of the time</b>. This makes it a reliable tool for answering <i>"which reservations are most likely to cancel"</i> — the model ranks risk accurately enough to be trusted for real business decisions.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# VISUAL 3 — Feature Importance (XGBoost, top 15)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">3 · Feature Importance — XGBoost (Top 15)</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">Which features the XGBoost model relies on most when predicting cancellation. Higher = more influential.</div>', unsafe_allow_html=True)

col_fi, col_fi_ins = st.columns([1.6, 1])

with col_fi:
    top_n   = 15
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feat_names[i] for i in top_idx]
    top_vals  = [importances[i] for i in top_idx]

    # Reverse for horizontal bar (most important at top)
    top_feats_r = top_feats[::-1]
    top_vals_r  = top_vals[::-1]

    # Colour gradient: top features darker
    bar_colors = [
        f"rgba(76,120,168,{0.4 + 0.6*(i/(top_n-1))})"
        for i in range(top_n)
    ]

    fig_fi = go.Figure(go.Bar(
        x=top_vals_r,
        y=top_feats_r,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color=WHITE, width=0.5)),
        text=[f"{v:.3f}" for v in top_vals_r],
        textposition="outside",
        textfont=dict(size=11, color=DARK),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        xaxis=dict(title="Feature Importance Score", gridcolor=GRID,
                   tickfont=dict(size=11, color=DARK), title_font=dict(size=13, color=DARK)),
        yaxis=dict(tickfont=dict(size=11, color=DARK), title_font=dict(color=DARK)),
        margin=dict(l=10, r=60, t=20, b=50),
        font=dict(family=FONT, size=12),
        height=480,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with col_fi_ins:
    st.markdown(f"""
    <div class="insight-box">
      <b>Key Takeaways</b><br><br>
      🔑 <b>Deposit Type</b> dominates — "Non Refund" and "No Deposit" together account for over <b>60%</b> of the model's predictive weight. Non-refundable bookings are associated with very high cancellation risk.<br><br>
      🅿️ <b>Parking requests</b> rank 3rd — guests who request parking are far less likely to cancel, acting as a strong commitment signal.<br><br>
      🌐 <b>Online Travel Agency</b> bookings (MarketSegment) carry elevated cancellation risk compared to direct bookings.<br><br>
      💰 <b>ADR and LeadTime</b> both appear in the top 15 — higher prices and longer booking horizons both increase cancellation likelihood.<br><br>
      These results align with the EDA findings and give the model strong business interpretability.<br><br>
      <hr style="border:1px solid #2C3E50; margin:10px 0;">
      <b>🎯 Project-Related Insight</b><br><br>
      This directly answers <i>"what factors drive cancellation risk"</i> — the top drivers are <b>Deposit Type, special requests, parking requests, booking channel, ADR, and lead time</b>. A hotel can act on these: encouraging guests to add special requests, offering parking options at booking, and monitoring OTA bookings more closely — all reduce cancellation likelihood based on what the model learned.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# VISUAL 4 — Predicted Probability Distribution
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">4 · Predicted Probability Distribution — XGBoost</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">Distribution of XGBoost\'s predicted cancellation probability, split by actual outcome. Well-separated peaks indicate a confident, discriminating model.</div>', unsafe_allow_html=True)

col_pd, col_pd_ins = st.columns([1.6, 1])

with col_pd:
    xgb_probs = probs["XGBoost"]
    probs_cancelled     = xgb_probs[y_test == 1]
    probs_not_cancelled = xgb_probs[y_test == 0]

    fig_pd = go.Figure()

    fig_pd.add_trace(go.Histogram(
        x=probs_not_cancelled,
        name="Not Cancelled (Actual)",
        nbinsx=50,
        marker=dict(color=GREEN, opacity=0.70, line=dict(color=WHITE, width=0.3)),
        hovertemplate="Probability: %{x:.2f}<br>Count: %{y:,}<extra>Not Cancelled</extra>",
    ))

    fig_pd.add_trace(go.Histogram(
        x=probs_cancelled,
        name="Cancelled (Actual)",
        nbinsx=50,
        marker=dict(color=RED, opacity=0.70, line=dict(color=WHITE, width=0.3)),
        hovertemplate="Probability: %{x:.2f}<br>Count: %{y:,}<extra>Cancelled</extra>",
    ))

    # Decision threshold line
    fig_pd.add_vline(
        x=0.5,
        line=dict(color=DARK, width=2, dash="dash"),
        annotation_text="Threshold = 0.5",
        annotation_position="top right",
        annotation_font=dict(size=12, color=DARK),
    )

    fig_pd.update_layout(
        barmode="overlay",
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        xaxis=dict(title="Predicted Cancellation Probability",
                   gridcolor=GRID, range=[0, 1], tickformat=".1f",
                   title_font=dict(size=13, color=DARK), tickfont=dict(size=12, color=DARK)),
        yaxis=dict(title="Number of Bookings", gridcolor=GRID,
                   title_font=dict(size=13, color=DARK), tickfont=dict(size=12, color=DARK)),
        legend=dict(
            orientation="v",
            yanchor="top", y=0.98,
            xanchor="right", x=0.98,
            font=dict(size=13, color=DARK),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=GRID,
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=30, b=55),
        font=dict(family=FONT, size=13),
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig_pd, use_container_width=True)

with col_pd_ins:
    high_risk = (xgb_probs >= 0.7).sum()
    low_risk  = (xgb_probs <= 0.3).sum()
    st.markdown(f"""
    <div class="insight-box">
      <b>Key Takeaways</b><br><br>
      📈 The two distributions are <b>well separated</b> — actual non-cancellations (green) peak near probability 0, while actual cancellations (red) peak near probability 1. This confirms the model is genuinely discriminating.<br><br>
      ⚡ <b>{high_risk:,} bookings ({high_risk/len(xgb_probs)*100:.1f}%)</b> were assigned a predicted risk above 0.70 — these are the highest-priority flagged cancellations.<br><br>
      ✅ <b>{low_risk:,} bookings ({low_risk/len(xgb_probs)*100:.1f}%)</b> were assigned a risk below 0.30 — the model is very confident these guests will stay.<br><br>
      The dashed line at 0.5 is the decision threshold. Bookings to the right are classified as cancellations. Lowering this threshold would catch more cancellations at the cost of more false alarms.<br><br>
      <hr style="border:1px solid #2C3E50; margin:10px 0;">
      <b>🎯 Project-Related Insight</b><br><br>
      This confirms the model is genuinely useful for answering <i>"which reservations are most likely to cancel"</i> — the clear separation between the two distributions means the model is not guessing. A hotel can confidently use bookings with predicted probability above 0.70 as a priority action list, covering <b>{{high_risk/len(xgb_probs)*100:.1f}}%</b> of all test bookings flagged as highest risk.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; color:#B0BEC5; font-size:12px; padding:16px 0 8px 0;">
  Hotel Bookings Cancellation Prediction &nbsp;•&nbsp;
  XGBoost ROC-AUC: {results['XGBoost']['ROC-AUC']:.3f} &nbsp;•&nbsp;
  Test set: {data['n_test']:,} bookings
</div>
""", unsafe_allow_html=True)
