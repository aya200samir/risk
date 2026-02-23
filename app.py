import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Judicial Integrity AI",
    layout="wide"
)
# =============================
# CUSTOM STYLING
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #111;
}
.metric-box {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)
# =============================
# TITLE
# =============================
st.title("⚖️ Judicial Integrity AI System")
st.markdown("AI-powered detection of abnormal sentencing patterns")
# =============================
# SIDEBAR
# =============================
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
# =============================
# FUNCTIONS
# =============================
def first_digit(n):
    n = abs(int(n))
    while n >= 10:
        n //= 10
    return n if n != 0 else 1
def compute_benford(df):
    df["first_digit"] = df["duration_months"].apply(first_digit)
    observed = df["first_digit"].value_counts(normalize=True).sort_index()
    expected = {d: np.log10(1 + 1/d) for d in range(1,10)}
    return observed, expected
def compute_risk(z_score, iso_pred, lof_pred):
    risk = 0
    if abs(z_score) > 20
        risk += 30
    if iso_pred == -1:
        risk += 35
    if lof_pred == -1:
        risk += 35
    return min(risk, 100)
# =============================
# MAIN LOGIC
# =============================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    # -------- Data Cleaning --------
    df = df.drop_duplicates()
    df = df.fillna(0)
    if "years" in df.columns and "months" in df.columns:
        df["duration_months"] = df["years"]*12 + df["months"]
    else:
        st.error("Dataset must contain 'years' and 'months' columns.")
        st.stop()
    # -------- Feature Engineering --------
    df["z_score"] = (
        df["duration_months"] - df["duration_months"].mean()
    ) / df["duration_months"].std()
    df["abs_dev"] = abs(
        df["duration_months"] - df["duration_months"].median()
    )
    features = ["duration_months", "z_score", "abs_dev"]
    X = df[features]
    # -------- Scaling --------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # -------- Models --------
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42
    )
    iso.fit(X_scaled)
    iso_pred = iso.predict(X_scaled)
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05
    )
    lof_pred = lof.fit_predict(X_scaled)
    df["iso_pred"] = iso_pred
    df["lof_pred"] = lof_pred
    # -------- Risk Score --------
    df["risk_score"] = df.apply(
        lambda row: compute_risk(
            row["z_score"],
            row["iso_pred"],
            row["lof_pred"]
        ), axis=1
    )
    # =============================
    # VISUALIZATION
    # =============================
    st.subheader("Sentence Distribution")
    fig = px.box(
        df,
        y="duration_months",
        points="all",
        title="Sentencing Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    # Risk Scatter
    st.subheader("Risk Score Distribution")
    fig2 = px.scatter(
        df,
        x=df.index,
        y="duration_months",
        color="risk_score",
        title="Anomaly Risk Mapping",
        color_continuous_scale="reds"
    )
    st.plotly_chart(fig2, use_container_width=True)
    # =============================
    # BENFORD LAW
    # =============================
    st.subheader("Benford's Law Analysis")
    observed, expected = compute_benford(df)
    benford_df = pd.DataFrame({
        "Digit": list(expected.keys()),
        "Expected": list(expected.values()),
        "Observed": [observed.get(d, 0) for d in expected.keys()]
    })
    fig3 = px.bar(
        benford_df,
        x="Digit",
        y=["Expected", "Observed"],
        barmode="group",
        title="Benford Distribution Comparison"
    )
    st.plotly_chart(fig3, use_container_width=True)
    # =============================
    # HIGH RISK CASES
    # =============================
    st.subheader("High Risk Cases 🚨")
    high_risk = df[df["risk_score"] > 70]
    st.dataframe(high_risk)
    # =============================
    # EXPLAINABILITY (SHAP)
    # =============================
    st.subheader("Model Explainability (SHAP)")
    explainer = shap.TreeExplainer(iso)
    shap_values = explainer.shap_values(X_scaled)
    fig4, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig4)
else:
    st.info("Please upload a dataset to begin analysis.")
