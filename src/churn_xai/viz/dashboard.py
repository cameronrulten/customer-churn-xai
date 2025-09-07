import json
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ART = Path("artifacts")
REP = Path("reports")

st.set_page_config(page_title="Churn XAI", layout="wide")
st.title("📉 Customer Churn — XAI Dashboard")

# Sidebar: API location
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
st.sidebar.write("Make sure the API is running.")

# Metrics summary
col1, col2, col3 = st.columns(3)
metrics = json.loads((ART/"metrics.json").read_text())
eval_path = ART / "eval_summary.json"
eval_json = json.loads(eval_path.read_text()) if eval_path.exists() else None

col1.metric("Chosen model", metrics["chosen_model"])
if eval_json:
    col2.metric("PR-AUC", f"{eval_json['ap']:.3f}")
    col3.metric("ROC-AUC", f"{eval_json['roc_auc']:.3f}")

# Plots
plot_cols = st.columns(3)
def img(path, col):
    p = REP / path if not str(path).startswith("reports/") else Path(path)
    if p.exists():
        col.image(str(p))
img("pr_curve_logreg.png", plot_cols[0])
img("calibration_logreg.png", plot_cols[1])
img("shap_bar_logreg.png", plot_cols[2])
img("pr_curve_xgboost.png", plot_cols[0])
img("calibration_xgboost.png", plot_cols[1])
img("shap_bar_xgboost.png", plot_cols[2])
img("eval_pr_curve.png", plot_cols[0])
img("eval_confusion.png", plot_cols[1])
img("eval_gains.png", plot_cols[2])

st.markdown("---")
st.header("🔮 Try a prediction")

schema = {"categorical": metrics.get("cats", []), "numeric": metrics.get("nums", [])}
form = st.form("predict")
inputs = {}

# Load some defaults from processed train (nice UX)
try:
    train = pd.read_parquet("data/processed/train.parquet")
    defaults = train.drop(columns=["Churn"]).iloc[0].to_dict()
except Exception:
    defaults = {}

for col in schema["numeric"]:
    val = defaults.get(col, 0.0)
    inputs[col] = form.number_input(col, value=float(val))

for col in schema["categorical"]:
    # build simple options from train if available
    options = sorted(train[col].dropna().unique().tolist()) if 'train' in locals() and col in train.columns else []
    val = defaults.get(col, options[0] if options else "")
    inputs[col] = form.selectbox(col, options=options, index=options.index(val) if val in options else 0) if options else form.text_input(col, value=str(val))

submitted = form.form_submit_button("Predict")
if submitted:
    with st.spinner("Calling API..."):
        try:
            r = httpx.post(f"{api_url}/predict", json={"features": inputs}, timeout=15.0)
            r.raise_for_status()
            data = r.json()
            st.success(f"Churn probability: {data['proba']:.3f}")
            tf = pd.DataFrame(data["top_features"])
            fig = px.bar(tf, x="impact", y="feature", orientation="h", title="Top drivers (|SHAP| grouped)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Request failed: {e}")
