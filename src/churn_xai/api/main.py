from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path("artifacts")
PROCESSED_DIR = Path("data/processed")
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

app = FastAPI(title="Customer Churn XAI API", version="0.1.0")

# ---------- models ----------

class PredictRequest(BaseModel):
    # Accept dynamic feature dict; pydantic won't validate unknown keys strictly
    features: Dict[str, Any] = Field(..., description="Raw input feature mapping")

class PredictResponse(BaseModel):
    proba: float
    top_features: List[Dict[str, float]]  # [{"feature":"Tenure","impact":0.12}, ...]
    used_model: str

# ---------- init ----------

def _unwrap_calibrated(pipe):
    calib = pipe.named_steps["clf"]
    if hasattr(calib, "calibrated_classifiers_") and calib.calibrated_classifiers_:
        first = calib.calibrated_classifiers_[0]
        return getattr(first, "estimator", getattr(first, "base_estimator", first))
    return getattr(calib, "estimator", getattr(calib, "base_estimator", calib))

def _make_explainer(pipe, X_background: pd.DataFrame):
    pre = pipe.named_steps["pre"]
    est = _unwrap_calibrated(pipe)
    Xt = pre.transform(X_background)
    # Choose explainer based on estimator type
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    if isinstance(est, LogisticRegression):
        explainer = shap.LinearExplainer(est, Xt, feature_dependence="independent")
    elif isinstance(est, XGBClassifier):
        explainer = shap.TreeExplainer(est, feature_perturbation="interventional")
    else:
        explainer = shap.Explainer(est, Xt)  # generic fallback
    # feature names out of preprocessor
    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(Xt.shape[1])]
    return explainer, feat_names

def _group_shap_by_base(feat_names: List[str], shap_values: np.ndarray, k: int = 8) -> List[Dict[str, float]]:
    """
    Group one-hot feature contributions back to the original column name.
    Assumes OHE names look like `Column_Value`. Numeric features are just `Column`.
    We aggregate by sum of absolute contributions per base column.
    """
    contrib: Dict[str, float] = {}
    for name, val in zip(feat_names, shap_values):
        base = name.split("_", 1)[0]  # "Contract_Month-to-month" -> "Contract"
        contrib[base] = contrib.get(base, 0.0) + float(abs(val))
    # top-k
    top = sorted(contrib.items(), key=lambda x: -x[1])[:k]
    return [{"feature": n, "impact": v} for n, v in top]

@app.on_event("startup")
def _load() -> None:
    global PIPE, EXPLAINER, FEAT_NAMES, CHOSEN_MODEL, SCHEMA
    meta = yaml.safe_load((ARTIFACTS_DIR / "metrics.json").read_text())
    CHOSEN_MODEL = meta["chosen_model"]
    PIPE = joblib.load(ARTIFACTS_DIR / f"model_{CHOSEN_MODEL}.joblib")
    SCHEMA = {"categorical": meta.get("cats", []), "numeric": meta.get("nums", [])}

    # small background sample from train for SHAP
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    Xb = train.drop(columns=[train.columns[-1]])  # target is last in our save
    Xb = Xb.sample(min(500, len(Xb)), random_state=42)
    EXPLAINER, FEAT_NAMES = _make_explainer(PIPE, Xb)

@app.get("/health")
def health():
    return {"status": "ok", "model": CHOSEN_MODEL}

@app.get("/metadata")
def metadata():
    return {"model": CHOSEN_MODEL, "features": SCHEMA}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Build one-row DataFrame with all expected raw columns
    cols = SCHEMA["categorical"] + SCHEMA["numeric"]
    row = {c: req.features.get(c, None) for c in cols}
    X = pd.DataFrame([row])

    proba = float(PIPE.predict_proba(X)[:, 1][0])

    # SHAP for the single sample on transformed space
    pre = PIPE.named_steps["pre"]
    Xt = pre.transform(X)
    shap_vals = EXPLAINER(Xt, check_additivity=False)
    single = np.array(shap_vals.values)[0]  # shape (n_features,)
    top = _group_shap_by_base(FEAT_NAMES, single, k=8)

    return PredictResponse(proba=proba, top_features=top, used_model=CHOSEN_MODEL)
