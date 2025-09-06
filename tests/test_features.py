from __future__ import annotations
from pathlib import Path
import json
import yaml
import joblib
import pandas as pd

def test_processed_exists():
    assert Path("data/processed/train.parquet").exists()
    assert Path("data/processed/test.parquet").exists()
    # quick shape sanity
    train = pd.read_parquet("data/processed/train.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    assert len(train) > 0 and len(test) > 0
    assert "Churn" in train.columns

def test_training_artifacts_exist():
    assert Path("artifacts").exists()
    metrics_path = Path("artifacts/metrics.json")
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "chosen_model" in metrics and metrics["chosen_model"] in {"logreg", "xgboost"}
    # model loads
    chosen = metrics["chosen_model"]
    model_path = Path(f"artifacts/model_{chosen}.joblib")
    pipe = joblib.load(model_path)
    assert hasattr(pipe, "predict_proba")

def test_reports_exist():
    reports = Path("reports")
    imgs = list(reports.glob("*.png"))
    assert len(imgs) >= 2  # PR + calibration at least
