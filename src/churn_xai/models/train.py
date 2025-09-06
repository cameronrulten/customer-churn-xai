from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

app = typer.Typer(help="Train baseline models (LogReg + XGBoost) with calibration & SHAP.")

def _load_processed(processed_dir: Path, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(processed_dir / "train.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    assert target in train.columns and target in test.columns, "Target missing."
    return train, test

def _preprocessor(cats, nums) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[("cat", cat_pipe, cats), ("num", num_pipe, nums)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def _unwrap_estimator(calib) -> object:
    """
    Return the underlying (pre-calibration) estimator from a CalibratedClassifierCV,
    handling version differences.
    """
    # Newer sklearn stores the original estimator on each calibrated classifier
    if hasattr(calib, "calibrated_classifiers_") and calib.calibrated_classifiers_:
        first = calib.calibrated_classifiers_[0]
        return getattr(first, "estimator", getattr(first, "base_estimator", first))
    # Fallbacks
    return getattr(calib, "estimator", getattr(calib, "base_estimator", calib))


def _calibrated(model, method: str, cv: int) -> CalibratedClassifierCV:
    # Wrap the base estimator with cross-validated calibration
    return CalibratedClassifierCV(estimator=model, method=method, cv=cv, n_jobs=None)

def _fit_and_eval(
    name: str,
    base_estimator,
    preproc: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method: str,
    cv_folds: int,
    reports_dir: Path,
) -> Dict:
    pipe = Pipeline(steps=[("pre", preproc), ("clf", _calibrated(base_estimator, method, cv_folds))])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    pr_auc = float(average_precision_score(y_test, proba))
    roc_auc = float(roc_auc_score(y_test, proba))

    # Curves
    precision, recall, _ = precision_recall_curve(y_test, proba)
    fig1 = plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall: {name} (AP={pr_auc:.3f})")
    pr_path = reports_dir / f"pr_curve_{name}.png"
    fig1.savefig(pr_path, bbox_inches="tight")
    plt.close(fig1)

    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig2 = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration: {name}")
    cal_path = reports_dir / f"calibration_{name}.png"
    fig2.savefig(cal_path, bbox_inches="tight")
    plt.close(fig2)

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "name": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "report": report,
        "pr_curve_path": str(pr_path),
        "calibration_path": str(cal_path),
        "pipeline": pipe,  # return for saving if best
    }

def _compute_shap(
    pipeline: Pipeline,
    model_name: str,
    X_sample: pd.DataFrame,
    reports_dir: Path,
) -> Dict[str, str]:
    """
    Compute SHAP summary plots for the fitted pipeline.
    Works for:
      - LogisticRegression via LinearExplainer
      - XGBClassifier via TreeExplainer
    """
    # Get the transformed feature matrix and feature names
    pre = pipeline.named_steps["pre"]
    calib = pipeline.named_steps["clf"]
    est = _unwrap_estimator(calib)
    # clf = pipeline.named_steps["clf"]
    
    # CalibratedClassifierCV wraps the estimator
    # est = clf.base_estimator

    Xt = pre.transform(X_sample)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(Xt.shape[1])]

    explainer = None
    if isinstance(est, LogisticRegression):
        explainer = shap.LinearExplainer(est, Xt, feature_dependence="independent")
    elif isinstance(est, XGBClassifier):
        explainer = shap.TreeExplainer(est, feature_perturbation="interventional")
    else:
        return {}

    shap_values = explainer(Xt, check_additivity=False)

    # Summary bar
    bar_fig = plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=20)
    bar_path = reports_dir / f"shap_bar_{model_name}.png"
    plt.title(f"SHAP Summary (bar): {model_name}")
    bar_fig.savefig(bar_path, bbox_inches="tight")
    plt.close(bar_fig)

    # Beeswarm
    swarm_fig = plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    swarm_path = reports_dir / f"shap_beeswarm_{model_name}.png"
    plt.title(f"SHAP Beeswarm: {model_name}")
    swarm_fig.savefig(swarm_path, bbox_inches="tight")
    plt.close(swarm_fig)

    return {"shap_bar": str(bar_path), "shap_beeswarm": str(swarm_path)}

@app.command()
def train(config: Path = typer.Option(Path("configs/model.yaml"), "--config", "-c")) -> None:
    """
    Train LogisticRegression and XGBoost with cross-validated calibration.
    Select the best by PR-AUC, save model + metrics + plots.
    """
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])
    artifacts_dir = Path(cfg["artifacts_dir"])
    reports_dir = Path(cfg["reports_dir"])
    for d in (artifacts_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    target = cfg["target"]
    train_df, test_df = _load_processed(processed_dir, target)
    y_train = train_df[target].astype(int)
    y_test = test_df[target].astype(int)
    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])

    # Determine categorical/numeric from saved schema (or infer)
    schema_path = processed_dir / "schema.yaml"
    if schema_path.exists():
        schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        cats, nums = schema["categorical"], schema["numeric"]
    else:
        cats = [c for c in X_train.columns if X_train[c].dtype == "object"]
        nums = [c for c in X_train.columns if c not in cats]

    preproc = _preprocessor(cats, nums)

    # Base estimators
    log_cfg = cfg["logreg"]
    xgb_cfg = cfg["xgboost"]

    logreg = LogisticRegression(
        C=log_cfg["C"], max_iter=log_cfg["max_iter"], class_weight=log_cfg["class_weight"], solver="lbfgs", #n_jobs=None
    ) #If we switch to solver="liblinear", we can pass n_jobs, otherwise it’s not needed here

    xgb = XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        max_depth=xgb_cfg["max_depth"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        reg_alpha=xgb_cfg["reg_alpha"],
        reg_lambda=xgb_cfg["reg_lambda"],
        random_state=cfg["random_state"],
        eval_metric="logloss",
        tree_method=xgb_cfg.get("tree_method", "auto"),
        enable_categorical=False,
        n_jobs=0,
    )

    # Fit & evaluate both
    results = []
    for name, est in [("logreg", logreg), ("xgboost", xgb)]:
        res = _fit_and_eval(
            name=name,
            base_estimator=est,
            preproc=preproc,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            method=cfg["calibration_method"],
            cv_folds=cfg["cv_folds"],
            reports_dir=reports_dir,
        )
        results.append(res)

    # Pick best by PR-AUC
    best = max(results, key=lambda d: d["pr_auc"])
    best_name = best["name"]
    best_pipe: Pipeline = best["pipeline"]

    # SHAP on a small, representative sample to keep it quick
    sample_n = min(1000, len(X_train))
    X_sample = X_train.sample(sample_n, random_state=cfg["random_state"])
    shap_paths = _compute_shap(best_pipe, best_name, X_sample, reports_dir)

    # Save model + metrics
    model_path = artifacts_dir / f"model_{best_name}.joblib"
    joblib.dump(best_pipe, model_path)

    metrics = {
        "chosen_model": best_name,
        "logreg": {k: v for k, v in results[0].items() if k != "pipeline"},
        "xgboost": {k: v for k, v in results[1].items() if k != "pipeline"},
        "shap_plots": shap_paths,
        "cats": cats,
        "nums": nums,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    typer.secho(f"Best model: {best_name} (PR-AUC={best['pr_auc']:.3f})", fg=typer.colors.GREEN)
    typer.secho(f"Saved model → {model_path}", fg=typer.colors.CYAN)
    typer.secho(f"Metrics/plots in {artifacts_dir} and {cfg['reports_dir']}", fg=typer.colors.CYAN)


if __name__ == "__main__":
    app()
