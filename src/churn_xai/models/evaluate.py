from __future__ import annotations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import yaml
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)

app = typer.Typer(help="Extra evaluation and plots for the chosen model.")

def _load_best(artifacts_dir: Path):
    import joblib
    meta = json.loads((artifacts_dir / "metrics.json").read_text())
    model_name = meta["chosen_model"]
    model_path = artifacts_dir / f"model_{model_name}.joblib"
    pipe = joblib.load(model_path)
    return model_name, pipe

@app.command()
def run(config: Path = typer.Option(Path("configs/eval.yaml"), "--config", "-c")) -> None:
    cfg = yaml.safe_load(config.read_text())
    processed_dir = Path(cfg["processed_dir"])
    artifacts_dir = Path(cfg["artifacts_dir"])
    reports_dir = Path(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    target = cfg["target"]
    thr = float(cfg.get("threshold", 0.5))

    test = pd.read_parquet(processed_dir / "test.parquet")
    y_test = test[target].astype(int).values
    X_test = test.drop(columns=[target])

    model_name, pipe = _load_best(artifacts_dir)
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= thr).astype(int)

    ap = float(average_precision_score(y_test, proba))
    roc = float(roc_auc_score(y_test, proba))

    # PR curve again (with threshold marker)
    p, r, t = precision_recall_curve(y_test, proba)
    fig = plt.figure()
    plt.step(r, p, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{model_name} — PR curve (AP={ap:.3f})")
    (reports_dir / "eval_pr_curve.png").unlink(missing_ok=True)
    fig.savefig(reports_dir / "eval_pr_curve.png", bbox_inches="tight"); plt.close(fig)

    # Confusion matrix at chosen threshold
    cm = confusion_matrix(y_test, y_pred)
    fig2 = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix @ {thr:.2f}")
    plt.xticks([0,1], ["No", "Yes"]); plt.yticks([0,1], ["No", "Yes"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    fig2.savefig(reports_dir / "eval_confusion.png", bbox_inches="tight"); plt.close(fig2)

    # Gains / lift style quick plot
    order = np.argsort(-proba)
    y_sorted = y_test[order]
    cumulative = np.cumsum(y_sorted) / y_sorted.sum()
    perc = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    fig3 = plt.figure()
    plt.plot(perc, cumulative)
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("Population fraction (descending score)")
    plt.ylabel("Cumulative capture of churners")
    plt.title("Cumulative Gains")
    fig3.savefig(reports_dir / "eval_gains.png", bbox_inches="tight"); plt.close(fig3)

    out = {
        "chosen_model": model_name,
        "threshold": thr,
        "ap": ap,
        "roc_auc": roc,
        "confusion_matrix": cm.tolist(),
        "reports": {
            "pr_curve": "reports/eval_pr_curve.png",
            "confusion": "reports/eval_confusion.png",
            "gains": "reports/eval_gains.png",
        },
    }
    (artifacts_dir / "eval_summary.json").write_text(json.dumps(out, indent=2))
    typer.secho("Evaluation complete → artifacts/eval_summary.json", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
