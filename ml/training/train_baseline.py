"""Train the baseline XGBoost + Isolation Forest models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.train import train_and_save  # noqa: E402

if __name__ == "__main__":
    metrics = train_and_save(n_samples=8000)
    print("Trained", metrics["model_version"])
    print("PR-AUC", round(metrics["pr_auc"], 3), "F1", round(metrics["f1"], 3), "ROC-AUC", round(metrics["roc_auc"], 3))
