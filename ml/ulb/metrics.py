from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if y_true.sum() else 0.0,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "n_legitimate": int((y_true == 0).sum()),
        "fraud_prevalence": float(y_true.mean()) if len(y_true) else None,
    }


def best_f1_threshold(y_true, y_prob) -> dict:
    """Pick a probability cutoff on a held-out split. Not applied to that same split for reporting."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    grid = np.unique(np.concatenate(([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9], np.quantile(y_prob, [0.5, 0.9, 0.95, 0.99]))))
    best = {"threshold": 0.5, "f1": -1.0}
    for t in grid:
        pred = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1}
    return best
