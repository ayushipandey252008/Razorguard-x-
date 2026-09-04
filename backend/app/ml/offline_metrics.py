"""Read committed ULB offline metrics without importing the training stack."""

from __future__ import annotations

import json

from app.config import REPO_ROOT
from app.ml.registry import MODEL_REGISTRY

ULB_METRICS_PATH = REPO_ROOT / "ml" / "evaluation" / "ulb_metrics.json"
CALIBRATION_METRICS_PATH = REPO_ROOT / "ml" / "evaluation" / "calibration_metrics.json"


def load_offline_ulb_metrics() -> dict:
    if not ULB_METRICS_PATH.exists():
        return {
            "available": False,
            "label": "OFFLINE EVALUATION",
            "track": "REAL_DATASET",
            "dataset": "ULB Credit Card Fraud Detection",
            "reason": "ulb_metrics.json has not been generated. Run PYTHONPATH=. python ml/training/train_ulb.py",
        }
    data = json.loads(ULB_METRICS_PATH.read_text())
    if data.get("skipped"):
        return {
            "available": False,
            "label": "OFFLINE EVALUATION",
            "track": "REAL_DATASET",
            "dataset": "ULB Credit Card Fraud Detection",
            "reason": data.get("reason") or "evaluation skipped",
        }
    family = data.get("booster_family") or ""
    model_name = "XGBoost" if family == "xgboost" else ("HistGradientBoosting" if family == "hist_gbm" else family or "unknown")
    half = data.get("metrics_at_half") or {}
    return {
        "available": True,
        "label": "OFFLINE EVALUATION",
        "track": data.get("track", "REAL_DATASET"),
        "dataset": "ULB Credit Card Fraud Detection",
        "dataset_id": data.get("dataset_id"),
        "model": model_name,
        "model_version": data.get("model_version"),
        "pr_auc": data.get("pr_auc"),
        "roc_auc": data.get("roc_auc"),
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "f1": data.get("f1"),
        "false_positive_rate": data.get("false_positive_rate"),
        "false_negative_rate": data.get("false_negative_rate"),
        "confusion_matrix": data.get("confusion_matrix"),
        "operating_point": "probability >= 0.5 on chronological test (original prevalence)",
        "val_tuned_threshold": data.get("val_threshold"),
        "val_tuned_precision": (data.get("metrics_at_val_threshold") or {}).get("precision"),
        "val_tuned_recall": (data.get("metrics_at_val_threshold") or {}).get("recall"),
        "val_tuned_f1": (data.get("metrics_at_val_threshold") or {}).get("f1"),
        "fraud_prevalence": half.get("fraud_prevalence"),
        "n_fraud": half.get("n_fraud"),
        "n_legitimate": half.get("n_legitimate"),
        "trained_at": data.get("trained_at"),
        "note": "Offline ULB metrics. Not live synthetic transaction scores. Not Razorpay data.",
    }


def load_offline_calibration() -> dict:
    if not CALIBRATION_METRICS_PATH.exists():
        return {
            "available": False,
            "label": "PROTOTYPE CALIBRATION",
            "not_industry_standard": True,
            "track": "REAL_DATASET",
            "reason": "calibration_metrics.json has not been generated. Run PYTHONPATH=. python ml/training/calibrate_ulb.py",
        }
    data = json.loads(CALIBRATION_METRICS_PATH.read_text())
    selected = data.get("selected_method")
    val = data.get("validation") or {}
    raw = val.get("raw") or {}
    cal = val.get(selected) or {}
    proto = data.get("prototype_operating_thresholds") or {}
    test = data.get("test_evaluation") or {}
    costs = proto.get("cost_assumptions") or {}
    return {
        "available": True,
        "label": "PROTOTYPE CALIBRATION",
        "not_industry_standard": True,
        "track": data.get("track", "REAL_DATASET"),
        "booster_model_version": data.get("booster_model_version"),
        "calibrated_model_version": data.get("calibrated_model_version"),
        "selected_method": selected,
        "selection_justification": data.get("selection_justification"),
        "raw": {
            "brier": raw.get("brier"),
            "log_loss": raw.get("log_loss"),
            "ece_uniform_10": raw.get("ece_uniform_10"),
        },
        "calibrated": {
            "method": selected,
            "brier": cal.get("brier"),
            "log_loss": cal.get("log_loss"),
            "ece_uniform_10": cal.get("ece_uniform_10"),
        },
        "test_once": {
            "raw_brier": (test.get("raw_probability") or {}).get("brier"),
            "raw_log_loss": (test.get("raw_probability") or {}).get("log_loss"),
            "raw_pr_auc": (test.get("raw_probability") or {}).get("pr_auc"),
            "calibrated_brier": (test.get("calibrated_probability") or {}).get("brier"),
            "calibrated_log_loss": (test.get("calibrated_probability") or {}).get("log_loss"),
            "pr_auc": (test.get("metrics_at_0_5") or {}).get("pr_auc"),
            "roc_auc": (test.get("metrics_at_0_5") or {}).get("roc_auc"),
            "note": "Chronological test evaluated once after freezing validation choices.",
        },
        "operating_thresholds": {
            "approve_below": proto.get("approve_below"),
            "review_from": proto.get("review_from"),
            "review_to": proto.get("review_to"),
            "block_above": proto.get("block_above"),
        },
        "cost_scenario": proto.get("cost_scenario"),
        "cost_assumptions": costs,
        "source": proto.get("source"),
        "notes": proto.get("notes") or [],
        "signal_note": "Calibrated probability is not the product final_risk_score.",
    }


IEEE_MANIFEST_PATH = REPO_ROOT / "ml" / "evaluation" / "ieee_experiment_manifest.json"
IEEE_AUDIT_PATH = REPO_ROOT / "ml" / "evaluation" / "ieee_data_audit.json"
IEEE_LEAKAGE_PATH = REPO_ROOT / "ml" / "evaluation" / "ieee_leakage_report.json"
IEEE_CROSS_PATH = REPO_ROOT / "ml" / "evaluation" / "ieee_cross_dataset.json"

IEEE_DISCLAIMER = (
    "The IEEE-CIS experiment is an offline public-dataset evaluation. "
    "It does not represent production payment-fraud performance."
)


def load_offline_ieee() -> dict:
    """Committed IEEE-CIS offline reports. Never mixed with ULB or live scores."""
    if not IEEE_MANIFEST_PATH.exists():
        return {
            "available": False,
            "dataset_available": False,
            "label": "OFFLINE PUBLIC DATASET EVALUATION",
            "status": "OFFLINE CANDIDATE",
            "track": "IEEE_CIS_OFFLINE",
            "active_live_model": "xgb-iforest-v1-calibrated",
            "disclaimer": IEEE_DISCLAIMER,
            "reason": "ieee_experiment_manifest.json has not been generated. Place train_transaction.csv and train_identity.csv in IEEE_DATA_DIR, then run PYTHONPATH=backend:. python ml/training/train_ieee.py. This prototype does not download the dataset.",
        }
    data = json.loads(IEEE_MANIFEST_PATH.read_text())
    source = data.get("source")
    public = source == "IEEE_CIS_CSV"
    audit = json.loads(IEEE_AUDIT_PATH.read_text()) if IEEE_AUDIT_PATH.exists() else {}
    leakage = json.loads(IEEE_LEAKAGE_PATH.read_text()) if IEEE_LEAKAGE_PATH.exists() else {}
    cross = json.loads(IEEE_CROSS_PATH.read_text()) if IEEE_CROSS_PATH.exists() else data.get("cross_dataset")
    frozen = data.get("frozen_test_metrics") or {}
    if not public:
        frozen = {**frozen, "not_ieee_cis_public_result": True}
    return {
        "available": True,
        "dataset_available": bool(data.get("dataset_available")),
        "label": "OFFLINE PUBLIC DATASET EVALUATION",
        "status": "OFFLINE CANDIDATE",
        "track": "IEEE_CIS_OFFLINE",
        "source": source,
        "disclaimer": IEEE_DISCLAIMER,
        "active_live_model": data.get("active_live_model") or "xgb-iforest-v1-calibrated",
        "ieee_status": "OFFLINE CANDIDATE",
        "auto_activated": False,
        "audit": audit,
        "split": data.get("split"),
        "leakage": leakage,
        "feature_families": data.get("feature_configuration"),
        "experiments": data.get("experiments_test"),
        "calibration": data.get("calibration"),
        "thresholds": data.get("thresholds"),
        "graph_ablation": data.get("graph_ablation"),
        "frozen_test_metrics": frozen,
        "candidates": data.get("candidates"),
        "shap": data.get("shap"),
        "cross_dataset": cross,
        "runtime_seconds": data.get("runtime_seconds"),
        "join": data.get("join"),
        "note": "IEEE-CIS metrics are not live scores, not ULB metrics, and not production fraud accuracy.",
    }
