"""Train IEEE-CIS offline candidates. Never writes live or ULB artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from app.ml.booster import make_classifier
from app.ml.ieee.constants import (
    BASELINE_VERSION,
    COMBINED_VERSION,
    GRAPH_VERSION,
    IEEE_MODEL_DIR,
    LIVE_MODEL_DIR,
    LIVE_MODEL_VERSION,
    RANDOM_SEED,
    TARGET_COLUMN,
    ULB_MODEL_DIR,
)
from app.ml.ieee.errors import IeeeDatasetError
from app.ml.ieee.evaluate import classification_metrics
from app.ml.ieee.features import EXPERIMENTS, FAMILY_MAP, columns_for_families
from app.ml.ieee.preprocessing import IeeePreprocessor

FORBIDDEN_WRITE_PATHS = [
    LIVE_MODEL_DIR / "xgb_fraud.joblib",
    LIVE_MODEL_DIR / "version.txt",
    LIVE_MODEL_DIR / "calibrator.joblib",
    LIVE_MODEL_DIR / "iforest.joblib",
    LIVE_MODEL_DIR / "metrics.json",
]


def _assert_not_live(path: Path) -> None:
    resolved = path.resolve()
    for forbidden in FORBIDDEN_WRITE_PATHS:
        if resolved == forbidden.resolve():
            raise IeeeDatasetError(f"Refusing to overwrite live artifact {forbidden}")
    ulb_root = ULB_MODEL_DIR.resolve()
    try:
        in_ulb = resolved.is_relative_to(ulb_root)
    except (ValueError, AttributeError):
        in_ulb = str(resolved).startswith(str(ulb_root))
    if in_ulb:
        raise IeeeDatasetError(f"Refusing to write into ULB model dir: {path}")


def predict_proba(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(X)[:, 1], dtype=float)
    scores = np.asarray(clf.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def train_experiment(
    name: str,
    families: list[str],
    train,
    val,
    test,
    seed: int = RANDOM_SEED,
    n_estimators: int | None = None,
) -> dict:
    available = list(train.columns)
    feature_columns = columns_for_families(families, available)
    pre = IeeePreprocessor()
    X_train = pre.fit_transform(train, feature_columns)
    X_val = pre.transform(val)
    X_test = pre.transform(test)
    y_train = train[TARGET_COLUMN].astype(int).to_numpy()
    y_val = val[TARGET_COLUMN].astype(int).to_numpy()
    y_test = test[TARGET_COLUMN].astype(int).to_numpy()
    pos = max(int(y_train.sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    spw = neg / pos
    clf, family, _ = make_classifier(scale_pos_weight=spw, seed=seed)
    if n_estimators is not None and hasattr(clf, "set_params"):
        params = {}
        if hasattr(clf, "n_estimators"):
            params["n_estimators"] = n_estimators
        if hasattr(clf, "max_iter"):
            params["max_iter"] = n_estimators
        if params:
            clf.set_params(**params)
    clf.fit(X_train, y_train)
    p_val = predict_proba(clf, X_val)
    p_test = predict_proba(clf, X_test)
    val_metrics = classification_metrics(y_val, p_val, threshold=0.5)
    test_metrics = classification_metrics(y_test, p_test, threshold=0.5)
    return {
        "experiment": name,
        "families": families,
        "feature_columns": feature_columns,
        "n_raw_features": len(feature_columns),
        "n_model_features": len(pre.feature_names),
        "preprocessor": pre,
        "clf": clf,
        "booster_family": family,
        "scale_pos_weight": spw,
        "class_imbalance_method": "scale_pos_weight / class_weight — SMOTE not used",
        "p_val": p_val,
        "p_test": p_test,
        "y_val": y_val,
        "y_test": y_test,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "preprocessing_meta": pre.to_meta(),
    }


def experiment_row(result: dict, split: str = "test") -> dict:
    m = result["val_metrics"] if split == "val" else result["test_metrics"]
    return {
        "Experiment": result["experiment"],
        "Features": "+".join(result["families"]),
        "n_features": result["n_model_features"],
        "PR-AUC": m.get("pr_auc"),
        "ROC-AUC": m.get("roc_auc"),
        "Precision": m.get("precision"),
        "Recall": m.get("recall"),
        "F1": m.get("f1"),
        "FPR": m.get("false_positive_rate"),
    }


def save_candidate(result: dict, version: str, extra: dict, model_dir: Path | None = None) -> dict:
    dest = Path(model_dir or IEEE_MODEL_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    joblib_path = dest / f"{version}.joblib"
    json_path = dest / f"{version}.json"
    _assert_not_live(joblib_path)
    _assert_not_live(json_path)
    artifact = {
        "clf": result["clf"],
        "preprocessor": result["preprocessor"],
        "feature_columns": result["feature_columns"],
        "families": result["families"],
        "version": version,
        "status": "CANDIDATE",
        "track": "IEEE_CIS_OFFLINE",
    }
    joblib.dump(artifact, joblib_path)
    meta = {
        "id": version,
        "status": "CANDIDATE",
        "deployed_to_live_pipeline": False,
        "track": "IEEE_CIS_OFFLINE",
        "dataset": extra.get("dataset"),
        "dataset_source": extra.get("source"),
        "dataset_version": extra.get("dataset_version"),
        "data_dir": extra.get("data_dir"),
        "feature_families": result["families"],
        "feature_count": result["n_model_features"],
        "train_rows": extra.get("train_rows"),
        "train_fraud_rows": extra.get("train_fraud_rows"),
        "validation_rows": extra.get("validation_rows"),
        "test_rows": extra.get("test_rows"),
        "metrics": result["test_metrics"],
        "val_metrics": result["val_metrics"],
        "artifact_path": str(joblib_path),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "live_model_unchanged": LIVE_MODEL_VERSION,
        "note": "OFFLINE CANDIDATE. Not routed to live scoring. Not production fraud accuracy.",
    }
    json_path.write_text(json.dumps(meta, indent=2, default=str))
    return meta


def version_for_experiment(name: str) -> str:
    if name.startswith("A_"):
        return BASELINE_VERSION
    if name.startswith("E_") or "graph" in name:
        return GRAPH_VERSION
    if name.startswith("F_"):
        return COMBINED_VERSION
    return f"ieee-xgb-{name.split('_')[0].lower()}-v1"


def family_lookup() -> dict[str, str]:
    out = {}
    for fam, cols in FAMILY_MAP.items():
        for c in cols:
            out[c] = fam
            out[f"{c}_freq"] = fam
            out[f"{c}_is_missing"] = fam
    return out
