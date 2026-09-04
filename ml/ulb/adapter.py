from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.ulb.constants import (
    DATASET_ID,
    DATASET_NAME,
    DOWNLOAD_URL,
    EVAL_DIR,
    EXPECTED_FILENAME,
    LEGACY_CSV,
    RANDOM_SEED,
    RAW_CSV,
    RAW_DIR,
    REPO_ROOT,
    TARGET_COLUMN,
    TRACK,
    ULB_MODEL_DIR,
)
from ml.ulb.features import ULBFeatureTransformer
from ml.ulb.metrics import best_f1_threshold, classification_metrics
from ml.ulb.model import make_ulb_classifier
from ml.ulb.preprocess import clean_ulb_frame, save_processed
from ml.ulb.reports import render_leakage_report, render_ulb_report, write_text
from ml.ulb.shap_eval import shap_summary
from ml.ulb.split import chronological_split, overlapping_row_count, split_summary, stratified_split
from ml.ulb.validate import validate_ulb_frame, write_validation_report


def _relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class ULBFraudDatasetAdapter:
    """Offline evaluation adapter for the ULB credit-card CSV.

    Does not participate in live transaction scoring. Does not write the
    synthetic product artifacts under `ml/models/xgb_fraud.joblib`.
    """

    def __init__(self, csv_path: Path | None = None, model_dir: Path | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else None
        self.model_dir = Path(model_dir) if model_dir else ULB_MODEL_DIR
        self.eval_dir = EVAL_DIR
        self._df: pd.DataFrame | None = None
        self._cleaned: pd.DataFrame | None = None
        self._clean_stats: dict = {}
        self._validation: dict = {}

    def resolve_csv(self) -> Path:
        if self.csv_path and self.csv_path.exists():
            return self.csv_path
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        if RAW_CSV.exists() and RAW_CSV.stat().st_size > 1_000_000:
            return RAW_CSV
        if LEGACY_CSV.exists() and LEGACY_CSV.stat().st_size > 1_000_000:
            shutil.copy2(LEGACY_CSV, RAW_CSV)
            return RAW_CSV
        raise FileNotFoundError(
            f"{EXPECTED_FILENAME} not found. Place it at {RAW_CSV} or run "
            "PYTHONPATH=. python ml/data/scripts/download_ulb.py"
        )

    def load(self) -> pd.DataFrame:
        path = self.resolve_csv()
        df = pd.read_csv(path)
        self.csv_path = path
        self._df = df
        return df

    def validate(self, df: pd.DataFrame | None = None) -> dict:
        frame = df if df is not None else self._df
        if frame is None:
            frame = self.load()
        report = validate_ulb_frame(frame, source=str(self.csv_path or "memory"))
        self._validation = report
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        write_validation_report(report, self.eval_dir / "ulb_validation.md")
        (self.eval_dir / "ulb_validation.json").write_text(json.dumps(report, indent=2))
        return report

    def preprocess(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        frame = df if df is not None else self._df
        if frame is None:
            frame = self.load()
        self.validate(frame)
        cleaned, stats = clean_ulb_frame(frame)
        save_processed(cleaned, stats)
        self._cleaned = cleaned
        self._clean_stats = stats
        return cleaned

    def train(self, df: pd.DataFrame | None = None) -> dict:
        cleaned = df if df is not None else self._cleaned
        if cleaned is None:
            cleaned = self.preprocess()
        train, val, test = chronological_split(cleaned)
        transformer = ULBFeatureTransformer()
        X_train = transformer.fit_transform(train)
        X_val = transformer.transform(val)
        X_test = transformer.transform(test)
        y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
        y_val = val[TARGET_COLUMN].to_numpy(dtype=int)
        y_test = test[TARGET_COLUMN].to_numpy(dtype=int)

        pos = max(int(y_train.sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        scale_pos_weight = neg / pos
        clf, family, version = make_ulb_classifier(scale_pos_weight, RANDOM_SEED)
        if family == "xgboost":
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            clf.fit(X_train, y_train)

        raw_val = clf.predict_proba(X_val)[:, 1]
        raw_test = clf.predict_proba(X_test)[:, 1]
        tuned = best_f1_threshold(y_val, raw_val)
        metrics_half = classification_metrics(y_test, raw_test, (raw_test >= 0.5).astype(int))
        metrics_tuned = classification_metrics(y_test, raw_test, (raw_test >= tuned["threshold"]).astype(int))

        shap = shap_summary(clf, X_test, transformer.feature_names_, y_test)

        strat_train, strat_val, strat_test = stratified_split(cleaned)
        strat_transformer = ULBFeatureTransformer()
        Xs_train = strat_transformer.fit_transform(strat_train)
        Xs_val = strat_transformer.transform(strat_val)
        Xs_test = strat_transformer.transform(strat_test)
        ys_train = strat_train[TARGET_COLUMN].to_numpy(dtype=int)
        ys_test = strat_test[TARGET_COLUMN].to_numpy(dtype=int)
        clf_s, family_s, _ = make_ulb_classifier(max(int((ys_train == 0).sum()), 1) / max(int(ys_train.sum()), 1), RANDOM_SEED)
        if family_s == "xgboost":
            clf_s.fit(Xs_train, ys_train, eval_set=[(strat_transformer.transform(strat_val), strat_val[TARGET_COLUMN].to_numpy(dtype=int))], verbose=False)
        else:
            clf_s.fit(Xs_train, ys_train)
        strat_prob = clf_s.predict_proba(Xs_test)[:, 1]
        strat_metrics = classification_metrics(ys_test, strat_prob, (strat_prob >= 0.5).astype(int))
        strat_summary = split_summary(strat_train, strat_val, strat_test, "stratified_random")

        leak = {
            "train_test_overlap": overlapping_row_count(train, test),
            "train_val_overlap": overlapping_row_count(train, val),
            "n_train_fit": int(len(train)),
            "resampling": "none — original training class distribution; scale_pos_weight only",
            "chronological_order_ok": split_summary(train, val, test, "chronological")["no_future_in_train"],
            "class_not_in_features": TARGET_COLUMN not in transformer.feature_names_,
            "imputer_fitted_on_train": True,
            "scaler_fitted_on_train": True,
        }

        trained_at = datetime.now(timezone.utc).isoformat()
        official_split = split_summary(train, val, test, "chronological")
        payload = {
            "track": TRACK,
            "skipped": False,
            "dataset_id": DATASET_ID,
            "source": DATASET_NAME,
            "download_url": DOWNLOAD_URL,
            "raw_path": _relpath(self.csv_path),
            "incompatible_with_product_pipeline": True,
            "label": "OFFLINE EVALUATION",
            "n_rows_cleaned": int(len(cleaned)),
            "duplicates_removed": int(self._clean_stats.get("exact_duplicates_removed", 0)),
            "cleaning_notes": self._clean_stats.get("notes") or [],
            "class_distribution": {
                "full_fraud": int(cleaned[TARGET_COLUMN].sum()),
                "full_legitimate": int((cleaned[TARGET_COLUMN] == 0).sum()),
                "full_prevalence": float(cleaned[TARGET_COLUMN].mean()),
            },
            "official_split": official_split,
            "split_comparison": {
                "chronological_pr_auc": metrics_half.get("pr_auc"),
                "stratified_pr_auc": strat_metrics.get("pr_auc"),
                "stratified_no_future_in_train": strat_summary["no_future_in_train"],
                "note": "Official artifact is chronological. Stratified numbers are a comparison only.",
            },
            "booster_family": family,
            "model_version": version,
            "scale_pos_weight": float(scale_pos_weight),
            "feature_names": transformer.feature_names_,
            "trained_at": trained_at,
            "val_threshold": tuned["threshold"],
            "metrics_at_half": metrics_half,
            "metrics_at_val_threshold": metrics_tuned,
            "shap": shap,
            "leakage": leak,
            "precision": metrics_half["precision"],
            "recall": metrics_half["recall"],
            "f1": metrics_half["f1"],
            "roc_auc": metrics_half["roc_auc"],
            "pr_auc": metrics_half["pr_auc"],
            "false_positive_rate": metrics_half["false_positive_rate"],
            "false_negative_rate": metrics_half["false_negative_rate"],
            "confusion_matrix": metrics_half["confusion_matrix"],
            "n_samples": metrics_half["n_samples"],
            "positive_rate": metrics_half["fraud_prevalence"],
        }
        self._save_artifacts(clf, transformer, payload)
        return payload

    def evaluate(self, payload: dict | None = None) -> dict:
        if payload is None:
            metrics_path = self.eval_dir / "ulb_metrics.json"
            if not metrics_path.exists():
                payload = self.train()
            else:
                payload = json.loads(metrics_path.read_text())
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        (self.eval_dir / "ulb_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
        (self.eval_dir / "ulb_shap.json").write_text(json.dumps(payload.get("shap") or {}, indent=2, default=str))
        write_text(self.eval_dir / "ulb_report.md", render_ulb_report(payload))
        write_text(self.eval_dir / "data_leakage_report.md", render_leakage_report(payload))
        # Backward-compatible alias used by an older test; still REAL_DATASET and still not the product model.
        public = {
            k: payload[k]
            for k in (
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "false_positive_rate",
                "false_negative_rate",
                "confusion_matrix",
                "n_samples",
                "positive_rate",
                "track",
                "skipped",
                "source",
                "incompatible_with_product_pipeline",
            )
            if k in payload
        }
        public["model_version"] = payload.get("model_version")
        public["label"] = "OFFLINE EVALUATION"
        public["note"] = (
            "REAL_DATASET / ULB offline evaluation. Not the live synthetic product model."
        )
        (self.eval_dir / "public_metrics.json").write_text(json.dumps(public, indent=2))
        return payload

    def run_full(self) -> dict:
        self.load()
        self.validate()
        self.preprocess()
        payload = self.train()
        return self.evaluate(payload)

    def load_model(self):
        path = self.model_dir / "model.joblib"
        pre = self.model_dir / "preprocessor.joblib"
        if not path.exists() or not pre.exists():
            raise FileNotFoundError(f"ULB artifacts missing in {self.model_dir}")
        return joblib.load(path), joblib.load(pre)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        model, transformer = self.load_model()
        X = transformer.transform(df)
        proba = model.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise RuntimeError(f"Unexpected predict_proba shape {proba.shape}")
        return proba

    def _save_artifacts(self, model, transformer: ULBFeatureTransformer, payload: dict) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_dir / "model.joblib")
        joblib.dump(transformer, self.model_dir / "preprocessor.joblib")
        meta = {
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET_NAME,
            "track": TRACK,
            "model_version": payload["model_version"],
            "booster_family": payload["booster_family"],
            "trained_at": payload["trained_at"],
            "feature_names": payload["feature_names"],
            "scale_pos_weight": payload["scale_pos_weight"],
            "split": "chronological_70_15_15",
            "imbalance_strategy": payload["leakage"]["resampling"],
            "metrics": {
                "pr_auc": payload["pr_auc"],
                "roc_auc": payload["roc_auc"],
                "precision": payload["precision"],
                "recall": payload["recall"],
                "f1": payload["f1"],
            },
            "synthetic_product_model_untouched": True,
            "raw_data_not_stored": True,
        }
        (self.model_dir / "feature_names.json").write_text(json.dumps(payload["feature_names"], indent=2))
        (self.model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        (self.model_dir / "metrics.json").write_text(json.dumps(payload["metrics_at_half"], indent=2))
        (self.model_dir / "version.txt").write_text(str(payload["model_version"]))


def load_offline_ulb_metrics() -> dict:
    path = EVAL_DIR / "ulb_metrics.json"
    if not path.exists():
        return {
            "available": False,
            "label": "OFFLINE EVALUATION",
            "track": TRACK,
            "reason": "ml/evaluation/ulb_metrics.json not generated yet. Run PYTHONPATH=. python ml/training/train_ulb.py",
        }
    data = json.loads(path.read_text())
    family = data.get("booster_family") or ""
    model_name = "XGBoost" if family == "xgboost" else ("HistGradientBoosting" if family == "hist_gbm" else family)
    return {
        "available": True,
        "label": "OFFLINE EVALUATION",
        "track": data.get("track", TRACK),
        "dataset": DATASET_NAME,
        "dataset_id": data.get("dataset_id", DATASET_ID),
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
        "fraud_prevalence": (data.get("metrics_at_half") or {}).get("fraud_prevalence"),
        "n_fraud": (data.get("metrics_at_half") or {}).get("n_fraud"),
        "n_legitimate": (data.get("metrics_at_half") or {}).get("n_legitimate"),
        "trained_at": data.get("trained_at"),
        "note": "Offline ULB metrics. Not live synthetic transaction scores. Not Razorpay data.",
    }
