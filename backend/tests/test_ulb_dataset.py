"""ULB dataset adapter tests. Use in-memory fixtures — not the 150MB CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.ulb.adapter import ULBFraudDatasetAdapter
from ml.ulb.constants import PCA_COLUMNS, REQUIRED_COLUMNS
from ml.ulb.errors import DatasetValidationError
from ml.ulb.features import ULBFeatureTransformer
from ml.ulb.metrics import classification_metrics
from ml.ulb.preprocess import clean_ulb_frame
from ml.ulb.split import chronological_split, overlapping_row_count, stratified_split
from ml.ulb.validate import validate_ulb_frame


def make_ulb_frame(n: int = 120, n_fraud: int = 12, seed: int = 0, duplicates: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 50_000, n))
    data = {"Time": time}
    for col in PCA_COLUMNS:
        data[col] = rng.normal(0, 1, n)
    data["Amount"] = rng.uniform(0, 500, n)
    y = np.zeros(n, dtype=int)
    y[:n_fraud] = 1
    rng.shuffle(y)
    data["Class"] = y
    df = pd.DataFrame(data)
    if duplicates:
        df = pd.concat([df, df.iloc[:duplicates]], ignore_index=True)
    return df


def test_schema_validation_passes_on_complete_frame():
    report = validate_ulb_frame(make_ulb_frame())
    assert report["ok"] is True
    assert report["class_counts"]["fraud"] == 12
    assert report["missing_columns"] == []


def test_missing_columns_fail_loudly():
    df = make_ulb_frame().drop(columns=["V1", "Amount"])
    with pytest.raises(DatasetValidationError, match="Missing required columns"):
        validate_ulb_frame(df)


def test_invalid_labels_fail_loudly():
    df = make_ulb_frame()
    df.loc[0, "Class"] = 2
    with pytest.raises(DatasetValidationError, match="0 or 1"):
        validate_ulb_frame(df)


def test_duplicate_detection_counts_exact_copies():
    df = make_ulb_frame(duplicates=5)
    report = validate_ulb_frame(df)
    assert report["duplicate_rows"] == 5
    cleaned, stats = clean_ulb_frame(df)
    assert stats["exact_duplicates_removed"] == 5
    assert len(cleaned) == len(df) - 5


def test_preprocess_does_not_drop_fraud_outliers():
    df = make_ulb_frame()
    df.loc[df.index[df["Class"] == 1][0], "Amount"] = 1_000_000
    cleaned, stats = clean_ulb_frame(df)
    assert int(cleaned["Class"].sum()) == int(df["Class"].sum())
    assert float(cleaned["Amount"].max()) >= 1_000_000
    assert stats["exact_duplicates_removed"] == 0


def test_chronological_split_has_no_future_in_train():
    df = make_ulb_frame(n=200, n_fraud=20)
    train, val, test = chronological_split(df)
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()
    assert overlapping_row_count(train, test) == 0


def test_stratified_split_can_place_later_time_in_train():
    df = make_ulb_frame(n=200, n_fraud=20, seed=1)
    train, val, test = stratified_split(df)
    # Later rows can land in train — the failure mode chronological split avoids.
    assert train["Time"].max() > test["Time"].min()
    assert overlapping_row_count(train, test) == 0


def test_scaler_fitted_only_on_train():
    df = make_ulb_frame(n=200, n_fraud=20)
    train, val, test = chronological_split(df)
    transformer = ULBFeatureTransformer()
    transformer.fit(train)
    time_idx = transformer.feature_names_.index("Time")
    # StandardScaler mean for Time is stored in scaler.mean_[0] because SCALE_COLUMNS starts with Time
    assert transformer.scaler.mean_[0] == pytest.approx(float(train["Time"].mean()), rel=1e-6)
    full_mean = float(pd.concat([train, val, test])["Time"].mean())
    assert transformer.scaler.mean_[0] != pytest.approx(full_mean, rel=1e-4) or len(train) > 0
    X_test = transformer.transform(test)
    assert X_test.shape == (len(test), len(transformer.feature_names_))
    assert "Class" not in transformer.feature_names_


def test_train_test_leakage_protection_on_tiny_adapter(tmp_path):
    df = make_ulb_frame(n=180, n_fraud=18, duplicates=3)
    csv = tmp_path / "creditcard.csv"
    df.to_csv(csv, index=False)
    adapter = ULBFraudDatasetAdapter(csv_path=csv, model_dir=tmp_path / "model")
    adapter.eval_dir = tmp_path / "eval"
    payload = adapter.run_full()
    assert payload["leakage"]["train_test_overlap"] == 0
    assert payload["leakage"]["resampling"].startswith("none")
    assert payload["track"] == "REAL_DATASET"
    assert payload["model_version"] in {"ulb-xgb-v1", "ulb-hgb-v1"}
    assert "pr_auc" in payload
    assert payload["incompatible_with_product_pipeline"] is True


def test_model_loading_and_prediction_shape(tmp_path):
    df = make_ulb_frame(n=150, n_fraud=15)
    csv = tmp_path / "creditcard.csv"
    df.to_csv(csv, index=False)
    adapter = ULBFraudDatasetAdapter(csv_path=csv, model_dir=tmp_path / "model")
    adapter.eval_dir = tmp_path / "eval"
    adapter.run_full()
    proba = adapter.predict_proba(df.head(10))
    assert proba.shape == (10, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    model, pre = adapter.load_model()
    assert hasattr(model, "predict_proba")
    meta = json.loads((tmp_path / "model" / "metadata.json").read_text())
    assert meta["raw_data_not_stored"] is True
    assert meta["synthetic_product_model_untouched"] is True


def test_evaluation_metrics_keys():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.9])
    m = classification_metrics(y, p, (p >= 0.5).astype(int))
    for key in (
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_positive_rate",
        "false_negative_rate",
        "confusion_matrix",
        "n_fraud",
        "n_legitimate",
        "fraud_prevalence",
    ):
        assert key in m
    assert "accuracy" not in m
    assert REQUIRED_COLUMNS[-1] == "Class"


def test_offline_reader_is_labeled_real_dataset():
    sys.path.insert(0, str(ROOT / "backend"))
    from app.ml.offline_metrics import load_offline_ulb_metrics

    data = load_offline_ulb_metrics()
    assert data["label"] == "OFFLINE EVALUATION"
    assert data.get("track") == "REAL_DATASET"
    assert "dataset" in data
