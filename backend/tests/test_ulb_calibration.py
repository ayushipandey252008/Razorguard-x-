"""ULB probability / threshold calibration. Fixtures only — not the 150MB CSV."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.ulb.adapter import ULBFraudDatasetAdapter
from ml.ulb.calibration import clip_proba, fit_calibrators, select_calibration_method
from ml.ulb.calibration_pipeline import run_ulb_calibration
from ml.ulb.constants import PCA_COLUMNS
from ml.ulb.decisions import assert_thresholds_ordered, pick_prototype_operating_point, three_way_decision
from ml.ulb.split import chronological_split


def make_chrono_both_classes(n: int = 300, n_fraud: int = 45, seed: int = 0) -> pd.DataFrame:
    """Spread fraud across time so train/val/test each contain positives."""
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 50_000, n))
    data = {"Time": time}
    for col in PCA_COLUMNS:
        data[col] = rng.normal(0, 1, n)
    data["Amount"] = rng.uniform(1, 500, n)
    y = np.zeros(n, dtype=int)
    y[np.linspace(0, n - 1, n_fraud, dtype=int)] = 1
    data["Class"] = y
    return pd.DataFrame(data)


def test_fit_calibrators_signature_has_no_test_argument():
    params = list(inspect.signature(fit_calibrators).parameters)
    assert params == ["raw_val", "y_val"]


def test_select_method_signature_is_validation_diagnostics_only():
    params = list(inspect.signature(select_calibration_method).parameters)
    assert params == ["val_diagnostics"]


def test_threshold_opt_signature_has_no_test_argument():
    params = inspect.signature(pick_prototype_operating_point).parameters
    assert "y_test" not in params
    assert "raw_test" not in params


def test_calibrators_fit_only_on_validation_labels(monkeypatch):
    rng = np.random.default_rng(1)
    y_val = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1] * 10)
    y_test = np.ones(len(y_val), dtype=int)
    raw_val = np.clip(0.08 + 0.75 * y_val + rng.normal(0, 0.05, len(y_val)), 0.01, 0.99)
    seen = []

    orig_lr = LogisticRegression.fit
    orig_iso = IsotonicRegression.fit

    def lr_fit(self, X, y, *args, **kwargs):
        seen.append(("sigmoid", np.asarray(y).copy()))
        return orig_lr(self, X, y, *args, **kwargs)

    def iso_fit(self, X, y, *args, **kwargs):
        seen.append(("isotonic", np.asarray(y).copy()))
        return orig_iso(self, X, y, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", lr_fit)
    monkeypatch.setattr(IsotonicRegression, "fit", iso_fit)
    fitted = fit_calibrators(raw_val, y_val)
    assert fitted.test_labels_used is False
    assert len(seen) == 2
    for _, y in seen:
        assert y.shape == y_val.shape
        assert np.array_equal(y, y_val)
        assert not np.array_equal(y, y_test)
        assert int(y.sum()) == int(y_val.sum())


def test_calibrated_probabilities_stay_in_unit_interval():
    y_val = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1] * 8)
    raw_val = np.linspace(0.0, 1.0, len(y_val))
    fitted = fit_calibrators(raw_val, y_val)
    raw_out = np.array([-0.2, 0.3, 1.7])
    for method in ("raw", "sigmoid", "isotonic"):
        p = fitted.transform(raw_out, method)
        assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all((clip_proba(raw_out) >= 0.0) & (clip_proba(raw_out) <= 1.0))


def test_three_way_decision_bands_and_ordering():
    p = np.array([0.05, 0.30, 0.40, 0.80, 0.95])
    d = three_way_decision(p, 0.30, 0.80)
    assert list(d) == ["APPROVE", "REVIEW", "REVIEW", "BLOCK", "BLOCK"]
    with pytest.raises(ValueError, match="T_REVIEW < T_BLOCK"):
        three_way_decision(p, 0.9, 0.4)
    with pytest.raises(ValueError):
        assert_thresholds_ordered(0.5, 0.5)


def test_select_calibration_uses_brier_not_wishful_naming():
    val = {
        "raw": {"brier": 0.20, "log_loss": 1.0, "ece_uniform_10": 0.2, "ece_quantile_10": 0.2, "n_unique_predictions": 10},
        "sigmoid": {"brier": 0.10, "log_loss": 0.5, "ece_uniform_10": 0.1, "ece_quantile_10": 0.1, "n_unique_predictions": 10},
        "isotonic": {"brier": 0.11, "log_loss": 0.4, "ece_uniform_10": 0.05, "ece_quantile_10": 0.05, "n_unique_predictions": 10},
    }
    choice = select_calibration_method(val)
    assert choice["selected_method"] == "sigmoid"


def test_pipeline_does_not_fit_or_tune_on_test(tmp_path):
    df = make_chrono_both_classes()
    csv = tmp_path / "creditcard.csv"
    df.to_csv(csv, index=False)
    adapter = ULBFraudDatasetAdapter(csv_path=csv, model_dir=tmp_path / "model")
    adapter.eval_dir = tmp_path / "eval"
    adapter.preprocess()
    train, val, test = chronological_split(adapter._cleaned)
    test_hash = int(pd.util.hash_pandas_object(test, index=False).sum())
    val_n = len(val)
    val_pos = int(val["Class"].sum())
    test_n = len(test)
    test_pos = int(test["Class"].sum())
    assert val_pos > 0 and test_pos > 0

    payload = run_ulb_calibration(adapter, train_if_missing=True)
    meth = payload["methodology"]
    assert meth["test_used_for_calibrator_fit"] is False
    assert meth["test_used_for_method_selection"] is False
    assert meth["test_used_for_threshold_selection"] is False
    assert meth["calibrator_fit_n"] == val_n
    assert meth["calibrator_fit_n_positive"] == val_pos
    assert meth["test_n"] == test_n
    assert payload["official_split"]["test"]["fraud"] == test_pos

    _, _, test_after = chronological_split(adapter._cleaned)
    assert int(pd.util.hash_pandas_object(test_after, index=False).sum()) == test_hash
    assert (
        payload["prototype_operating_thresholds"]["approve_below"]
        < payload["prototype_operating_thresholds"]["block_above"]
    )
    assert payload["test_evaluation"]["calibrated_probability"]["within_unit_interval"] is True
    assert payload["booster_model_version"] in {"ulb-xgb-v1", "ulb-hgb-v1"}
    assert payload["calibrated_model_version"] == f"{payload['booster_model_version']}-calibrated"
    assert payload["synthetic_model_untouched"] is True
    assert payload["label"] == "PROTOTYPE CALIBRATION"
    assert (tmp_path / "eval" / "calibration_report.md").exists()
    assert (tmp_path / "eval" / "calibration_metrics.json").exists()
    assert not (tmp_path / "model" / "xgb_fraud.joblib").exists()
    assert not (tmp_path / "model" / "iforest.joblib").exists()


def test_offline_api_includes_prototype_calibration_payload():
    from app.ml.offline_metrics import load_offline_calibration
    from app.ml.registry import MODEL_REGISTRY

    data = load_offline_calibration()
    assert data["label"] == "PROTOTYPE CALIBRATION"
    assert data["not_industry_standard"] is True
    ids = {row["id"] for row in MODEL_REGISTRY}
    assert "xgb-iforest-v1-calibrated" in ids
    assert "ulb-xgb-v1" in ids
    assert "ulb-xgb-v1-calibrated" in ids
    live = next(r for r in MODEL_REGISTRY if r["id"] == "xgb-iforest-v1-calibrated")
    ulb = next(r for r in MODEL_REGISTRY if r["id"] == "ulb-xgb-v1-calibrated")
    assert live["deployed_to_live_pipeline"] is True
    assert ulb["deployed_to_live_pipeline"] is False
    assert live["final_output_is_probability"] is False
