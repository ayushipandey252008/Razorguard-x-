"""Calibration robustness audit tests. Fixtures only — no 150MB CSV, no test-label fitting."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.ulb.adapter import ULBFraudDatasetAdapter
from ml.ulb.constants import PCA_COLUMNS
from ml.ulb.robustness import nested_holdout, pr_auc_tiebroken, staircase_diagnostics
from ml.ulb.robustness_audit import PROTECTED_ARTIFACTS, run_calibration_robustness_audit
from ml.ulb.split import chronological_split


ALLOWED = {
    "isotonic",
    "sigmoid/Platt",
    "raw probabilities with documented calibration limitation",
}


def make_chrono_both_classes(n: int = 300, n_fraud: int = 45, seed: int = 0):
    import pandas as pd

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


def test_robustness_helpers_have_no_test_parameters():
    from ml.ulb import robustness, robustness_audit

    for fn in (
        robustness.nested_holdout,
        robustness.kfold_oof,
        robustness.bootstrap_frozen_maps,
        robustness.train_fit_val_eval,
        robustness.recommend,
        robustness_audit.run_calibration_robustness_audit,
    ):
        params = inspect.signature(fn).parameters
        assert "y_test" not in params
        assert "raw_test" not in params
        assert "test" not in params


def test_nested_holdout_fits_only_on_fit_indices(monkeypatch):
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 0, 1] * 40)
    raw = np.clip(0.1 + 0.7 * y + rng.normal(0, 0.05, len(y)), 0.01, 0.99)
    y_test = np.ones(len(y), dtype=int)
    seen = []
    orig_iso = IsotonicRegression.fit
    orig_lr = LogisticRegression.fit

    def iso_fit(self, X, y_fit, *a, **k):
        seen.append(np.asarray(y_fit).copy())
        return orig_iso(self, X, y_fit, *a, **k)

    def lr_fit(self, X, y_fit, *a, **k):
        seen.append(np.asarray(y_fit).copy())
        return orig_lr(self, X, y_fit, *a, **k)

    monkeypatch.setattr(IsotonicRegression, "fit", iso_fit)
    monkeypatch.setattr(LogisticRegression, "fit", lr_fit)
    nested_holdout(raw, y, n_splits=3, test_size=0.3, seed=1)
    assert seen
    for captured in seen:
        assert captured.shape[0] < len(y)
        assert not np.array_equal(captured, y_test)
        assert captured.sum() >= 1


def test_staircase_tiebreak_recovers_ranking():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    stepped = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    diag = staircase_diagnostics(y, raw, stepped)
    assert diag["monotone_nondecreasing_in_raw"] is True
    assert diag["n_unique_calibrated"] == 2
    assert diag["pr_auc_calibrated"] <= diag["pr_auc_raw"] + 1e-12
    assert pr_auc_tiebroken(y, stepped, raw) >= diag["pr_auc_calibrated"] - 1e-12
    assert diag["drop_explained_by_ties"] is True


def test_audit_does_not_touch_protected_artifacts_or_test_labels(tmp_path, monkeypatch):
    df = make_chrono_both_classes()
    csv = tmp_path / "creditcard.csv"
    df.to_csv(csv, index=False)
    adapter = ULBFraudDatasetAdapter(csv_path=csv, model_dir=tmp_path / "model")
    adapter.eval_dir = tmp_path / "eval"
    adapter.run_full()
    eval_dir = tmp_path / "eval"
    planted = {"do_not_touch": True, "selected_method": "isotonic"}
    (eval_dir / "calibration_metrics.json").write_text(json.dumps(planted))
    protected = {name: (eval_dir / name).read_bytes() if (eval_dir / name).exists() else None for name in PROTECTED_ARTIFACTS}

    seen_y_lens = []
    orig_fit = LogisticRegression.fit

    def lr_fit(self, X, y, *a, **k):
        seen_y_lens.append(len(np.asarray(y)))
        return orig_fit(self, X, y, *a, **k)

    monkeypatch.setattr(LogisticRegression, "fit", lr_fit)
    _train, val, test = chronological_split(adapter._cleaned)
    test_n = len(test)
    val_n = len(val)
    payload = run_calibration_robustness_audit(
        adapter, n_boot=8, n_nested=4, n_folds=3, seed=0, write=True
    )
    assert payload["methodology"]["test_scored"] is False
    assert payload["methodology"]["test_used_for_fit"] is False
    assert payload["methodology"]["test_used_for_recommendation"] is False
    assert payload["recommendation"]["recommended_calibration_method"] in ALLOWED
    assert (eval_dir / "calibration_robustness.json").exists()
    assert (eval_dir / "calibration_robustness_report.md").exists()
    assert json.loads((eval_dir / "calibration_metrics.json").read_text()) == planted
    for name, before in protected.items():
        path = eval_dir / name
        if before is None:
            continue
        assert path.read_bytes() == before
    assert seen_y_lens
    # Fits are on train, full val, or val subsets — never the isolated test length unless it equals val.
    if test_n != val_n:
        assert test_n not in seen_y_lens
    assert payload.get("protected_artifacts_unchanged") is True
