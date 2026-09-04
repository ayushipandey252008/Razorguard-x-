"""IEEE-CIS calibration robustness audit tests. Synthetic scores only — no IEEE CSVs, no fixtures-as-results."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.ml.ieee.calibration_robustness import (
    DECISIONS,
    kfold_oof,
    nested_holdout,
    pr_auc_tiebroken,
    recommend,
    staircase_diagnostics,
    temporal_pretest_holdout,
    train_fit_val_eval,
)
from app.ml.ieee.calibration_robustness_audit import (
    PROTECTED_REPO_PATHS,
    build_payload_from_scores,
    run_calibration_robustness_audit,
)
from app.ml.ieee.constants import LIVE_MODEL_VERSION


def _separated_scores(n: int = 800, n_fraud: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[np.linspace(0, n - 1, n_fraud, dtype=int)] = 1
    raw = np.clip(np.where(y == 1, rng.normal(0.72, 0.12, n), rng.normal(0.18, 0.10, n)), 0.01, 0.99)
    times = np.arange(n, dtype=int)
    return raw, y, times


def test_robustness_helpers_have_no_test_parameters():
    from app.ml.ieee import calibration_robustness, calibration_robustness_audit

    for fn in (
        calibration_robustness.nested_holdout,
        calibration_robustness.kfold_oof,
        calibration_robustness.bootstrap_frozen_maps,
        calibration_robustness.train_fit_val_eval,
        calibration_robustness.temporal_pretest_holdout,
        calibration_robustness.recommend,
        calibration_robustness_audit.build_payload_from_scores,
        calibration_robustness_audit.run_calibration_robustness_audit,
        calibration_robustness_audit.score_frozen_combined_pretest,
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


def test_temporal_pretest_holdout_is_strictly_earlier():
    raw, y, times = _separated_scores()
    hold = temporal_pretest_holdout(raw, y, times, cut_frac=0.5, seed=0)
    assert hold.get("skipped") is not True
    assert hold["fit_time_max"] < hold["eval_time_min"]
    assert hold["strict_temporal"] is True
    assert hold["lowest_brier_method"] in {"raw", "sigmoid", "isotonic"}


def test_recommend_returns_allowed_decision():
    raw, y, times = _separated_scores()
    n_train = 500
    payload = {
        "nested_holdout_validation": nested_holdout(raw[n_train:], y[n_train:], n_splits=8, test_size=0.3, seed=0),
        "kfold_oof_validation": kfold_oof(raw[n_train:], y[n_train:], n_splits=5, seed=0),
        "train_fit_val_eval": train_fit_val_eval(raw[:n_train], y[:n_train], raw[n_train:], y[n_train:]),
        "staircase_in_sample_validation": staircase_diagnostics(
            y[n_train:], raw[n_train:], np.round(raw[n_train:], 1)
        ),
        "in_sample_validation": {
            "raw": {"brier": 0.04, "n_unique_predictions": 300, "n_positive": int(y[n_train:].sum())},
            "sigmoid": {"brier": 0.03, "n_unique_predictions": 300},
            "isotonic": {"brier": 0.02, "n_unique_predictions": 40},
        },
        "temporal_pretest_holdouts": [
            temporal_pretest_holdout(raw, y, times, cut_frac=0.5, seed=0),
            temporal_pretest_holdout(raw, y, times, cut_frac=0.75, seed=0),
        ],
    }
    rec = recommend(payload)
    assert rec["decision"] in DECISIONS
    assert rec["test_used_for_decision"] is False


def test_recommend_prefer_sigmoid_when_isotonic_collapses():
    payload = {
        "nested_holdout_validation": {
            "n_splits_used": 20,
            "test_size": 0.3,
            "methods": {
                "raw": {"brier": {"mean": 0.04}, "pr_auc": {"mean": 0.40}, "n_unique": {"mean": 5000}},
                "sigmoid": {"brier": {"mean": 0.03}, "pr_auc": {"mean": 0.40}, "n_unique": {"mean": 5000}},
                "isotonic": {"brier": {"mean": 0.035}, "pr_auc": {"mean": 0.30}, "n_unique": {"mean": 8}},
            },
            "win_rates": {"isotonic_brier_lt_sigmoid": 0.2},
            "paired_differences": {
                "brier_isotonic_minus_sigmoid": {"mean": 0.005, "p025": 0.001, "p975": 0.009},
                "pr_auc_isotonic_minus_raw": {"mean": -0.08, "p025": -0.10, "p975": -0.05},
            },
        },
        "kfold_oof_validation": {
            "pooled_oof": {
                "raw": {"brier": 0.04, "pr_auc": 0.40, "n_unique_predictions": 5000},
                "sigmoid": {"brier": 0.03, "pr_auc": 0.40, "n_unique_predictions": 5000},
                "isotonic": {"brier": 0.036, "pr_auc": 0.31, "n_unique_predictions": 9},
            }
        },
        "train_fit_val_eval": {
            "train_positives": 200,
            "methods": {
                "raw": {"brier": 0.04, "pr_auc": 0.41},
                "sigmoid": {"brier": 0.031, "pr_auc": 0.41},
                "isotonic": {"brier": 0.037, "pr_auc": 0.32},
            },
        },
        "staircase_in_sample_validation": {
            "n_unique_raw": 8000,
            "n_unique_calibrated": 8,
            "unique_ratio": 8 / 8000,
            "monotone_nondecreasing_in_raw": True,
            "pr_auc_drop": 0.08,
            "pr_auc_drop_after_tiebreak": 0.0,
            "drop_explained_by_ties": True,
        },
        "in_sample_validation": {
            "raw": {"brier": 0.04, "n_unique_predictions": 8000, "n_positive": 80},
            "sigmoid": {"brier": 0.03, "n_unique_predictions": 8000},
            "isotonic": {"brier": 0.02, "n_unique_predictions": 8},
        },
        "temporal_pretest_holdouts": [{"lowest_brier_method": "sigmoid", "skipped": False}],
    }
    rec = recommend(payload)
    assert rec["decision"] == "PREFER_SIGMOID"


def test_recommend_inconclusive_when_brier_and_ranking_disagree():
    payload = {
        "nested_holdout_validation": {
            "n_splits_used": 20,
            "test_size": 0.3,
            "methods": {
                "raw": {"brier": {"mean": 0.04}, "pr_auc": {"mean": 0.42}, "n_unique": {"mean": 8000}},
                "sigmoid": {"brier": {"mean": 0.032}, "pr_auc": {"mean": 0.42}, "n_unique": {"mean": 8000}},
                "isotonic": {"brier": {"mean": 0.028}, "pr_auc": {"mean": 0.35}, "n_unique": {"mean": 40}},
            },
            "win_rates": {"isotonic_brier_lt_sigmoid": 1.0},
            "paired_differences": {
                "brier_isotonic_minus_sigmoid": {"mean": -0.004, "p025": -0.006, "p975": -0.002},
                "pr_auc_isotonic_minus_raw": {"mean": -0.07, "p025": -0.09, "p975": -0.04},
            },
        },
        "kfold_oof_validation": {
            "pooled_oof": {
                "raw": {"brier": 0.04, "pr_auc": 0.42, "n_unique_predictions": 8000},
                "sigmoid": {"brier": 0.032, "pr_auc": 0.42, "n_unique_predictions": 8000},
                "isotonic": {"brier": 0.027, "pr_auc": 0.34, "n_unique_predictions": 45},
            }
        },
        "train_fit_val_eval": {
            "train_positives": 200,
            "methods": {
                "raw": {"brier": 0.04, "pr_auc": 0.43},
                "sigmoid": {"brier": 0.033, "pr_auc": 0.43},
                "isotonic": {"brier": 0.029, "pr_auc": 0.36},
            },
        },
        "staircase_in_sample_validation": {
            "n_unique_raw": 8000,
            "n_unique_calibrated": 45,
            "unique_ratio": 45 / 8000,
            "monotone_nondecreasing_in_raw": True,
            "pr_auc_drop": 0.07,
            "pr_auc_drop_after_tiebreak": 0.0,
            "drop_explained_by_ties": True,
        },
        "in_sample_validation": {
            "raw": {"brier": 0.04, "n_unique_predictions": 8000, "n_positive": 80},
            "sigmoid": {"brier": 0.03, "n_unique_predictions": 8000},
            "isotonic": {"brier": 0.02, "n_unique_predictions": 45},
        },
        "temporal_pretest_holdouts": [{"lowest_brier_method": "isotonic", "skipped": False}],
    }
    rec = recommend(payload)
    assert rec["decision"] == "INCONCLUSIVE_KEEP_CURRENT"


def test_audit_from_scores_does_not_touch_protected_artifacts(tmp_path):
    raw, y, times = _separated_scores(n=400, n_fraud=60, seed=1)
    n_train = 250
    official = {
        "train": {"n": n_train, "fraud": int(y[:n_train].sum()), "prevalence": float(y[:n_train].mean())},
        "validation": {"n": 150, "fraud": int(y[n_train:].sum()), "prevalence": float(y[n_train:].mean())},
        "test": {"n": 150, "fraud": 10, "prevalence": 0.06},
    }
    before = {p: (p.read_bytes() if p.exists() else None) for p in PROTECTED_REPO_PATHS}
    live_version = Path(PROTECTED_REPO_PATHS[1])
    payload = run_calibration_robustness_audit(
        eval_dir=tmp_path,
        n_boot=6,
        n_nested=4,
        n_folds=3,
        seed=0,
        write=True,
        scores={
            "raw_train": raw[:n_train],
            "y_train": y[:n_train],
            "raw_val": raw[n_train:],
            "y_val": y[n_train:],
            "times_pre": times,
            "official_split": official,
            "split_matches_phase9_manifest": False,
        },
    )
    assert payload["recommendation"]["decision"] in DECISIONS
    assert payload["methodology"]["test_scored"] is False
    assert payload["methodology"]["xgboost_refit"] is False
    assert payload["live_model_unchanged"] == LIVE_MODEL_VERSION
    assert (tmp_path / "ieee_calibration_robustness.json").exists()
    assert (tmp_path / "ieee_calibration_robustness_report.md").exists()
    doc = json.loads((tmp_path / "ieee_calibration_robustness.json").read_text())
    assert "historical_phase9_frozen_test" in doc
    assert doc["historical_phase9_frozen_test"]["quoted_not_recomputed"] is True
    for path, prev in before.items():
        now = path.read_bytes() if path.exists() else None
        assert now == prev
    assert payload.get("integrity", {}).get("protected_artifacts_unchanged") is True
    assert live_version.exists()
    assert live_version.read_text().strip() == LIVE_MODEL_VERSION


def test_build_payload_has_no_test_scores_in_methodology():
    raw, y, times = _separated_scores(n=300, n_fraud=50, seed=2)
    n_train = 180
    official = {
        "train": {"n": n_train, "fraud": int(y[:n_train].sum()), "prevalence": float(y[:n_train].mean())},
        "validation": {"n": 120, "fraud": int(y[n_train:].sum()), "prevalence": float(y[n_train:].mean())},
        "test": {"n": 120, "fraud": 8, "prevalence": 0.06},
    }
    payload = build_payload_from_scores(
        raw_train=raw[:n_train],
        y_train=y[:n_train],
        raw_val=raw[n_train:],
        y_val=y[n_train:],
        times_pre=times,
        official_split=official,
        n_boot=5,
        n_nested=3,
        n_folds=3,
        seed=1,
    )
    assert payload["methodology"]["test_used_for_fit"] is False
    assert payload["methodology"]["test_used_for_recommendation"] is False
    assert payload["recommendation"]["test_used_for_decision"] is False
    assert "raw_test" not in payload
    assert "y_test" not in payload
