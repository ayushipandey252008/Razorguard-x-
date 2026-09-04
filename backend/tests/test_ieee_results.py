"""ieee_results.json must track the latest pipeline outcome, not a stale STOPPED file."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.ieee.constants import LIVE_MODEL_VERSION
from app.ml.ieee.fixture import make_ieee_fixture
from app.ml.ieee.pipeline import run_ieee_pipeline
from app.ml.ieee.report import persist_ieee_results_from_eval_dir, write_ieee_results


def _stopped_doc() -> dict:
    return {
        "status": "STOPPED",
        "official_ieee_cis_result": False,
        "source": "MISSING",
        "dataset_available": False,
        "reason": "stale STOPPED fixture for regression",
        "active_live_model": LIVE_MODEL_VERSION,
    }


def test_successful_pipeline_overwrites_stale_stopped_report(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    write_ieee_results(_stopped_doc(), eval_dir)
    stale = json.loads((eval_dir / "ieee_results.json").read_text())
    assert stale["status"] == "STOPPED"

    txn, ident = make_ieee_fixture()
    run_ieee_pipeline(
        eval_dir=eval_dir,
        model_dir=tmp_path / "models",
        allow_fixture=True,
        transaction=txn,
        identity=ident,
        n_estimators=8,
        write_reports=True,
    )
    fresh = json.loads((eval_dir / "ieee_results.json").read_text())
    assert fresh["status"] != "STOPPED"
    assert fresh["source"] == "SYNTHETIC_FIXTURE_NOT_IEEE_CIS"
    assert fresh["official_ieee_cis_result"] is False
    assert isinstance(fresh["experiments_test"], list) and fresh["experiments_test"]
    assert fresh["active_live_model"] == LIVE_MODEL_VERSION
    assert fresh["ieee_status"] == "OFFLINE CANDIDATE"
    md = (eval_dir / "ieee_results.md").read_text()
    assert "OFFLINE PUBLIC DATASET EVALUATION" in md
    assert "does not represent production payment-fraud performance" in md.lower()
    assert "**Status:** STOPPED" not in md


def test_persist_from_manifest_replaces_stopped_without_training(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    write_ieee_results(_stopped_doc(), eval_dir)
    manifest = {
        "source": "IEEE_CIS_CSV",
        "dataset_available": True,
        "max_rows": None,
        "label": "OFFLINE PUBLIC DATASET EVALUATION",
        "disclaimer": "The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.",
        "split": {
            "train": {"n": 10, "fraud": 1, "prevalence": 0.1, "time_min": 1, "time_max": 5},
            "validation": {"n": 5, "fraud": 1, "prevalence": 0.2, "time_min": 6, "time_max": 8},
            "test": {"n": 5, "fraud": 1, "prevalence": 0.2, "time_min": 9, "time_max": 12},
            "constraints": {"max_train_lt_min_validation": True, "max_validation_lt_min_test": True},
        },
        "experiments_test": [
            {
                "Experiment": "A_transaction_only",
                "Features": "transaction",
                "PR-AUC": 0.4,
                "ROC-AUC": 0.7,
                "Precision": 0.1,
                "Recall": 0.2,
                "F1": 0.15,
                "FPR": 0.01,
            }
        ],
        "graph_ablation": {"without_graph": {"pr_auc": 0.3}, "with_graph": {"pr_auc": 0.35}, "improved": {"pr_auc": True}},
        "calibration": {"selection": {"selected_method": "sigmoid"}},
        "thresholds": {"approve_below": 0.1, "block_above": 0.9, "source": "validation_only"},
        "frozen_test_metrics": {"pr_auc": 0.41, "roc_auc": 0.72},
        "shap": {"available": False, "reason": "not run in this stub"},
        "candidates": [{"id": "ieee-xgb-combined-v1", "status": "CANDIDATE"}],
        "active_live_model": LIVE_MODEL_VERSION,
        "ieee_status": "OFFLINE CANDIDATE",
        "auto_activated": False,
        "runtime_seconds": 1.5,
        "leakage_all_passed": True,
        "join": {"identity_coverage": 0.5},
    }
    audit = {
        "dataset": "IEEE-CIS Fraud Detection",
        "transaction": {
            "n_rows": 20,
            "n_columns": 4,
            "target_distribution": {"positive": 3, "negative": 17, "prevalence": 0.15},
            "duplicate_transaction_ids": 0,
            "duplicate_rows": 0,
        },
        "identity": {"n_rows": 8, "n_columns": 3, "duplicate_transaction_ids": 0},
        "joined": {"n_rows": 20, "n_columns": 6, "numerical_columns": ["a"], "categorical_columns": ["b"]},
        "join": {"identity_coverage": 0.5},
    }
    (eval_dir / "ieee_experiment_manifest.json").write_text(json.dumps(manifest))
    (eval_dir / "ieee_data_audit.json").write_text(json.dumps(audit))
    (eval_dir / "ieee_leakage_report.json").write_text(
        json.dumps({"all_passed": True, "checks": [{"id": "target_leakage", "passed": True, "detail": "ok"}], "excluded_features": {"isFraud": "target"}})
    )
    payload = persist_ieee_results_from_eval_dir(eval_dir)
    assert payload["status"] == "COMPLETED"
    assert payload["official_ieee_cis_result"] is True
    assert payload["frozen_test_metrics"]["pr_auc"] == 0.41
    written = json.loads((eval_dir / "ieee_results.json").read_text())
    assert written["status"] != "STOPPED"
    assert "STOPPED" not in (eval_dir / "ieee_results.md").read_text().split("**Status:**")[1].splitlines()[0]


def test_missing_data_results_are_stopped(tmp_path: Path):
    payload = run_ieee_pipeline(data_dir=tmp_path, eval_dir=tmp_path / "eval", allow_fixture=False, write_reports=True)
    assert payload["dataset_available"] is False
    results = json.loads((tmp_path / "eval" / "ieee_results.json").read_text())
    assert results["status"] == "STOPPED"
    assert results["official_ieee_cis_result"] is False
