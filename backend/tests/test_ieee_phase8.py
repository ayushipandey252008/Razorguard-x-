"""Phase 8 IEEE-CIS offline track. Uses a tiny fixture — not IEEE-CIS public results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.ml.ieee.adapter import (
    audit_table,
    ieee_files_present,
    join_transaction_identity,
    setup_payload,
)
from app.ml.ieee.constants import (
    EVAL_DIR,
    JOIN_KEY,
    LIVE_MODEL_DIR,
    LIVE_MODEL_VERSION,
    TARGET_COLUMN,
    TIME_COLUMN,
    ULB_METRICS_PATH,
)
from app.ml.ieee.evaluate import fit_calibrators, select_calibration_method, select_three_way_thresholds
from app.ml.ieee.features import EXPERIMENTS, add_behavioral_features, add_transaction_timing, columns_for_families
from app.ml.ieee.fixture import make_ieee_fixture
from app.ml.ieee.graph_features import add_graph_features, graph_feature_uses_future
from app.ml.ieee.leakage import audit_leakage
from app.ml.ieee.pipeline import run_ieee_pipeline
from app.ml.ieee.preprocessing import IeeePreprocessor
from app.ml.ieee.split import chronological_split, verify_temporal_order
from app.ml.ieee.train import FORBIDDEN_WRITE_PATHS, _assert_not_live
from app.ml.registry import MODEL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tables():
    return make_ieee_fixture(n=90, n_identity=55, n_fraud=14, seed=0)


def test_setup_message_when_missing(tmp_path: Path):
    payload = setup_payload(tmp_path)
    assert payload["dataset_available"] is False
    assert "does not download" in payload["setup_message"].lower() or "not download" in payload["setup_message"]
    assert ieee_files_present(tmp_path) is False


def test_dataset_audit_and_join(tables):
    txn, ident = tables
    txn_audit = audit_table(txn, "transaction")
    assert txn_audit["n_rows"] == len(txn)
    assert txn_audit["duplicate_transaction_ids"] >= 1
    assert txn_audit["target_distribution"]["positive"] == 14
    joined, doc = join_transaction_identity(txn, ident)
    assert doc["join_key"] == JOIN_KEY
    assert doc["target_source"].startswith("transaction")
    assert doc["n_transaction_before"] == len(txn)
    assert TARGET_COLUMN in joined.columns
    assert "identity_present" in joined.columns
    assert 0 < doc["identity_coverage"] < 1
    assert doc["join_not_on_target"] is True


def test_temporal_split_strict(tables):
    txn, ident = tables
    joined, _ = join_transaction_identity(txn, ident)
    joined = joined.drop_duplicates(JOIN_KEY, keep="first")
    train, val, test = chronological_split(joined)
    verify_temporal_order(train, val, test)
    assert train[TIME_COLUMN].max() < val[TIME_COLUMN].min()
    assert val[TIME_COLUMN].max() < test[TIME_COLUMN].min()
    assert len(train) and len(val) and len(test)


def test_leakage_detection(tables):
    txn, ident = tables
    joined, _ = join_transaction_identity(txn, ident)
    joined = joined.drop_duplicates(JOIN_KEY, keep="first")
    joined = add_graph_features(add_behavioral_features(add_transaction_timing(joined)))
    train, val, test = chronological_split(joined)
    feats = columns_for_families(EXPERIMENTS["F_combined"], list(train.columns))
    report = audit_leakage(joined, train, val, test, "train_only", feats)
    assert report["all_passed"] is True
    assert JOIN_KEY not in feats
    assert TARGET_COLUMN not in feats
    assert TIME_COLUMN not in feats
    assert "TransactionID" in report["excluded_features"]

    leaked = audit_leakage(joined, train, val, test, "full_data", feats + [TARGET_COLUMN])
    assert leaked["all_passed"] is False


def test_preprocessing_train_only_and_missingness(tables):
    txn, ident = tables
    joined, _ = join_transaction_identity(txn, ident)
    joined = joined.drop_duplicates(JOIN_KEY, keep="first")
    train, val, test = chronological_split(joined)
    cols = ["TransactionAmt", "ProductCD", "card4", "id_01", "identity_present"]
    pre = IeeePreprocessor()
    X_train = pre.fit_transform(train, cols)
    X_val = pre.transform(val)
    assert pre.fitted
    assert X_train.shape[1] == X_val.shape[1]
    assert pre.to_meta()["target_encoding_used"] is False
    assert pre.to_meta()["zero_fill_all_missing"] is False
    unseen = val.copy()
    unseen["ProductCD"] = "UNSEEN_CODE"
    X_unseen = pre.transform(unseen)
    assert np.isfinite(X_unseen).all()


def test_feature_family_selection():
    cols = columns_for_families(["transaction", "card"], ["TransactionAmt", "ProductCD", "card1", "DeviceType"])
    assert "TransactionAmt" in cols
    assert "card1" in cols
    assert "DeviceType" not in cols


def test_graph_temporal_safety(tables):
    txn, ident = tables
    joined, _ = join_transaction_identity(txn, ident)
    joined = joined.drop_duplicates(JOIN_KEY, keep="first")
    featurized = add_graph_features(joined.sort_values(TIME_COLUMN).reset_index(drop=True))
    mid = len(featurized) // 2
    assert graph_feature_uses_future(featurized, mid) is False
    # Inject a future leak and ensure the checker fires.
    leaked = featurized.copy()
    leaked.loc[5, "graph_card_degree"] = 10_000
    assert graph_feature_uses_future(leaked, 5) is True
    # Later rows must not decrease prior fraud when earlier labels exist on shared card.
    shared = featurized[featurized["card1"] == 1001]
    if len(shared) >= 2:
        degrees = shared["graph_card_degree"].tolist()
        assert degrees == sorted(degrees) or degrees[0] <= degrees[-1]


def test_full_pipeline_fixture_artifacts(tmp_path: Path, tables):
    txn, ident = tables
    eval_dir = tmp_path / "eval"
    model_dir = tmp_path / "models"
    ulb_before = _sha(ULB_METRICS_PATH) if ULB_METRICS_PATH.exists() else None
    cal_path = EVAL_DIR / "calibration_metrics.json"
    cal_before = _sha(cal_path) if cal_path.exists() else None
    version_path = LIVE_MODEL_DIR / "version.txt"
    version_before = version_path.read_text() if version_path.exists() else None
    live_model = LIVE_MODEL_DIR / "xgb_fraud.joblib"
    live_before = _sha(live_model) if live_model.exists() else None

    payload = run_ieee_pipeline(
        eval_dir=eval_dir,
        model_dir=model_dir,
        allow_fixture=True,
        transaction=txn,
        identity=ident,
        n_estimators=8,
        write_reports=True,
    )
    assert payload["source"] == "SYNTHETIC_FIXTURE_NOT_IEEE_CIS"
    assert payload["dataset_available"] is False
    assert payload["active_live_model"] == LIVE_MODEL_VERSION
    assert payload["ieee_status"] == "OFFLINE CANDIDATE"
    assert payload["auto_activated"] is False
    assert payload["leakage_all_passed"] is True
    assert (eval_dir / "ieee_data_audit.json").exists()
    assert (eval_dir / "ieee_leakage_report.json").exists()
    assert (eval_dir / "ieee_experiment_manifest.json").exists()
    results = json.loads((eval_dir / "ieee_results.json").read_text())
    assert results["status"] != "STOPPED"
    assert results["official_ieee_cis_result"] is False
    assert "not available" not in str(results["experiments_test"][0]["PR-AUC"])
    md = (eval_dir / "ieee_results.md").read_text()
    assert "STOPPED" not in md.split("**Status:**")[1].split("\n")[0]
    assert "OFFLINE PUBLIC DATASET EVALUATION" in md
    assert (model_dir / "ieee-xgb-baseline-v1.joblib").exists()
    assert (model_dir / "ieee-xgb-combined-v1.json").exists()
    meta = json.loads((model_dir / "ieee-xgb-combined-v1.json").read_text())
    assert meta["status"] == "CANDIDATE"
    assert meta["deployed_to_live_pipeline"] is False
    table = payload["experiments_test"]
    assert {row["Experiment"] for row in table} >= {"A_transaction_only", "F_combined"}
    frozen = payload["frozen_test_metrics"]
    assert frozen.get("not_ieee_cis_public_result") is True
    cal = payload["calibration"]["selection"]
    assert cal["selected_method"] in {"raw", "sigmoid", "isotonic"}
    assert payload["thresholds"]["source"] == "validation_only"

    if ulb_before:
        assert _sha(ULB_METRICS_PATH) == ulb_before
    if cal_before:
        assert _sha(cal_path) == cal_before
    if version_before is not None:
        assert version_path.read_text() == version_before
        assert version_before.strip() == LIVE_MODEL_VERSION
    if live_before:
        assert _sha(live_model) == live_before
    assert not (LIVE_MODEL_DIR / "ieee-xgb-combined-v1.joblib").exists()


def test_candidate_isolation_refuses_live_paths():
    assert LIVE_MODEL_DIR.resolve() == (REPO_ROOT / "ml" / "models").resolve()
    for path in FORBIDDEN_WRITE_PATHS:
        with pytest.raises(Exception):
            _assert_not_live(path)


def test_registry_ieee_not_live():
    ids = {row["id"] for row in MODEL_REGISTRY}
    assert "xgb-iforest-v1-calibrated" in ids
    assert "ieee-xgb-baseline-v1" in ids
    live = next(r for r in MODEL_REGISTRY if r["id"] == "xgb-iforest-v1-calibrated")
    assert live["deployed_to_live_pipeline"] is True
    for ieee_id in ("ieee-xgb-baseline-v1", "ieee-xgb-combined-v1", "ieee-xgb-graph-v1"):
        row = next(r for r in MODEL_REGISTRY if r["id"] == ieee_id)
        assert row["deployed_to_live_pipeline"] is False
        assert row.get("status") == "CANDIDATE"


def test_threshold_and_calibration_fit_on_val_only():
    rng = np.random.default_rng(0)
    y_val = np.array([0] * 20 + [1] * 6)
    p_val = np.clip(np.where(y_val == 1, 0.7, 0.2) + rng.normal(0, 0.05, size=26), 0, 1)
    y_test = np.array([0] * 10 + [1] * 3)
    p_test = np.clip(np.where(y_test == 1, 0.65, 0.25) + rng.normal(0, 0.05, size=13), 0, 1)
    cals = fit_calibrators(p_val, y_val)
    assert cals.test_labels_used is False
    diag = {
        "raw": {"brier": 0.2, "log_loss": 0.5, "ece_uniform_10": 0.1, "n_unique_predictions": 20},
        "sigmoid": {"brier": 0.18, "log_loss": 0.4, "ece_uniform_10": 0.08, "n_unique_predictions": 20},
        "isotonic": {"brier": 0.01, "log_loss": 0.2, "ece_uniform_10": 0.01, "n_unique_predictions": 3},
    }
    selected = select_calibration_method(diag)
    assert selected["selected_method"] != "isotonic"
    thr = select_three_way_thresholds(y_val, p_val)
    assert thr["source"] == "validation_only"
    _ = cals.transform(p_test, selected["selected_method"])


def test_missing_dataset_pipeline_does_not_fake_metrics(tmp_path: Path):
    payload = run_ieee_pipeline(data_dir=tmp_path, eval_dir=tmp_path / "eval", allow_fixture=False, write_reports=True)
    assert payload["dataset_available"] is False
    assert "frozen_test_metrics" not in payload or payload.get("source") == "MISSING"
    assert payload.get("setup_message")
