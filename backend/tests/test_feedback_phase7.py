"""Phase 7: analyst feedback, drift, candidate training, scenario evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import REPO_ROOT
from app.database import SessionLocal, init_db
from app.events.factory import connect_event_bus, reset_event_bus_for_tests
from app.main import app
from app.ml.drift import compute_drift_from_rows, population_stability_index, should_emit_alert
from app.ml.features import row_to_features
from app.ml.feedback_dataset import (
    EVAL_SCENARIO_PREFIX,
    assert_no_future_leakage,
    is_scenario_eval_tag,
    temporal_split,
    validate_dataset,
)
from app.ml.scenarios.generators import GROUND_TRUTH, generate_scenario
from app.ml.scenarios.metrics import overall_metrics, scenario_matrix
from app.ml.scenarios.runner import run_scenario_evaluation
from app.ml.train_feedback import LIVE_MODEL_VERSION, train_candidate
from app.models.feedback import AnalystFeedback
from app.models.outbox import OutboxEvent
from app.models.risk import RiskAssessment
from app.ml.predictor import model_service


ULB_FILES = [
    REPO_ROOT / "ml" / "evaluation" / "ulb_metrics.json",
    REPO_ROOT / "ml" / "evaluation" / "calibration_metrics.json",
    REPO_ROOT / "ml" / "models" / "version.txt",
]


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _login(client: TestClient, email="admin@razorguard.local", password="prototype-pass"):
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _labeled_record(i: int, y: int, hours: int) -> dict:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hours)
    raw = {
        "amount": 200 + i * 10,
        "account_age_days": 100,
        "failed_attempts": 0 if y == 0 else 4,
        "transaction_velocity": 1 if y == 0 else 8,
        "previous_transaction_count": 20,
        "previous_average_amount": 200,
        "current_device_known": y == 0,
        "current_location_known": y == 0,
        "timestamp": ts,
        "payment_method": "UPI",
        "merchant_category": "GROCERY",
    }
    return {
        "feedback_id": f"fb_{i}",
        "investigation_id": f"inv_{i}",
        "transaction_id": f"txn_{i}",
        "analyst_decision": "CONFIRM_FRAUD" if y else "CONFIRM_LEGITIMATE",
        "actual_outcome": "FRAUD" if y else "LEGITIMATE",
        "y": y,
        "model_prediction_decision": "BLOCK" if y else "APPROVE",
        "model_version": LIVE_MODEL_VERSION,
        "feedback_created_at": ts,
        "scenario_tag": "live",
        "features": row_to_features(raw),
        "raw": raw,
    }


def test_psi_low_moderate_high():
    import numpy as np

    ref = np.random.default_rng(0).normal(10, 1, 400)
    low = population_stability_index(ref, ref)
    high = np.random.default_rng(3).normal(25, 1.0, 400)
    drifted = population_stability_index(ref, high)
    assert low["status"] == "LOW"
    assert drifted["status"] == "HIGH"


def test_drift_insufficient_and_stable_windows():
    now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    few = [
        (
            SimpleNamespace(
                amount=100,
                transaction_velocity=1,
                timestamp=now,
                current_device_known=True,
                current_location_known=True,
            ),
            SimpleNamespace(ml_score=20, final_risk_score=20),
        )
    ]
    ins = compute_drift_from_rows(few, min_samples=20)
    assert ins["status"] == "insufficient"

    rows = []
    for i in range(40):
        ts = now + timedelta(minutes=i)
        rows.append(
            (
                SimpleNamespace(
                    amount=100,
                    transaction_velocity=1,
                    timestamp=ts,
                    current_device_known=True,
                    current_location_known=True,
                ),
                SimpleNamespace(ml_score=20, final_risk_score=20),
            )
        )
    stable = compute_drift_from_rows(rows, min_samples=20)
    assert stable["status"] in {"stable", "warning"}
    assert should_emit_alert("stable", None) is False
    assert should_emit_alert("drift", None) is True
    assert should_emit_alert("drift", datetime.now(timezone.utc), cooldown_seconds=3600) is False


def test_temporal_split_and_scenario_exclusion():
    records = [_labeled_record(i, i % 2, i) for i in range(10)]
    records[3]["scenario_tag"] = f"{EVAL_SCENARIO_PREFIX}stolen_account"
    assert is_scenario_eval_tag(records[3]["scenario_tag"])
    clean = [r for r in records if not is_scenario_eval_tag(r.get("scenario_tag"))]
    train, eval_rows = temporal_split(clean)
    assert_no_future_leakage(train, eval_rows)
    assert max(r["feedback_created_at"] for r in train) <= min(r["feedback_created_at"] for r in eval_rows)
    leaked = list(train)
    leaked[-1] = dict(leaked[-1], feedback_created_at=eval_rows[-1]["feedback_created_at"] + timedelta(hours=2))
    with pytest.raises(ValueError):
        assert_no_future_leakage(leaked, eval_rows)
    needs = dict(records[0], analyst_decision="NEEDS_REVIEW", y=1)
    report = validate_dataset([needs])
    assert report["ok"] is False


def test_candidate_training_does_not_touch_live(tmp_path):
    before = {p: _file_hash(p) for p in ULB_FILES}
    live = tmp_path / "live"
    live.mkdir()
    (live / "xgb_fraud.joblib").write_bytes(b"live")
    (live / "version.txt").write_text(LIVE_MODEL_VERSION)
    records = [_labeled_record(i, i % 2, i) for i in range(24)]
    result = train_candidate(
        records,
        artifact_root=tmp_path / "feedback",
        live_model_dir=live,
        current_predict_fn=lambda raw: {"ml_probability": 0.2},
        min_rows=12,
    )
    assert result["ok"] is True
    print("CANDIDATE_METRICS", json.dumps(result["candidate_metrics"]))
    assert result["status"] == "CANDIDATE"
    assert result["active_model_unchanged"] is True
    assert result["version"].startswith("xgb-feedback-")
    assert (tmp_path / "feedback" / result["version"] / "model.joblib").exists()
    assert (live / "xgb_fraud.joblib").read_bytes() == b"live"
    assert (live / "version.txt").read_text() == LIVE_MODEL_VERSION
    after = {p: _file_hash(p) for p in ULB_FILES}
    assert before == after


def test_scenario_generation_is_deterministic():
    a = generate_scenario("stolen_account", 4, seed=7)
    b = generate_scenario("stolen_account", 4, seed=7)
    assert [r["amount"] for r in a] == [r["amount"] for r in b]
    assert all(r["expected_fraud"] == GROUND_TRUTH["stolen_account"] for r in a)
    assert all(r["scenario_tag"].startswith(EVAL_SCENARIO_PREFIX) for r in a)
    legit = generate_scenario("normal_payment", 3, seed=1)
    assert all(r["expected_fraud"] == 0 for r in legit)
    rows = [
        {"scenario": "normal_payment", "expected_fraud": 0, "decision": "APPROVE"},
        {"scenario": "stolen_account", "expected_fraud": 1, "decision": "BLOCK"},
        {"scenario": "stolen_account", "expected_fraud": 1, "decision": "APPROVE"},
    ]
    metrics = overall_metrics(rows)
    assert metrics["n"] == 3
    assert "fraud_catch_rate" in metrics
    matrix = scenario_matrix(rows)
    assert {r["scenario"] for r in matrix} == {"normal_payment", "stolen_account"}


@pytest.mark.asyncio
async def test_graph_and_investigation_scenario_eval():
    await init_db()
    from app.ml.predictor import model_service as ms
    from app.services.bootstrap import ensure_seeded

    ms.load_or_train()
    await ensure_seeded()
    async with SessionLocal() as db:
        result = await run_scenario_evaluation(
            db,
            counts={"normal_payment": 2, "shared_device": 3, "stolen_account": 2},
            seed=3,
            run_investigations=True,
            max_investigations=1,
        )
    assert result["label"] == "SYNTHETIC SCENARIO EVALUATION"
    assert result["n"] == 7
    print("SCENARIO_METRICS", json.dumps(result["overall"]))
    print("SCENARIO_MATRIX", json.dumps(result["scenario_matrix"]))
    assert result["overall"]["n"] == 7
    graph_rows = [g for g in result["graph"] if g["scenario"] == "shared_device"]
    assert graph_rows
    assert graph_rows[0]["graph_backend"]
    if result["investigations"]:
        g = result["investigations"][0]["grounding"]
        assert "tool_trace_complete" in g
        assert g["has_limitations"] is True


def test_feedback_api_validation_persistence_duplicate_and_event():
    before = {p: _file_hash(p) for p in ULB_FILES}
    with TestClient(app) as client:
        headers = _login(client)
        viewer = _login(client, "viewer@razorguard.local", "prototype-pass")
        denied = client.post(
            "/api/v1/feedback",
            json={"investigation_id": "missing", "decision": "CONFIRM_FRAUD", "reason": "nope"},
            headers=viewer,
        )
        assert denied.status_code == 403
        missing = client.post(
            "/api/v1/feedback",
            json={"investigation_id": "not-real", "decision": "CONFIRM_FRAUD", "reason": "case missing here"},
            headers=headers,
        )
        assert missing.status_code == 404
        bad = client.post(
            "/api/v1/feedback",
            json={"investigation_id": "x", "decision": "APPROVE", "reason": "wrong enum"},
            headers=headers,
        )
        assert bad.status_code == 422
        sim = client.post(
            "/api/v1/simulation/run",
            json={"scenario": "stolen_account", "count": 1},
            headers=headers,
        )
        assert sim.status_code == 200, sim.text
        inv_id = sim.json()["transactions"][0]["investigation_id"]
        txn_id = sim.json()["transactions"][0]["transaction_id"]
        assert inv_id
        created = client.post(
            "/api/v1/feedback",
            json={"investigation_id": inv_id, "decision": "CONFIRM_FRAUD", "reason": "matches stolen-account pattern"},
            headers={**headers, "X-Correlation-ID": "corr-feedback-7"},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["analyst_decision"] == "CONFIRM_FRAUD"
        assert body["actual_outcome"] == "FRAUD"
        assert body["historical_risk_unchanged"] is True
        assert body["transaction_id"] == txn_id
        listed = client.get(f"/api/v1/feedback?investigation_id={inv_id}", headers=headers)
        assert listed.status_code == 200
        assert listed.json()["n"] >= 1
        dup = client.post(
            "/api/v1/feedback",
            json={"investigation_id": inv_id, "decision": "CONFIRM_LEGITIMATE", "reason": "second try should conflict"},
            headers=headers,
        )
        assert dup.status_code == 409
        status = client.get("/api/v1/ml/model-status", headers=headers)
        assert status.status_code == 200
        assert status.json()["active_model"]["version"]
        assert status.json()["auto_activation"] is False
        drift = client.get("/api/v1/ml/drift", headers=headers)
        assert drift.status_code == 200
        assert drift.json()["status"] in {"stable", "warning", "drift", "insufficient"}
        events = client.get("/api/v1/events/status", headers=headers).json()
        matched = [
            e
            for e in events["recent_events"]
            if e.get("correlation_id") == "corr-feedback-7" and e["event_type"] == "analyst-feedback-recorded"
        ]
        assert matched
        eval_run = client.post(
            "/api/v1/simulation/evaluate",
            json={"scenarios": ["normal_payment", "card_testing"], "count_per_scenario": 2, "seed": 11},
            headers=headers,
        )
        assert eval_run.status_code == 200, eval_run.text
        assert eval_run.json()["not_public_dataset"] is True
        assert eval_run.json()["n"] == 4
        viewer_train = client.post("/api/v1/ml/train-feedback", headers=viewer)
        assert viewer_train.status_code == 403
    after = {p: _file_hash(p) for p in ULB_FILES}
    assert before == after
    assert model_service.version == LIVE_MODEL_VERSION or model_service.version.startswith("xgb-iforest")


@pytest.mark.asyncio
async def test_feedback_does_not_rewrite_risk_row():
    await init_db()
    from sqlalchemy import select

    from app.ml.predictor import model_service as ms
    from app.services.bootstrap import ensure_seeded

    await connect_event_bus()
    ms.load_or_train()
    await ensure_seeded()
    with TestClient(app) as client:
        headers = _login(client)
        sim = client.post("/api/v1/simulation/run", json={"scenario": "card_testing", "count": 1}, headers=headers)
        txn_id = sim.json()["transactions"][0]["transaction_id"]
        inv_id = sim.json()["transactions"][0]["investigation_id"]
        async with SessionLocal() as db:
            before = (
                await db.execute(select(RiskAssessment).where(RiskAssessment.transaction_id == txn_id))
            ).scalar_one()
            before_decision, before_score = before.decision, before.final_risk_score
        created = client.post(
            "/api/v1/feedback",
            json={"investigation_id": inv_id, "decision": "NEEDS_REVIEW", "reason": "not enough to label fraud"},
            headers=headers,
        )
        assert created.status_code == 200, created.text
        assert created.json()["actual_outcome"] is None
        async with SessionLocal() as db:
            after = (
                await db.execute(select(RiskAssessment).where(RiskAssessment.transaction_id == txn_id))
            ).scalar_one()
            outbox = list(
                (
                    await db.execute(
                        select(OutboxEvent).where(OutboxEvent.event_type == "analyst-feedback-recorded")
                    )
                ).scalars().all()
            )
            labels = list(
                (
                    await db.execute(select(AnalystFeedback).where(AnalystFeedback.investigation_id == inv_id))
                ).scalars().all()
            )
        assert after.decision == before_decision
        assert after.final_risk_score == before_score
        assert outbox
        assert labels and labels[0].actual_outcome is None
    await reset_event_bus_for_tests()
