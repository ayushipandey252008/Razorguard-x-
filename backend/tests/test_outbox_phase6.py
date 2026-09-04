"""Phase 6 transactional outbox: atomic enqueue, worker, retries, crash recovery."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.database import SessionLocal, init_db
from app.events.factory import connect_event_bus, reset_event_bus_for_tests
from app.events.outbox import (
    backoff_delay_seconds,
    claim_pending,
    insert_outbox,
    mark_retry_or_failed,
    release_stale_processing,
)
from app.events.outbox_worker import drain_outbox_batch
from app.events.schemas import AlertCreated, TransactionCreated
from app.main import app
from app.models.eventing import FailedEvent
from app.models.outbox import (
    OUTBOX_FAILED,
    OUTBOX_PENDING,
    OUTBOX_PROCESSING,
    OUTBOX_PUBLISHED,
    OutboxEvent,
)
from app.models.risk import RiskAssessment
from app.schemas.common import TransactionCreate
from app.services.pipeline import process_transaction
from app.utils.ids import new_id
from app.utils.logging import Timer


def _login(client: TestClient, email="admin@razorguard.local", password="prototype-pass"):
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _txn_payload(**extra):
    body = {
        "user_id": extra.get("user_id", "usr_ananya"),
        "merchant_id": extra.get("merchant_id", "m_groc_01"),
        "amount": extra.get("amount", 620),
        "currency": "INR",
        "device_id": extra.get("device_id", "dev_ananya_phone"),
        "ip_address": extra.get("ip_address", "10.10.1.14"),
        "location": extra.get("location", "Bengaluru"),
        "payment_method": "UPI",
        "failed_attempts": extra.get("failed_attempts", 0),
        "transaction_velocity": extra.get("transaction_velocity", 1),
        "current_device_known": extra.get("current_device_known", True),
        "current_location_known": extra.get("current_location_known", True),
        "scenario_tag": extra.get("scenario_tag", "normal"),
    }
    return TransactionCreate(**body)


@pytest.mark.asyncio
async def test_outbox_model_and_unique_event_id():
    await init_db()
    event = TransactionCreated(
        correlation_id="corr-model",
        transaction_id="txn_model",
        payload={"amount": 1, "currency": "INR"},
    )
    async with SessionLocal() as db:
        row = await insert_outbox(db, event, aggregate_type="transaction", aggregate_id="txn_model")
        await db.commit()
        assert row.status == OUTBOX_PENDING
        assert row.event_id == event.event_id
        assert row.schema_version == "1"
        assert row.attempts == 0
        assert row.payload.get("event_type") == "transaction-created"
        assert "payment_identifier" not in (row.payload or {})
    async with SessionLocal() as db:
        with pytest.raises(Exception):
            await insert_outbox(db, event, aggregate_type="transaction", aggregate_id="txn_model")
            await db.commit()


@pytest.mark.asyncio
async def test_rollback_removes_outbox_event():
    await init_db()
    event = TransactionCreated(
        event_id=new_id(),
        correlation_id="corr-rb",
        transaction_id="txn_rb",
        payload={"amount": 1, "currency": "INR"},
    )
    async with SessionLocal() as db:
        await insert_outbox(db, event, aggregate_type="transaction", aggregate_id="txn_rb")
        await db.flush()
        await db.rollback()
    async with SessionLocal() as db:
        n = (
            await db.execute(select(func.count()).select_from(OutboxEvent).where(OutboxEvent.event_id == event.event_id))
        ).scalar()
        assert int(n or 0) == 0


@pytest.mark.asyncio
async def test_atomic_risk_and_outbox_then_crash_before_publish():
    """Primary reliability demo: commit without publish, then worker recovers."""
    await init_db()
    from app.ml.predictor import model_service
    from app.services.bootstrap import ensure_seeded

    await connect_event_bus()
    model_service.load_or_train()
    await ensure_seeded()
    async with SessionLocal() as db:
        timer = Timer()
        result = await process_transaction(db, _txn_payload(device_id="dev_outbox_crash"), drain_outbox=False)
        insert_ms = timer.ms()
    txn_id = result["transaction"].transaction_id
    assert result["risk"].decision in {"APPROVE", "REVIEW", "BLOCK"}
    async with SessionLocal() as db:
        risk = (
            await db.execute(select(RiskAssessment).where(RiskAssessment.transaction_id == txn_id))
        ).scalar_one_or_none()
        rows = list(
            (
                await db.execute(select(OutboxEvent).where(OutboxEvent.aggregate_id == txn_id))
            ).scalars()
        )
    assert risk is not None
    assert rows
    assert all(r.status == OUTBOX_PENDING for r in rows)
    assert all(r.correlation_id for r in rows)
    print(f"PROTOTYPE_METRIC outbox_insert_and_score_ms={insert_ms}")
    print(f"PROTOTYPE_METRIC outbox_enqueue_ms={result['stage_latency_ms'].get('outbox_enqueue')}")

    # Shared pytest SQLite can already hold pending rows from earlier cases.
    # One default batch may not include this transaction — drain until it does.
    published_total = 0
    worker_ms = 0.0
    for _ in range(25):
        async with SessionLocal() as db:
            rows = list(
                (
                    await db.execute(select(OutboxEvent).where(OutboxEvent.aggregate_id == txn_id))
                ).scalars()
            )
        if rows and all(r.status == OUTBOX_PUBLISHED for r in rows):
            break
        drain = await drain_outbox_batch(limit=100)
        published_total += int(drain.get("published") or 0)
        worker_ms += float(drain.get("latency_ms") or 0)
    print(f"PROTOTYPE_METRIC outbox_worker_ms={worker_ms}")
    assert published_total >= 1
    async with SessionLocal() as db:
        rows = list(
            (
                await db.execute(select(OutboxEvent).where(OutboxEvent.aggregate_id == txn_id))
            ).scalars()
        )
    assert rows
    assert all(r.status == OUTBOX_PUBLISHED for r in rows)
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_worker_retry_backoff_and_max_attempts(monkeypatch):
    await init_db()
    await connect_event_bus()
    event = TransactionCreated(
        event_id=new_id(),
        correlation_id="corr-retry",
        transaction_id="txn_retry",
        payload={"amount": 1, "currency": "INR"},
    )
    async with SessionLocal() as db:
        row = await insert_outbox(db, event, aggregate_type="transaction", aggregate_id="txn_retry")
        await db.commit()
        oid = row.id

    async def boom(event, **kwargs):
        return {"ok": False, "event_bus": "inprocess", "error": "simulated_transport"}

    monkeypatch.setattr("app.events.outbox_worker.get_event_bus", lambda: SimpleNamespace(publish=boom, name="inprocess"))
    monkeypatch.setattr(
        "app.events.outbox.backoff_delay_seconds",
        lambda attempts, base=None: 30.0,
    )
    first = await mark_retry_or_failed(oid, "simulated_transport", 0)
    assert first == OUTBOX_PENDING
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.id == oid))).scalar_one()
        assert row.attempts == 1
        assert row.last_error
        assert row.status == OUTBOX_PENDING
        assert row.available_at > row.created_at

    monkeypatch.setattr(
        "app.events.outbox.get_settings",
        lambda: SimpleNamespace(outbox_max_attempts=2, outbox_retry_backoff_seconds=1.0),
    )
    final = await mark_retry_or_failed(oid, "still_down", 1)
    assert final == OUTBOX_FAILED
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.id == oid))).scalar_one()
        assert row.status == OUTBOX_FAILED
        assert row.attempts == 2
    await reset_event_bus_for_tests()


def test_retry_backoff_grows():
    assert backoff_delay_seconds(1, base=1) == 1
    assert backoff_delay_seconds(2, base=1) == 2
    assert backoff_delay_seconds(3, base=1) == 4
    assert backoff_delay_seconds(10, base=1) == 60


@pytest.mark.asyncio
async def test_stale_processing_recovery():
    await init_db()
    event = TransactionCreated(
        event_id=new_id(),
        correlation_id="corr-stale",
        transaction_id="txn_stale",
        payload={"amount": 1, "currency": "INR"},
    )
    async with SessionLocal() as db:
        await insert_outbox(db, event, aggregate_type="transaction", aggregate_id="txn_stale")
        await db.commit()
    async with SessionLocal() as db:
        claimed = await claim_pending(db, limit=50)
        await db.commit()
    assert any(c["event_id"] == event.event_id for c in claimed)
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.event_id == event.event_id))).scalar_one()
        assert row.status == OUTBOX_PROCESSING
        recovered = await release_stale_processing(db, stale_seconds=0)
        await db.commit()
    assert recovered >= 1
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.event_id == event.event_id))).scalar_one()
        assert row.status == OUTBOX_PENDING


@pytest.mark.asyncio
async def test_malformed_outbox_goes_to_failed_and_dlq():
    await init_db()
    await connect_event_bus()
    oid = new_id()
    eid = new_id()
    async with SessionLocal() as db:
        db.add(
            OutboxEvent(
                id=oid,
                event_id=eid,
                event_type="transaction-created",
                schema_version="1",
                correlation_id="corr-bad",
                aggregate_type="transaction",
                aggregate_id="txn_bad",
                payload={"not": "a valid event"},
                status=OUTBOX_PENDING,
            )
        )
        await db.commit()
    result = await drain_outbox_batch()
    assert result["malformed"] >= 1
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.id == oid))).scalar_one()
        assert row.status == OUTBOX_FAILED
        failed = list((await db.execute(select(FailedEvent).where(FailedEvent.event_id == eid))).scalars())
        assert failed
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_kafka_unavailable_leaves_outbox_pending(monkeypatch):
    await init_db()
    event = TransactionCreated(
        event_id=new_id(),
        correlation_id="corr-kdown",
        transaction_id="txn_kdown",
        payload={"amount": 1, "currency": "INR"},
    )
    async with SessionLocal() as db:
        await insert_outbox(db, event, aggregate_type="transaction", aggregate_id="txn_kdown")
        await db.commit()
    monkeypatch.setattr(
        "app.events.outbox_worker.event_bus_status",
        lambda: {"configured": "kafka", "kafka_connected": False, "active": "inprocess", "fallback": True},
    )
    result = await drain_outbox_batch()
    assert result.get("skipped") is True
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.event_id == event.event_id))).scalar_one()
        assert row.status == OUTBOX_PENDING


@pytest.mark.asyncio
async def test_inprocess_worker_publishes_and_is_idempotent():
    await init_db()
    await connect_event_bus()
    event = AlertCreated(
        event_id=new_id(),
        correlation_id="corr-idem",
        transaction_id=f"txn_idem_{new_id()[:8]}",
        payload={"alert_id": new_id(), "decision": "BLOCK", "risk_level": "BLOCK", "kind": "block"},
    )
    async with SessionLocal() as db:
        await insert_outbox(db, event, aggregate_type="transaction", aggregate_id=event.transaction_id)
        await db.commit()
    first = await drain_outbox_batch()
    second = await drain_outbox_batch()
    assert first["published"] >= 1
    assert second["claimed"] == 0
    from app.events.consumers import process_event

    again = await process_event(event)
    assert again["duplicate"] is True
    await reset_event_bus_for_tests()


def test_outbox_status_api_and_admin_retry_authorization():
    with TestClient(app) as client:
        headers = _login(client)
        r = client.get("/api/v1/events/outbox/status", headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "pending" in body
        assert "processing" in body
        assert "published" in body
        assert "failed" in body
        assert body.get("durable_event_delivery") is True
        blob = json.dumps(body).lower()
        assert "password" not in blob
        assert "payload" not in body
        status = client.get("/api/v1/events/status", headers=headers)
        assert status.status_code == 200
        assert "outbox" in status.json()
        viewer = _login(client, email="viewer@razorguard.local", password="prototype-pass")
        denied = client.post("/api/v1/events/outbox/not-a-real-id/retry", headers=viewer)
        assert denied.status_code == 403
        missing = client.post("/api/v1/events/outbox/not-a-real-id/retry", headers=headers)
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_admin_retry_failed_and_409_if_not_failed():
    await init_db()
    await connect_event_bus()
    event = TransactionCreated(
        event_id=new_id(),
        correlation_id="corr-admin-retry",
        transaction_id="txn_admin_retry",
        payload={"amount": 1, "currency": "INR"},
    )
    async with SessionLocal() as db:
        row = await insert_outbox(db, event, aggregate_type="transaction", aggregate_id="txn_admin_retry")
        row.status = OUTBOX_FAILED
        row.last_error = "exhausted"
        await db.commit()
    with TestClient(app) as client:
        headers = _login(client)
        pending_retry = client.post(f"/api/v1/events/outbox/{event.event_id}/retry", headers=headers)
        assert pending_retry.status_code == 200, pending_retry.text
        assert pending_retry.json()["event_id"] == event.event_id
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.event_id == event.event_id))).scalar_one()
        assert row.status == OUTBOX_PUBLISHED
        row.status = OUTBOX_PUBLISHED
        await db.commit()
    with TestClient(app) as client:
        headers = _login(client)
        conflict = client.post(f"/api/v1/events/outbox/{event.event_id}/retry", headers=headers)
        assert conflict.status_code == 409
    await reset_event_bus_for_tests()


def test_http_transaction_still_sync_and_outbox_drains(monkeypatch):
    with TestClient(app) as client:
        headers = {**_login(client), "X-Correlation-ID": "corr-http-outbox"}
        timer = Timer()
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_ananya",
                "merchant_id": "m_groc_01",
                "amount": 620,
                "currency": "INR",
                "device_id": "dev_ananya_phone",
                "ip_address": "10.10.1.14",
                "location": "Bengaluru",
                "payment_method": "UPI",
                "failed_attempts": 0,
                "transaction_velocity": 1,
                "current_device_known": True,
                "current_location_known": True,
                "scenario_tag": "normal",
            },
            headers=headers,
        )
        api_ms = timer.ms()
        assert r.status_code == 200, r.text
        assert r.json()["risk"]["decision"] == "APPROVE"
        print(f"PROTOTYPE_METRIC transaction_api_with_outbox_ms={api_ms}")
        events = client.get("/api/v1/events/status", headers=headers).json()
        matched = [e for e in events["recent_events"] if e.get("correlation_id") == "corr-http-outbox"]
        assert {e["event_type"] for e in matched} >= {"transaction-created", "risk-scored"}
        outbox = client.get("/api/v1/events/outbox/status", headers=headers).json()
        assert outbox["published"] >= 1


def test_crash_simulation_via_skipped_drain_then_admin_drain(monkeypatch):
    async def noop():
        return None

    monkeypatch.setattr("app.services.pipeline.drain_after_commit", noop)
    with TestClient(app) as client:
        headers = _login(client)
        before = client.get("/api/v1/events/outbox/status", headers=headers).json()
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_ananya",
                "merchant_id": "m_groc_01",
                "amount": 620,
                "device_id": "dev_crash_http",
                "ip_address": "10.10.1.14",
                "location": "Bengaluru",
                "failed_attempts": 0,
                "transaction_velocity": 1,
                "current_device_known": True,
                "current_location_known": True,
                "scenario_tag": "normal",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        txn_id = r.json()["transaction"]["transaction_id"]
        assert r.json()["risk"]["decision"]
        mid = client.get("/api/v1/events/outbox/status", headers=headers).json()
        assert mid["pending"] > before["pending"]
        drain = client.post("/api/v1/events/outbox/drain", headers=headers)
        assert drain.status_code == 200, drain.text
        assert drain.json()["published"] >= 1
        after = client.get("/api/v1/events/outbox/status", headers=headers).json()
        assert after["published"] >= mid["published"] + 1
        viewer = _login(client, "viewer@razorguard.local", "prototype-pass")
        assert client.post("/api/v1/events/outbox/drain", headers=viewer).status_code == 403


def test_worker_module_importable():
    from app.workers import event_consumer, outbox

    assert callable(event_consumer.main)
    assert callable(outbox.main)
