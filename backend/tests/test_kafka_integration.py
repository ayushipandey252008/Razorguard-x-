"""Real-broker Kafka integration. Opt-in: RUN_KAFKA_TESTS=1.

These tests skip when the Compose broker is not reachable. They do not
require Kubernetes, Prisma, or a second message bus.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest
from sqlalchemy import select

from app.database import SessionLocal, init_db
from app.events.consumers import HANDLERS, persist_alert, process_event, reset_consumer_memory
from app.events.factory import connect_event_bus, event_bus_status, reset_event_bus_for_tests
from app.events.kafka_bus import KafkaEventBus, KafkaUnavailable
from app.events.outbox import insert_outbox
from app.events.outbox_worker import drain_outbox_batch
from app.events.schemas import (
    AlertCreated,
    AnalystFeedbackRecorded,
    InvestigationCompleted,
    InvestigationCreated,
    RiskScored,
    TransactionCreated,
)
from app.events.serialize import deserialize_event, serialize_event
from app.events.topics import topic_for_event, topic_names
from app.models.eventing import Alert, FailedEvent, ProcessedEvent
from app.models.outbox import OUTBOX_PENDING, OUTBOX_PUBLISHED, OutboxEvent
from app.utils.ids import new_id

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_KAFKA_TESTS") != "1",
    reason="Kafka integration tests are opt-in",
)

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def _live_settings(**overrides):
    from tests.test_events_phase5 import _kafka_settings

    return _kafka_settings(
        kafka_bootstrap_servers=BOOTSTRAP,
        kafka_connect_timeout_seconds=15,
        kafka_publish_timeout_ms=5000,
        event_bus_fallback=False,
        kafka_group_transactions=f"rgx-itest-txn-{new_id()[:8]}",
        kafka_group_risk=f"rgx-itest-risk-{new_id()[:8]}",
        kafka_group_investigations=f"rgx-itest-inv-{new_id()[:8]}",
        kafka_group_alerts=f"rgx-itest-alert-{new_id()[:8]}",
        kafka_group_feedback=f"rgx-itest-fb-{new_id()[:8]}",
        **overrides,
    )


async def _consume_event(topic: str, event_id: str, timeout: float = 20.0):
    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=BOOTSTRAP,
        group_id=f"rgx-itest-read-{event_id[:12]}",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        request_timeout_ms=20000,
    )
    await consumer.start()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            batches = await consumer.getmany(timeout_ms=1000)
            for messages in batches.values():
                for msg in messages:
                    try:
                        event = deserialize_event(msg.value)
                    except Exception:
                        continue
                    if event.event_id == event_id:
                        return event
        return None
    finally:
        await consumer.stop()


async def _consume_raw(topic: str, needle: bytes, timeout: float = 20.0):
    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=BOOTSTRAP,
        group_id=f"rgx-itest-raw-{new_id()[:8]}",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        request_timeout_ms=20000,
    )
    await consumer.start()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            batches = await consumer.getmany(timeout_ms=1000)
            for messages in batches.values():
                for msg in messages:
                    if needle in (msg.value or b""):
                        return msg.value
        return None
    finally:
        await consumer.stop()


async def _wait_processed(event_id: str, timeout: float = 20.0) -> ProcessedEvent | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with SessionLocal() as db:
            row = (
                await db.execute(select(ProcessedEvent).where(ProcessedEvent.event_id == event_id))
            ).scalar_one_or_none()
            if row is not None:
                return row
        await asyncio.sleep(0.25)
    return None


@pytest.fixture
async def kafka_bus():
    bus = KafkaEventBus(
        BOOTSTRAP,
        settings=_live_settings(),
        fallback_bus=None,
        use_fallback_on_publish=False,
    )
    try:
        await bus.connect()
    except KafkaUnavailable as exc:
        pytest.skip(f"Kafka broker not reachable: {exc}")
    try:
        yield bus
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_broker_connect_and_all_topics_roundtrip(kafka_bus):
    names = topic_names()
    events = [
        TransactionCreated(
            correlation_id="corr-all-txn",
            transaction_id=f"txn_all_{new_id()[:8]}",
            payload={"amount": 10, "currency": "INR"},
        ),
        RiskScored(
            correlation_id="corr-all-risk",
            transaction_id=f"txn_all_{new_id()[:8]}",
            payload={"decision": "APPROVE", "final_risk_score": 1},
        ),
        InvestigationCreated(
            correlation_id="corr-all-invc",
            transaction_id=f"txn_all_{new_id()[:8]}",
            payload={"investigation_id": new_id(), "status": "OPEN"},
        ),
        InvestigationCompleted(
            correlation_id="corr-all-invd",
            transaction_id=f"txn_all_{new_id()[:8]}",
            payload={"investigation_id": new_id(), "recommendation": "BLOCK"},
        ),
        AlertCreated(
            correlation_id="corr-all-alert",
            transaction_id=f"txn_all_{new_id()[:8]}",
            payload={"alert_id": new_id(), "decision": "BLOCK", "kind": "block", "risk_level": "BLOCK"},
        ),
        AnalystFeedbackRecorded(
            correlation_id="corr-all-fb",
            transaction_id=f"txn_all_{new_id()[:8]}",
            payload={"investigation_id": new_id(), "decision": "BLOCK"},
        ),
    ]
    latencies = []
    for event in events:
        result = await kafka_bus.publish(event, allow_fallback=False)
        assert result["ok"] is True
        assert result["event_bus"] == "kafka"
        latencies.append(result.get("latency_ms"))
        got = await _consume_event(topic_for_event(event.event_type), event.event_id)
        assert got is not None, f"missing consume for {event.event_type}"
        assert got.event_id == event.event_id
        assert got.correlation_id == event.correlation_id
        assert got.transaction_id == event.transaction_id
    dlq_marker = f"dlq-probe-{new_id()}".encode()
    await kafka_bus._producer.send_and_wait(names["dlq"], dlq_marker)
    raw = await _consume_raw(names["dlq"], dlq_marker)
    assert raw == dlq_marker
    print(f"PROTOTYPE_METRIC kafka_topic_publish_ms={latencies}")


@pytest.mark.asyncio
async def test_outbox_worker_publishes_to_kafka_and_marks_published(kafka_bus):
    await init_db()
    await reset_event_bus_for_tests()
    settings = _live_settings()
    bus = await connect_event_bus(settings, start_consumers=False)
    assert event_bus_status()["kafka_connected"] is True
    assert bus.name == "kafka"
    event = TransactionCreated(
        correlation_id="corr-outbox-k",
        transaction_id=f"txn_obx_{new_id()[:8]}",
        payload={"amount": 3, "currency": "INR"},
    )
    async with SessionLocal() as db:
        await insert_outbox(db, event, aggregate_type="transaction", aggregate_id=event.transaction_id)
        await db.commit()
    result = await drain_outbox_batch()
    assert result["published"] >= 1
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.event_id == event.event_id))).scalar_one()
        assert row.status == OUTBOX_PUBLISHED
        assert row.correlation_id == "corr-outbox-k"
    got = await _consume_event(topic_names()["transactions"], event.event_id)
    assert got is not None
    assert got.correlation_id == event.correlation_id
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_transaction_pipeline_outbox_to_kafka_to_consumer():
    """Real path: score + persist → outbox → Kafka → consumer, same event_id/correlation_id."""
    await init_db()
    reset_consumer_memory()
    await reset_event_bus_for_tests()
    from app.ml.predictor import model_service
    from app.schemas.common import TransactionCreate
    from app.services.bootstrap import ensure_seeded
    from app.services.pipeline import process_transaction

    settings = _live_settings()
    await connect_event_bus(settings, start_consumers=True)
    status = event_bus_status()
    assert status["kafka_connected"] is True
    assert status.get("transport", {}).get("consumers", 0) >= 5
    model_service.load_or_train()
    await ensure_seeded()
    payload = TransactionCreate(
        user_id="usr_ring_a",
        merchant_id="m_watch_01",
        amount=22000,
        currency="INR",
        device_id=f"dev_kafka_e2e_{new_id()[:8]}",
        ip_address="203.0.113.200",
        location="Dubai",
        payment_method="UPI",
        failed_attempts=5,
        transaction_velocity=11,
        current_device_known=True,
        current_location_known=False,
        scenario_tag="fraud_ring",
    )
    async with SessionLocal() as db:
        result = await process_transaction(db, payload)
    txn_id = result["transaction"].transaction_id
    decision = result["risk"].decision
    assert decision in {"APPROVE", "REVIEW", "BLOCK"}
    async with SessionLocal() as db:
        rows = list(
            (await db.execute(select(OutboxEvent).where(OutboxEvent.aggregate_id == txn_id))).scalars()
        )
    assert rows
    assert all(r.status == OUTBOX_PUBLISHED for r in rows)
    types = {r.event_type for r in rows}
    assert "transaction-created" in types
    assert "risk-scored" in types
    corr_ids = {r.correlation_id for r in rows}
    assert len(corr_ids) == 1
    corr = next(iter(corr_ids))
    for row in rows:
        got = await _consume_event(topic_for_event(row.event_type), row.event_id)
        assert got is not None, f"kafka missing {row.event_type}"
        assert got.event_id == row.event_id
        assert got.correlation_id == corr
        processed = await _wait_processed(row.event_id)
        assert processed is not None, f"consumer missed {row.event_type}"
        assert processed.correlation_id == corr
    if decision in {"BLOCK", "REVIEW"}:
        assert "alert-created" in types
        async with SessionLocal() as db:
            alerts = list((await db.execute(select(Alert).where(Alert.transaction_id == txn_id))).scalars())
        assert len(alerts) == 1
        assert alerts[0].correlation_id == corr
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_all_handler_event_types_are_consumed():
    await init_db()
    reset_consumer_memory()
    await reset_event_bus_for_tests()
    settings = _live_settings()
    bus = await connect_event_bus(settings, start_consumers=True)
    txn_id = f"txn_types_{new_id()[:8]}"
    corr = f"corr-types-{new_id()[:8]}"
    events = [
        TransactionCreated(correlation_id=corr, transaction_id=txn_id, payload={"amount": 1, "currency": "INR"}),
        RiskScored(correlation_id=corr, transaction_id=txn_id, payload={"decision": "REVIEW", "final_risk_score": 55}),
        InvestigationCreated(
            correlation_id=corr,
            transaction_id=txn_id,
            payload={"investigation_id": new_id(), "status": "OPEN"},
        ),
        InvestigationCompleted(
            correlation_id=corr,
            transaction_id=txn_id,
            payload={"investigation_id": new_id(), "recommendation": "BLOCK"},
        ),
        AlertCreated(
            correlation_id=corr,
            transaction_id=txn_id,
            payload={"alert_id": new_id(), "decision": "REVIEW", "kind": "review", "risk_level": "REVIEW"},
        ),
        AnalystFeedbackRecorded(
            correlation_id=corr,
            transaction_id=txn_id,
            payload={"investigation_id": new_id(), "decision": "BLOCK"},
        ),
    ]
    for event in events:
        result = await bus.publish(event, allow_fallback=False)
        assert result["ok"] is True
        processed = await _wait_processed(event.event_id)
        assert processed is not None, event.event_type
        assert processed.correlation_id == corr
        assert processed.event_id == event.event_id
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_outbox_stays_pending_when_kafka_producer_closed():
    await init_db()
    await reset_event_bus_for_tests()
    settings = _live_settings()
    bus = await connect_event_bus(settings, start_consumers=False)
    event = TransactionCreated(
        correlation_id="corr-closed-prod",
        transaction_id=f"txn_clp_{new_id()[:8]}",
        payload={"amount": 2, "currency": "INR"},
    )
    async with SessionLocal() as db:
        await insert_outbox(db, event, aggregate_type="transaction", aggregate_id=event.transaction_id)
        await db.commit()
    await bus.close()
    result = await drain_outbox_batch()
    assert result.get("published", 0) == 0
    async with SessionLocal() as db:
        row = (await db.execute(select(OutboxEvent).where(OutboxEvent.event_id == event.event_id))).scalar_one()
        assert row.status == OUTBOX_PENDING
        assert int(row.attempts or 0) >= 1
        assert row.last_error
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_kafka_consumer_processes_alert_and_is_idempotent():
    await init_db()
    reset_consumer_memory()
    await reset_event_bus_for_tests()
    settings = _live_settings()
    bus = await connect_event_bus(settings, start_consumers=True)
    assert event_bus_status()["active"] == "kafka"
    txn_id = f"txn_idemk_{new_id()[:8]}"
    event = AlertCreated(
        correlation_id="corr-idem-k",
        transaction_id=txn_id,
        payload={"alert_id": new_id(), "decision": "BLOCK", "kind": "block", "risk_level": "BLOCK"},
    )
    first = await bus.publish(event, allow_fallback=False)
    assert first["ok"] is True
    processed = await _wait_processed(event.event_id)
    assert processed is not None
    assert processed.correlation_id == "corr-idem-k"
    async with SessionLocal() as db:
        alerts = list((await db.execute(select(Alert).where(Alert.transaction_id == txn_id))).scalars())
    assert len(alerts) == 1
    second = await bus.publish(event, allow_fallback=False)
    assert second["ok"] is True
    again = await process_event(event)
    assert again["duplicate"] is True
    async with SessionLocal() as db:
        alerts = list((await db.execute(select(Alert).where(Alert.transaction_id == txn_id))).scalars())
    assert len(alerts) == 1
    await persist_alert(event)
    async with SessionLocal() as db:
        alerts = list((await db.execute(select(Alert).where(Alert.transaction_id == txn_id))).scalars())
    assert len(alerts) == 1
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_malformed_event_goes_to_dlq_without_infinite_retry(kafka_bus):
    await init_db()
    reset_consumer_memory()
    await reset_event_bus_for_tests()
    settings = _live_settings()
    bus = await connect_event_bus(settings, start_consumers=True)
    marker = f"malformed-{new_id()}"
    raw = f'{{"not": "an-event", "marker": "{marker}"}}'.encode()
    topic = topic_names()["alerts"]
    await bus._producer.send_and_wait(topic, raw)
    dlq = await _consume_raw(topic_names()["dlq"], marker.encode(), timeout=25)
    assert dlq is not None
    deadline = time.monotonic() + 20
    matched = []
    while time.monotonic() < deadline:
        async with SessionLocal() as db:
            rows = list((await db.execute(select(FailedEvent))).scalars())
        matched = [
            r
            for r in rows
            if marker in str((r.payload or {})) or marker in (r.error_reason or "")
        ]
        if matched:
            break
        await asyncio.sleep(0.25)
    assert matched, "malformed event was not persisted to failed_events"
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_handler_failure_retries_then_dlq(monkeypatch):
    await init_db()
    reset_consumer_memory()
    attempts = {"n": 0}

    async def boom(event):
        attempts["n"] += 1
        raise RuntimeError("controlled_handler_failure")

    monkeypatch.setitem(HANDLERS, "alert-created", boom)
    await reset_event_bus_for_tests()
    settings = _live_settings()
    bus = await connect_event_bus(settings, start_consumers=True)
    event = AlertCreated(
        correlation_id="corr-retry-k",
        transaction_id=f"txn_retry_{new_id()[:8]}",
        payload={"alert_id": new_id(), "decision": "BLOCK", "kind": "block", "risk_level": "BLOCK"},
    )
    result = await bus.publish(event, allow_fallback=False)
    assert result["ok"] is True
    deadline = time.monotonic() + 25
    failed = []
    while time.monotonic() < deadline:
        async with SessionLocal() as db:
            failed = list(
                (await db.execute(select(FailedEvent).where(FailedEvent.event_id == event.event_id))).scalars()
            )
        if failed:
            break
        await asyncio.sleep(0.25)
    assert failed
    assert failed[0].retry_count == 2
    assert "controlled_handler_failure" in (failed[0].error_reason or "")
    assert attempts["n"] >= 2
    dlq = await _consume_raw(topic_names()["dlq"], event.event_id.encode(), timeout=15)
    assert dlq is not None
    await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_kafka_publish_failure_is_not_marked_ok():
    settings = _live_settings()
    bus = KafkaEventBus(
        BOOTSTRAP,
        settings=settings,
        fallback_bus=None,
        use_fallback_on_publish=False,
    )
    try:
        await bus.connect()
    except KafkaUnavailable as exc:
        pytest.skip(f"Kafka broker not reachable: {exc}")
    await bus.close()
    event = TransactionCreated(
        correlation_id="corr-fail-pub",
        transaction_id=f"txn_fail_{new_id()[:8]}",
        payload={"amount": 1, "currency": "INR"},
    )
    result = await bus.publish(event, allow_fallback=False)
    assert result["ok"] is False
    assert result.get("event_bus") == "kafka"


def test_http_still_works_when_kafka_configured_but_down(monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import get_settings
    from app.main import app

    monkeypatch.setenv("EVENT_BUS", "kafka")
    monkeypatch.setenv("EVENT_BUS_FALLBACK", "true")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:1")
    monkeypatch.setenv("KAFKA_CONNECT_TIMEOUT_SECONDS", "1")
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/login",
                json={"email": "admin@razorguard.local", "password": "prototype-pass"},
            )
            assert login.status_code == 200, login.text
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
            r = client.post(
                "/api/v1/transactions",
                json={
                    "user_id": "usr_ananya",
                    "merchant_id": "m_groc_01",
                    "amount": 620,
                    "device_id": "dev_kafka_down_api",
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
            assert r.json()["risk"]["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
            health = client.get("/api/v1/health").json()
            assert health["event_bus"]["configured"] == "kafka"
            assert health["event_bus"]["fallback"] is True
            assert health["event_bus"]["kafka_connected"] is False
    finally:
        get_settings.cache_clear()
