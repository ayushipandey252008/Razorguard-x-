"""Phase 5 EventBus: schemas, in-process bus, Kafka fallback, idempotency."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.events.base import FORBIDDEN_PAYLOAD_KEYS, sanitize_payload
from app.events.bus import EventBus
from app.events.consumers import persist_alert, process_event, reset_consumer_memory
from app.events.factory import connect_event_bus, event_bus_status, reset_event_bus_for_tests
from app.events.inprocess_bus import InProcessEventBus
from app.events.kafka_bus import KafkaEventBus
from app.events.metrics import reset_metrics
from app.events.schemas import (
    AlertCreated,
    RiskScored,
    TransactionCreated,
    parse_event,
)
from app.events.serialize import MalformedEventError, deserialize_event, serialize_event
from app.events.topics import topic_for_event, topic_names
from app.main import app
from app.utils.ids import new_id
from app.utils.logging import Timer
from app.utils.redact import contains_secret


def _login(client: TestClient):
    r = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@razorguard.local", "password": "prototype-pass"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _kafka_settings(**overrides):
    ns = SimpleNamespace(
        event_bus="kafka",
        event_bus_fallback=True,
        kafka_bootstrap_servers="127.0.0.1:1",
        kafka_connect_timeout_seconds=1,
        kafka_publish_timeout_ms=200,
        kafka_topic_transactions="transactions",
        kafka_topic_risk_results="risk-results",
        kafka_topic_investigations="investigations",
        kafka_topic_alerts="alerts",
        kafka_topic_feedback="feedback",
        kafka_topic_dlq="events-dlq",
        kafka_group_risk="rgx-risk-results",
        kafka_group_investigations="rgx-investigations",
        kafka_group_alerts="rgx-alerts",
        kafka_group_feedback="rgx-feedback",
        kafka_group_transactions="rgx-transactions",
        event_alert_on_block=True,
        event_alert_on_review=True,
        event_consumer_in_api=False,
        outbox_enabled=True,
        outbox_drain_after_commit=True,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_event_schema_validation_and_required_fields():
    event = TransactionCreated(
        correlation_id="corr-1",
        transaction_id="txn_1",
        payload={"user_id": "u1", "amount": 10, "currency": "INR"},
    )
    assert event.event_id
    assert event.event_type == "transaction-created"
    assert event.schema_version == "1"
    assert event.correlation_id == "corr-1"
    assert event.transaction_id == "txn_1"
    parsed = parse_event(event.model_dump(mode="json"))
    assert isinstance(parsed, TransactionCreated)
    with pytest.raises(ValidationError):
        TransactionCreated(correlation_id="x")  # type: ignore[call-arg]


def test_payload_strips_secrets_and_payment_identifiers():
    payload = sanitize_payload(
        {
            "amount": 99,
            "payment_identifier": "4111111111111111",
            "api_key": "sk-live-abcdefghijklmnopqrstuvwxyz",
            "password": "secret",
            "authorization": "Bearer abc",
            "user_id": "usr_1",
        }
    )
    assert payload["amount"] == 99
    assert payload["user_id"] == "usr_1"
    assert "payment_identifier" not in payload
    for key in FORBIDDEN_PAYLOAD_KEYS:
        assert key not in payload
    event = RiskScored(
        correlation_id="c",
        transaction_id="t",
        payload={"decision": "APPROVE", "api_key": "sk-live-abcdefghijklmnopqrstuvwxyz"},
    )
    blob = json.dumps(event.model_dump(mode="json"))
    assert "sk-live-" not in blob
    assert "payment_identifier" not in blob
    assert contains_secret(event.model_dump()) is False


@pytest.mark.asyncio
async def test_event_bus_interface_and_inprocess_pubsub():
    bus = InProcessEventBus()
    assert isinstance(bus, EventBus)
    await bus.connect()
    seen: list[str] = []

    async def handler(event):
        seen.append(event.event_type)

    bus.subscribe("risk-scored", handler)
    event = RiskScored(
        correlation_id="corr-bus",
        transaction_id="txn_bus",
        payload={"decision": "APPROVE", "final_risk_score": 12},
    )
    result = await bus.publish(event)
    assert result["ok"] is True
    assert result["event_bus"] == "inprocess"
    assert seen == ["risk-scored"]
    assert bus.published[0].correlation_id == "corr-bus"
    await bus.close()


def test_kafka_consumers_include_transactions_topic():
    source = inspect.getsource(KafkaEventBus.start_consumers)
    assert "kafka_topic_transactions" in source
    assert "kafka_group_transactions" in source
    assert "kafka_topic_feedback" in source


def test_kafka_admin_closes_with_close_not_stop():
    source = inspect.getsource(KafkaEventBus._ensure_topics)
    assert "admin.close()" in source
    assert "admin.stop()" not in source


def test_compose_kafka_image_is_pullable_legacy_tag():
    compose = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    text = compose.read_text()
    assert "bitnamilegacy/kafka:3.9.0" in text
    assert "image: bitnami/kafka:3.9\n" not in text


def test_kafka_serialization_roundtrip_without_broker():
    event = AlertCreated(
        correlation_id="corr-ser",
        transaction_id="txn_ser",
        payload={"alert_id": "al1", "decision": "BLOCK", "risk_level": "BLOCK", "kind": "block"},
    )
    raw = serialize_event(event)
    restored = deserialize_event(raw)
    assert restored.event_id == event.event_id
    assert restored.event_type == "alert-created"
    assert restored.correlation_id == "corr-ser"
    assert b"password" not in raw
    assert topic_for_event("alert-created") == topic_names()["alerts"]
    assert topic_for_event("risk-scored") == topic_names()["risk-results"]


def test_malformed_event_handling():
    with pytest.raises(MalformedEventError):
        deserialize_event(b"not-json")
    with pytest.raises(MalformedEventError):
        deserialize_event(json.dumps(["list"]))
    with pytest.raises(ValidationError):
        parse_event({"event_type": "transaction-created", "correlation_id": "c"})


@pytest.mark.asyncio
async def test_duplicate_event_idempotency():
    reset_consumer_memory()
    reset_metrics()
    event = AlertCreated(
        event_id=new_id(),
        correlation_id="corr-dup",
        transaction_id=f"txn_dup_{new_id()[:8]}",
        payload={"alert_id": "al-dup", "decision": "BLOCK", "risk_level": "BLOCK", "kind": "block"},
    )
    first = await process_event(event)
    second = await process_event(event)
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    third = await persist_alert(event)
    fourth = await persist_alert(event)
    assert third == fourth


@pytest.mark.asyncio
async def test_kafka_unavailable_falls_back_to_inprocess():
    await reset_event_bus_for_tests()
    try:
        bus = await connect_event_bus(_kafka_settings())
        status = event_bus_status()
        assert status["configured"] == "kafka"
        assert status["active"] == "inprocess"
        assert status["fallback"] is True
        assert status["kafka_connected"] is False
        assert status["reason"] == "connection unavailable"
        assert bus.name == "inprocess"
        received: list[str] = []

        async def handler(event):
            received.append(event.event_id)

        bus.subscribe("transaction-created", handler)
        event = TransactionCreated(
            correlation_id="corr-fb",
            transaction_id="txn_fb",
            payload={"amount": 1, "currency": "INR"},
        )
        result = await bus.publish(event)
        assert result["ok"] is True
        assert event.event_id in received
    finally:
        await reset_event_bus_for_tests()


@pytest.mark.asyncio
async def test_kafka_publish_failure_uses_inprocess_fallback():
    inner = InProcessEventBus()
    await inner.connect()
    received: list[str] = []

    async def handler(event):
        received.append(event.event_type)

    inner.subscribe("transaction-created", handler)
    bus = KafkaEventBus(
        "127.0.0.1:1",
        settings=_kafka_settings(),
        fallback_bus=inner,
        use_fallback_on_publish=True,
    )
    event = TransactionCreated(
        correlation_id="corr-kf",
        transaction_id="txn_kf",
        payload={"amount": 2, "currency": "INR"},
    )
    result = await bus.publish(event)
    assert result.get("fallback") is True
    assert result["ok"] is True
    assert received == ["transaction-created"]
    await inner.close()


def test_consumers_do_not_touch_graph_or_scoring():
    source = inspect.getsource(inspect.getmodule(process_event))
    assert "ingest_transaction" not in source
    assert "graph_store" not in source
    assert "combine_scores" not in source
    assert "model_service" not in source


def test_health_and_events_status_hide_credentials():
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        body = health.json()
        assert body["event_bus"]["configured"] in {"inprocess", "kafka"}
        assert body["event_bus"]["active"] in {"inprocess", "kafka"}
        assert "kafka_connected" in body["event_bus"]
        blob = json.dumps(body).lower()
        assert "password" not in blob or "seed_admin_password" not in blob
        assert "secret_key" not in blob
        headers = _login(client)
        status = client.get("/api/v1/events/status", headers=headers)
        assert status.status_code == 200
        data = status.json()
        assert "topics" in data
        assert data["topics"]["transactions"]
        assert "bootstrap_servers" in json.dumps(data) or data["active"] == "inprocess"
        assert "sasl" not in json.dumps(data).lower()
        assert status.json().get("kafka_password") is None


def test_demo_a_normal_transaction_events_and_correlation():
    with TestClient(app) as client:
        headers = {
            **_login(client),
            "X-Correlation-ID": "demo-a-corr",
            "X-Request-ID": "demo-a-corr",
        }
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
        assert r.headers.get("X-Correlation-ID") == "demo-a-corr"
        status = client.get("/api/v1/events/status", headers=headers).json()
        types = [e["event_type"] for e in status["recent_events"]]
        assert "transaction-created" in types
        assert "risk-scored" in types
        matched = [e for e in status["recent_events"] if e.get("correlation_id") == "demo-a-corr"]
        assert {e["event_type"] for e in matched} >= {"transaction-created", "risk-scored"}
        assert "alert-created" not in {e["event_type"] for e in matched}
        print(f"PROTOTYPE_METRIC inprocess_transaction_api_ms={api_ms}")
        print(f"PROTOTYPE_METRIC publish_last_ms={status['prototype_latency_ms']['publish_last_ms']}")
        print(f"PROTOTYPE_METRIC consume_last_ms={status['prototype_latency_ms']['consume_last_ms']}")


def test_demo_b_high_risk_alert_and_investigation_events():
    with TestClient(app) as client:
        headers = {**_login(client), "X-Correlation-ID": "demo-b-corr"}
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_ring_a",
                "merchant_id": "m_watch_01",
                "amount": 22000,
                "currency": "INR",
                "device_id": "dev_farm_01",
                "ip_address": "203.0.113.200",
                "location": "Dubai",
                "payment_method": "UPI",
                "failed_attempts": 5,
                "transaction_velocity": 11,
                "current_device_known": True,
                "current_location_known": False,
                "scenario_tag": "fraud_ring",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["risk"]["decision"] == "BLOCK"
        assert r.json()["investigation_id"]
        status = client.get("/api/v1/events/status", headers=headers).json()
        matched = {e["event_type"] for e in status["recent_events"] if e.get("correlation_id") == "demo-b-corr"}
        assert "transaction-created" in matched
        assert "risk-scored" in matched
        assert "alert-created" in matched
        assert "investigation-created" in matched


def test_demo_c_investigation_completed_event():
    with TestClient(app) as client:
        headers = {**_login(client), "X-Correlation-ID": "demo-c-corr"}
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_leila",
                "merchant_id": "m_fash_01",
                "amount": 3200,
                "currency": "INR",
                "device_id": "dev_leila_travel",
                "ip_address": "198.51.100.40",
                "location": "Singapore",
                "payment_method": "CARD_TOKEN",
                "failed_attempts": 3,
                "transaction_velocity": 5,
                "current_device_known": False,
                "current_location_known": False,
                "scenario_tag": "moderate_review",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        inv_id = r.json()["investigation_id"]
        run = client.post(f"/api/v1/investigations/{inv_id}/run", headers=headers)
        assert run.status_code == 200, run.text
        decide = client.post(
            f"/api/v1/investigations/{inv_id}/decision",
            json={"decision": "BLOCK", "reason": "prototype analyst decision"},
            headers=headers,
        )
        assert decide.status_code == 200, decide.text
        status = client.get("/api/v1/events/status", headers=headers).json()
        matched = {e["event_type"] for e in status["recent_events"] if e.get("correlation_id") == "demo-c-corr"}
        assert "investigation-created" in matched
        assert "investigation-completed" in matched
        assert "analyst-feedback-recorded" in matched
        completed = [
            e
            for e in status["recent_events"]
            if e.get("correlation_id") == "demo-c-corr" and e["event_type"] == "investigation-completed"
        ]
        assert completed


@pytest.mark.asyncio
async def test_inprocess_publish_latency_prototype_bound():
    bus = InProcessEventBus()
    await bus.connect()
    event = TransactionCreated(
        correlation_id="corr-lat",
        transaction_id="txn_lat",
        payload={"amount": 1, "currency": "INR"},
    )
    timer = Timer()
    await bus.publish(event)
    ms = timer.ms()
    await bus.close()
    print(f"PROTOTYPE_METRIC inprocess_publish_ms={ms}")
    assert ms < 500


@pytest.mark.skipif(os.environ.get("RUN_KAFKA_TESTS") != "1", reason="Kafka integration tests are opt-in")
@pytest.mark.asyncio
async def test_optional_kafka_broker_roundtrip():
    from app.config import get_settings
    from app.events.kafka_bus import KafkaUnavailable

    settings = get_settings()
    bus = KafkaEventBus(
        os.environ.get("KAFKA_BOOTSTRAP_SERVERS", settings.kafka_bootstrap_servers),
        settings=_kafka_settings(
            kafka_bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", settings.kafka_bootstrap_servers),
            kafka_connect_timeout_seconds=15,
            kafka_publish_timeout_ms=5000,
            event_bus_fallback=False,
        ),
        fallback_bus=None,
        use_fallback_on_publish=False,
    )
    try:
        await bus.connect()
    except KafkaUnavailable:
        pytest.skip("Kafka broker not reachable")
    try:
        event = RiskScored(
            correlation_id="corr-kafka",
            transaction_id="txn_kafka",
            payload={"decision": "APPROVE", "final_risk_score": 1},
        )
        timer = Timer()
        result = await bus.publish(event)
        publish_ms = timer.ms()
        assert result["ok"] is True
        print(f"PROTOTYPE_METRIC kafka_publish_ms={publish_ms}")
    finally:
        await bus.close()
