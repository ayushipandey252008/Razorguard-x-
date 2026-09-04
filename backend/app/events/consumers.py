"""Minimal consumers: validate, skip duplicates, never mutate the graph or scoring."""

from __future__ import annotations

from typing import Any

from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal
from app.events.dlq import record_failed_event
from app.events.idempotency import claim_event, reset_memory_idempotency
from app.events.metrics import record_consume, record_duplicate
from app.events.schemas import DomainEvent
from app.models.eventing import Alert
from app.utils.ids import new_id
from app.utils.logging import Timer, get_logger

log = get_logger("events.consumers")

MAX_HANDLER_ATTEMPTS = 2

# In-memory alert index for unit tests that never open SQL.
_memory_alerts: dict[tuple[str, str], str] = {}


def reset_consumer_memory() -> None:
    reset_memory_idempotency()
    _memory_alerts.clear()


async def _unclaim(event_id: str) -> None:
    from app.events.idempotency import _memory_seen
    from app.models.eventing import ProcessedEvent
    from sqlalchemy import delete

    _memory_seen.discard(event_id)
    try:
        async with SessionLocal() as db:
            await db.execute(delete(ProcessedEvent).where(ProcessedEvent.event_id == event_id))
            await db.commit()
    except Exception:
        pass


async def persist_alert(event: DomainEvent) -> str | None:
    """Insert an alert. Unique (event_id) and (transaction_id, kind) prevent duplicates."""
    payload = event.payload or {}
    decision = str(payload.get("decision") or "")
    kind = str(payload.get("kind") or ("block" if decision == "BLOCK" else "review"))
    alert_id = str(payload.get("alert_id") or new_id())
    txn_id = event.transaction_id or str(payload.get("transaction_id") or "")
    key = (txn_id, kind)
    if key in _memory_alerts:
        return _memory_alerts[key]
    try:
        async with SessionLocal() as db:
            db.add(
                Alert(
                    id=alert_id,
                    source_event_id=event.event_id,
                    transaction_id=txn_id,
                    kind=kind,
                    decision=decision or kind.upper(),
                    risk_level=str(payload.get("risk_level") or decision or kind),
                    correlation_id=event.correlation_id,
                )
            )
            await db.commit()
            _memory_alerts[key] = alert_id
            return alert_id
    except IntegrityError:
        _memory_alerts[key] = alert_id
        return alert_id
    except Exception as exc:
        if key in _memory_alerts:
            return _memory_alerts[key]
        _memory_alerts[key] = alert_id
        log.warning("alert_persist_memory_only", error=type(exc).__name__, correlation_id=event.correlation_id)
        return alert_id


async def handle_risk_scored(event: DomainEvent) -> None:
    log.info(
        "consumed_risk_scored",
        event_id=event.event_id,
        correlation_id=event.correlation_id,
        transaction_id=event.transaction_id,
        decision=(event.payload or {}).get("decision"),
    )


async def handle_investigation_event(event: DomainEvent) -> None:
    """Observe investigation events. Do not insert investigation rows or touch the graph."""
    log.info(
        "consumed_investigation_event",
        event_id=event.event_id,
        event_type=event.event_type,
        correlation_id=event.correlation_id,
        transaction_id=event.transaction_id,
        investigation_id=(event.payload or {}).get("investigation_id"),
    )


async def handle_alert_created(event: DomainEvent) -> None:
    alert_id = await persist_alert(event)
    log.info(
        "consumed_alert_created",
        event_id=event.event_id,
        alert_id=alert_id,
        correlation_id=event.correlation_id,
        transaction_id=event.transaction_id,
        decision=(event.payload or {}).get("decision"),
    )


async def handle_feedback(event: DomainEvent) -> None:
    log.info(
        "consumed_analyst_feedback",
        event_id=event.event_id,
        correlation_id=event.correlation_id,
        transaction_id=event.transaction_id,
        decision=(event.payload or {}).get("decision"),
    )


async def handle_transaction_created(event: DomainEvent) -> None:
    log.info(
        "consumed_transaction_created",
        event_id=event.event_id,
        correlation_id=event.correlation_id,
        transaction_id=event.transaction_id,
    )


async def handle_drift(event: DomainEvent) -> None:
    log.info(
        "consumed_model_drift",
        event_id=event.event_id,
        correlation_id=event.correlation_id,
        status=(event.payload or {}).get("status"),
    )


HANDLERS = {
    "transaction-created": handle_transaction_created,
    "risk-scored": handle_risk_scored,
    "investigation-created": handle_investigation_event,
    "investigation-completed": handle_investigation_event,
    "alert-created": handle_alert_created,
    "analyst-feedback-recorded": handle_feedback,
    "model-drift-detected": handle_drift,
}


async def process_event(event: DomainEvent) -> dict[str, Any]:
    """Idempotent consumer entry. Safe to call more than once for the same event_id."""
    timer = Timer()
    first = await claim_event(event.event_id, event.event_type, event.correlation_id)
    if not first:
        record_duplicate()
        record_consume(timer.ms())
        return {"ok": True, "duplicate": True, "event_id": event.event_id}

    handler = HANDLERS.get(event.event_type)
    last_error: Exception | None = None
    for attempt in range(1, MAX_HANDLER_ATTEMPTS + 1):
        try:
            if handler:
                await handler(event)
            record_consume(timer.ms())
            return {"ok": True, "duplicate": False, "event_id": event.event_id, "attempt": attempt}
        except Exception as exc:
            last_error = exc
            log.warning(
                "event_handler_retry",
                event_id=event.event_id,
                event_type=event.event_type,
                correlation_id=event.correlation_id,
                attempt=attempt,
                error=type(exc).__name__,
            )
    await _unclaim(event.event_id)
    await record_failed_event(
        event_id=event.event_id,
        event_type=event.event_type,
        correlation_id=event.correlation_id,
        error_reason=str(last_error) if last_error else "handler failed",
        payload={"event_type": event.event_type, "transaction_id": event.transaction_id},
        retry_count=MAX_HANDLER_ATTEMPTS,
    )
    record_consume(timer.ms())
    return {"ok": False, "event_id": event.event_id, "error": type(last_error).__name__ if last_error else "error"}


def register_default_handlers(bus) -> None:
    for event_type, handler in HANDLERS.items():
        bus.subscribe(event_type, process_event)
