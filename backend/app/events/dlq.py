"""Lightweight failed-event log plus optional Kafka dead-letter topic."""

from __future__ import annotations

from typing import Any

from app.database import SessionLocal
from app.events.metrics import record_failed
from app.models.eventing import FailedEvent
from app.utils.ids import new_id
from app.utils.logging import get_logger
from app.utils.redact import redact_secrets

log = get_logger("events.dlq")


async def record_failed_event(
    *,
    event_id: str | None,
    event_type: str | None,
    error_reason: str,
    correlation_id: str | None = None,
    payload: dict[str, Any] | None = None,
    retry_count: int = 0,
) -> None:
    record_failed(event_type)
    safe_payload = redact_secrets(payload) if payload else None
    log.warning(
        "event_failed",
        event_id=event_id,
        event_type=event_type,
        correlation_id=correlation_id,
        error_reason=error_reason,
        retry_count=retry_count,
    )
    try:
        async with SessionLocal() as db:
            db.add(
                FailedEvent(
                    id=new_id(),
                    event_id=event_id,
                    event_type=event_type,
                    correlation_id=correlation_id,
                    error_reason=error_reason[:2000],
                    retry_count=retry_count,
                    payload=safe_payload,
                )
            )
            await db.commit()
    except Exception as exc:
        log.warning("failed_event_persist_skipped", error=type(exc).__name__)
