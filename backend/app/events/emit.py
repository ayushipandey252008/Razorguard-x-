"""Domain events: enqueue on the caller's DB session; the outbox worker publishes."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.events.outbox import insert_outbox, record_insert_ms
from app.events.outbox_worker import drain_outbox_batch
from app.events.schemas import (
    AlertCreated,
    AnalystFeedbackRecorded,
    DomainEvent,
    InvestigationCompleted,
    InvestigationCreated,
    ModelDriftDetected,
    RiskScored,
    TransactionCreated,
)
from app.utils.ids import new_id
from app.utils.logging import Timer, get_logger

log = get_logger("events.emit")


async def enqueue_event(
    db: AsyncSession,
    event: DomainEvent,
    *,
    aggregate_type: str,
    aggregate_id: str,
) -> None:
    timer = Timer()
    await insert_outbox(db, event, aggregate_type=aggregate_type, aggregate_id=aggregate_id)
    record_insert_ms(timer.ms())


async def drain_after_commit() -> None:
    """Best-effort publish after the domain transaction committed. Never raises."""
    settings = get_settings()
    if not settings.outbox_enabled or not settings.outbox_drain_after_commit:
        return
    try:
        await drain_outbox_batch()
    except Exception as exc:
        log.warning("outbox_drain_after_commit_failed", error=type(exc).__name__)


async def enqueue_transaction_created(
    db: AsyncSession,
    event: TransactionCreated,
) -> None:
    await enqueue_event(
        db, event, aggregate_type="transaction", aggregate_id=event.transaction_id
    )


async def enqueue_risk_scored(
    db: AsyncSession,
    *,
    transaction_id: str,
    correlation_id: str,
    payload: dict[str, Any],
) -> RiskScored:
    event = RiskScored(
        correlation_id=correlation_id,
        transaction_id=transaction_id,
        payload=payload,
    )
    await enqueue_event(db, event, aggregate_type="transaction", aggregate_id=transaction_id)
    return event


async def enqueue_investigation_created(
    db: AsyncSession,
    *,
    transaction_id: str,
    correlation_id: str,
    payload: dict[str, Any],
) -> InvestigationCreated:
    event = InvestigationCreated(
        correlation_id=correlation_id,
        transaction_id=transaction_id,
        payload=payload,
    )
    await enqueue_event(
        db,
        event,
        aggregate_type="investigation",
        aggregate_id=str(payload.get("investigation_id") or transaction_id),
    )
    return event


async def enqueue_investigation_completed(
    db: AsyncSession,
    *,
    transaction_id: str,
    correlation_id: str,
    payload: dict[str, Any],
) -> InvestigationCompleted:
    event = InvestigationCompleted(
        correlation_id=correlation_id,
        transaction_id=transaction_id,
        payload=payload,
    )
    await enqueue_event(
        db,
        event,
        aggregate_type="investigation",
        aggregate_id=str(payload.get("investigation_id") or transaction_id),
    )
    return event


async def enqueue_alert_created(
    db: AsyncSession,
    *,
    transaction_id: str,
    correlation_id: str,
    decision: str,
    risk_level: str,
    extra: dict[str, Any] | None = None,
) -> AlertCreated | None:
    settings = get_settings()
    if decision == "BLOCK" and not settings.event_alert_on_block:
        return None
    if decision == "REVIEW" and not settings.event_alert_on_review:
        return None
    if decision not in {"BLOCK", "REVIEW"}:
        return None
    kind = "block" if decision == "BLOCK" else "review"
    payload = {
        "alert_id": new_id(),
        "transaction_id": transaction_id,
        "decision": decision,
        "risk_level": risk_level,
        "kind": kind,
        **(extra or {}),
    }
    event = AlertCreated(
        correlation_id=correlation_id,
        transaction_id=transaction_id,
        payload=payload,
    )
    await enqueue_event(db, event, aggregate_type="transaction", aggregate_id=transaction_id)
    return event


async def enqueue_analyst_feedback(
    db: AsyncSession,
    *,
    transaction_id: str,
    correlation_id: str,
    payload: dict[str, Any],
) -> AnalystFeedbackRecorded:
    event = AnalystFeedbackRecorded(
        correlation_id=correlation_id,
        transaction_id=transaction_id,
        payload=payload,
    )
    await enqueue_event(
        db,
        event,
        aggregate_type="investigation",
        aggregate_id=str(payload.get("investigation_id") or transaction_id),
    )
    return event


def should_alert(decision: str) -> bool:
    settings = get_settings()
    if decision == "BLOCK":
        return settings.event_alert_on_block
    if decision == "REVIEW":
        return settings.event_alert_on_review
    return False


async def enqueue_model_drift(
    db: AsyncSession,
    *,
    correlation_id: str,
    payload: dict[str, Any],
) -> ModelDriftDetected:
    event = ModelDriftDetected(
        correlation_id=correlation_id,
        payload=payload,
    )
    await enqueue_event(db, event, aggregate_type="model", aggregate_id=str(payload.get("status") or "drift"))
    return event
