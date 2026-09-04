"""Enqueue, claim, and inspect outbox rows. Publishing happens in outbox_worker."""

from __future__ import annotations

from datetime import datetime, timedelta
from threading import Lock
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import SessionLocal
from app.events.schemas import DomainEvent
from app.models.outbox import (
    OUTBOX_FAILED,
    OUTBOX_PENDING,
    OUTBOX_PROCESSING,
    OUTBOX_PUBLISHED,
    OutboxEvent,
)
from app.utils.ids import new_id, utcnow
from app.utils.logging import get_logger
from app.utils.redact import redact_secrets

log = get_logger("events.outbox")

_lock = Lock()
_last_success: datetime | None = None
_last_failure: datetime | None = None
_last_failure_reason: str | None = None
_insert_ms: list[float] = []
_process_ms: list[float] = []


def record_insert_ms(ms: float) -> None:
    with _lock:
        _insert_ms.append(ms)
        if len(_insert_ms) > 50:
            del _insert_ms[0]


def record_process_ms(ms: float) -> None:
    with _lock:
        _process_ms.append(ms)
        if len(_process_ms) > 50:
            del _process_ms[0]


def record_publish_success() -> None:
    global _last_success
    with _lock:
        _last_success = utcnow()


def record_publish_failure(reason: str) -> None:
    global _last_failure, _last_failure_reason
    with _lock:
        _last_failure = utcnow()
        _last_failure_reason = reason[:500]


def memory_status() -> dict[str, Any]:
    with _lock:
        return {
            "last_successful_publish": _last_success.isoformat() if _last_success else None,
            "last_failure": _last_failure.isoformat() if _last_failure else None,
            "last_failure_reason": _last_failure_reason,
            "prototype_latency_ms": {
                "note": "Prototype measurements from this process only. Not production throughput.",
                "insert_samples": list(_insert_ms),
                "insert_last_ms": _insert_ms[-1] if _insert_ms else None,
                "worker_samples": list(_process_ms),
                "worker_last_ms": _process_ms[-1] if _process_ms else None,
            },
        }


def reset_outbox_memory() -> None:
    global _last_success, _last_failure, _last_failure_reason
    with _lock:
        _last_success = None
        _last_failure = None
        _last_failure_reason = None
        _insert_ms.clear()
        _process_ms.clear()


def backoff_delay_seconds(attempts: int, base: float | None = None) -> float:
    settings = get_settings()
    base = base if base is not None else settings.outbox_retry_backoff_seconds
    exponent = max(0, int(attempts) - 1)
    return float(min(base * (2**exponent), 60.0))


def _is_postgres(db: AsyncSession) -> bool:
    try:
        bind = db.get_bind()
        name = getattr(getattr(bind, "dialect", None), "name", "") or ""
        if name:
            return name.startswith("postgres")
    except Exception:
        pass
    return not get_settings().is_sqlite


def _store_dt(dt: datetime) -> datetime:
    """SQLite stores naive timestamps; avoid tz-aware vs tz-naive comparisons."""
    if get_settings().is_sqlite and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


async def insert_outbox(
    db: AsyncSession,
    event: DomainEvent,
    *,
    aggregate_type: str,
    aggregate_id: str,
) -> OutboxEvent:
    """Insert an outbox row on the caller's session. Must be committed with domain state."""
    envelope = redact_secrets(event.model_dump(mode="json"))
    row = OutboxEvent(
        id=new_id(),
        event_id=event.event_id,
        event_type=event.event_type,
        schema_version=event.schema_version,
        correlation_id=event.correlation_id,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        payload=envelope,
        status=OUTBOX_PENDING,
        attempts=0,
        available_at=_store_dt(utcnow()),
        last_error=None,
    )
    db.add(row)
    await db.flush()
    log.info(
        "outbox_enqueued",
        event_id=event.event_id,
        event_type=event.event_type,
        correlation_id=event.correlation_id,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
    )
    return row


async def release_stale_processing(db: AsyncSession, stale_seconds: int | None = None) -> int:
    settings = get_settings()
    stale = stale_seconds if stale_seconds is not None else settings.outbox_stale_processing_seconds
    cutoff = _store_dt(utcnow() - timedelta(seconds=max(0, int(stale))))
    result = await db.execute(
        update(OutboxEvent)
        .where(OutboxEvent.status == OUTBOX_PROCESSING, OutboxEvent.available_at <= cutoff)
        .values(
            status=OUTBOX_PENDING,
            last_error="stale PROCESSING recovered (worker crash or timeout)",
            available_at=_store_dt(utcnow()),
        )
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount or 0)


async def claim_pending(db: AsyncSession, limit: int | None = None) -> list[dict[str, Any]]:
    """Atomically claim a batch. Caller must commit before publishing to EventBus."""
    settings = get_settings()
    batch = limit if limit is not None else settings.outbox_batch_size
    now = _store_dt(utcnow())
    recovered = await release_stale_processing(db)
    if recovered:
        log.info("outbox_stale_recovered", count=recovered)

    stmt = (
        select(OutboxEvent)
        .where(OutboxEvent.status == OUTBOX_PENDING, OutboxEvent.available_at <= now)
        .order_by(OutboxEvent.created_at.asc())
        .limit(max(1, int(batch)))
    )
    if _is_postgres(db):
        stmt = stmt.with_for_update(skip_locked=True)

    rows = list((await db.execute(stmt)).scalars().all())
    claimed: list[OutboxEvent] = []
    if _is_postgres(db):
        for row in rows:
            row.status = OUTBOX_PROCESSING
            row.available_at = now
            claimed.append(row)
        await db.flush()
    else:
        for row in rows:
            result = await db.execute(
                update(OutboxEvent)
                .where(OutboxEvent.id == row.id, OutboxEvent.status == OUTBOX_PENDING)
                .values(status=OUTBOX_PROCESSING, available_at=now)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount:
                claimed.append(row)
        await db.flush()

    snapshots = [
        {
            "id": row.id,
            "event_id": row.event_id,
            "event_type": row.event_type,
            "correlation_id": row.correlation_id,
            "payload": dict(row.payload or {}),
            "attempts": int(row.attempts or 0),
        }
        for row in claimed
    ]
    return snapshots


async def mark_published(outbox_id: str) -> None:
    async with SessionLocal() as db:
        await db.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id == outbox_id)
            .values(status=OUTBOX_PUBLISHED, published_at=utcnow(), last_error=None)
        )
        await db.commit()
    record_publish_success()


async def mark_retry_or_failed(outbox_id: str, error: str, attempts_before: int) -> str:
    settings = get_settings()
    next_attempts = attempts_before + 1
    max_attempts = max(1, int(settings.outbox_max_attempts))
    safe_error = (error or "publish failed")[:2000]
    if next_attempts >= max_attempts:
        async with SessionLocal() as db:
            await db.execute(
                update(OutboxEvent)
                .where(OutboxEvent.id == outbox_id)
                .values(
                    status=OUTBOX_FAILED,
                    attempts=next_attempts,
                    last_error=safe_error,
                    available_at=utcnow(),
                )
            )
            await db.commit()
        record_publish_failure(safe_error)
        return OUTBOX_FAILED
    delay = backoff_delay_seconds(next_attempts)
    async with SessionLocal() as db:
        await db.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id == outbox_id)
            .values(
                status=OUTBOX_PENDING,
                attempts=next_attempts,
                last_error=safe_error,
                available_at=utcnow() + timedelta(seconds=delay),
            )
        )
        await db.commit()
    record_publish_failure(safe_error)
    return OUTBOX_PENDING


async def mark_failed_permanent(outbox_id: str, error: str, attempts_before: int = 0) -> None:
    async with SessionLocal() as db:
        await db.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id == outbox_id)
            .values(
                status=OUTBOX_FAILED,
                attempts=attempts_before + 1,
                last_error=(error or "permanent failure")[:2000],
                available_at=utcnow(),
            )
        )
        await db.commit()
    record_publish_failure(error)


async def retry_failed(event_id: str) -> OutboxEvent | None:
    async with SessionLocal() as db:
        row = (
            await db.execute(select(OutboxEvent).where(OutboxEvent.event_id == event_id))
        ).scalar_one_or_none()
        if row is None:
            return None
        if row.status != OUTBOX_FAILED:
            return row
        row.status = OUTBOX_PENDING
        row.attempts = 0
        row.available_at = _store_dt(utcnow())
        await db.commit()
        await db.refresh(row)
        return row


async def outbox_aggregate_status() -> dict[str, Any]:
    settings = get_settings()
    async with SessionLocal() as db:
        rows = (
            await db.execute(
                select(OutboxEvent.status, func.count()).group_by(OutboxEvent.status)
            )
        ).all()
        counts = {OUTBOX_PENDING: 0, OUTBOX_PROCESSING: 0, OUTBOX_PUBLISHED: 0, OUTBOX_FAILED: 0}
        for status, n in rows:
            counts[str(status)] = int(n)
        oldest = (
            await db.execute(
                select(func.min(OutboxEvent.created_at)).where(OutboxEvent.status == OUTBOX_PENDING)
            )
        ).scalar_one_or_none()
    age_s = None
    if oldest is not None:
        created = oldest
        if created.tzinfo is None:
            from datetime import timezone

            created = created.replace(tzinfo=timezone.utc)
        age_s = round((utcnow() - created).total_seconds(), 3)
    mem = memory_status()
    return redact_secrets(
        {
            "enabled": settings.outbox_enabled,
            "durable_event_delivery": True,
            "pending": counts[OUTBOX_PENDING],
            "processing": counts[OUTBOX_PROCESSING],
            "published": counts[OUTBOX_PUBLISHED],
            "failed": counts[OUTBOX_FAILED],
            "oldest_pending_age_seconds": age_s,
            **mem,
            "delivery": "at-least-once",
            "prototype": True,
        }
    )
