"""Outbox poller: claim after DB commit, then publish through EventBus.

Never publishes while a domain transaction is open. Delivery is at-least-once.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import get_settings
from app.database import SessionLocal
from app.events.dlq import record_failed_event
from app.events.factory import EventBusError, event_bus_status, get_event_bus
from app.events.outbox import (
    claim_pending,
    mark_failed_permanent,
    mark_published,
    mark_retry_or_failed,
    record_process_ms,
)
from app.events.serialize import MalformedEventError, deserialize_event
from app.models.outbox import OUTBOX_FAILED
from app.utils.logging import Timer, get_logger

log = get_logger("events.outbox_worker")

_task: asyncio.Task | None = None


def _transport_accepted(result: dict[str, Any] | None) -> bool:
    if not result or not result.get("ok"):
        return False
    if result.get("fallback"):
        return False
    status = event_bus_status()
    if status.get("configured") == "kafka" and result.get("event_bus") != "kafka":
        return False
    return True


def _kafka_configured_but_down() -> bool:
    status = event_bus_status()
    return status.get("configured") == "kafka" and not status.get("kafka_connected")


async def drain_outbox_batch(limit: int | None = None) -> dict[str, Any]:
    """Publish one claimed batch. Safe to call after a domain commit or from the poller."""
    settings = get_settings()
    if not settings.outbox_enabled:
        return {"ok": True, "skipped": True, "reason": "outbox_disabled"}
    if _kafka_configured_but_down():
        log.warning(
            "outbox_skip_kafka_unavailable",
            reason="EVENT_BUS=kafka but broker is down; leaving rows PENDING",
        )
        return {"ok": False, "skipped": True, "reason": "kafka_unavailable"}

    timer = Timer()
    async with SessionLocal() as db:
        claimed = await claim_pending(db, limit=limit)
        await db.commit()

    published = 0
    retried = 0
    failed = 0
    malformed = 0
    for row in claimed:
        try:
            event = deserialize_event(row["payload"])
        except MalformedEventError as exc:
            await mark_failed_permanent(row["id"], f"malformed:{exc}", row["attempts"])
            await record_failed_event(
                event_id=row.get("event_id"),
                event_type=row.get("event_type"),
                correlation_id=row.get("correlation_id"),
                error_reason=f"malformed:{exc}",
                payload={"outbox_id": row["id"]},
                retry_count=row["attempts"] + 1,
            )
            malformed += 1
            failed += 1
            log.warning(
                "outbox_malformed",
                event_id=row.get("event_id"),
                correlation_id=row.get("correlation_id"),
                error=str(exc),
            )
            continue
        try:
            bus = get_event_bus()
            result = await bus.publish(event, allow_fallback=False)
        except EventBusError as exc:
            status = await mark_retry_or_failed(row["id"], str(exc), row["attempts"])
            retried += 1 if status != OUTBOX_FAILED else 0
            failed += 1 if status == OUTBOX_FAILED else 0
            continue
        except TypeError:
            bus = get_event_bus()
            result = await bus.publish(event)
        except Exception as exc:
            status = await mark_retry_or_failed(row["id"], type(exc).__name__, row["attempts"])
            retried += 1 if status != OUTBOX_FAILED else 0
            failed += 1 if status == OUTBOX_FAILED else 0
            log.warning(
                "outbox_publish_error",
                event_id=row.get("event_id"),
                correlation_id=row.get("correlation_id"),
                error=type(exc).__name__,
            )
            continue
        if _transport_accepted(result):
            await mark_published(row["id"])
            published += 1
            log.info(
                "outbox_published",
                event_id=event.event_id,
                event_type=event.event_type,
                correlation_id=event.correlation_id,
                event_bus=result.get("event_bus"),
            )
        else:
            reason = str(result.get("error") or result.get("kafka_error") or "transport_rejected")
            status = await mark_retry_or_failed(row["id"], reason, row["attempts"])
            retried += 1 if status != OUTBOX_FAILED else 0
            failed += 1 if status == OUTBOX_FAILED else 0
            log.warning(
                "outbox_transport_retry",
                event_id=row.get("event_id"),
                correlation_id=row.get("correlation_id"),
                error=reason,
                status=status,
            )
    elapsed = timer.ms()
    if claimed:
        record_process_ms(elapsed)
    return {
        "ok": True,
        "claimed": len(claimed),
        "published": published,
        "retried": retried,
        "failed": failed,
        "malformed": malformed,
        "latency_ms": elapsed,
    }


async def run_outbox_loop(stop: asyncio.Event | None = None) -> None:
    settings = get_settings()
    interval = max(50, int(settings.outbox_poll_interval_ms)) / 1000.0
    log.info("outbox_worker_started", poll_interval_s=interval, batch_size=settings.outbox_batch_size)
    while True:
        if stop is not None and stop.is_set():
            break
        try:
            await drain_outbox_batch()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("outbox_worker_tick_failed")
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


async def start_outbox_worker() -> asyncio.Task | None:
    global _task
    settings = get_settings()
    if not settings.outbox_enabled:
        return None
    if _task is not None and not _task.done():
        return _task
    _task = asyncio.create_task(run_outbox_loop(), name="outbox-worker")
    return _task


async def stop_outbox_worker() -> None:
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except (asyncio.CancelledError, Exception):
        pass
    _task = None
