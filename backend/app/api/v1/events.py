"""Event bus and outbox status for operators. No credentials, no payload dumps."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_current_user, require_roles
from app.events.factory import event_bus_status
from app.events.outbox import outbox_aggregate_status, retry_failed
from app.events.outbox_worker import drain_outbox_batch
from app.models.app_user import AppUser
from app.utils.redact import redact_secrets

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/status")
async def events_status(user: AppUser = Depends(get_current_user)):
    status = event_bus_status()
    outbox = await outbox_aggregate_status()
    return redact_secrets(
        {
            "configured": status.get("configured"),
            "active": status.get("active"),
            "fallback": status.get("fallback"),
            "kafka_connected": status.get("kafka_connected"),
            "reason": status.get("reason"),
            "topics": status.get("topics"),
            "topic_documentation": status.get("topic_documentation"),
            "event_counts": status.get("event_counts"),
            "failed_counts": status.get("failed_counts"),
            "duplicate_skips": status.get("duplicate_skips"),
            "recent_events": status.get("recent_events"),
            "prototype_latency_ms": status.get("prototype_latency_ms"),
            "transport": status.get("transport"),
            "outbox": {
                "enabled": outbox.get("enabled"),
                "durable_event_delivery": True,
                "pending": outbox.get("pending"),
                "processing": outbox.get("processing"),
                "published": outbox.get("published"),
                "failed": outbox.get("failed"),
                "oldest_pending_age_seconds": outbox.get("oldest_pending_age_seconds"),
                "last_successful_publish": outbox.get("last_successful_publish"),
                "last_failure": outbox.get("last_failure"),
            },
            "prototype": True,
            "note": "Optional Kafka transport. Synchronous risk decisions do not wait on the broker.",
        }
    )


@router.get("/outbox/status")
async def outbox_status(user: AppUser = Depends(get_current_user)):
    return await outbox_aggregate_status()


@router.post("/outbox/drain")
async def drain_outbox(user: AppUser = Depends(require_roles("ADMIN"))):
    """Admin/dev: publish pending outbox rows now. Does not open a domain transaction."""
    return await drain_outbox_batch()


@router.post("/outbox/{event_id}/retry")
async def retry_outbox_event(
    event_id: str,
    user: AppUser = Depends(require_roles("ADMIN")),
):
    row = await retry_failed(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Outbox event not found")
    if row.status != "PENDING":
        raise HTTPException(
            status_code=409,
            detail="Only FAILED outbox events can be retried",
        )
    await drain_outbox_batch()
    return redact_secrets(
        {
            "ok": True,
            "event_id": row.event_id,
            "status": row.status,
            "attempts": row.attempts,
        }
    )
