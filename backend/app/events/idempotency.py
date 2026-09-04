"""Claim event_id once. Duplicate deliveries must not create side effects."""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal
from app.models.eventing import ProcessedEvent
from app.utils.logging import get_logger

log = get_logger("events.idempotency")

# Process-local guard used when SQL is unavailable (unit tests without lifespan).
_memory_seen: set[str] = set()


def reset_memory_idempotency() -> None:
    _memory_seen.clear()


async def claim_event(event_id: str, event_type: str, correlation_id: str | None = None) -> bool:
    """Return True if this event should be processed; False if it is a duplicate."""
    if not event_id:
        return True
    if event_id in _memory_seen:
        log.info(
            "duplicate_event_skipped",
            event_id=event_id,
            event_type=event_type,
            correlation_id=correlation_id,
        )
        return False
    try:
        async with SessionLocal() as db:
            db.add(
                ProcessedEvent(
                    event_id=event_id,
                    event_type=event_type,
                    correlation_id=correlation_id,
                )
            )
            await db.commit()
    except IntegrityError:
        log.info(
            "duplicate_event_skipped",
            event_id=event_id,
            event_type=event_type,
            correlation_id=correlation_id,
        )
        return False
    except Exception as exc:
        # Unit tests without a mapped table still get in-memory uniqueness.
        log.warning("idempotency_sql_unavailable", error=type(exc).__name__)
        if event_id in _memory_seen:
            return False
    _memory_seen.add(event_id)
    return True
