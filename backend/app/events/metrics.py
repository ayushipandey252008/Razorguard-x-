"""In-process counters and recent-event ring buffer for the prototype status API."""

from __future__ import annotations

from collections import Counter, deque
from threading import Lock
from typing import Any

from app.events.schemas import DomainEvent
from app.utils.redact import redact_secrets

_lock = Lock()
_counts: Counter[str] = Counter()
_failed: Counter[str] = Counter()
_duplicates: int = 0
_publish_ms: deque[float] = deque(maxlen=50)
_consume_ms: deque[float] = deque(maxlen=50)
_recent: deque[dict[str, Any]] = deque(maxlen=40)


def record_publish(event: DomainEvent, latency_ms: float) -> None:
    with _lock:
        _counts[event.event_type] += 1
        _publish_ms.append(latency_ms)
        _recent.appendleft(
            redact_secrets(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "correlation_id": event.correlation_id,
                    "transaction_id": event.transaction_id,
                    "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                    "schema_version": event.schema_version,
                }
            )
        )


def record_consume(latency_ms: float) -> None:
    with _lock:
        _consume_ms.append(latency_ms)


def record_duplicate() -> None:
    global _duplicates
    with _lock:
        _duplicates += 1


def record_failed(event_type: str | None = None) -> None:
    with _lock:
        _failed[event_type or "unknown"] += 1


def snapshot() -> dict[str, Any]:
    with _lock:
        publish = list(_publish_ms)
        consume = list(_consume_ms)
        return {
            "event_counts": dict(_counts),
            "failed_counts": dict(_failed),
            "duplicate_skips": _duplicates,
            "recent_events": list(_recent),
            "prototype_latency_ms": {
                "note": "Prototype measurements from this process only. Not production throughput.",
                "publish_samples": publish,
                "publish_last_ms": publish[-1] if publish else None,
                "consume_samples": consume,
                "consume_last_ms": consume[-1] if consume else None,
            },
        }


def reset_metrics() -> None:
    global _duplicates
    with _lock:
        _counts.clear()
        _failed.clear()
        _duplicates = 0
        _publish_ms.clear()
        _consume_ms.clear()
        _recent.clear()
