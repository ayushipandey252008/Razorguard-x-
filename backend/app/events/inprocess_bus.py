"""In-memory EventBus for local development and tests. No Kafka process required."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.events.bus import EventHandler
from app.events.metrics import record_publish
from app.events.schemas import DomainEvent
from app.utils.logging import Timer, get_logger

log = get_logger("events.inprocess")


class InProcessEventBus:
    name = "inprocess"

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._connected = False
        self.published: list[DomainEvent] = []

    async def connect(self) -> None:
        self._connected = True
        log.info("event_bus_connected", event_bus="inprocess")

    async def close(self) -> None:
        self._connected = False
        self._handlers.clear()

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: DomainEvent, **kwargs) -> dict[str, Any]:
        timer = Timer()
        self.published.append(event)
        handlers = list(self._handlers.get(event.event_type, [])) + list(self._handlers.get("*", []))
        errors: list[str] = []
        for handler in handlers:
            try:
                await handler(event)
            except Exception as exc:
                errors.append(f"{handler.__name__}:{type(exc).__name__}")
                log.warning(
                    "inprocess_handler_failed",
                    event_id=event.event_id,
                    event_type=event.event_type,
                    correlation_id=event.correlation_id,
                    handler=getattr(handler, "__name__", "handler"),
                    error=type(exc).__name__,
                )
        latency_ms = timer.ms()
        record_publish(event, latency_ms)
        log.info(
            "event_published",
            event_bus="inprocess",
            event_id=event.event_id,
            event_type=event.event_type,
            correlation_id=event.correlation_id,
            transaction_id=event.transaction_id,
            latency_ms=latency_ms,
        )
        return {"ok": True, "event_bus": "inprocess", "latency_ms": latency_ms, "errors": errors}

    def status(self) -> dict[str, Any]:
        return {
            "name": "inprocess",
            "connected": self._connected,
            "handler_types": sorted(self._handlers.keys()),
        }
