"""Select in-process or Kafka EventBus. Kafka is optional and may fall back."""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.events.consumers import register_default_handlers
from app.events.inprocess_bus import InProcessEventBus
from app.events.metrics import snapshot as metrics_snapshot
from app.events.topics import topic_documentation, topic_names
from app.utils.logging import get_logger
from app.utils.redact import redact_secrets

log = get_logger("events.factory")

_bus = None
_status: dict[str, Any] = {
    "configured": "inprocess",
    "active": "inprocess",
    "fallback": False,
    "kafka_connected": False,
    "reason": None,
}


class EventBusError(RuntimeError):
    pass


def event_bus_status() -> dict[str, Any]:
    extra = _bus.status() if _bus is not None else {}
    return redact_secrets(
        {
            **_status,
            **metrics_snapshot(),
            "topics": topic_names(),
            "topic_documentation": topic_documentation(),
            "transport": extra,
        }
    )


async def connect_event_bus(settings=None, *, start_consumers: bool | None = None):
    global _bus, _status
    await close_event_bus()
    settings = settings or get_settings()
    configured = (settings.event_bus or "inprocess").strip().lower()
    if configured in {"", "inprocess", "memory", "local"}:
        bus = InProcessEventBus()
        await bus.connect()
        register_default_handlers(bus)
        _bus = bus
        _status = {
            "configured": "inprocess",
            "active": "inprocess",
            "fallback": False,
            "kafka_connected": False,
            "reason": None,
        }
        log.info("event_bus_selected", configured="inprocess", active="inprocess")
        return bus

    if configured != "kafka":
        raise EventBusError(f"Unknown EVENT_BUS '{configured}'. Use inprocess or kafka.")

    from app.events.kafka_bus import KafkaEventBus, KafkaUnavailable

    fallback = InProcessEventBus()
    await fallback.connect()
    register_default_handlers(fallback)
    kafka_bus = KafkaEventBus(
        settings.kafka_bootstrap_servers,
        settings=settings,
        fallback_bus=fallback if settings.event_bus_fallback else None,
        use_fallback_on_publish=settings.event_bus_fallback,
    )
    try:
        await kafka_bus.connect()
    except KafkaUnavailable as exc:
        if settings.event_bus_fallback:
            log.warning(
                "kafka_unavailable_fallback_inprocess",
                reason="connection unavailable",
                error=type(exc).__name__,
            )
            _bus = fallback
            _status = {
                "configured": "kafka",
                "active": "inprocess",
                "fallback": True,
                "kafka_connected": False,
                "reason": "connection unavailable",
            }
            return fallback
        _status = {
            "configured": "kafka",
            "active": "kafka",
            "fallback": False,
            "kafka_connected": False,
            "reason": "connection unavailable",
        }
        raise EventBusError(
            "EVENT_BUS=kafka but Kafka is unavailable. "
            "Set EVENT_BUS_FALLBACK=true for local in-process fallback, "
            "or start Kafka (see docs/event-driven-architecture.md)."
        ) from exc

    try:
        should_start = settings.event_consumer_in_api if start_consumers is None else start_consumers
        if should_start:
            await kafka_bus.start_consumers()
    except Exception as exc:
        log.warning("kafka_consumers_unavailable", error=type(exc).__name__)
    _bus = kafka_bus
    _status = {
        "configured": "kafka",
        "active": "kafka",
        "fallback": False,
        "kafka_connected": True,
        "reason": None,
    }
    log.info("event_bus_selected", configured="kafka", active="kafka")
    return kafka_bus


def get_event_bus():
    if _bus is None:
        raise EventBusError("Event bus is not connected")
    return _bus


async def close_event_bus() -> None:
    global _bus
    if _bus is None:
        return
    try:
        await _bus.close()
    except Exception:
        pass
    _bus = None


async def reset_event_bus_for_tests() -> None:
    from app.events.consumers import reset_consumer_memory
    from app.events.metrics import reset_metrics
    from app.events.outbox import reset_outbox_memory

    await close_event_bus()
    reset_metrics()
    reset_consumer_memory()
    reset_outbox_memory()
