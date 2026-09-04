"""Optional Kafka / in-process domain events. Not the WebSocket live-wire bus."""

from app.events.bus import EventBus
from app.events.factory import event_bus_status, get_event_bus
from app.events.schemas import DomainEvent

__all__ = ["EventBus", "DomainEvent", "event_bus_status", "get_event_bus"]
