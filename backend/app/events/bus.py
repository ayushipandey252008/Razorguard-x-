"""EventBus protocol. Application code depends on this, not Kafka clients."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from app.events.schemas import DomainEvent

EventHandler = Callable[[DomainEvent], Awaitable[None]]


@runtime_checkable
class EventBus(Protocol):
    name: str

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def publish(self, event: DomainEvent, **kwargs: Any) -> dict[str, Any]:
        """Publish without blocking the caller indefinitely.

        Implementations must not raise on transport failure. They return a
        small status dict (ok, latency_ms, fallback, error).
        `allow_fallback` may be passed by the outbox worker; in-process ignores it.
        """

    def subscribe(self, event_type: str, handler: EventHandler) -> None: ...

    def status(self) -> dict[str, Any]: ...
