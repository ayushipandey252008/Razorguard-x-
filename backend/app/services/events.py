from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("events")


class EventBus:
    """Redis pub/sub when available, in-process fan-out otherwise."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._redis = None
        self.backend = "memory"

    async def connect(self) -> None:
        settings = get_settings()
        if not settings.redis_url:
            return
        try:
            import redis.asyncio as redis

            client = redis.from_url(settings.redis_url, decode_responses=True)
            await client.ping()
            self._redis = client
            self.backend = "redis"
            log.info("redis_connected")
        except Exception as exc:
            log.warning("redis_unavailable", error=str(exc))
            self._redis = None
            self.backend = "memory"

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers[channel].append(q)
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        if q in self._subscribers.get(channel, []):
            self._subscribers[channel].remove(q)

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, default=str)
        if self._redis is not None:
            try:
                await self._redis.publish(channel, message)
            except Exception as exc:
                log.warning("redis_publish_failed", error=str(exc))
        for q in list(self._subscribers.get(channel, [])):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    pass


event_bus = EventBus()
TXN_CHANNEL = "rgx.transactions"
