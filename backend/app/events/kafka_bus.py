"""Optional Kafka EventBus. Application code must not import this module for publishing."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from app.config import get_settings
from app.events.bus import EventHandler
from app.events.consumers import process_event
from app.events.dlq import record_failed_event
from app.events.metrics import record_publish
from app.events.schemas import DomainEvent
from app.events.serialize import MalformedEventError, deserialize_event, serialize_event
from app.events.topics import topic_for_event, topic_names
from app.utils.logging import Timer, get_logger
from app.utils.redact import redact_secrets

log = get_logger("events.kafka")

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
except ImportError:  # pragma: no cover - optional dependency present in requirements
    AIOKafkaProducer = None  # type: ignore[misc, assignment]
    AIOKafkaConsumer = None  # type: ignore[misc, assignment]


class KafkaUnavailable(RuntimeError):
    pass


class KafkaEventBus:
    name = "kafka"

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        *,
        settings=None,
        fallback_bus=None,
        use_fallback_on_publish: bool = True,
    ) -> None:
        cfg = settings or get_settings()
        self.bootstrap_servers = bootstrap_servers or cfg.kafka_bootstrap_servers
        self._fallback_bus = fallback_bus
        self._use_fallback_on_publish = use_fallback_on_publish
        self._producer = None
        self._consumers: list[Any] = []
        self._tasks: list[asyncio.Task] = []
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self.connected = False
        self._connect_error: str | None = None
        self.publish_timeout_s = max(0.2, cfg.kafka_publish_timeout_ms / 1000.0)
        self.connect_timeout_s = max(1, int(cfg.kafka_connect_timeout_seconds))
        self._settings = cfg

    async def connect(self) -> None:
        if AIOKafkaProducer is None:
            self._connect_error = "aiokafka not installed"
            raise KafkaUnavailable(self._connect_error)
        cfg = self._settings
        request_timeout_ms = max(int(cfg.kafka_publish_timeout_ms), 15000)
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            client_id="razorguard-x-producer",
            request_timeout_ms=request_timeout_ms,
            retry_backoff_ms=200,
            acks=1,
            linger_ms=5,
        )
        try:
            await asyncio.wait_for(producer.start(), timeout=self.connect_timeout_s)
        except Exception as exc:
            self._connect_error = type(exc).__name__
            try:
                await producer.stop()
            except Exception:
                pass
            raise KafkaUnavailable(f"Kafka connection unavailable: {type(exc).__name__}") from exc
        self._producer = producer
        self.connected = True
        self._connect_error = None
        await self._ensure_topics()
        log.info(
            "event_bus_connected",
            event_bus="kafka",
            bootstrap_servers=self.bootstrap_servers,
        )

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for consumer in self._consumers:
            try:
                await consumer.stop()
            except Exception:
                pass
        self._consumers.clear()
        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception:
                pass
        self._producer = None
        self.connected = False

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: DomainEvent, **kwargs) -> dict[str, Any]:
        timer = Timer()
        topic = topic_for_event(event.event_type)
        body = serialize_event(event)
        allow_fallback = kwargs.get("allow_fallback", True)
        if self._producer is None or not self.connected:
            return await self._fail_publish(
                event, timer, "kafka_not_connected", topic, allow_fallback=allow_fallback
            )
        try:
            await asyncio.wait_for(
                self._producer.send_and_wait(topic, body, key=event.event_id.encode("utf-8")),
                timeout=self.publish_timeout_s,
            )
        except Exception as exc:
            return await self._fail_publish(
                event, timer, type(exc).__name__, topic, allow_fallback=allow_fallback
            )
        latency_ms = timer.ms()
        record_publish(event, latency_ms)
        log.info(
            "event_published",
            event_bus="kafka",
            event_id=event.event_id,
            event_type=event.event_type,
            correlation_id=event.correlation_id,
            transaction_id=event.transaction_id,
            topic=topic,
            latency_ms=latency_ms,
        )
        return {"ok": True, "event_bus": "kafka", "topic": topic, "latency_ms": latency_ms}

    async def _fail_publish(
        self,
        event: DomainEvent,
        timer: Timer,
        reason: str,
        topic: str,
        *,
        allow_fallback: bool = True,
    ) -> dict[str, Any]:
        latency_ms = timer.ms()
        log.warning(
            "kafka_publish_failed",
            event_id=event.event_id,
            event_type=event.event_type,
            correlation_id=event.correlation_id,
            topic=topic,
            error=reason,
            latency_ms=latency_ms,
        )
        await record_failed_event(
            event_id=event.event_id,
            event_type=event.event_type,
            correlation_id=event.correlation_id,
            error_reason=f"kafka_publish:{reason}",
            payload={"topic": topic, "transaction_id": event.transaction_id},
        )
        if allow_fallback and self._use_fallback_on_publish and self._fallback_bus is not None:
            result = await self._fallback_bus.publish(event)
            result["fallback"] = True
            result["kafka_error"] = reason
            return result
        record_publish(event, latency_ms)
        return {"ok": False, "event_bus": "kafka", "error": reason, "latency_ms": latency_ms}

    async def publish_dlq(self, raw: bytes, reason: str, event_id: str | None = None) -> None:
        if self._producer is None:
            return
        topic = topic_names()["dlq"]
        preview = (raw or b"")[:500].decode("utf-8", errors="replace")
        envelope = redact_secrets(
            {
                "reason": reason,
                "event_id": event_id,
                "raw_size": len(raw or b""),
                "raw_preview": preview,
            }
        )
        try:
            await asyncio.wait_for(
                self._producer.send_and_wait(topic, serialize_event_dict(envelope)),
                timeout=self.publish_timeout_s,
            )
        except Exception as exc:
            log.warning("dlq_publish_failed", error=type(exc).__name__, reason=reason)

    async def _ensure_topics(self) -> None:
        """Create configured topics so the first produce is not racing auto-create."""
        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic
        except ImportError:  # pragma: no cover
            return
        cfg = self._settings
        names = [
            cfg.kafka_topic_transactions,
            cfg.kafka_topic_risk_results,
            cfg.kafka_topic_investigations,
            cfg.kafka_topic_alerts,
            cfg.kafka_topic_feedback,
            cfg.kafka_topic_dlq,
        ]
        admin = AIOKafkaAdminClient(
            bootstrap_servers=self.bootstrap_servers,
            client_id="razorguard-x-admin",
            request_timeout_ms=max(int(self._settings.kafka_publish_timeout_ms), 15000),
        )
        try:
            await asyncio.wait_for(admin.start(), timeout=self.connect_timeout_s)
            existing = set()
            try:
                existing = set(await admin.list_topics())
            except Exception:
                existing = set()
            missing = [name for name in names if name not in existing]
            if missing:
                topics = [NewTopic(name=name, num_partitions=1, replication_factor=1) for name in missing]
                try:
                    await admin.create_topics(topics)
                except Exception as exc:
                    log.warning("kafka_topic_create_failed", error=type(exc).__name__, topics=missing)
            log.info("kafka_topics_ready", topics=names)
        except Exception as exc:
            log.warning("kafka_topic_ensure_failed", error=type(exc).__name__)
        finally:
            try:
                await admin.close()
            except Exception:
                pass

    async def start_consumers(self) -> None:
        if not self.connected or AIOKafkaConsumer is None:
            return
        cfg = self._settings
        specs = [
            ("transactions", cfg.kafka_topic_transactions, getattr(cfg, "kafka_group_transactions", "rgx-transactions")),
            ("risk", cfg.kafka_topic_risk_results, cfg.kafka_group_risk),
            ("investigations", cfg.kafka_topic_investigations, cfg.kafka_group_investigations),
            ("alerts", cfg.kafka_topic_alerts, cfg.kafka_group_alerts),
            ("feedback", cfg.kafka_topic_feedback, cfg.kafka_group_feedback),
        ]
        consumer_timeout_ms = max(int(cfg.kafka_publish_timeout_ms) + 5000, 20000)
        for label, topic, group in specs:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group,
                enable_auto_commit=False,
                auto_offset_reset="earliest",
                client_id=f"razorguard-x-{label}",
                request_timeout_ms=consumer_timeout_ms,
            )
            try:
                await asyncio.wait_for(consumer.start(), timeout=self.connect_timeout_s)
            except Exception as exc:
                log.warning("kafka_consumer_start_failed", consumer=label, error=type(exc).__name__)
                try:
                    await consumer.stop()
                except Exception:
                    pass
                continue
            self._consumers.append(consumer)
            self._tasks.append(asyncio.create_task(self._consume_loop(label, consumer), name=f"kafka-{label}"))
            log.info("kafka_consumer_started", consumer=label, topic=topic, group=group)

    async def _consume_loop(self, label: str, consumer) -> None:
        try:
            async for msg in consumer:
                await self._handle_message(label, msg)
                try:
                    await consumer.commit()
                except Exception as exc:
                    log.warning("kafka_commit_failed", consumer=label, error=type(exc).__name__)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("kafka_consumer_stopped", consumer=label)

    async def _handle_message(self, label: str, msg) -> None:
        try:
            event = deserialize_event(msg.value)
        except MalformedEventError as exc:
            preview = (msg.value or b"")[:200].decode("utf-8", errors="replace")
            await record_failed_event(
                event_id=None,
                event_type=None,
                error_reason=f"malformed:{exc}",
                payload={
                    "consumer": label,
                    "topic": getattr(msg, "topic", None),
                    "raw_preview": preview,
                },
            )
            await self.publish_dlq(msg.value or b"", f"malformed:{exc}")
            log.warning(
                "malformed_event",
                consumer=label,
                topic=getattr(msg, "topic", None),
                error=str(exc),
            )
            return
        log.info(
            "kafka_event_received",
            consumer=label,
            event_id=event.event_id,
            event_type=event.event_type,
            correlation_id=event.correlation_id,
            transaction_id=event.transaction_id,
        )
        result = await process_event(event)
        extra_handlers = list(self._handlers.get(event.event_type, []))
        for handler in extra_handlers:
            try:
                await handler(event)
            except Exception as exc:
                log.warning(
                    "kafka_extra_handler_failed",
                    event_id=event.event_id,
                    correlation_id=event.correlation_id,
                    error=type(exc).__name__,
                )
        if not result.get("ok"):
            await self.publish_dlq(msg.value or b"", str(result.get("error")), event.event_id)

    def status(self) -> dict[str, Any]:
        return {
            "name": "kafka",
            "connected": self.connected,
            "bootstrap_servers": self.bootstrap_servers,
            "connect_error": self._connect_error,
            "consumers": len(self._consumers),
        }


def serialize_event_dict(data: dict[str, Any]) -> bytes:
    import json

    return json.dumps(redact_secrets(data), default=str).encode("utf-8")
