"""JSON serialization for domain events. Independent of Kafka clients."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.events.schemas import DomainEvent, parse_event
from app.utils.redact import redact_secrets


class MalformedEventError(ValueError):
    pass


def serialize_event(event: DomainEvent) -> bytes:
    payload = redact_secrets(event.model_dump(mode="json"))
    return json.dumps(payload, default=str).encode("utf-8")


def deserialize_event(raw: bytes | str | dict[str, Any]) -> DomainEvent:
    try:
        if isinstance(raw, dict):
            data = raw
        else:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            data = json.loads(text)
        if not isinstance(data, dict):
            raise MalformedEventError("event JSON must be an object")
        return parse_event(data)
    except MalformedEventError:
        raise
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise MalformedEventError(f"malformed event: {type(exc).__name__}") from exc
