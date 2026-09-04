"""Shared envelope fields, correlation IDs, and payload sanitization.

Application code should not put secrets, PANs, or payment identifiers into events.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.utils.ids import new_id
from app.utils.redact import SECRET_KEY_FRAGMENTS, redact_secrets

SCHEMA_VERSION = "1"

# Extra keys that must never appear in event payloads (beyond redact_secrets).
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "payment_identifier",
        "pan",
        "cvv",
        "cvc",
        "card_number",
        "cardnumber",
        "account_number",
        "iban",
        "pin",
        "otp",
        "llm_api_key",
        "openai_api_key",
        "secret_key",
        "password",
        "passwd",
        "authorization",
        "bearer",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
    }
)


def current_correlation_id() -> str:
    ctx = structlog.contextvars.get_contextvars()
    return str(ctx.get("correlation_id") or ctx.get("request_id") or new_id())


def bind_correlation_id(correlation_id: str) -> None:
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def _is_forbidden_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in FORBIDDEN_PAYLOAD_KEYS:
        return True
    return any(frag in lowered for frag in SECRET_KEY_FRAGMENTS)


def sanitize_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Drop secret-shaped keys and redact remaining string values."""
    if not payload:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_forbidden_key(str(key)):
            continue
        cleaned[str(key)] = value
    return redact_secrets(cleaned)
