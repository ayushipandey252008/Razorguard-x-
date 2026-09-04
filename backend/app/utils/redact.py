"""Redact secrets from logs, traces, and API payloads.

Never log API keys, bearer tokens, or raw card-like payment identifiers.
"""

from __future__ import annotations

import re
from typing import Any

SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "secret",
    "password",
    "passwd",
    "token",
    "openai",
    "llm_api_key",
)

# OpenAI-style secret prefixes; never emit these in logs or responses.
_SECRET_VALUE = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._\-]+)",
    re.IGNORECASE,
)

REDACTED = "[redacted]"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(frag in lowered for frag in SECRET_KEY_FRAGMENTS)


def redact_text(value: str) -> str:
    return _SECRET_VALUE.sub(REDACTED, value)


def redact_secrets(obj: Any) -> Any:
    """Return a copy with secret keys and secret-shaped strings removed."""
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if _is_secret_key(str(key)):
                out[key] = REDACTED
            else:
                out[key] = redact_secrets(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact_secrets(item) for item in obj]
    return obj


def contains_secret(obj: Any) -> bool:
    if isinstance(obj, str):
        return bool(_SECRET_VALUE.search(obj))
    if isinstance(obj, dict):
        for key, value in obj.items():
            if _is_secret_key(str(key)) and value not in (None, "", REDACTED):
                if isinstance(value, str) and len(value) > 0:
                    return True
            if contains_secret(value):
                return True
        return False
    if isinstance(obj, (list, tuple)):
        return any(contains_secret(item) for item in obj)
    return False
