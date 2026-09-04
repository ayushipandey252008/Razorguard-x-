"""OpenAI-compatible Chat Completions provider with tool calling."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from app.agents.provider import configured_llm_key
from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.redact import redact_text

log = get_logger("llm")

REMOTE_TIMEOUT_SECONDS = 45.0
LOCAL_TIMEOUT_SECONDS = 180.0
LOCAL_MAX_TOKENS = 768


class OpenAIProvider:
    """OpenAI Chat Completions (or a compatible base URL)."""

    name = "llm"
    supports_tool_calling = True

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = configured_llm_key()
        self.model = settings.llm_model
        self.base_url = (settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")

    def _is_loopback(self) -> bool:
        host = urlparse(self.base_url).hostname
        return host in {"127.0.0.1", "localhost", "::1"}

    def _request_timeout(self) -> float:
        if self._is_loopback():
            return LOCAL_TIMEOUT_SECONDS
        return REMOTE_TIMEOUT_SECONDS

    async def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        if not self._api_key:
            raise RuntimeError("OpenAI provider has no API key configured")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.1,
        }
        if self._is_loopback():
            body["max_tokens"] = LOCAL_MAX_TOKENS
        try:
            async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                choice = resp.json()["choices"][0]["message"]
                return choice
        except httpx.HTTPStatusError as exc:
            # Do not include request headers (they hold the API key).
            status = exc.response.status_code if exc.response is not None else "unknown"
            log.warning("llm_http_error", status=status, model=self.model)
            raise RuntimeError(f"LLM HTTP error status={status}") from None
        except httpx.HTTPError as exc:
            log.warning("llm_transport_error", error=redact_text(type(exc).__name__))
            raise RuntimeError("LLM transport error") from None
