"""Provider-independent LLM interface.

Future providers can implement LLMProvider without changing the investigator.
"""

from __future__ import annotations

from typing import Protocol

from app.config import get_settings


class LLMProvider(Protocol):
    name: str
    model: str | None
    supports_tool_calling: bool

    async def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict: ...


def configured_llm_key() -> str | None:
    settings = get_settings()
    key = (settings.openai_api_key or settings.llm_api_key or "").strip()
    return key or None


def llm_is_configured() -> bool:
    settings = get_settings()
    provider = (settings.llm_provider or "none").strip().lower()
    if provider in {"none", "", "deterministic", "deterministic_fallback", "fallback"}:
        return False
    return configured_llm_key() is not None


def get_provider() -> LLMProvider:
    from app.agents.fallback_provider import FallbackProvider
    from app.agents.openai_provider import OpenAIProvider

    if llm_is_configured():
        return OpenAIProvider()
    return FallbackProvider()
