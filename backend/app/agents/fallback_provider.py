"""Deterministic provider used when no LLM key is configured."""

from __future__ import annotations


class FallbackProvider:
    name = "deterministic_fallback"
    model = None
    supports_tool_calling = False

    async def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        raise RuntimeError("FallbackProvider does not call an LLM")
