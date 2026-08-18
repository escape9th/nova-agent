"""Deterministic in-memory LLM for tests and offline development.

Useful to drive the agent loop without spending API credits, and to make the
test suite fully hermetic.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import BaseLLM, ChatResponse, Message


class MockLLM(BaseLLM):
    def __init__(self, responses: Sequence[ChatResponse] | None = None):
        self.responses: list[ChatResponse] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> ChatResponse:
        self.calls.append(
            {"messages": list(messages), "tools": list(tools) if tools else None}
        )
        if self.responses:
            return self.responses.pop(0)

        # Default behaviour: echo the last user message, useful as a smoke test.
        last = next((m for m in reversed(messages) if m.role == "user"), None)
        return ChatResponse(content=f"echo: {last.content if last else ''}")
