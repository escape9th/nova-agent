"""Provider-agnostic LLM interface.

Only ``chat`` is required; ``stream`` and ``embed`` have sensible defaults so a
new provider (or a test double) is cheap to add.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


@dataclass
class ToolCall:
    """A single function call requested by the model."""

    id: str
    name: str
    arguments: str  # JSON-encoded string, kept as-is for protocol fidelity

    def arguments_dict(self) -> dict[str, Any]:
        return json.loads(self.arguments or "{}")


@dataclass
class Message:
    """A chat message in any of the standard roles.

    ``tool_calls`` is only meaningful on assistant messages; ``tool_call_id``
    only on tool messages.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ChatResponse:
    """Normalised completion result, independent of any provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # provider-native object, for debugging

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class BaseLLM(ABC):
    """Interface every provider adapter must implement."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        response_format: dict | None = None,
    ) -> ChatResponse:
        """Return a single completion for the given conversation history.

        ``tools`` is a list of OpenAI-style tool specifications. When provided,
        the model may return ``tool_calls`` instead of (or alongside) text.
        """

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
    ) -> Iterator[ChatResponse]:
        """Yield completion chunks.

        Text is yielded incrementally; tool calls (if any) are accumulated and
        yielded in a final chunk with ``finish_reason == "tool_calls"``.
        The default implementation degrades to a single non-streaming chunk.
        """
        yield self.chat(messages, tools=tools, temperature=temperature)

    async def achat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        response_format: dict | None = None,
    ) -> ChatResponse:
        """Async variant of :meth:`chat`.

        The default runs the synchronous :meth:`chat` in a worker thread, so any
        adapter works out of the box. Adapters may override it for true async I/O.
        """
        import asyncio

        return await asyncio.to_thread(
            self.chat,
            messages,
            tools=tools,
            temperature=temperature,
            stop=stop,
            response_format=response_format,
        )

    def json(self, messages: Sequence[Message], temperature: float = 0.0) -> dict:
        """Ask the model for a JSON object and parse it into a dict.

        Uses JSON mode (``response_format={"type": "json_object"}``). The caller
        should make the prompt ask for JSON explicitly, as most providers require
        that for JSON mode.
        """
        import json

        response = self.chat(
            messages, response_format={"type": "json_object"}, temperature=temperature
        )
        return json.loads(response.content or "{}")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return an embedding vector per input text.

        Optional — providers without an embeddings endpoint may raise
        ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support embeddings")
