"""Adapter over any OpenAI-compatible Chat Completions endpoint.

Works unchanged against OpenAI, Aliyun Bailian (Qwen), DeepSeek, Moonshot and
similar providers — they all speak the same protocol, so the framework only
needs to know this one dialect.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Sequence

from openai import OpenAI

from .. import config
from .base import BaseLLM, ChatResponse, Message, ToolCall


def to_openai_message(msg: Message) -> dict[str, Any]:
    """Convert an internal :class:`Message` to an OpenAI API message dict."""
    d: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.name is not None:
        d["name"] = msg.name
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    return d


def from_openai_choice(choice: Any) -> ChatResponse:
    """Convert one OpenAI choice into a normalised :class:`ChatResponse`."""
    message = choice.message
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
        for tc in (message.tool_calls or [])
    ]
    return ChatResponse(
        content=message.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
    )


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: OpenAI | None = None,
        timeout: float = 60.0,
        **client_kwargs: Any,
    ):
        self.model = model or config.get_model()
        if client is None:
            api_key = api_key or config.get_api_key()
            base_url = base_url or config.get_base_url()
            if not api_key:
                raise ValueError(
                    "api_key is required — pass it, or set the NOVA_API_KEY env var."
                )
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                **client_kwargs,
            )
        self.client = client
        # Keep the resolved settings so the async client can be built lazily.
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client_kwargs = client_kwargs
        self._async_client: Any = None

    def _request_kwargs(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None,
        temperature: float,
        stop: Sequence[str] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai_message(m) for m in messages],
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        if stop:
            kwargs["stop"] = list(stop)
        return kwargs

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> ChatResponse:
        completion = self.client.chat.completions.create(
            **self._request_kwargs(messages, tools, temperature, stop)
        )
        response = from_openai_choice(completion.choices[0])
        if completion.usage is not None:
            response.usage = completion.usage.model_dump()
        response.raw = completion
        return response

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                **self._client_kwargs,
            )
        return self._async_client

    async def achat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> ChatResponse:
        """Truly async completion via the AsyncOpenAI client."""
        if self._api_key is None:  # custom client without key — fall back to sync
            return await super().achat(messages, tools=tools, temperature=temperature, stop=stop)
        completion = await self._get_async_client().chat.completions.create(
            **self._request_kwargs(messages, tools, temperature, stop)
        )
        response = from_openai_choice(completion.choices[0])
        if completion.usage is not None:
            response.usage = completion.usage.model_dump()
        response.raw = completion
        return response

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
    ) -> Iterator[ChatResponse]:
        stream = self.client.chat.completions.create(
            stream=True,
            **self._request_kwargs(messages, tools, temperature, None),
        )

        # Tool-call fragments arrive across multiple chunks, keyed by index.
        # We accumulate them and emit the completed calls on the final chunk.
        pending: dict[int, dict[str, str]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is None:
                continue

            if delta.tool_calls:
                for part in delta.tool_calls:
                    slot = pending.setdefault(
                        part.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if part.id:
                        slot["id"] = part.id
                    fn = part.function
                    if fn:
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments

            tool_calls: list[ToolCall] = []
            if choice.finish_reason == "tool_calls":
                tool_calls = [
                    ToolCall(id=s["id"], name=s["name"], arguments=s["arguments"])
                    for s in pending.values()
                ]

            yield ChatResponse(
                content=delta.content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = config.get_embed_model() or self.model
        result = self.client.embeddings.create(model=model, input=list(texts))
        return [item.embedding for item in result.data]
