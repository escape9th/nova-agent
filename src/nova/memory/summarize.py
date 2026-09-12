"""Long-conversation memory with automatic summarization.

A plain sliding-window buffer *drops* old messages entirely. This class instead
compresses the oldest turns into a running summary using an LLM, so the model
keeps the gist of a long conversation without blowing up the context window.
"""

from __future__ import annotations

from ..llm.base import BaseLLM, Message
from ..prompts.templates import SUMMARIZE_SYSTEM


class SummarizationMemory:
    """Keeps a bounded history by summarising old turns.

    When the raw message count exceeds ``trigger``, the messages before the most
    recent ``keep_recent`` are collapsed into a summary (via an LLM call). The
    summary is injected back into the context as a system note, so the model
    still "remembers" what happened earlier without the full transcript.
    """

    def __init__(
        self,
        llm: BaseLLM,
        trigger: int = 24,
        keep_recent: int = 12,
        summary_prompt: str = SUMMARIZE_SYSTEM,
    ):
        if keep_recent >= trigger:
            raise ValueError("keep_recent must be smaller than trigger")
        self.llm = llm
        self.trigger = trigger
        self.keep_recent = keep_recent
        self.summary_prompt = summary_prompt
        self._summary: str = ""
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self.trigger:
            self._compress()

    def add_user(self, content: str) -> None:
        self.add(Message.user(content))

    def add_assistant(self, content: str | None, tool_calls=None) -> None:
        self.add(Message.assistant(content, tool_calls))

    def add_tool(self, content: str, tool_call_id: str, name: str | None = None) -> None:
        self.add(Message.tool(content, tool_call_id, name))

    def _compress(self) -> None:
        anchor = 1 if (self._messages and self._messages[0].role == "system") else 0
        # Never summarise away the system message, and never touch the most
        # recent keep_recent messages — those are the "working memory".
        old = self._messages[anchor : len(self._messages) - self.keep_recent]
        recent = self._messages[len(self._messages) - self.keep_recent :]
        if not old:
            return
        self._summary = self._summarize(old)
        self._messages = self._messages[:anchor] + recent

    def _summarize(self, old: list[Message]) -> str:
        transcript = "\n".join(f"{m.role}: {m.content or ''}" for m in old)
        response = self.llm.chat(
            [
                Message.system(self.summary_prompt),
                Message.user(f"Transcript of earlier conversation:\n\n{transcript}"),
            ],
            temperature=0.0,
        )
        return response.content or ""

    def messages(self) -> list[Message]:
        """Current history, with the running summary injected as context."""
        result = list(self._messages)
        if self._summary:
            insert_at = 1 if (result and result[0].role == "system") else 0
            result.insert(
                insert_at,
                Message.system(f"[Earlier conversation summary]\n{self._summary}"),
            )
        return result

    def to_messages(self) -> list[Message]:
        return self.messages()

    def clear(self) -> None:
        self._summary = ""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
