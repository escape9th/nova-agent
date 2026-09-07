"""Short-term memory: a sliding-window conversation buffer.

This is the "working memory" of an agent — the raw message history that gets
sent back to the model on every step, trimmed so it never grows unbounded.
"""

from __future__ import annotations

from ..llm.base import Message


class ConversationBufferMemory:
    """Keeps the most recent ``max_messages`` messages of a conversation.

    Trimming respects two invariants so the history always remains a valid
    OpenAI message sequence:

    * the leading system message is never dropped;
    * a ``tool`` result is never split from the assistant message whose
      ``tool_calls`` requested it (otherwise the ``tool_call_id`` would dangle).
    """

    def __init__(self, max_messages: int = 40):
        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def add_user(self, content: str) -> None:
        self.add(Message.user(content))

    def add_assistant(self, content: str | None, tool_calls=None) -> None:
        self.add(Message.assistant(content, tool_calls))

    def add_tool(self, content: str, tool_call_id: str, name: str | None = None) -> None:
        self.add(Message.tool(content, tool_call_id, name))

    def add_many(self, messages) -> None:
        for m in messages:
            self._messages.append(m)
        self._trim()

    def messages(self) -> list[Message]:
        """A snapshot of the current history (a copy, so callers may mutate it)."""
        return list(self._messages)

    def to_messages(self) -> list[Message]:
        return self.messages()

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def _trim(self) -> None:
        if len(self._messages) <= self.max_messages:
            return

        # The leading system message (if any) lives outside the sliding window.
        anchor = 1 if self._messages[0].role == "system" else 0
        keep = self.max_messages - anchor
        start = len(self._messages) - keep

        # Never start the window on a tool result: it must stay attached to the
        # assistant message whose tool_calls requested it. Walk back to that
        # call. The kept window may end up slightly larger than max_messages —
        # correctness over strictness.
        while start > anchor and self._messages[start].role == "tool":
            start -= 1
        start = max(start, anchor)

        self._messages = self._messages[:anchor] + self._messages[start:]
