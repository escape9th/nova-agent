"""Agent interface and shared result/event types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class AgentEvent:
    """One observable step of an agent run.

    Common ``type`` values: ``tool_call``, ``tool_result``, ``answer``,
    ``error``. Higher-level agents may emit additional types (``plan``,
    ``step``) — consumers should treat unknown types as informational.
    """

    type: str
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """The final outcome of a run: the answer plus a replayable event trace."""

    answer: str
    events: list[AgentEvent] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0


class BaseAgent(ABC):
    """Minimal contract every agent implements."""

    name: str = "agent"

    @abstractmethod
    def run(self, task: str) -> AgentResult:
        """Execute a task and return the final answer with an event trace."""

    def run_stream(self, task: str) -> Iterator[AgentEvent]:
        """Yield events live as the agent works.

        The default implementation runs to completion and replays the trace;
        agents that can emit events incrementally should override this.
        """
        yield from self.run(task).events

    async def arun(self, task: str) -> AgentResult:
        """Async variant of :meth:`run`.

        The default runs :meth:`run` in a worker thread; agents that can do true
        async I/O should override it.
        """
        import asyncio

        return await asyncio.to_thread(self.run, task)
