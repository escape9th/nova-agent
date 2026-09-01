"""Multi-agent orchestration via a supervisor.

The supervisor is driven by the *same* tool-calling mechanism as a single
agent — its only tool is ``delegate``, which runs a named member agent on a
subtask. This is deliberate: it shows that "calling a tool" and "calling another
agent" are the same shape, and keeps the framework conceptually small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..llm.base import BaseLLM, Message
from ..prompts.templates import TEAM_SYSTEM
from ..tools.base import Tool
from ..tools.registry import ToolRegistry
from .base import AgentEvent, AgentResult, BaseAgent


@dataclass
class TeamMember:
    name: str
    role: str  # natural-language description of what this member is good at
    agent: BaseAgent


class Team(BaseAgent):
    name = "team"

    def __init__(
        self,
        members: Sequence[TeamMember],
        llm: BaseLLM,
        system_prompt: str | None = None,
        max_rounds: int = 10,
        temperature: float = 0.0,
    ):
        self.members: dict[str, TeamMember] = {m.name: m for m in members}
        self.llm = llm
        self.system_prompt = system_prompt or TEAM_SYSTEM
        self.max_rounds = max_rounds
        self.temperature = temperature

    def _delegate(self, member: str, task: str) -> str:
        target = self.members.get(member)
        if target is None:
            return f"Error: unknown member '{member}'. Available: {', '.join(self.members)}"
        return target.agent.run(task).answer

    def run(self, task: str) -> AgentResult:
        roster = "\n".join(f"- {m.name}: {m.role}" for m in self.members.values())
        system = f"{self.system_prompt}\n\nAvailable specialists:\n{roster}"

        delegate = Tool(
            name="delegate",
            description="Delegate a subtask to a named specialist and get their answer.",
            parameters={
                "type": "object",
                "properties": {
                    "member": {"type": "string", "description": "Name of the specialist to delegate to."},
                    "task": {"type": "string", "description": "The subtask to give them."},
                },
                "required": ["member", "task"],
            },
            func=self._delegate,
        )
        registry = ToolRegistry([delegate])

        messages = [Message.system(system), Message.user(task)]
        events: list[AgentEvent] = []
        tool_calls = 0

        for i in range(self.max_rounds):
            response = self.llm.chat(messages, tools=registry.specs(), temperature=self.temperature)
            if response.tool_calls:
                messages.append(Message.assistant(response.content, response.tool_calls))
                events.append(
                    AgentEvent(
                        type="tool_call",
                        content=response.content or "",
                        data={
                            "iterations": i + 1,
                            "calls": [
                                {"name": tc.name, "arguments": tc.arguments}
                                for tc in response.tool_calls
                            ],
                        },
                    )
                )
                for tc in response.tool_calls:
                    result = registry.execute(tc.name, tc.arguments)
                    tool_calls += 1
                    messages.append(Message.tool(result, tc.id, tc.name))
                    events.append(
                        AgentEvent(type="tool_result", content=result, data={"name": tc.name})
                    )
            else:
                content = response.content or ""
                events.append(AgentEvent(type="answer", content=content, data={"iterations": i + 1}))
                return AgentResult(answer=content, events=events, iterations=i + 1, tool_calls=tool_calls)

        message = f"supervisor stopped after {self.max_rounds} rounds without a final answer"
        events.append(AgentEvent(type="error", content=message))
        return AgentResult(answer=message, events=events, iterations=self.max_rounds, tool_calls=tool_calls)
