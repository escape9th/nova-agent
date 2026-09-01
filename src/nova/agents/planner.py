"""Plan-and-execute: plan first, then execute each step, then synthesize.

This is a classic agent pattern for multi-step tasks: a planner breaks the goal
into ordered steps (via structured output), an executor runs each step (reusing
:class:`ReActAgent` so steps can still use tools), and a final pass merges the
step results into one answer.
"""

from __future__ import annotations

from typing import Sequence

from ..llm.base import BaseLLM, Message
from ..memory.buffer import ConversationBufferMemory
from ..prompts.templates import PLANNER_SYSTEM, SYNTHESIZE_SYSTEM
from ..schemas.models import PLAN_SCHEMA, Plan, PlanStep
from ..tools.base import Tool
from ..tools.registry import ToolRegistry
from .base import AgentEvent, AgentResult, BaseAgent
from .react import ReActAgent


class PlanAndExecuteAgent(BaseAgent):
    name = "plan_execute"

    def __init__(
        self,
        llm: BaseLLM,
        tools: Sequence | ToolRegistry | None = None,
        memory: ConversationBufferMemory | None = None,
        planner_llm: BaseLLM | None = None,
        executor: BaseAgent | None = None,
        max_steps: int = 8,
        temperature: float = 0.0,
    ):
        self.llm = llm
        self.planner_llm = planner_llm or llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools or [])
        self.memory = memory
        self.executor = executor or ReActAgent(llm, self.tools, memory=memory, temperature=temperature)
        self.max_steps = max_steps
        self.temperature = temperature

    def _make_plan(self, task: str) -> Plan:
        # The planner is forced to emit a plan by calling submit_plan. We parse
        # the tool call directly; the tool's func is therefore never executed.
        submit = Tool(
            name="submit_plan",
            description="Submit the ordered list of steps that solve the task.",
            parameters=PLAN_SCHEMA,
            func=lambda **kwargs: kwargs,
        )
        messages = [Message.system(PLANNER_SYSTEM), Message.user(task)]
        response = self.planner_llm.chat(
            messages, tools=[submit.to_openai_spec()], temperature=self.temperature
        )
        for tc in response.tool_calls:
            if tc.name == "submit_plan":
                return Plan.model_validate(tc.arguments_dict())

        # Fallback: model ignored the tool — treat the whole task as one step.
        return Plan(steps=[PlanStep(description=task)])

    def run(self, task: str) -> AgentResult:
        events: list[AgentEvent] = []
        plan = self._make_plan(task)
        events.append(
            AgentEvent(
                type="plan",
                content=f"{len(plan.steps)} step(s)",
                data={"steps": [s.description for s in plan.steps]},
            )
        )

        step_results: list[str] = []
        total_tool_calls = 0
        steps = plan.steps[: self.max_steps]
        for idx, step in enumerate(steps, start=1):
            events.append(AgentEvent(type="step", content=step.description, data={"step": idx}))
            result = self.executor.run(step.description)
            events.extend(result.events)
            total_tool_calls += result.tool_calls
            step_results.append(f"Step {idx}: {step.description}\nResult: {result.answer}")

        synthesis = [
            Message.system(SYNTHESIZE_SYSTEM),
            Message.user(f"Task: {task}\n\n" + "\n\n".join(step_results)),
        ]
        final = self.llm.chat(synthesis, temperature=self.temperature)
        answer = final.content or "\n\n".join(r.split("\nResult: ", 1)[-1] for r in step_results)
        events.append(AgentEvent(type="answer", content=answer))
        return AgentResult(
            answer=answer,
            events=events,
            iterations=len(steps),
            tool_calls=total_tool_calls,
        )
