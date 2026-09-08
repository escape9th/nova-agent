from __future__ import annotations

import json

from nova.agents.planner import PlanAndExecuteAgent
from nova.llm.base import ChatResponse, ToolCall


def _plan_response(steps: list[str]) -> ChatResponse:
    args = json.dumps({"steps": [{"description": s} for s in steps]})
    return ChatResponse(tool_calls=[ToolCall(id="p1", name="submit_plan", arguments=args)])


def test_planner_parses_plan_and_executes(scripted_llm):
    llm = scripted_llm(
        _plan_response(["step one", "step two"]),
        ChatResponse(content="result one"),  # executor, step 1
        ChatResponse(content="result two"),  # executor, step 2
        ChatResponse(content="final synthesis"),
    )
    agent = PlanAndExecuteAgent(llm)
    result = agent.run("do the thing")
    assert result.answer == "final synthesis"
    assert result.iterations == 2


def test_planner_falls_back_to_single_step_without_tool_call(scripted_llm):
    llm = scripted_llm(
        ChatResponse(content="ignoring the plan tool"),  # planner ignores submit_plan
        ChatResponse(content="done"),  # executor
        ChatResponse(content="final"),
    )
    agent = PlanAndExecuteAgent(llm)
    result = agent.run("task")
    assert result.answer == "final"
    assert result.iterations == 1  # fallback produced a single-step plan
