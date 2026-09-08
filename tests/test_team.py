from __future__ import annotations

from nova.agents.react import ReActAgent
from nova.agents.team import Team, TeamMember
from nova.llm.base import ChatResponse, ToolCall


def test_team_delegates_and_answers(scripted_llm):
    supervisor = scripted_llm(
        ChatResponse(
            tool_calls=[
                ToolCall(id="d1", name="delegate", arguments='{"member": "joker", "task": "tell a joke"}')
            ]
        ),
        ChatResponse(content="Here is your joke: why did the snake cross the road?"),
    )
    worker = scripted_llm(ChatResponse(content="A joke about snakes."))

    team = Team(
        members=[TeamMember("joker", "tells jokes", ReActAgent(worker))],
        llm=supervisor,
    )
    result = team.run("tell me a joke")
    assert result.answer == "Here is your joke: why did the snake cross the road?"
    assert result.tool_calls == 1


def test_team_unknown_member_returns_error_to_supervisor(scripted_llm):
    supervisor = scripted_llm(
        ChatResponse(
            tool_calls=[
                ToolCall(id="d1", name="delegate", arguments='{"member": "ghost", "task": "x"}')
            ]
        ),
        ChatResponse(content="I could not do that."),
    )
    team = Team(members=[], llm=supervisor)
    result = team.run("task")
    assert result.answer == "I could not do that."
    # the failed delegation still counts as a tool call
    assert result.tool_calls == 1
