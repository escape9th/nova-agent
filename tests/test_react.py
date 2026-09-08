from __future__ import annotations

from nova.agents.react import ReActAgent
from nova.llm.base import ChatResponse, ToolCall
from nova.memory.buffer import ConversationBufferMemory
from nova.tools.builtin import calculator
from nova.tools.registry import ToolRegistry


def test_react_calls_tool_then_answers(scripted_llm):
    llm = scripted_llm(
        ChatResponse(
            tool_calls=[ToolCall(id="c1", name="calculator", arguments='{"expression": "2+2"}')]
        ),
        ChatResponse(content="The answer is 4."),
    )
    agent = ReActAgent(llm, tools=ToolRegistry([calculator]))
    result = agent.run("what is 2+2?")
    assert result.answer == "The answer is 4."
    assert result.tool_calls == 1
    assert result.iterations == 2


def test_react_answers_without_tools(scripted_llm):
    llm = scripted_llm(ChatResponse(content="Hello!"))
    agent = ReActAgent(llm)
    result = agent.run("hi")
    assert result.answer == "Hello!"
    assert result.tool_calls == 0
    assert result.iterations == 1


def test_react_stops_after_max_iterations(scripted_llm):
    tool_call_resp = ChatResponse(
        tool_calls=[ToolCall(id="c1", name="calculator", arguments='{"expression": "1+1"}')]
    )
    llm = scripted_llm(*([tool_call_resp] * 10))
    agent = ReActAgent(llm, tools=ToolRegistry([calculator]), max_iterations=3)
    result = agent.run("loop forever")
    assert result.answer.startswith("stopped after")
    assert result.tool_calls == 3


def test_react_persists_multi_turn_memory(scripted_llm):
    llm = scripted_llm(ChatResponse(content="answer 1"), ChatResponse(content="answer 2"))
    agent = ReActAgent(llm, memory=ConversationBufferMemory())
    agent.run("first question")
    agent.run("second question")

    second_messages = llm.calls[1]["messages"]
    contents = [m.content for m in second_messages]
    assert "first question" in contents  # previous user turn is remembered
    assert "second question" in contents
