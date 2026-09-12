from __future__ import annotations

import pytest

from nova.agents.react import ReActAgent
from nova.llm.base import ChatResponse, Message, ToolCall
from nova.llm.mock import MockLLM
from nova.memory.buffer import ConversationBufferMemory
from nova.tools.builtin import calculator
from nova.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mock_llm_achat_falls_back_to_sync():
    llm = MockLLM([ChatResponse(content="hi")])
    resp = await llm.achat([Message.user("hello")])
    assert resp.content == "hi"


@pytest.mark.asyncio
async def test_react_arun_calls_tool_then_answers():
    llm = MockLLM(
        [
            ChatResponse(
                tool_calls=[ToolCall(id="c1", name="calculator", arguments='{"expression": "2+2"}')]
            ),
            ChatResponse(content="The answer is 4."),
        ]
    )
    agent = ReActAgent(llm, tools=ToolRegistry([calculator]))
    result = await agent.arun("what is 2+2?")
    assert result.answer == "The answer is 4."
    assert result.tool_calls == 1
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_react_arun_persists_multi_turn_memory():
    llm = MockLLM([ChatResponse(content="a1"), ChatResponse(content="a2")])
    agent = ReActAgent(llm, memory=ConversationBufferMemory())
    await agent.arun("first question")
    await agent.arun("second question")
    contents = [m.content for m in llm.calls[1]["messages"]]
    assert "first question" in contents
    assert "second question" in contents
