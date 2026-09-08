from __future__ import annotations

from types import SimpleNamespace

from nova.llm.base import Message, ToolCall
from nova.llm.openai_compat import from_openai_choice, to_openai_message


def test_to_openai_message_basic():
    assert to_openai_message(Message.user("hi")) == {"role": "user", "content": "hi"}


def test_to_openai_message_with_tool_calls():
    msg = Message.assistant(None, [ToolCall(id="c1", name="calc", arguments="{}")])
    d = to_openai_message(msg)
    assert d["tool_calls"][0]["function"]["name"] == "calc"
    assert d["tool_calls"][0]["type"] == "function"


def test_to_openai_message_tool_role():
    d = to_openai_message(Message.tool("result", "c1", "calc"))
    assert d == {"role": "tool", "content": "result", "tool_call_id": "c1", "name": "calc"}


def test_from_openai_choice_plain():
    choice = SimpleNamespace(
        message=SimpleNamespace(content="hi", tool_calls=None), finish_reason="stop"
    )
    resp = from_openai_choice(choice)
    assert resp.content == "hi"
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []


def test_from_openai_choice_with_tool_calls():
    fn = SimpleNamespace(name="calc", arguments='{"expression": "1+1"}')
    tc = SimpleNamespace(id="c1", function=fn)
    choice = SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[tc]), finish_reason="tool_calls"
    )
    resp = from_openai_choice(choice)
    assert resp.tool_calls[0].name == "calc"
    assert resp.tool_calls[0].arguments_dict() == {"expression": "1+1"}


def test_tool_call_arguments_dict():
    tc = ToolCall(id="x", name="f", arguments='{"a": 1}')
    assert tc.arguments_dict() == {"a": 1}
