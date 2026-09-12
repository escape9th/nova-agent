from __future__ import annotations

from nova.llm.base import ChatResponse, Message


def test_llm_json_parses_object(scripted_llm):
    llm = scripted_llm(ChatResponse(content='{"name": "Nova", "stars": 5}'))
    result = llm.json([Message.user("describe nova as json")])
    assert result == {"name": "Nova", "stars": 5}


def test_llm_json_sets_json_mode(scripted_llm):
    llm = scripted_llm(ChatResponse(content="{}"))
    llm.json([Message.user("output json")])
    assert llm.calls[0]["response_format"] == {"type": "json_object"}
