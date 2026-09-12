from __future__ import annotations

import pytest

from nova.llm.base import ChatResponse, Message
from nova.memory.summarize import SummarizationMemory


def test_summarization_memory_collapses_history(scripted_llm):
    llm = scripted_llm(ChatResponse(content="Earlier: user asked about Python and tools."))
    mem = SummarizationMemory(llm, trigger=6, keep_recent=3)
    mem.add(Message.system("sys"))
    for i in range(6):
        mem.add(Message.user(f"q{i}"))

    msgs = mem.messages()
    assert msgs[0].role == "system"
    system_contents = [m.content or "" for m in msgs if m.role == "system"]
    assert any("Earlier:" in c for c in system_contents)
    # only system + keep_recent messages remain in the raw buffer
    assert len(mem) <= 1 + 3
    # the summarization LLM was called exactly once
    assert len(llm.calls) == 1


def test_summarization_memory_no_summary_under_trigger(scripted_llm):
    llm = scripted_llm()
    mem = SummarizationMemory(llm, trigger=10, keep_recent=3)
    mem.add(Message.system("sys"))
    mem.add(Message.user("hello"))
    assert len(llm.calls) == 0
    assert len(mem.messages()) == 2


def test_summarization_rejects_bad_keep_recent(scripted_llm):
    with pytest.raises(ValueError):
        SummarizationMemory(scripted_llm(), trigger=5, keep_recent=5)
