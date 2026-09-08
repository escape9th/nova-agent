from __future__ import annotations

from nova.llm.base import Message, ToolCall
from nova.memory.buffer import ConversationBufferMemory
from nova.memory.vector import VectorStore, cosine, hash_embed


def test_buffer_trims_to_max_messages():
    mem = ConversationBufferMemory(max_messages=4)
    for i in range(10):
        mem.add(Message.user(f"msg{i}"))
    assert len(mem) <= 4
    assert mem.messages()[-1].content == "msg9"


def test_buffer_keeps_system_message():
    mem = ConversationBufferMemory(max_messages=2)
    mem.add(Message.system("sys"))
    mem.add(Message.user("a"))
    mem.add(Message.user("b"))
    assert mem.messages()[0].role == "system"


def test_buffer_does_not_orphan_tool_results():
    call = ToolCall(id="c1", name="calc", arguments="{}")
    mem = ConversationBufferMemory(max_messages=4)
    mem.add(Message.system("sys"))
    mem.add(Message.user("q"))
    mem.add(Message.assistant(None, [call]))
    mem.add(Message.tool("r1", "c1"))
    mem.add(Message.tool("r2", "c1"))
    mem.add(Message.assistant("done"))

    msgs = mem.messages()
    assert msgs[0].role == "system"
    # the window must not begin with a dangling tool result
    assert msgs[1].role != "tool"
    # the assistant message that issued the tool call is still present
    assert any(m.role == "assistant" and m.tool_calls for m in msgs)


def test_buffer_clear():
    mem = ConversationBufferMemory()
    mem.add(Message.user("hi"))
    mem.clear()
    assert len(mem) == 0


def test_vector_store_retrieves_similar():
    store = VectorStore()
    store.add("python programming language")
    store.add("beautiful sunset at the beach")
    hits = store.query("coding in python", top_k=1)
    assert "python" in hits[0]["text"]


def test_hash_embed_is_normalized():
    v = hash_embed("hello world")
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_cosine_of_identical_vectors_is_one():
    v = [1.0, 0.0, 0.0]
    assert cosine(v, v) == 1.0
