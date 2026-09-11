# Architecture

Nova is a small layered framework. Each layer depends only on the one below it,
so you can read the whole thing top-to-bottom.

## Layer overview

```
agents/     ReActAgent, PlanAndExecuteAgent, Team   (orchestration)
tools/      Tool, ToolRegistry, builtin             (capabilities)
memory/     ConversationBufferMemory, VectorStore   (context)
llm/        BaseLLM, OpenAICompatLLM, MockLLM       (model access)
```

## 1. `llm/` — model access

`BaseLLM` is the single interface everything else depends on:

- `chat(messages, tools=None)` → a normalised `ChatResponse`
- `stream(...)` → an iterator of chunks (token streaming)
- `embed(texts)` → embedding vectors (optional)

`OpenAICompatLLM` adapts this to any OpenAI-compatible Chat Completions
endpoint. Because OpenAI, Qwen (Bailian), DeepSeek and Moonshot all speak the
same protocol, one adapter covers all of them. `MockLLM` returns scripted
responses so the rest of the framework can be tested without a network.

Internal `Message` / `ToolCall` dataclasses are converted to and from the
provider's wire format by `to_openai_message` / `from_openai_choice`.

## 2. `tools/` — capabilities

A `Tool` pairs a plain callable with the JSON Schema that describes its
arguments. The `@tool` decorator generates that schema automatically from type
hints (`int` → `{"type": "integer"}`, `Optional[int]` → optional integer, …).

`ToolRegistry` is the agent's *only* handle on tools: it produces the OpenAI
tool specs sent to the model, and executes a named tool call. Crucially,
execution errors are caught and returned as text — a failing tool becomes an
*observation* the model can react to, instead of crashing the loop.

## 3. `memory/` — context

- **`ConversationBufferMemory`** is short-term memory: a sliding window over
  the raw message history, trimmed so tool results are never split from the
  assistant call that requested them.
- **`VectorStore`** is long-term memory: it embeds text chunks (with a
  dependency-free feature-hashing embedder by default, or a real model via
  `llm.embed`) and retrieves by cosine similarity — a minimal, transparent RAG.

## 4. `agents/` — orchestration

### ReActAgent

The heart of the framework. The loop:

1. Send `system + history + user` (plus tool specs) to the model.
2. If the model returns tool calls, execute them and append the results as
   `tool` messages.
3. Repeat until the model answers directly or the iteration budget runs out.

This is what "agentic" behaviour *is*: the model decides to act, we perform the
action, and feed the observation back so the model can reason further.

### PlanAndExecuteAgent

For multi-step tasks, a two-stage approach:

1. **Plan** — a planner LLM is forced to emit a JSON plan by calling a
   `submit_plan` tool (structured output via function calling).
2. **Execute** — each step is run by a `ReActAgent`, so steps can still use tools.
3. **Synthesize** — a final pass merges the step results into one answer.

### Team

Multi-agent orchestration via a supervisor. The supervisor's *only* tool is
`delegate(member, task)`, which runs a named specialist and returns its answer.
This deliberately reuses the tool-calling mechanism — "calling a tool" and
"calling another agent" are the same shape.

## Design choices worth noting

- **From scratch, on purpose.** Wrapping LangChain would hide the mechanisms
  Nova exists to teach. The whole core is a few hundred lines.
- **Errors are data.** Tool failures are returned to the model as text rather
  than raised, which is what lets agents self-correct.
- **Streaming is explicit.** The agent loop uses non-streaming `chat` for
  correct tool-call handling; token streaming is available via `llm.stream`.
- **Trust boundary is visible.** Builtin tools run with the caller's privileges;
  sandboxing is left explicit rather than pretending to be solved.
