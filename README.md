[English](README.md) · [简体中文](README_zh.md)

# Nova

A minimal, readable, **from-scratch AI agent framework**.

> **Keywords:** AI Agent · 智能体 · LLM · 大模型 · Tool Calling · Function Calling · RAG · Multi-Agent · 多智能体 · ReAct · 向量检索 · Memory · 记忆 · Planning · 规划 · Python · Qwen 通义千问 · DeepSeek · Ollama

Nova implements the core mechanisms behind modern AI agents — the reasoning
loop, tool calling, memory, planning and multi-agent orchestration — directly
on top of the OpenAI-compatible protocol, with no heavyweight framework
dependencies. The entire core is a few hundred lines of plain Python, written
so you can read and understand every part.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/escape9th/nova-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/escape9th/nova-agent/actions/workflows/ci.yml)

## Why Nova?

- **Learn the internals.** Every mechanism — the ReAct loop, function calling,
  a vector store — is implemented in ~50–100 lines you can actually read.
- **Provider-agnostic.** Works with OpenAI, Aliyun Bailian (Qwen), DeepSeek,
  Moonshot and any other OpenAI-compatible endpoint.
- **Zero magic.** No LangChain, no CrewAI. Nova shows you *what* an agent
  framework does, not just how to call one.

## Features

- **LLM abstraction** — a small provider interface plus an OpenAI-compatible
  adapter and a deterministic mock for offline tests.
- **Tool system** — a `@tool` decorator that infers JSON Schema from type
  hints, plus a registry and a set of safe builtin tools.
- **Memory** — a sliding-window conversation buffer (short-term) and a
  dependency-free vector store with feature-hashing embeddings (long-term RAG).
- **ReAct agent** — the classic Reason + Act loop with tool calling and an
  iteration budget.
- **Plan-and-execute** — structured planning (via forced tool output) followed
  by step execution and synthesis.
- **Multi-agent team** — a supervisor that delegates to specialists using the
  same tool-calling mechanism.
- **Streaming** — token-level streaming and event-level agent traces.
- **CLI, examples and tests** — runnable demos and a hermetic test suite.

## Installation

```bash
git clone https://github.com/escape9th/nova-agent.git
cd nova-agent
pip install -e ".[dev]"
```

## Quick start

```bash
# 1. configure your provider (copy and edit)
cp .env.example .env

# 2. run a one-shot task
nova run "27 * 43 等于多少？"

# 3. or start an interactive session
nova chat
```

```python
from nova.llm.openai_compat import OpenAICompatLLM
from nova.agents.react import ReActAgent
from nova.tools.registry import ToolRegistry
from nova.tools.builtin import calculator, get_datetime

agent = ReActAgent(
    llm=OpenAICompatLLM(),
    tools=ToolRegistry([calculator, get_datetime]),
)

result = agent.run("(12 + 3) * 5 是多少？")
print(result.answer)
```

## Architecture

```
                    ┌─────────────────────────────┐
                    │          CLI / API          │
                    └──────────────┬──────────────┘
                                   │
   ┌───────────────────────────────┼───────────────────────────────┐
   │                               │                               │
┌──▼────────────┐       ┌──────────▼──────────┐       ┌────────────▼──┐
│  ReActAgent   │       │ PlanAndExecuteAgent │       │     Team      │
│  reason+act   │       │   plan + execute    │       │  supervisor   │
└──┬────────────┘       └──────────┬──────────┘       └────────────┬──┘
   │           agents              │                              │
   └───────────────────────────────┼──────────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
  ┌────▼─────┐              ┌──────▼──────┐             ┌───────▼──────┐
  │  Tools   │              │   Memory    │             │      LLM     │
  │ registry │              │ buffer/vec  │             │ OpenAI-compat│
  └──────────┘              └─────────────┘             └──────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a component-by-component
walkthrough.

## Examples

| Example | What it shows |
| --- | --- |
| [examples/01_chat.py](examples/01_chat.py) | Basic chat via the LLM layer |
| [examples/02_tools.py](examples/02_tools.py) | Registering and calling tools |
| [examples/03_react.py](examples/03_react.py) | The ReAct tool-using loop |
| [examples/04_memory.py](examples/04_memory.py) | Vector store / RAG retrieval |
| [examples/05_planner.py](examples/05_planner.py) | Plan-and-execute |
| [examples/06_team.py](examples/06_team.py) | Multi-agent supervision |
| [examples/07_streaming.py](examples/07_streaming.py) | Token streaming |

## Running tests

```bash
pytest
```

The suite is fully hermetic — it drives a `MockLLM` with scripted responses,
so it needs no API key and no network.

## Security note

Builtin tools (file I/O, URL fetching) run with the caller's privileges. Nova
keeps them intentionally plain so the trust boundary is visible; sandbox them
before exposing an agent to untrusted input.

## License

[MIT](LICENSE)
