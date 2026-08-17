# Nova

A minimal, readable, from-scratch AI agent framework.

Nova implements the core mechanisms behind modern AI agents — tool calling,
memory, reasoning loops and multi-agent orchestration — directly on top of the
OpenAI-compatible protocol, with no heavyweight framework dependencies. The
goal is a small codebase that is easy to read, easy to extend, and honest about
how agents actually work under the hood.

> **Status:** early development. API may change before 1.0.

## Features (planned)

- [ ] LLM abstraction layer (OpenAI-compatible: OpenAI / Qwen / DeepSeek / ...)
- [ ] Tool registry with builtin tools
- [ ] Short-term & long-term memory
- [ ] ReAct reasoning loop
- [ ] Plan-and-execute agent
- [ ] Multi-agent orchestration
- [ ] Streaming output
- [ ] Test suite

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env   # then fill in your API key
```

More to come.
