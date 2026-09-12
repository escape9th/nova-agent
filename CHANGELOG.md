# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-09-12

### Added

- `SummarizationMemory` — LLM-based compression of long conversations.
- `web_search` tool backed by DuckDuckGo (no API key required).
- `OllamaLLM` adapter for running local models offline.
- FastAPI HTTP server (`nova.server`) exposing the agent over REST.
- Chinese/English keywords in README and package metadata for discoverability.

### Changed

- Test suite expanded from 27 to 36 tests.

## [0.6.0] - 2026-09-11

### Added

- Full README and architecture guide.
- GitHub Actions CI workflow.
- Test suite (27 tests, fully hermetic).

### Fixed

- Multi-turn chat now persists user turns in memory.
- Buffer trimming no longer drops the leading system message.
- `@tool` supports positional calls and resolves PEP 563 string annotations.
- Agents reuse a provided (possibly empty) memory instead of treating it as missing.

## [0.5.0] - 2026-09-04

### Added

- CLI: `nova run` and `nova chat`.
- Runnable examples for every component.

## [0.4.0] - 2026-09-01

### Added

- Plan-and-execute agent with structured planning.
- Multi-agent team with a supervisor.

## [0.3.0] - 2026-08-27

### Added

- ReAct agent: the tool-using reasoning loop.

## [0.2.0] - 2026-08-24

### Added

- Tool registry with JSON-Schema inference from type hints.
- Builtin tools (calculator, datetime, file I/O, URL fetch).
- Conversation buffer and dependency-free vector store.

## [0.1.0] - 2026-08-17

### Added

- Project scaffolding (packaging, license, gitignore).
- Provider-agnostic LLM layer (OpenAI-compatible adapter + mock).
