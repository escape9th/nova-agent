"""CLI entry point.

Usage:
    nova run  "27 * 43 等于多少？"
    nova chat
    nova --version

Set NOVA_API_KEY / NOVA_BASE_URL / NOVA_MODEL (or copy .env.example to .env).
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .agents.base import AgentEvent
from .agents.react import ReActAgent
from .llm.openai_compat import OpenAICompatLLM
from .memory.buffer import ConversationBufferMemory
from .tools.builtin import calculator, fetch_url, get_datetime, list_files, read_file, write_file
from .tools.registry import ToolRegistry

DEFAULT_TOOLS = [calculator, get_datetime, read_file, write_file, list_files, fetch_url]


def _make_agent(args: argparse.Namespace, memory: ConversationBufferMemory | None = None) -> ReActAgent:
    llm = OpenAICompatLLM(model=args.model)
    tools = ToolRegistry(DEFAULT_TOOLS)
    return ReActAgent(
        llm,
        tools,
        memory=memory,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
    )


def _print_event(event: AgentEvent) -> None:
    if event.type == "tool_call":
        for call in event.data.get("calls", []):
            print(f"  → {call['name']}({call['arguments']})")
    elif event.type == "tool_result":
        preview = " ".join(event.content.split())
        if len(preview) > 120:
            preview = preview[:120] + " …"
        print(f"      {preview}")
    elif event.type == "answer":
        print(f"\n{event.content}")
    elif event.type == "error":
        print(f"\n[error] {event.content}")
    elif event.type in ("plan", "step"):
        print(f"\n[{event.type}] {event.content}")


def _cmd_run(args: argparse.Namespace) -> int:
    agent = _make_agent(args)
    for event in agent.run_stream(args.task):
        _print_event(event)
    print()
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    memory = ConversationBufferMemory()
    agent = _make_agent(args, memory=memory)
    print("Nova chat — type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            task = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task.lower() in {"exit", "quit"}:
            break
        for event in agent.run_stream(task):
            _print_event(event)
        print()
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="model name (default: $NOVA_MODEL)")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nova", description="Nova — a from-scratch AI agent framework"
    )
    parser.add_argument("--version", action="version", version=f"nova {__version__}")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run a single task with the agent")
    run_p.add_argument("task")
    _add_common(run_p)

    chat_p = sub.add_parser("chat", help="start an interactive chat session")
    _add_common(chat_p)

    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "chat":
            return _cmd_chat(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Set NOVA_API_KEY / NOVA_BASE_URL / NOVA_MODEL, or copy .env.example to .env.",
            file=sys.stderr,
        )
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
