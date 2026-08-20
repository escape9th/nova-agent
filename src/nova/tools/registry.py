"""Tool registry: the single source of truth for which tools an agent may call."""

from __future__ import annotations

import json
from typing import Any, Sequence

from .base import Tool


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def specs(self) -> list[dict[str, Any]]:
        """OpenAI-style tool specifications for all registered tools."""
        return [t.to_openai_spec() for t in self._tools.values()]

    def execute(self, name: str, arguments: str | dict) -> str:
        """Run a tool and return its result as a string.

        Exceptions are caught and returned as error text, so a failing tool
        becomes an *observation* the model can react to, instead of crashing
        the whole agent loop.
        """
        tool = self.get(name)
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        try:
            result = tool.func(**arguments)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the model
            return f"Error: {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)
