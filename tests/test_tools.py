from __future__ import annotations

import pytest

from nova.tools.base import tool
from nova.tools.builtin import calculator
from nova.tools.registry import ToolRegistry


def test_calculator_evaluates_arithmetic():
    assert calculator("2 + 3 * 4") == "14"
    assert calculator("(2 + 3) * 4") == "20"
    assert calculator("2 ** 10") == "1024"


def test_calculator_rejects_dangerous_input():
    # __import__ is an ast.Call node, which the whitelist does not allow.
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo hi')")


def test_tool_decorator_infers_schema():
    @tool
    def add(a: int, b: int = 0) -> int:
        """Add two numbers."""
        return a + b

    fn = add.to_openai_spec()["function"]
    assert fn["name"] == "add"
    assert fn["parameters"]["properties"]["a"]["type"] == "integer"
    assert fn["parameters"]["required"] == ["a"]  # b has a default, so optional


def test_registry_executes_tools():
    registry = ToolRegistry([calculator])
    assert registry.execute("calculator", {"expression": "1+1"}) == "2"
    assert registry.execute("calculator", '{"expression": "1+1"}') == "2"


def test_registry_raises_on_unknown_tool():
    registry = ToolRegistry([calculator])
    with pytest.raises(KeyError):
        registry.execute("does_not_exist", {})


def test_registry_turns_tool_errors_into_observations():
    registry = ToolRegistry([calculator])
    result = registry.execute("calculator", {"expression": "1 / 0"})
    assert result.startswith("Error:")
