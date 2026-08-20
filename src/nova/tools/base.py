"""Tool definition and the ``@tool`` decorator.

A :class:`Tool` wraps a plain Python callable together with the JSON Schema
that describes its arguments — that schema is what we hand to the model so it
can produce valid function calls.
"""

from __future__ import annotations

import inspect
import types as pytypes
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object
    func: Callable[..., Any]

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def to_openai_spec(self) -> dict[str, Any]:
        """Return this tool as an OpenAI-style tool specification."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Best-effort mapping from a Python type hint to a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] / X | None -> schema of X
    if origin in (Union, pytypes.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list or origin is list:
        items = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": items}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _make_tool(func: Callable[..., Any], name: str | None, description: str | None) -> Tool:
    sig = inspect.signature(func)
    doc = (func.__doc__ or "").strip()
    desc = description or (doc.splitlines()[0] if doc else name or func.__name__)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        properties[pname] = _annotation_to_schema(param.annotation)
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return Tool(
        name=name or func.__name__,
        description=desc,
        parameters=schema,
        func=func,
    )


def tool(func: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
    """Decorate a function to turn it into a :class:`Tool`.

    The JSON Schema is inferred from type hints; the first line of the docstring
    becomes the tool description (used by the model to decide when to call it).

        @tool
        def add(a: int, b: int) -> int:
            "Add two integers."
            return a + b
    """
    if func is not None:  # used bare as ``@tool``
        return _make_tool(func, name, description)

    def decorator(f: Callable[..., Any]) -> Tool:
        return _make_tool(f, name, description)

    return decorator
