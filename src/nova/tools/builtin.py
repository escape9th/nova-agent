"""A small set of safe, dependency-light builtin tools.

These cover the common needs of a demo agent: arithmetic, time, local file I/O
and fetching a web page. They run with the privileges of the calling process —
in a real deployment you would sandbox them; Nova deliberately keeps them plain
so the security boundary is obvious rather than hidden.
"""

from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from pathlib import Path

from .base import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


@tool(description="Evaluate a safe arithmetic expression, e.g. '2 * (3 + 4) / 5'.")
def calculator(expression: str) -> str:
    """Evaluate arithmetic without the dangers of ``eval`` (AST-whitelisted)."""

    def _eval(node: ast.AST):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return str(_eval(ast.parse(expression, mode="eval")))


@tool(description="Get the current local date and time.")
def get_datetime() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@tool(description="Read a text file from the local filesystem and return its content.")
def read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        return f"Error: file not found: {path}"
    return p.read_text(encoding="utf-8", errors="replace")


@tool(description="Write text content to a file, creating parent directories as needed.")
def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


@tool(description="List the names of files and directories inside a directory.")
def list_files(directory: str) -> str:
    p = Path(directory).expanduser()
    if not p.is_dir():
        return f"Error: not a directory: {directory}"
    entries = sorted(e.name for e in p.iterdir())
    return "\n".join(entries) if entries else "(empty)"


def _strip_html(text: str) -> str:
    text = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@tool(description="Fetch a URL over HTTP(S) and return its visible text content (truncated).")
def fetch_url(url: str) -> str:
    import httpx

    with httpx.Client(follow_redirects=True, timeout=15.0) as client:
        response = client.get(url)
        response.raise_for_status()
    return _strip_html(response.text)[:4000]
