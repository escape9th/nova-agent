"""Expose a Nova agent over HTTP with FastAPI.

This turns the framework from a Python library into a small web service, so an
agent can be called from any frontend, script or app over plain HTTP.

Run::

    pip install "nova-agent[server]"
    uvicorn nova.server:app --host 0.0.0.0 --port 8000

Then::

    curl -X POST http://localhost:8000/run \
         -H "Content-Type: application/json" \
         -d '{"task": "27 * 43 等于多少？"}'
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import __version__
from .agents.react import ReActAgent
from .llm.openai_compat import OpenAICompatLLM
from .tools.builtin import (
    calculator,
    fetch_url,
    get_datetime,
    list_files,
    read_file,
    web_search,
    write_file,
)
from .tools.registry import ToolRegistry

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The HTTP server requires FastAPI. Install it with: pip install \"nova-agent[server]\""
    ) from exc

DEFAULT_TOOLS = [
    calculator,
    get_datetime,
    read_file,
    write_file,
    list_files,
    fetch_url,
    web_search,
]


class RunRequest(BaseModel):
    task: str = Field(..., description="The task or question to run.")
    max_iterations: int = 8
    temperature: float = 0.0


class EventOut(BaseModel):
    type: str
    content: str = ""


class RunResponse(BaseModel):
    answer: str
    iterations: int
    tool_calls: int
    events: list[EventOut] = []


app = FastAPI(
    title="Nova Agent",
    description="A from-scratch AI agent framework served over HTTP.",
    version=__version__,
)


def _make_agent(max_iterations: int, temperature: float) -> ReActAgent:
    return ReActAgent(
        OpenAICompatLLM(),
        ToolRegistry(DEFAULT_TOOLS),
        max_iterations=max_iterations,
        temperature=temperature,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run_task(req: RunRequest) -> RunResponse:
    result = _make_agent(req.max_iterations, req.temperature).run(req.task)
    return RunResponse(
        answer=result.answer,
        iterations=result.iterations,
        tool_calls=result.tool_calls,
        events=[EventOut(type=e.type, content=e.content) for e in result.events],
    )
