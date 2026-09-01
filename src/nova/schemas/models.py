"""Pydantic models for structured outputs.

These give us *typed* validation of what the model returns, independent of the
raw JSON the model actually emits.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    description: str = Field(..., description="A single concrete, self-contained action step")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(..., description="Ordered list of steps to execute")


# Provider-safe JSON schema for the plan. We don't hand pydantic's own
# ``model_json_schema()`` to the model because it contains ``$defs``/``$ref``
# which some OpenAI-compatible endpoints reject inside function-calling
# parameters. This flat schema is understood everywhere; ``Plan`` above still
# validates the parsed result.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
        }
    },
    "required": ["steps"],
}
