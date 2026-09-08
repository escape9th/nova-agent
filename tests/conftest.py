"""Shared test fixtures.

Everything here is hermetic: tests drive :class:`MockLLM` with scripted
responses, so no API key or network is ever required.
"""

from __future__ import annotations

import pytest

from nova.llm.base import ChatResponse
from nova.llm.mock import MockLLM


@pytest.fixture
def scripted_llm():
    """Factory that builds a MockLLM which pops one response per chat() call."""

    def build(*responses: ChatResponse) -> MockLLM:
        return MockLLM(list(responses))

    return build
