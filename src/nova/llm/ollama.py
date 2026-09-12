"""Convenience adapter for Ollama (local models).

Ollama exposes an OpenAI-compatible API at ``http://localhost:11434/v1``, so
this is a thin wrapper around :class:`OpenAICompatLLM` with the right defaults.
It lets you run agents fully offline with local models such as ``qwen2.5`` or
``llama3.2`` — no cloud API key required.
"""

from __future__ import annotations

from .openai_compat import OpenAICompatLLM

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaLLM(OpenAICompatLLM):
    """OpenAI-compatible adapter pre-configured for a local Ollama server.

    Usage::

        from nova.llm.ollama import OllamaLLM
        llm = OllamaLLM(model="qwen2.5")   # any model you've `ollama pull`ed
    """

    def __init__(
        self,
        model: str = "qwen2.5",
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        **kwargs,
    ):
        # Ollama needs no API key; pass a dummy so the parent's non-empty key
        # check doesn't block local usage (the local server ignores it).
        super().__init__(api_key="ollama", base_url=base_url, model=model, **kwargs)
