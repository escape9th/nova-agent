from __future__ import annotations

from nova.llm.ollama import OllamaLLM
from nova.llm.openai_compat import OpenAICompatLLM


def test_ollama_llm_is_openai_compat_with_defaults():
    llm = OllamaLLM()
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.model == "qwen2.5"
    # points at the local Ollama server, not a cloud endpoint
    assert "11434" in str(llm.client.base_url)


def test_ollama_llm_custom_model():
    llm = OllamaLLM(model="llama3.2")
    assert llm.model == "llama3.2"
