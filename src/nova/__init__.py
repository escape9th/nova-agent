"""Nova — a minimal, from-scratch AI agent framework.

The whole point of Nova is to implement the *internals* of AI agents — the
reasoning loop, tool calling, memory and orchestration — directly on top of the
OpenAI-compatible protocol, instead of hiding them behind a heavyweight
framework. Every component is deliberately small and readable.
"""

from __future__ import annotations

__version__ = "0.4.0"

try:  # load .env if present (and python-dotenv is installed)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass
