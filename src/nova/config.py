"""Central place for reading configuration from the environment.

All settings are optional and read lazily so the library can also be used
programmatically (by passing values directly to constructors).
"""

from __future__ import annotations

import os


def get_api_key() -> str | None:
    return os.getenv("NOVA_API_KEY")


def get_base_url() -> str | None:
    return os.getenv("NOVA_BASE_URL")


def get_model() -> str:
    return os.getenv("NOVA_MODEL", "qwen-plus")


def get_embed_model() -> str | None:
    return os.getenv("NOVA_EMBED_MODEL")
