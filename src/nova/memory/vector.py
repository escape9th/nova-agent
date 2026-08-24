"""Long-term memory: a tiny in-memory vector store with a dependency-free embedder.

This implements the same *idea* behind vector databases (embed text -> store
vectors -> retrieve by cosine similarity) in ~60 lines, so the mechanism is
fully visible. For real workloads you would swap the embedder for an embedding
model and the storage for a proper vector DB; the interface stays the same.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class MemoryItem:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def hash_embed(text: str, dim: int = 256) -> list[float]:
    """Deterministic feature-hashing embedding (no model, no API call).

    Every token hashes to one signed contribution in a fixed-size vector, so
    texts that share tokens end up with similar vectors. Crude compared to a
    real embedding model, but free, deterministic and good enough for tests
    and offline demos.
    """
    vec = [0.0] * dim
    for token in re.findall(r"\w+", text.lower()):
        h = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 64) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def llm_embedder(llm) -> Callable[[str], list[float]]:
    """Adapt a Nova :class:`BaseLLM`'s ``embed`` method to the store's signature."""
    return lambda text: llm.embed([text])[0]


class VectorStore:
    """An in-memory store of text chunks searchable by semantic similarity."""

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None, dim: int = 256):
        self._embed: Callable[[str], list[float]] = embed_fn or (lambda t: hash_embed(t, dim))
        self._items: list[MemoryItem] = []
        self._counter = 0

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        self._counter += 1
        item_id = str(self._counter)
        self._items.append(
            MemoryItem(
                id=item_id,
                text=text,
                embedding=self._embed(text),
                metadata=metadata or {},
            )
        )
        return item_id

    def add_many(
        self, texts: Sequence[str], metadatas: Sequence[dict[str, Any]] | None = None
    ) -> list[str]:
        metas = metadatas or [{} for _ in texts]
        return [self.add(t, m) for t, m in zip(texts, metas)]

    def query(self, text: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Return the ``top_k`` most similar stored items, best first."""
        q = self._embed(text)
        scored = sorted(
            ((cosine(q, item.embedding), item) for item in self._items),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            {"id": item.id, "text": item.text, "score": round(score, 4), "metadata": item.metadata}
            for score, item in scored[:top_k]
        ]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
