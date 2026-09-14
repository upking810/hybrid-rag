"""Embedding client. Vectors are L2-normalized so cosine similarity = dot product."""
from __future__ import annotations

import numpy as np
from openai import OpenAI

from . import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts, returning a (n, dim) float32 array of unit vectors."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client().embeddings.create(model=config.EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    arr = np.asarray(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
