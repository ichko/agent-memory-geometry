"""Context subset indices for the fixed-store selection comparison."""

from __future__ import annotations

import random

import numpy as np


def select_indices(query: np.ndarray, memories: np.ndarray, k: int, condition: str,
                   seed: int = 0, replicate: int = 0) -> np.ndarray:
    """Match the proximity/random ordering of the published experiment.

    Native system rankings must come from the user's own memory system.
    This function does not reproduce a backend-specific native retriever.
    """
    if k < 1 or condition not in {"all", "closest_k", "farthest_k", "random_k"}:
        raise ValueError("Use positive k and a supported condition")
    query = np.asarray(query, dtype=np.float32)
    memories = np.asarray(memories, dtype=np.float32)
    if memories.ndim != 2 or query.shape != (memories.shape[1],):
        raise ValueError("Expected query (d,) and memories (n, d)")
    if not np.isfinite(query).all() or not np.isfinite(memories).all():
        raise ValueError("Embeddings must be finite")
    n = len(memories)
    if condition == "all" or n <= k:
        return np.arange(n)
    if condition == "random_k":
        return np.asarray(random.Random(seed * 100 + replicate).sample(range(n), k))
    qn = query / (np.linalg.norm(query) + 1e-12)
    mn = memories / (np.linalg.norm(memories, axis=1, keepdims=True) + 1e-12)
    order = np.argsort(-(mn @ qn))
    return order[:k] if condition == "closest_k" else order[-k:]
