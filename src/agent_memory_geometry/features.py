"""Compute geometry from user-supplied numeric embeddings, without API calls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .metrics import all_features


def validate_inputs(turns, memories, sessions, query, k=5):
    """Reject incompatible inputs before the quadratic-distance computation."""
    if k < 1:
        raise ValueError("k must be positive")
    if turns.ndim != 2 or memories.ndim != 2:
        raise ValueError("turns and memories must have shape (n, d)")
    if not len(turns) or not len(memories) or turns.shape[1] == 0:
        raise ValueError("turns and memories must be nonempty")
    if turns.shape[1] != memories.shape[1]:
        raise ValueError("All embeddings must share one dimension")
    if sessions.shape != (len(turns),) or sessions.dtype.kind not in "iu":
        raise ValueError("sessions must contain one integer label per turn")
    if query.shape != (turns.shape[1],):
        raise ValueError("query must have shape (d,)")
    for array in (turns, memories, query):
        if not np.isfinite(array).all():
            raise ValueError("Embeddings must be finite")
        if np.any(np.linalg.norm(np.atleast_2d(array), axis=1) == 0):
            raise ValueError("Zero embeddings are not supported")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ with turns, memories, sessions, query")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        turns, memories, sessions, query = [data[name] for name in ("turns", "memories", "sessions", "query")]
    validate_inputs(turns, memories, sessions, query, args.k)
    result = all_features(turns, memories, sessions, q=query, k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
