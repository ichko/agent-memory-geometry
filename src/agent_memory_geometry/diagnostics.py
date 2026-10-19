"""Descriptive score gaps and answer-anchor proxies, not verified mechanisms."""

from __future__ import annotations

import numpy as np


def reachability_gaps(rows):
    """Distance means conditioned on correctness, so this requires labels."""
    out = []
    for qt in sorted({r["question_type"] for r in rows}):
        sub = [r for r in rows if r["question_type"] == qt]
        correct = [float(r["q_nn_mem"]) for r in sub if int(r["score"]) == 1]
        wrong = [float(r["q_nn_mem"]) for r in sub if int(r["score"]) == 0]
        out.append(dict(question_type=qt, n=len(sub), mean_correct=float(np.mean(correct)),
                        mean_wrong=float(np.mean(wrong)), gap=float(np.mean(wrong) - np.mean(correct))))
    return out


def anchor_bins(rows, percentile=75):
    """Replicate descriptive bins using thresholds from correct outcomes.

    An answer-derived anchor is not an entailment annotation. These bins do not
    identify extraction, retrieval, or reasoning failures.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must lie between zero and 100")
    y = np.asarray([int(r["score"]) for r in rows])
    answer = np.asarray([float(r["answer_distance"]) for r in rows])
    query = np.asarray([float(r["query_distance"]) for r in rows])
    if not (y == 1).any():
        raise ValueError("Correct outcomes are required to define proxy thresholds")
    answer_threshold = float(np.percentile(answer[y == 1], percentile))
    query_threshold = float(np.percentile(query[y == 1], percentile))
    types = np.asarray([r["question_type"] for r in rows])
    out = []
    for qt in [*sorted(set(types)), "ALL"]:
        selected = (y == 0) & ((types == qt) if qt != "ALL" else True)
        a, q = answer[selected], query[selected]
        answer_distant = a > answer_threshold
        query_distant = ~answer_distant & (q > query_threshold)
        out.append(dict(question_type=qt, percentile=percentile, n_failed=int(selected.sum()),
                        answer_threshold=answer_threshold, query_threshold=query_threshold,
                        answer_distant=int(answer_distant.sum()), query_distant=int(query_distant.sum()),
                        neither=int((~answer_distant & ~query_distant).sum())))
    return out
