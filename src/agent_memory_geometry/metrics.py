"""Geometric metrics over the (turn, memory, query) embedding manifold.

Pure functions, no I/O. All distances are *cosine distance* in the
original embedding space (not 2D projection). Input matrices are
``(n, d)`` with rows = points.

Convention: we operate on unit-normalized vectors. The functions tolerate
non-normalized input by normalizing internally.

The five metric families:

    1. Conversation structure   — describes the turn manifold (same across
                                  strategies for a given example).
    2. Memory–manifold coverage — how well memories cover the turn cloud.
    3. Memory dispersion        — internal structure of the memory set.
    4. Query reachability       — query proximity to memories and turns.
    5. Memory–session alignment — do memories preserve session structure?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


# ── helpers ─────────────────────────────────────────────────────────────


def _normalize(X: NDArray[np.floating]) -> NDArray[np.float32]:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def cosine_distance_matrix(
    X: NDArray[np.floating], Y: NDArray[np.floating]
) -> NDArray[np.float32]:
    """Pairwise cosine distance, shape ``(len(X), len(Y))``, range ``[0, 2]``."""
    Xn = _normalize(X)
    Yn = _normalize(Y)
    sim = Xn @ Yn.T
    return (1.0 - sim).astype(np.float32)


def _safe_min(values: NDArray[np.floating]) -> float:
    if values.size == 0:
        return float("nan")
    return float(values.min())


def calibrated_eps(T: NDArray[np.floating], percentile: float = 25.0) -> float:
    """ε calibrated to the per-example pairwise turn-distance distribution.

    Default: 25th percentile of pairwise turn distances. Scale-free across
    examples with different conversation densities.
    """
    if T.shape[0] < 2:
        return float("nan")
    D = cosine_distance_matrix(T, T)
    # Take strict upper triangle to avoid diagonal zeros.
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.percentile(D[iu], percentile))


# ── family 1: conversation structure ────────────────────────────────────


def silhouette_score_cosine(
    X: NDArray[np.floating], labels: Iterable[int]
) -> float:
    """Mean silhouette over points with cosine distance.

    Returns NaN if fewer than 2 distinct labels or fewer than 2 points
    per cluster on average.
    """
    labels_arr = np.asarray(list(labels))
    if labels_arr.size != X.shape[0] or X.shape[0] < 2:
        return float("nan")
    unique = np.unique(labels_arr)
    if unique.size < 2:
        return float("nan")
    D = cosine_distance_matrix(X, X)
    sils = []
    for i in range(X.shape[0]):
        own = labels_arr[i]
        own_mask = (labels_arr == own) & (np.arange(X.shape[0]) != i)
        if not own_mask.any():
            continue
        a = float(D[i, own_mask].mean())
        b = math.inf
        for other in unique:
            if other == own:
                continue
            other_mask = labels_arr == other
            if not other_mask.any():
                continue
            b = min(b, float(D[i, other_mask].mean()))
        if not math.isfinite(b):
            continue
        denom = max(a, b)
        if denom <= 0:
            continue
        sils.append((b - a) / denom)
    if not sils:
        return float("nan")
    return float(np.mean(sils))


def intra_inter_session_ratio(
    T: NDArray[np.floating], session_labels: Iterable[int]
) -> float:
    """mean(intra-session distance) / mean(inter-session distance).

    Lower = sessions are more distinct in embedding space. NaN if only
    one session.
    """
    labels = np.asarray(list(session_labels))
    if labels.size != T.shape[0]:
        return float("nan")
    D = cosine_distance_matrix(T, T)
    same = labels[:, None] == labels[None, :]
    diff = ~same
    np.fill_diagonal(same, False)
    if not same.any() or not diff.any():
        return float("nan")
    return float(D[same].mean() / D[diff].mean())


def gmm_effective_clusters(T: NDArray[np.floating], max_k: int = 10) -> float:
    """Number of clusters chosen by BIC among ``[1, max_k]`` GMM fits.

    Uses cosine-normalized inputs. Cheap proxy for "effective topical
    diversity" of the conversation.
    """
    from sklearn.mixture import GaussianMixture

    if T.shape[0] < 2:
        return float("nan")
    Xn = _normalize(T)
    max_k = min(max_k, T.shape[0])
    best_k, best_bic = 1, math.inf
    for k in range(1, max_k + 1):
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=0,
                reg_covar=1e-4,
            ).fit(Xn)
            bic = gm.bic(Xn)
        except (ValueError, FloatingPointError):
            continue
        if bic < best_bic:
            best_bic = bic
            best_k = k
    return float(best_k)


@dataclass
class ConversationFeatures:
    n_turns: int
    n_sessions: int
    turn_silhouette: float
    intra_inter_ratio: float
    effective_clusters: float


def conversation_features(
    T: NDArray[np.floating], session_labels: Iterable[int]
) -> ConversationFeatures:
    labels = np.asarray(list(session_labels))
    return ConversationFeatures(
        n_turns=int(T.shape[0]),
        n_sessions=int(np.unique(labels).size) if labels.size else 0,
        turn_silhouette=silhouette_score_cosine(T, labels),
        intra_inter_ratio=intra_inter_session_ratio(T, labels),
        effective_clusters=gmm_effective_clusters(T),
    )


# ── family 2: memory–manifold coverage ──────────────────────────────────


def _turn_to_mem_nn(
    T: NDArray[np.floating], M: NDArray[np.floating]
) -> NDArray[np.float32]:
    """For each turn, distance to its nearest memory. Empty M → array of NaNs."""
    if T.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if M.shape[0] == 0:
        return np.full(T.shape[0], np.nan, dtype=np.float32)
    D = cosine_distance_matrix(T, M)
    return D.min(axis=1).astype(np.float32)


def mean_nn_t_to_m(T: NDArray[np.floating], M: NDArray[np.floating]) -> float:
    nn = _turn_to_mem_nn(T, M)
    return float(np.nanmean(nn)) if nn.size else float("nan")


def max_nn_t_to_m(T: NDArray[np.floating], M: NDArray[np.floating]) -> float:
    nn = _turn_to_mem_nn(T, M)
    return float(np.nanmax(nn)) if nn.size else float("nan")


def coverage_at_eps(
    T: NDArray[np.floating], M: NDArray[np.floating], eps: float
) -> float:
    """Fraction of turns within ε (cosine distance) of some memory."""
    nn = _turn_to_mem_nn(T, M)
    if nn.size == 0 or not math.isfinite(eps):
        return float("nan")
    return float((nn < eps).mean())


def hausdorff_t_to_m(
    T: NDArray[np.floating], M: NDArray[np.floating]
) -> float:
    """Symmetric Hausdorff distance between turn and memory sets."""
    if T.shape[0] == 0 or M.shape[0] == 0:
        return float("nan")
    D = cosine_distance_matrix(T, M)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def per_session_coverage(
    T: NDArray[np.floating],
    M: NDArray[np.floating],
    session_labels: Iterable[int],
) -> tuple[float, float]:
    """Mean and worst (max) of per-session mean-NN-distance.

    Returns ``(mean_session_coverage, worst_session_coverage)``.
    """
    labels = np.asarray(list(session_labels))
    if labels.size != T.shape[0] or M.shape[0] == 0:
        return float("nan"), float("nan")
    nn = _turn_to_mem_nn(T, M)
    per: list[float] = []
    for s in np.unique(labels):
        mask = labels == s
        if mask.any():
            per.append(float(nn[mask].mean()))
    if not per:
        return float("nan"), float("nan")
    return float(np.mean(per)), float(np.max(per))


@dataclass
class CoverageFeatures:
    mean_nn: float
    max_nn: float
    coverage_at_eps: float
    hausdorff: float
    mean_session_cov: float
    worst_session_cov: float


def coverage_features(
    T: NDArray[np.floating],
    M: NDArray[np.floating],
    session_labels: Iterable[int],
    eps: float,
) -> CoverageFeatures:
    mean_s, worst_s = per_session_coverage(T, M, session_labels)
    return CoverageFeatures(
        mean_nn=mean_nn_t_to_m(T, M),
        max_nn=max_nn_t_to_m(T, M),
        coverage_at_eps=coverage_at_eps(T, M, eps),
        hausdorff=hausdorff_t_to_m(T, M),
        mean_session_cov=mean_s,
        worst_session_cov=worst_s,
    )


# ── family 3: memory dispersion / quality ───────────────────────────────


def effective_rank(M: NDArray[np.floating]) -> float:
    """Entropy-based effective rank (Roy & Vetterli, 2007).

    ``exp(H(s))`` where ``s_i`` are normalized singular values. ≈ 1 for
    rank-1 matrices, ≈ min(n, d) for isotropic clouds.
    """
    if M.shape[0] < 2:
        return float("nan")
    Mn = _normalize(M)
    try:
        s = np.linalg.svd(Mn, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s = s[s > 1e-12]
    if s.size == 0:
        return float("nan")
    p = s / s.sum()
    h = -np.sum(p * np.log(p))
    return float(math.exp(h))


def mean_pairwise_distance(M: NDArray[np.floating]) -> float:
    if M.shape[0] < 2:
        return float("nan")
    D = cosine_distance_matrix(M, M)
    iu = np.triu_indices(D.shape[0], k=1)
    return float(D[iu].mean())


def redundancy_at_eps(M: NDArray[np.floating], eps: float) -> float:
    """Fraction of memories with another memory within ε."""
    if M.shape[0] < 2 or not math.isfinite(eps):
        return float("nan")
    D = cosine_distance_matrix(M, M)
    np.fill_diagonal(D, np.inf)
    has_neighbor = (D < eps).any(axis=1)
    return float(has_neighbor.mean())


@dataclass
class MemoryFeatures:
    n_memories: int
    effective_rank: float
    mean_pairwise_distance: float
    redundancy_at_eps: float


def memory_features(M: NDArray[np.floating], eps: float) -> MemoryFeatures:
    return MemoryFeatures(
        n_memories=int(M.shape[0]),
        effective_rank=effective_rank(M),
        mean_pairwise_distance=mean_pairwise_distance(M),
        redundancy_at_eps=redundancy_at_eps(M, eps),
    )


# ── family 4: query reachability (per-question) ─────────────────────────


def query_nn_distance(q: NDArray[np.floating], M: NDArray[np.floating]) -> float:
    if M.shape[0] == 0:
        return float("nan")
    d = cosine_distance_matrix(q.reshape(1, -1), M).ravel()
    return float(d.min())


def query_topk_mean(
    q: NDArray[np.floating], M: NDArray[np.floating], k: int = 5
) -> float:
    if M.shape[0] == 0:
        return float("nan")
    d = cosine_distance_matrix(q.reshape(1, -1), M).ravel()
    k_eff = min(k, d.size)
    return float(np.sort(d)[:k_eff].mean())


def query_memory_density(
    q: NDArray[np.floating], M: NDArray[np.floating], eps: float
) -> int:
    if M.shape[0] == 0 or not math.isfinite(eps):
        return 0
    d = cosine_distance_matrix(q.reshape(1, -1), M).ravel()
    return int((d < eps).sum())


def query_to_turn_distance(
    q: NDArray[np.floating], T: NDArray[np.floating]
) -> float:
    if T.shape[0] == 0:
        return float("nan")
    d = cosine_distance_matrix(q.reshape(1, -1), T).ravel()
    return float(d.min())


@dataclass
class QueryFeatures:
    q_nn_mem: float
    q_topk_mean_mem: float
    q_memory_density: int
    q_nn_turn: float


def query_features(
    q: NDArray[np.floating],
    T: NDArray[np.floating],
    M: NDArray[np.floating],
    eps: float,
    k: int = 5,
) -> QueryFeatures:
    return QueryFeatures(
        q_nn_mem=query_nn_distance(q, M),
        q_topk_mean_mem=query_topk_mean(q, M, k),
        q_memory_density=query_memory_density(q, M, eps),
        q_nn_turn=query_to_turn_distance(q, T),
    )


# ── family 5: memory–session alignment ──────────────────────────────────


def memory_inherited_labels(
    M: NDArray[np.floating],
    T: NDArray[np.floating],
    session_labels: Iterable[int],
) -> NDArray[np.int_]:
    """Assign each memory the session label of its nearest turn."""
    labels = np.asarray(list(session_labels))
    if M.shape[0] == 0 or T.shape[0] == 0 or labels.size != T.shape[0]:
        return np.array([], dtype=int)
    D = cosine_distance_matrix(M, T)
    nn_idx = D.argmin(axis=1)
    return labels[nn_idx]


def bridging_memory_fraction(
    M: NDArray[np.floating],
    T: NDArray[np.floating],
    session_labels: Iterable[int],
) -> float:
    """Fraction of memories whose 1st- and 2nd-nearest turns are in
    different sessions — a memory that "bridges" two sessions."""
    labels = np.asarray(list(session_labels))
    if M.shape[0] == 0 or T.shape[0] < 2 or labels.size != T.shape[0]:
        return float("nan")
    D = cosine_distance_matrix(M, T)
    order = np.argsort(D, axis=1)
    nn1 = labels[order[:, 0]]
    nn2 = labels[order[:, 1]]
    return float((nn1 != nn2).mean())


@dataclass
class AlignmentFeatures:
    memory_silhouette: float
    bridging_fraction: float


def alignment_features(
    M: NDArray[np.floating],
    T: NDArray[np.floating],
    session_labels: Iterable[int],
) -> AlignmentFeatures:
    inherited = memory_inherited_labels(M, T, session_labels)
    sil = silhouette_score_cosine(M, inherited) if inherited.size else float("nan")
    return AlignmentFeatures(
        memory_silhouette=sil,
        bridging_fraction=bridging_memory_fraction(M, T, session_labels),
    )


# ── one-call aggregator ─────────────────────────────────────────────────


def all_features(
    T: NDArray[np.floating],
    M: NDArray[np.floating],
    session_labels: Iterable[int],
    eps: float | None = None,
    q: NDArray[np.floating] | None = None,
    k: int = 5,
) -> dict[str, float | int]:
    """Compute all five families. ``q`` is optional; if None, family 4 is skipped.

    The default ``eps`` is the 25th percentile of pairwise turn distances.
    We also compute tighter coverage and redundancy ε levels (p5, p10)
    because the p25 versions saturate at 1.0 on LongMemEval and are
    uninformative.
    """
    if eps is None:
        eps = calibrated_eps(T, percentile=25.0)
    eps_p10 = calibrated_eps(T, percentile=10.0)
    eps_p5 = calibrated_eps(T, percentile=5.0)
    conv = conversation_features(T, session_labels)
    cov = coverage_features(T, M, session_labels, eps)
    mem = memory_features(M, eps)
    aln = alignment_features(M, T, session_labels)
    out: dict[str, float | int] = {
        # ε levels used for coverage / density metrics
        "eps_p25": eps,
        "eps_p10": eps_p10,
        "eps_p5": eps_p5,
        # family 1
        "n_turns": conv.n_turns,
        "n_sessions": conv.n_sessions,
        "turn_silhouette": conv.turn_silhouette,
        "intra_inter_ratio": conv.intra_inter_ratio,
        "effective_clusters": conv.effective_clusters,
        # family 2 — coverage at three ε levels
        "cov_mean_nn": cov.mean_nn,
        "cov_max_nn": cov.max_nn,
        "cov_at_eps": cov.coverage_at_eps,
        "cov_at_eps_p10": coverage_at_eps(T, M, eps_p10),
        "cov_at_eps_p5": coverage_at_eps(T, M, eps_p5),
        "cov_hausdorff": cov.hausdorff,
        "cov_mean_session": cov.mean_session_cov,
        "cov_worst_session": cov.worst_session_cov,
        # family 3 — dispersion + redundancy at three ε levels
        "mem_count": mem.n_memories,
        "mem_eff_rank": mem.effective_rank,
        "mem_mean_pairwise": mem.mean_pairwise_distance,
        "mem_redundancy_at_eps": mem.redundancy_at_eps,
        "mem_redundancy_at_eps_p10": redundancy_at_eps(M, eps_p10),
        "mem_redundancy_at_eps_p5": redundancy_at_eps(M, eps_p5),
        # family 5
        "mem_silhouette": aln.memory_silhouette,
        "bridging_fraction": aln.bridging_fraction,
    }
    if q is not None:
        qf = query_features(q, T, M, eps, k)
        out.update(
            {
                "q_nn_mem": qf.q_nn_mem,
                "q_topk_mean_mem": qf.q_topk_mean_mem,
                "q_memory_density": qf.q_memory_density,
                "q_memory_density_p10": query_memory_density(q, M, eps_p10),
                "q_memory_density_p5": query_memory_density(q, M, eps_p5),
                "q_nn_turn": qf.q_nn_turn,
            }
        )
    return out
