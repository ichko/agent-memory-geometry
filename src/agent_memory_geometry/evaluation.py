"""Published five-fold grouped logistic evaluation and fixed-OOF bootstrap.

Historical protocol: column-mean imputation precedes cross-validation.
Scaling uses training folds only. See docs/protocol.md for limitations.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "longmemeval"
TABLES_DIR = PROJECT_ROOT / "outputs" / "evaluation"

logger = logging.getLogger("predict")

# The 25 features used in the published model.
FAM_CONVERSATION = [
    "n_turns",
    "n_sessions",
    "turn_silhouette",
    "intra_inter_ratio",
    "effective_clusters",
]
FAM_COVERAGE = [
    "cov_mean_nn",
    "cov_max_nn",
    "cov_hausdorff",
    "cov_mean_session",
    "cov_worst_session",
    # Multi-ε coverage — the p25 version saturates at 1.0 on LongMemEval;
    # p10 and p5 are informative.
    "cov_at_eps_p10",
    "cov_at_eps_p5",
]
FAM_MEMORY = [
    "mem_count",
    "mem_eff_rank",
    "mem_mean_pairwise",
    # Multi-ε redundancy — same saturation argument as above.
    "mem_redundancy_at_eps_p10",
    "mem_redundancy_at_eps_p5",
]
FAM_QUERY = [
    "q_nn_mem",
    "q_topk_mean_mem",
    "q_memory_density",
    "q_memory_density_p10",
    "q_memory_density_p5",
    "q_nn_turn",
]
FAM_ALIGNMENT = [
    "mem_silhouette",
    "bridging_fraction",
]

ALL_GEOMETRY = (
    FAM_CONVERSATION
    + FAM_COVERAGE
    + FAM_MEMORY
    + FAM_QUERY
    + FAM_ALIGNMENT
)

# Strategy one-hot columns are appended to each row in ``main`` so they
# can be used as ordinary numeric features. The exact names depend on
# which strategies appear in the CSV; populated lazily.
STRATEGY_PREFIX = "_strategy_"

# Models compared in the summary.
MODELS: dict[str, list[str]] = {
    "covariates_only": FAM_CONVERSATION,
    "coverage_only": FAM_COVERAGE,
    "memory_only": FAM_MEMORY,
    "query_only": FAM_QUERY,
    "alignment_only": FAM_ALIGNMENT,
    "geometry_all": ALL_GEOMETRY,
    "strategy_only": [],  # filled in main once strategy columns are known
    "covariates_plus_strategy": list(FAM_CONVERSATION),
    "geometry_plus_strategy": list(ALL_GEOMETRY),
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(x: Any) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    X = np.array(
        [[_to_float(r.get(f)) for f in features] for r in rows], dtype=np.float64
    )
    # Mean-impute NaNs column-wise.
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isfinite(col)
        if not m.any():
            X[:, j] = 0.0
        else:
            X[~m, j] = float(col[m].mean())
    return X


def _scores(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([_to_float(r.get("score")) for r in rows], dtype=np.float64)


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC. Returns NaN if only one class present."""
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U formulation.
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average rank handling for ties.
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(ranks[pos].sum())
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _grouped_cv_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray | None:
    """Out-of-fold predictions via *group-aware* stratified k-fold CV.

    StratifiedGroupKFold keeps all rows of each question in one fold.
    This controls the source of leakage from repeated questions across systems.

    Standardisation is fit on the training fold only. Returns the OOF
    probability vector, or ``None`` if the split is infeasible.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    n_groups = int(np.unique(groups).size)
    if (
        X.shape[0] < n_folds * 2
        or y.sum() < n_folds
        or (1 - y).sum() < n_folds
        or n_groups < n_folds
    ):
        return None
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    try:
        splits = list(sgkf.split(X, y, groups))
    except ValueError:
        return None
    for train_idx, test_idx in splits:
        # Guard against a fold whose training side is single-class.
        if y[train_idx].sum() == 0 or (1 - y[train_idx]).sum() == 0:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(
            scaler.transform(X[train_idx]), y[train_idx]
        )
        oof[test_idx] = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
    if np.isnan(oof).all():
        return None
    return oof


def _auroc_with_group_ci(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate + 95% CI of grouped-CV AUROC.

    The point estimate is the AUROC of a single grouped-CV pass. The CI is a
    *group-level* bootstrap of the held-out predictions: we resample whole
    question groups with replacement and recompute AUROC on the OOF vector.
    The interval conditions on fixed OOF predictions. It does not account for
    model retraining, feature selection, or the historical global imputation.
    """
    oof = _grouped_cv_oof(X, y, groups, n_folds=n_folds, seed=seed)
    if oof is None:
        return float("nan"), float("nan"), float("nan")
    valid = ~np.isnan(oof)
    yv = y[valid]
    pv = oof[valid]
    gv = groups[valid]
    point = _auroc(yv, pv)
    if not math.isfinite(point) or n_boot <= 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    uniq = np.unique(gv)
    g2idx = {g: np.where(gv == g)[0] for g in uniq}
    samples: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([g2idx[g] for g in chosen])
        a = _auroc(yv[idx], pv[idx])
        if math.isfinite(a):
            samples.append(a)
    if not samples:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return point, lo, hi


def _group_rows(
    rows: list[dict[str, Any]], col: str
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r.get(col, "")), []).append(r)
    return out


def _feature_importance(
    rows: list[dict[str, Any]], features: list[str]
) -> dict[str, float]:
    """Standardized LR coefficients on all data (no CV) — for ranking only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = _matrix(rows, features)
    y = _scores(rows)
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if y.sum() < 5 or (1 - y).sum() < 5:
        return {f: float("nan") for f in features}
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(Xs, y)
    return dict(zip(features, clf.coef_[0].astype(float)))


# ── orchestration ───────────────────────────────────────────────────────


def _eval_models(
    rows: list[dict[str, Any]],
    group_label: str,
    out: list[dict[str, Any]],
    n_boot: int = 200,
) -> None:
    y = _scores(rows)
    mask = np.isfinite(y)
    y = y[mask].astype(int)
    if y.size < 30 or y.sum() < 5 or (1 - y).sum() < 5:
        logger.info(
            "Skipping %s (n=%d, pos=%d, neg=%d)",
            group_label,
            int(mask.sum()),
            int(y.sum()) if y.size else 0,
            int((1 - y).sum()) if y.size else 0,
        )
        return
    sub_rows = [r for r, m in zip(rows, mask) if m]
    # Repeated rows of the same question stay in one fold across all systems.
    groups = np.array([str(r.get("question_id", i)) for i, r in enumerate(sub_rows)])
    for model_name, features in MODELS.items():
        X = _matrix(sub_rows, features)
        point, lo, hi = _auroc_with_group_ci(
            X, y, groups, n_boot=n_boot, n_folds=5
        )
        out.append(
            {
                "group": group_label,
                "model": model_name,
                "n_features": len(features),
                "n": int(y.size),
                "auroc": round(point, 4) if math.isfinite(point) else "",
                "ci_lo": round(lo, 4) if math.isfinite(lo) else "",
                "ci_hi": round(hi, 4) if math.isfinite(hi) else "",
            }
        )
        logger.info(
            "%-32s %-18s n=%4d  AUROC=%.3f [%.3f, %.3f]",
            group_label,
            model_name,
            int(y.size),
            point if math.isfinite(point) else float("nan"),
            lo if math.isfinite(lo) else float("nan"),
            hi if math.isfinite(hi) else float("nan"),
        )


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = ["group", "model", "n_features", "n", "auroc", "ci_lo", "ci_hi"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_importance_csv(coefs: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not coefs:
        path.write_text("")
        return
    groups = list(coefs.keys())
    features = sorted({f for g in coefs.values() for f in g.keys()})
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", *groups])
        for feat in features:
            w.writerow(
                [feat, *[f"{coefs[g].get(feat, float('nan')):.4f}" for g in groups]]
            )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DATA_DIR / "geometry_text-embedding-3-small.csv"))
    p.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to output CSV filenames (e.g. '_large').",
    )
    args = p.parse_args()
    suf = args.suffix

    rows = _read_csv(Path(args.csv))
    if not rows:
        logger.error("Empty input: %s", args.csv)
        return 1
    logger.info("Loaded %d rows", len(rows))

    # ── Add strategy one-hot columns ────────────────────────────────────
    strategies = sorted({str(r.get("strategy", "")) for r in rows if r.get("strategy")})
    strategy_cols: list[str] = []
    for s in strategies:
        col = f"{STRATEGY_PREFIX}{s}"
        strategy_cols.append(col)
        for r in rows:
            r[col] = 1 if str(r.get("strategy", "")) == s else 0
    # Inject strategy cols into the relevant model definitions.
    MODELS["strategy_only"] = list(strategy_cols)
    MODELS["covariates_plus_strategy"] = list(FAM_CONVERSATION) + strategy_cols
    MODELS["geometry_plus_strategy"] = list(ALL_GEOMETRY) + strategy_cols
    logger.info(
        "Detected %d strategies; appended one-hot columns: %s",
        len(strategies),
        strategy_cols,
    )

    summary: list[dict[str, Any]] = []

    # 1) Overall
    logger.info("\n=== Overall ===")
    _eval_models(rows, "overall", summary, n_boot=args.n_boot)

    # 2) Per strategy
    logger.info("\n=== Per strategy ===")
    for strategy, sub in _group_rows(rows, "strategy").items():
        _eval_models(sub, f"strategy={strategy}", summary, n_boot=args.n_boot)

    # 3) Per question type
    logger.info("\n=== Per question type ===")
    for qt, sub in sorted(_group_rows(rows, "question_type").items()):
        _eval_models(sub, f"qtype={qt}", summary, n_boot=args.n_boot)

    _write_summary_csv(summary, args.output_dir / f"auroc_summary{suf}.csv")

    # 4) Feature importance (geometry_all, by group)
    logger.info("\n=== Feature importance (standardized LR coefs) ===")
    importance: dict[str, dict[str, float]] = {}
    importance["overall"] = _feature_importance(rows, ALL_GEOMETRY)
    for strategy, sub in _group_rows(rows, "strategy").items():
        importance[f"strategy={strategy}"] = _feature_importance(sub, ALL_GEOMETRY)
    for qt, sub in sorted(_group_rows(rows, "question_type").items()):
        importance[f"qtype={qt}"] = _feature_importance(sub, ALL_GEOMETRY)

    # Print top-by-|coef| for overall
    overall = importance["overall"]
    print("\n  overall — top 10 by |coef|:")
    for name, coef in sorted(overall.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"    {name:30s} {coef:+.3f}")

    _write_importance_csv(importance, args.output_dir / f"feature_importance{suf}.csv")

    print(
        f"\nWrote: {args.output_dir}/auroc_summary{suf}.csv, "
        f"feature_importance{suf}.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
