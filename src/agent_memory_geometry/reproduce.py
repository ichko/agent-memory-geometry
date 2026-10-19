"""Recompute the main offline checks from the released numerical tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import evaluation as E
from .diagnostics import anchor_bins, reachability_gaps


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No output rows for {path.name}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_delta(y, first, second, groups, n_boot=1000, seed=0):
    """AUROC(second) minus AUROC(first), with fixed-OOF group bootstrap."""
    valid = np.isfinite(first) & np.isfinite(second)
    y, first, second, groups = y[valid], first[valid], second[valid], groups[valid]
    point = E._auroc(y, second) - E._auroc(y, first)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([by_group[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        d = E._auroc(y[idx], second[idx]) - E._auroc(y[idx], first[idx])
        if np.isfinite(d):
            deltas.append(d)
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas else (float("nan"), float("nan"))
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi)}


def add_covariates(rows, path):
    covariates = E._read_csv(path)
    lookup = {(r["question_id"], r["strategy"]): r for r in covariates}
    if len(lookup) != len(covariates):
        raise ValueError("Duplicate question covariates")
    types = sorted({r["question_type"] for r in rows})
    systems = sorted({r["strategy"] for r in rows})
    for r in rows:
        r.update(lookup[(r["question_id"], r["strategy"])])
        for t in types:
            r[f"qtype_{t}"] = float(r["question_type"] == t)
        for s in systems:
            r[f"system_{s}"] = float(r["strategy"] == s)
    difficulty = [f"qtype_{t}" for t in types] + [f"qpca64_{j}" for j in range(64)] + ["q_len_chars"]
    identity = [f"system_{s}" for s in systems]
    return difficulty, identity


def fit(rows, features):
    y = E._scores(rows)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary correctness labels")
    y = y.astype(int)
    groups = np.asarray([r["question_id"] for r in rows])
    p = E._grouped_cv_oof(E._matrix(rows, features), y, groups)
    if p is None or not np.isfinite(p).all():
        raise ValueError("Could not obtain predictions for every row")
    return y, p, groups


def context_summary(path):
    """Random subsets: average seed accuracies, not a new independent sample."""
    groups = defaultdict(list)
    for r in E._read_csv(path):
        if r["score"] not in {"0", "1"}:
            continue
        groups[(r.get("answer_model", "gpt-4o-mini"), int(r["k"]), r["condition"])].append(int(r["score"]))
    out = []
    random_groups = defaultdict(list)
    for (model, k, condition), values in sorted(groups.items()):
        if condition.startswith("random_k"):
            random_groups[(model, k)].append(values)
        else:
            out.append(dict(answer_model=model, k=k, condition=condition, n=len(values),
                            accuracy=float(np.mean(values)), seed_sd=0.0, n_seeds=1))
    for (model, k), samples in sorted(random_groups.items()):
        accuracies = [float(np.mean(s)) for s in samples]
        out.append(dict(answer_model=model, k=k, condition="random_k", n=sum(map(len, samples)) // len(samples),
                        accuracy=float(np.mean(accuracies)), seed_sd=float(np.std(accuracies)), n_seeds=len(samples)))
    return out


def selective_curve(y, p):
    order = np.argsort(-p)
    return [(float(c), float(np.mean(y[order[:max(1, round(c * len(y)))]])))
            for c in np.linspace(0.3, 1.0, 71)]


def make_plots(out, curves, labels, per_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in curves.items():
        ax.plot([c for c, _ in curve], [1 - a for _, a in curve], label=name)
    ax.axhline(1 - labels.mean(), color="gray", linestyle="--", label="random deferral (expected)")
    ax.set(xlabel="Coverage (fraction answered)", ylabel="Error rate among retained answers")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.pdf")
    plt.close(fig)

    geometry = [r for r in per_type if r["model"] == "geometry"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(geometry))
    ax.bar(xs, [r["auroc"] for r in geometry], color="#4e79a7")
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xticks(xs, [r["question_type"] for r in geometry], rotation=25, ha="right", fontsize=8)
    ax.set(ylabel="Grouped out-of-fold AUROC", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out / "per_type.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=E.PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=E.PROJECT_ROOT / "outputs/reproduction")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--check-paper", action="store_true", help="Check selected published estimates to 0.002 tolerance")
    args = parser.parse_args()
    if args.n_boot < 0:
        parser.error("--n-boot must be nonnegative")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = E._read_csv(args.data_dir / "longmemeval/geometry_text-embedding-3-small.csv")
    difficulty, identity = add_covariates(rows, args.data_dir / "longmemeval/question_covariates.csv")
    correspondence = E.FAM_COVERAGE + E.FAM_MEMORY + E.FAM_ALIGNMENT
    models = {"system_identity": identity, "difficulty": difficulty, "geometry": E.ALL_GEOMETRY,
              "difficulty+identity": difficulty + identity,
              "difficulty+identity+geometry": difficulty + identity + E.ALL_GEOMETRY,
              "reachability": E.FAM_QUERY, "correspondence": correspondence}
    predictions, summary = {}, []
    for name, features in models.items():
        y, p, g = fit(rows, features)
        predictions[name] = p
        auc = E._auroc(y, p)
        summary.append(dict(model=name, n=len(rows), n_features=len(features), auroc=auc))
        print(f"{name}: {auc:.6f}", flush=True)
    write_csv(out / "pooled.csv", summary)
    write_csv(out / "reachability_gaps.csv", reachability_gaps(rows))
    anchors = E._read_csv(args.data_dir / "longmemeval/answer_anchors.csv")
    write_csv(out / "answer_anchor_proxies.csv", anchor_bins(anchors, 75) + anchor_bins(anchors, 90))
    labels = {(r["question_id"], r["strategy"]): int(r["score"]) for r in rows}
    confidence = E._read_csv(args.data_dir / "longmemeval/confidence.csv")
    confidence_summary = []
    for field in ["confidence", "p_true"]:
        available = [r for r in confidence if np.isfinite(float(r[field]))]
        cy = np.asarray([labels[(r["question_id"], r["strategy"])] for r in available])
        cp = np.asarray([float(r[field]) for r in available])
        confidence_summary.append(dict(model=field, n=len(available), auroc=E._auroc(cy, cp)))
    write_csv(out / "confidence.csv", confidence_summary)
    paired = []
    for first, second in [("difficulty+identity", "difficulty+identity+geometry"),
                          ("reachability", "geometry"), ("correspondence", "geometry"),
                          ("correspondence", "reachability")]:
        paired.append({"comparison": f"{second} minus {first}",
                       **paired_delta(y, predictions[first], predictions[second], g, args.n_boot)})
    write_csv(out / "paired_differences.csv", paired)
    per_type = []
    for qt in sorted({r["question_type"] for r in rows}):
        subset = [r for r in rows if r["question_type"] == qt]
        for name, features in [("geometry", E.ALL_GEOMETRY), ("system_identity", identity),
                               ("memory_count", ["mem_count"]), ("reachability", E.FAM_QUERY),
                               ("correspondence", correspondence)]:
            a, lo, hi = E._auroc_with_group_ci(E._matrix(subset, features), E._scores(subset),
                                              np.asarray([r["question_id"] for r in subset]), args.n_boot)
            per_type.append(dict(question_type=qt, model=name, n=len(subset), auroc=a, ci_lo=lo, ci_hi=hi))
    write_csv(out / "per_type.csv", per_type)

    loso = []
    X = E._matrix(rows, E.ALL_GEOMETRY)
    systems = np.asarray([r["strategy"] for r in rows])
    for held in sorted(set(systems)):
        tr, te = systems != held, systems == held
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1, max_iter=3000).fit(scaler.transform(X[tr]), y[tr])
        p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        loso.append(dict(system=held, n=int(te.sum()), auroc=E._auroc(y[te], p)))
    write_csv(out / "leave_one_system_out.csv", loso)

    embedders = []
    for path in sorted((args.data_dir / "longmemeval").glob("geometry_*.csv")):
        current = E._read_csv(path)
        ey, ep, _ = fit(current, E.ALL_GEOMETRY)
        embedders.append(dict(embedder=path.stem.removeprefix("geometry_"), n=len(current), auroc=E._auroc(ey, ep)))
    write_csv(out / "embedders.csv", embedders)
    for benchmark in ["longmemeval", "locomo"]:
        write_csv(out / f"selection_{benchmark}.csv", context_summary(args.data_dir / f"context_selection/{benchmark}.csv"))

    locomo = E._read_csv(args.data_dir / "locomo/nearest_turn.csv")
    categories = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    lsummary = []
    for category, name in categories.items():
        subset = [r for r in locomo if int(r["category"]) == category]
        ly = np.asarray([int(r["score"]) for r in subset])
        distances = np.asarray([float(r["nearest_turn_distance"]) for r in subset])
        lsummary.append(dict(category=name, n=len(subset), accuracy=float(ly.mean()),
                             distance_gap=float(distances[ly == 0].mean() - distances[ly == 1].mean()),
                             auroc=E._auroc(ly, -distances)))
    write_csv(out / "locomo_categories.csv", lsummary)
    curves = {name: selective_curve(y, predictions[name]) for name in ["geometry", "difficulty", "difficulty+identity", "difficulty+identity+geometry"]}
    write_csv(out / "selective_answering.csv", [dict(model=name, coverage=c, accuracy=a)
                                               for name, curve in curves.items() for c, a in curve])
    if not args.no_plots:
        make_plots(out, curves, y, per_type)
    checks = {"rows": len(rows), "questions": len(set(g)),
              "empty_memory_representations": sum(float(r["mem_count"]) == 0 for r in rows),
              "undefined_feature_entries": int(sum(not np.isfinite(E._to_float(r[f])) for r in rows for f in E.ALL_GEOMETRY)),
              "loso_mean": float(np.mean([r["auroc"] for r in loso])),
              "embedder_mean": float(np.mean([r["auroc"] for r in embedders])),
              "embedder_sample_sd": float(np.std([r["auroc"] for r in embedders], ddof=1))}
    if args.check_paper:
        expected = {"geometry": .640, "difficulty": .655, "difficulty+identity": .679,
                    "difficulty+identity+geometry": .681, "reachability": .598, "correspondence": .580}
        for r in summary:
            if r["model"] in expected and abs(r["auroc"] - expected[r["model"]]) > .002:
                raise AssertionError(f"Published estimate mismatch: {r['model']}")
        assert (checks["rows"], checks["questions"], checks["undefined_feature_entries"]) == (2367, 500, 2495)
        assert abs(min(r["auroc"] for r in loso) - .467) < .002
        ssa = next(r for r in per_type if r["question_type"] == "single-session-assistant" and r["model"] == "geometry")
        assert abs(ssa["auroc"] - .841) < .002
        singlehop = next(r for r in lsummary if r["category"] == "single-hop")
        assert abs(singlehop["auroc"] - .603) < .002
        assert sum(r["n"] for r in lsummary) == 1986
        checks["published_checks_passed"] = True
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Wrote results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
