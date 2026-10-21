"""Build a local conversation atlas with real embeddings and shared projections.

Inputs are public benchmark JSON and optional saved memory snapshots. Outputs
contain source text and belong under ignored outputs/, never in site/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

MODEL = "redis/langcache-embed-v3-small"


def collect(dataset: list[dict], runs: list[tuple[str, Path]], memory_limit: int = 40) -> dict:
    texts: list[str] = []
    text_index: dict[str, int] = {}

    def point(text: str) -> int:
        text = str(text).strip()
        if text not in text_index:
            text_index[text] = len(texts)
            texts.append(text)
        return text_index[text]

    sessions, histories, session_index = [], [], {}
    by_question = {}
    for index, row in enumerate(dataset):
        history = {"id": f"h{index + 1}", "question": row["question"], "type": row.get("question_type", ""), "sessions": [], "query": point(row["question"]), "memories": {}, "memoryTotals": {}}
        for position, raw in enumerate(row["haystack_sessions"]):
            turns = [(str(turn.get("role", "unknown")), str(turn.get("content", "")).strip()) for turn in raw if str(turn.get("content", "")).strip()]
            if not turns:
                continue
            key = hashlib.sha256(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
            if key not in session_index:
                session_index[key] = len(sessions)
                dates = row.get("haystack_dates") or []
                label = next((text for role, text in turns if role == "user"), turns[0][1])
                sessions.append({"id": f"c{len(sessions) + 1}", "label": label[:140], "date": str(dates[position]) if position < len(dates) else "", "turns": [{"p": point(text), "role": role} for role, text in turns], "histories": []})
            session_id = session_index[key]
            history["sessions"].append(session_id)
            if index not in sessions[session_id]["histories"]:
                sessions[session_id]["histories"].append(index)
        histories.append(history)
        by_question[str(row.get("question_id"))] = history

    systems = []
    for index, (label, path) in enumerate(runs):
        system_id = f"m{index + 1}"
        systems.append({"id": system_id, "label": label, "sampleLimit": memory_limit})
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                history = by_question.get(str(row.get("question_id")))
                if history is None:
                    continue
                memories = list(dict.fromkeys(memory.strip() for memory in row.get("memories", []) if isinstance(memory, str) and memory.strip()))
                history["memoryTotals"][system_id] = len(memories)
                if len(memories) > memory_limit:
                    memories = [memories[i] for i in np.linspace(0, len(memories) - 1, memory_limit, dtype=int)]
                history["memories"][system_id] = [point(memory) for memory in memories]
    return {"schemaVersion": 2, "title": "LongMemEval-small", "source": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned", "embedding": {"model": MODEL, "kind": "analysis", "maxTokens": 128, "note": "Shared analysis embeddings, not the memory systems’ native index vectors. Long turns are truncated by the model at 128 tokens."}, "texts": texts, "sessions": sessions, "histories": histories, "systems": systems}


def embed(bundle: dict, output: Path, device: str) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL, device=device, trust_remote_code=False)
    model.max_seq_length = 128
    dimension = model.get_sentence_embedding_dimension()
    revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    bundle["embedding"].update({"revision": revision, "dimensions": dimension})
    texts = bundle["texts"]
    vectors = np.empty((len(texts), dimension), dtype=np.float32)
    cache = sqlite3.connect(output / "embedding-cache.sqlite")
    cache.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
    keys = [hashlib.sha256(f"{MODEL}\0{revision}\0128\0{text}".encode()).hexdigest() for text in texts]
    missing = []
    for start in range(0, len(texts), 700):
        batch = keys[start:start + 700]
        saved = dict(cache.execute(f"SELECT key,vector FROM embeddings WHERE key IN ({','.join('?' for _ in batch)})", batch))
        for offset, key in enumerate(batch):
            if key in saved:
                vectors[start + offset] = np.frombuffer(saved[key], dtype=np.float32)
            else:
                missing.append(start + offset)
    print(f"Embed: {len(texts):,} texts; {len(texts) - len(missing):,} cached; {len(missing):,} new; device={device}", flush=True)
    begin = time.monotonic()
    for start in range(0, len(missing), 512):
        indices = missing[start:start + 512]
        values = model.encode([texts[i] for i in indices], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        for i, value in zip(indices, values):
            vectors[i] = value
            cache.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)", (keys[i], value.tobytes()))
        cache.commit()
        done = min(start + len(indices), len(missing))
        if start % 2048 == 0 or done == len(missing):
            print(f"Embedded {done:,}/{len(missing):,} · {done / max(time.monotonic() - begin, .01):.0f} texts/s", flush=True)
    cache.close()
    np.save(output / "vectors.npy", vectors)
    (output / "source.json").write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    return vectors


def project(bundle: dict, vectors: np.ndarray, output: Path) -> None:
    import umap
    from sklearn.decomposition import PCA

    print(f"Project: {len(vectors):,} real embedding vectors", flush=True)
    dimension = min(64, vectors.shape[0] - 1, vectors.shape[1])
    pca = PCA(n_components=dimension, svd_solver="randomized", random_state=42, iterated_power=3)
    reduced = pca.fit_transform(vectors)
    print("PCA complete. Computing UMAP neighborhoods and layout…", flush=True)
    reducer = umap.UMAP(n_components=2, n_neighbors=min(20, len(vectors) - 1), min_dist=.12, metric="cosine", random_state=42, n_jobs=1, n_epochs=150, low_memory=True, verbose=True)
    coordinates = reducer.fit_transform(reduced)
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP produced nonfinite coordinates")
    texts = bundle.pop("texts")
    bundle["projections"] = {
        "umap": {"method": "UMAP", "metric": "cosine", "neighbors": 20, "minDist": .12, "seed": 42, "epochs": 150, "preprocessing": f"PCA to {dimension} dimensions for the neighborhood search", "fitPoints": len(vectors)},
        "pca": {"method": "PCA", "variance": float(pca.explained_variance_ratio_[:2].sum()), "fitPoints": len(vectors)},
    }
    bundle["points"] = [{"text": text, "umap": [round(float(x), 5) for x in xy], "pca": [round(float(x), 6) for x in pc[:2]]} for text, xy, pc in zip(texts, coordinates, reduced)]
    target = output / "atlas.json"
    temporary = output / "atlas.pending.json"
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(target)
    print(f"Atlas ready: {len(bundle['histories'])} histories; {len(bundle['sessions']):,} conversations; {sum(len(s['turns']) for s in bundle['sessions']):,} turns; {target.stat().st_size / 1e6:.1f} MB", flush=True)


def add_3d(output: Path) -> None:
    """Add independently fitted 3D coordinates without changing the existing 2D map."""
    import umap
    from sklearn.decomposition import PCA

    target = output / "atlas.json"
    bundle = json.loads(target.read_text())
    vectors = np.load(output / "vectors.npy", mmap_mode="r")
    if len(vectors) != len(bundle["points"]):
        raise ValueError("The existing atlas and vector rows do not match")
    dimension = min(64, vectors.shape[0] - 1, vectors.shape[1])
    if dimension < 3:
        raise ValueError("A real 3D projection needs at least four points and three embedding dimensions")
    print(f"Adding 3D projections for {len(vectors):,} cached real vectors; no embedding calls", flush=True)
    pca = PCA(n_components=dimension, svd_solver="randomized", random_state=42, iterated_power=3)
    reduced = pca.fit_transform(vectors)
    for point, pc in zip(bundle["points"], reduced):
        point["pca3d"] = [round(float(value), 6) for value in pc[:3]]
    bundle["projections3d"] = {"pca": {"method": "PCA", "dimensions": 3, "variance": float(pca.explained_variance_ratio_[:3].sum()), "fitPoints": len(vectors)}}
    temporary = output / "atlas.pending.json"
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(target)
    print("3D PCA saved. Fitting a separate 3D UMAP layout…", flush=True)
    reducer = umap.UMAP(n_components=3, n_neighbors=min(20, len(vectors) - 1), min_dist=.12, metric="cosine", random_state=42, n_jobs=1, n_epochs=150, low_memory=True, verbose=True)
    coordinates = reducer.fit_transform(reduced)
    if not np.isfinite(coordinates).all():
        raise ValueError("3D UMAP produced nonfinite coordinates")
    for point, xyz in zip(bundle["points"], coordinates):
        point["umap3d"] = [round(float(value), 5) for value in xyz]
    bundle["projections3d"]["umap"] = {"method": "UMAP", "dimensions": 3, "metric": "cosine", "neighbors": 20, "minDist": .12, "seed": 42, "epochs": 150, "preprocessing": f"PCA to {dimension} dimensions for the neighborhood search", "fitPoints": len(vectors)}
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(target)
    print("Both real 3D layouts ready; existing 2D coordinates preserved", flush=True)


def add_tsne(output: Path, perplexity: float = 30, jobs: int = 4, *, iterations: int = 1000, exaggeration: float = 1, layout: str | None = None, metric: str = "euclidean", pca_dimensions: int = 64, set_default: bool = False) -> None:
    """Fit all cached points together, preserving every existing layout."""
    from openTSNE import TSNE
    from sklearn.decomposition import PCA

    target = output / "atlas.json"
    bundle = json.loads(target.read_text())
    vectors = np.load(output / "vectors.npy", mmap_mode="r")
    if len(vectors) != len(bundle["points"]) or not np.isfinite(vectors).all():
        raise ValueError("Cached vectors must be finite and match atlas point rows")
    if not 0 < perplexity < len(vectors) / 3:
        raise ValueError("Perplexity must be positive and less than one third of the point count")
    if iterations < 1 or not np.isfinite(exaggeration) or exaggeration < 1:
        raise ValueError("Iterations must be positive and exaggeration must be at least one")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("t-SNE metric must be euclidean or cosine")
    if pca_dimensions < 0:
        raise ValueError("PCA dimensions must be nonnegative; zero uses original embeddings")
    if layout is not None and (not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", layout) or layout == "original"):
        raise ValueError("Use a short lowercase layout name other than reserved 'original'")
    if layout and not bundle.get("projections", {}).get("tsne"):
        raise ValueError("A named comparison layout requires an existing base t-SNE fit")
    dimension = min(pca_dimensions, len(vectors) - 1, vectors.shape[1])
    print(f"t-SNE: fitting all {len(vectors):,} cached vectors; perplexity={perplexity}; iterations={iterations}; exaggeration={exaggeration}; layout={layout or 'default'}; no embedding calls", flush=True)
    reduced = PCA(n_components=dimension, svd_solver="randomized", random_state=42, iterated_power=3).fit_transform(vectors) if dimension else np.array(vectors, dtype=np.float32, copy=True)
    coordinates = np.asarray(TSNE(n_components=2, perplexity=perplexity, metric=metric, initialization="pca", neighbors="pynndescent", negative_gradient_method="fft", early_exaggeration_iter=250, n_iter=iterations, exaggeration=exaggeration, learning_rate="auto", n_jobs=jobs, random_state=42, verbose=True).fit(reduced))
    if coordinates.shape != (len(vectors), 2) or not np.isfinite(coordinates).all():
        raise ValueError("t-SNE produced invalid coordinates")
    # Read again: a separate correspondence export may have added metadata meanwhile.
    bundle = json.loads(target.read_text())
    if len(bundle["points"]) != len(coordinates):
        raise ValueError("The atlas point set changed while t-SNE was fitting")
    metadata = {"method": "t-SNE", "implementation": "openTSNE", "dimensions": 2, "metric": metric, "perplexity": perplexity, "seed": 42, "iterations": iterations, "earlyExaggerationIterations": 250, "exaggeration": exaggeration, "initialization": "pca", "learningRate": "auto", "negativeGradient": "fft", "neighbors": "pynndescent", "preprocessing": f"PCA to {dimension} dimensions for neighborhood search" if dimension else "Original embeddings; no PCA preprocessing of neighborhoods", "pcaDimensions": dimension, "fitPoints": len(vectors)}
    key = f"tsne_{layout}" if layout else "tsne"
    for point, xy in zip(bundle["points"], coordinates):
        point[key] = [round(float(value), 6) for value in xy]
    if layout:
        variants = bundle.setdefault("tsneLayouts", {})
        variants.setdefault("original", {**bundle["projections"]["tsne"], "pointKey": "tsne", "label": "Original"})
        variants[layout] = {**metadata, "pointKey": key, "label": f"{layout.capitalize()} · P{perplexity:g}"}
        if set_default or "tsneDefault" not in bundle:
            bundle["tsneDefault"] = layout
    else:
        bundle.setdefault("projections", {})["tsne"] = metadata
        if "original" in bundle.get("tsneLayouts", {}):
            bundle["tsneLayouts"]["original"] = {**metadata, "pointKey": "tsne", "label": "Original"}
    temporary = output / "atlas-tsne.pending.json"
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(target)
    print("Real 2D t-SNE ready; UMAP/PCA and all point identities preserved", flush=True)


def export_correspondence(output: Path) -> None:
    """Export original-space cosine matrices separately, one file per history."""
    target = output / "atlas.json"
    bundle = json.loads(target.read_text())
    vectors = np.load(output / "vectors.npy", mmap_mode="r")
    if len(vectors) != len(bundle["points"]) or vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Cached vectors must be finite and match atlas point rows")
    digest = hashlib.sha256(json.dumps(bundle["embedding"], sort_keys=True).encode())
    for point in bundle["points"]:
        text = point["text"].encode()
        digest.update(len(text).to_bytes(8, "little")); digest.update(text)
    fingerprint = digest.hexdigest()
    folder = output / "correspondence"
    folder.mkdir(exist_ok=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Cannot compute cosine similarity for a zero embedding")
    normalized = np.asarray(vectors / norms, dtype=np.float32)
    print(f"Correspondence: {len(bundle['histories'])} histories; original {vectors.shape[1]}D cosine; no projection distances", flush=True)
    for index, history in enumerate(bundle["histories"]):
        rows = [[session, turn, item["p"]] for session in history["sessions"] for turn, item in enumerate(bundle["sessions"][session]["turns"])]
        turns = normalized[[row[2] for row in rows]]
        systems = {}
        for system, memories in history["memories"].items():
            scores = np.clip(turns @ normalized[memories].T, -1, 1) if memories else np.empty((len(rows), 0))
            systems[system] = {"memories": memories, "total": history.get("memoryTotals", {}).get(system, len(memories)), "scores": np.round(scores.astype(np.float64), 6).ravel().tolist()}
        payload = {"schemaVersion": 1, "fingerprint": fingerprint, "history": index, "metric": "cosine", "dimensions": vectors.shape[1], "rows": rows, "systems": systems}
        destination = folder / f"h{index + 1}.json"
        temporary = destination.with_suffix(".pending.json")
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        temporary.replace(destination)
        if index % 50 == 0 or index + 1 == len(bundle["histories"]):
            print(f"Correspondence saved: {index + 1}/{len(bundle['histories'])}", flush=True)
    bundle = json.loads(target.read_text())
    bundle["correspondence"] = {"schemaVersion": 1, "fingerprint": fingerprint, "metric": "cosine", "dimensions": vectors.shape[1], "files": "correspondence/h{history}.json", "precision": 6, "scope": "Exported memory overlays only; samples are not complete stores. Similarity is not verified provenance or factual coverage."}
    temporary = output / "atlas-correspondence.pending.json"
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(target)
    print("Correspondence ready; vectors stay private and are never served", flush=True)


def select_tsne_layout(output: Path, layout: str) -> None:
    target = output / "atlas.json"
    bundle = json.loads(target.read_text())
    if layout not in bundle.get("tsneLayouts", {}):
        raise ValueError("That t-SNE comparison layout has not been computed")
    bundle["tsneDefault"] = layout
    temporary = output / "atlas-default.pending.json"
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(target)
    print(f"Default t-SNE layout: {layout}; no refitting or coordinate changes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=ANSWERS_JSONL")
    parser.add_argument("--output", type=Path, default=Path("outputs/atlas"))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--memory-limit", type=int, default=40, help="Uniformly sample at most this many memories per history and system")
    parser.add_argument("--project-only", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--add-3d", action="store_true", help="Add 3D PCA and UMAP to an existing atlas using its cached vectors; preserve its 2D coordinates")
    parser.add_argument("--add-tsne", action="store_true", help="Add a shared 2D t-SNE layout from cached vectors, preserving existing projections")
    parser.add_argument("--perplexity", type=float, default=30, help="t-SNE perplexity (default: 30)")
    parser.add_argument("--iterations", type=int, default=1000, help="Normal t-SNE optimization iterations, after 250 early iterations (default: 1000)")
    parser.add_argument("--exaggeration", type=float, default=1, help="Late t-SNE attraction multiplier; changes apparent compactness (default: 1)")
    parser.add_argument("--tsne-metric", choices=["euclidean", "cosine"], default="euclidean", help="Neighborhood metric after PCA (default: euclidean)")
    parser.add_argument("--tsne-pca-dimensions", type=int, default=64, help="PCA dimensions for t-SNE neighborhood search; 0 uses original embeddings (default: 64)")
    parser.add_argument("--tsne-default", action="store_true", help="Make this named t-SNE comparison fit the viewer default")
    parser.add_argument("--tsne-layout", help="Save a named comparison layout instead of replacing the original t-SNE")
    parser.add_argument("--jobs", type=int, default=4, help="t-SNE CPU workers (default: 4)")
    parser.add_argument("--correspondence", action="store_true", help="Export original-space turn-to-memory cosine matrices for an existing atlas")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tsne_default and not args.add_tsne:
        if not args.tsne_layout:
            parser.error("--tsne-default requires --tsne-layout")
        select_tsne_layout(args.output, args.tsne_layout)
        return
    if args.add_3d or args.correspondence or args.add_tsne:
        if args.add_3d:
            add_3d(args.output)
        if args.correspondence:
            export_correspondence(args.output)
        if args.add_tsne:
            add_tsne(args.output, args.perplexity, args.jobs, iterations=args.iterations, exaggeration=args.exaggeration, layout=args.tsne_layout, metric=args.tsne_metric, pca_dimensions=args.tsne_pca_dimensions, set_default=args.tsne_default)
        return
    if args.project_only:
        bundle = json.loads((args.output / "source.json").read_text())
        vectors = np.load(args.output / "vectors.npy")
    else:
        if not args.dataset or not args.dataset.is_file():
            parser.error("Provide a local public benchmark file with --dataset")
        runs = []
        for value in args.run:
            if "=" not in value:
                parser.error("--run requires LABEL=ANSWERS_JSONL")
            label, filename = value.split("=", 1)
            path = Path(filename)
            if not label or not path.is_file():
                parser.error("Each --run requires a label and an existing JSONL file")
            runs.append((label, path))
        if args.memory_limit < 1:
            parser.error("--memory-limit must be positive")
        bundle = collect(json.loads(args.dataset.read_text()), runs, args.memory_limit)
        print(f"Collected {len(bundle['histories'])} histories; {len(bundle['sessions']):,} conversations; {len(bundle['texts']):,} distinct texts", flush=True)
        vectors = embed(bundle, args.output, args.device)
    if not args.embed_only:
        project(bundle, vectors, args.output)


if __name__ == "__main__":
    main()
