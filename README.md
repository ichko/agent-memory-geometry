# The Geometry of Agent Memory

Analysis code and numerical artifacts for:

**The Geometry of Agent Memory: Forecasting Retrieval-Bound Memory Failures, and Characterizing Its Limits**

Iliya Zhechev, Radoslav Ralev, Chris Coleman, Srijith Rajamohan, Jen Agarwal, Tyler Hutcherson, and Simba Khadder.

Accepted as a poster at PALM, NeurIPS 2026.

[arXiv](https://arxiv.org/abs/2610.26433) · [Paper and reviews](https://openreview.net/forum?id=xP0xvmQZhB) · [Paper PDF](paper/geometry_of_agent_memory.pdf) · [Live demo](https://www.izhv.eu/agent-memory-geometry/) · [Protocol](docs/protocol.md)

## What this study shows

We measure geometric relationships between conversation turns, memory texts, and queries. A supervised logistic classifier uses 25 features to predict answer correctness.

Geometry reaches 0.640 pooled AUROC on 2,367 LongMemEval question–system pairs. Question difficulty alone reaches 0.655.
After controlling for difficulty and system identity, geometry adds only +0.002 AUROC. Its reported confidence interval includes zero.

Performance varies by question type. The strongest category result also has a strong system-identity baseline.
These results characterize the evaluated representations. They do not establish a reliable deployment diagnostic or verify why individual answers fail.

## Memory Atlas explorer

[Open the live demo](https://www.izhv.eu/agent-memory-geometry/).

The static explorer under `site/` shows real conversation sessions as chronological paths in a shared embedding map.
UMAP is the default. Optional t-SNE and PCA provide comparison views. Toggle 2D/3D when real three-axis layouts are present; t-SNE is 2D only.
In 3D, drag to rotate, Shift-drag to pan, and scroll to zoom. Select a path to read its turns.
The viewer supports history filtering, text search, memory overlays, and wheel/trackpad navigation.
**Inspect memories** links a stored memory to its closest source turns and a turn-by-memory heatmap.
These use precomputed cosine similarities in the original embedding space, not projected distances.
Sampled overlays are labeled; these are candidate correspondences, not verified provenance or factual coverage.

```sh
python3 tools/serve_atlas.py --data outputs/atlas/atlas.json
```

Open `http://127.0.0.1:8766/` for the paper page, then follow the Atlas link (or open `/atlas.html` directly).
The local atlas uses the public `redis/langcache-embed-v3-small` model.
The viewer contains no synthetic fallback and makes no embedding API calls.
Raw text, embeddings, and generated atlas files stay under ignored `outputs/`, outside the publishable site directory.
See [docs/explorer.md](docs/explorer.md) to build an atlas from your own local benchmark files.

The [`demo` branch](https://github.com/ichko/agent-memory-geometry/tree/demo) contains the standalone public demo,
including all benchmark conversations, sampled memories, precomputed layouts, and original-space similarities.
To serve it, select **Settings → Pages → Deploy from a branch → demo → /(root)**.
Credential-shaped examples and local paths are masked in displayed text; plotted coordinates and similarity scores are unchanged.
Code and build tools remain on `main`; vectors, caches, backend configurations, and generated answers are not published.

## Reproduce the analysis

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run memory-geometry-reproduce --check-paper
```

Run these commands from the repository root. Reproduction uses the included numerical tables. It needs no credentials, external services, or paid API calls.

The command writes CSV results and plots under `outputs/reproduction/`:

- Pooled geometry, difficulty, and system-identity comparisons.
- Paired AUROC differences with question-group bootstrap intervals.
- Separately fitted per-type classifiers and a memory-count baseline.
- Leave-one-system-out evaluation and four-embedder comparison.
- Fixed-store context-selection results and LoCoMo category scores.
- Selective-answering curves from out-of-fold predictions.

For a faster point-estimate check:

```sh
uv run memory-geometry-reproduce --check-paper --n-boot 0 --no-plots
```

For the original nine-model grouped evaluation:

```sh
uv run memory-geometry-evaluate --n-boot 1000
```

## Analyze your own embeddings

Prepare a numeric NPZ file with these arrays:

- `turns`: shape `(n_turns, embedding_dimension)`.
- `memories`: shape `(n_memories, embedding_dimension)`.
- `sessions`: one integer session label per turn.
- `query`: shape `(embedding_dimension,)`.

All vectors must use the same embedding model and dimension. Do not use a two-dimensional visualization as input.

```sh
uv run memory-geometry-features example.npz --output outputs/features.csv --k 5
```

The command computes geometry without uploading data. It reports additional diagnostic measurements alongside the model's 25 selected features.
See `ALL_GEOMETRY` in `src/agent_memory_geometry/evaluation.py` for the selected feature names.
See `selection.py` for closest, farthest, and random context-subset selection.

## Release scope

This is an **offline analysis release**, not the original multi-backend benchmark harness.
It includes feature computation, evaluation, numerical results, paper sources, and published figures.

It excludes backend integrations, backend deployment tooling, raw benchmark conversations, memory text, generated answers, embedding caches, credentials, and the source repository's history.
The separate `demo` branch is an explicitly approved exception: it contains the public benchmark visualization data, not a backend.
The tables retain correctness labels and derived numerical features. Question IDs use consistent anonymous identifiers.

Reproducing memory construction, answer generation, or judging requires separate implementations and the original benchmark data.
The release does not claim end-to-end replay of those stages.

The published protocol uses global mean imputation and globally fitted question PCA before cross-validation.
The code retains that protocol for numerical reproduction. Read the limitations in [docs/protocol.md](docs/protocol.md) before reusing the evaluation.

## Paper source

The preprint source and bibliography are under `paper/`. The NeurIPS checklist is not part of this preprint.

```sh
cd paper
pdflatex main.tex
pdflatex main.tex
```

The included `main.bbl` avoids a required bibliography rebuild. If you edit references, run BibTeX before the final two LaTeX passes.
The published figure PDFs are included. The analysis plots generated by this package are new plots, not exact replicas of the paper figures.

## Data attribution and reuse

The numerical artifacts derive from [LongMemEval](https://github.com/xiaowu0162/LongMemEval) and [LoCoMo](https://github.com/snap-research/locomo).
See [data/README.md](data/README.md) for provenance and field definitions.
LoCoMo carries a CC BY-NC 4.0 noncommercial restriction. No blanket software license has been assigned to this repository.
See [NOTICE.md](NOTICE.md) before reuse. Cite the paper using [CITATION.cff](CITATION.cff).
