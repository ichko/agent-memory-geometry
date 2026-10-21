# Working on this release

Use Python 3.11 or 3.12 with uv. Run `uv sync --locked` for dependencies.

Tests: `uv run python -m unittest discover -s tests -v`.
Numerical checks: `uv run memory-geometry-reproduce --check-paper --n-boot 0 --no-plots`.
Publication check: `uv run python tools/check_release.py`.
Explorer tests: `node --test tests/explorer.test.mjs` (Node.js 20+).
Local explorer: `python3 tools/serve_atlas.py --data outputs/atlas/atlas.json`.
The explorer uses plain HTML/CSS/JavaScript. No build step or external browser dependencies are required.
`site/index.html` is the lightweight paper landing page; `site/atlas.html` opens the explorer. The landing page must not load atlas data or `app.js`.
The landing page uses warm white, serif headings, red accents, and thin rules. The Atlas is a compact tool with a small sans-serif header, tight controls, warm neutral chrome, and the same red UI accent. Do not apply the UI accent palette to the plot's category colors, marker sizes, or path styling.
Keep the landing-page title and subtitle compact, without forced line breaks. Omit poster links and placeholders until the authors provide their own poster. Never publish an unrelated design-reference PDF.
The optional `explorer` extra provides local Sentence Transformers and UMAP preprocessing.
It also includes openTSNE. Add a shared real 2D t-SNE fit with `uv run python tools/build_atlas.py --output outputs/atlas --add-tsne`.
Add original-space cosine matrices with `uv run python tools/build_atlas.py --output outputs/atlas --correspondence`.
The matrices are lazy-loaded per history from ignored `outputs/atlas/correspondence/`; never publish them automatically.
The user-approved public site lives on the standalone `demo` branch, published from its root. Source stays on `main`.
Build reviewed public data with `tools/package_demo.py --approve-public-data` into an empty ignored output directory.
It verifies public-source turns, redacts credential-shaped display examples, preserves coordinates/scores, and exports gzip blocks plus a checksum manifest. Do not copy vectors, caches, or private run files to `demo`.
Memory-to-turn links and heatmap colors must use these original-space scores, never projected distances. Call them candidate matches, not verified provenance or fact coverage.
Add real 3D layouts to an existing local atlas with `uv run python tools/build_atlas.py --output outputs/atlas --add-3d`.
This reuses cached vectors and must preserve the existing 2D coordinates. Never invent a depth axis.

Keep historical evaluation choices explicit. Do not silently replace preprocessing and claim exact reproduction.
Do not commit raw conversations, questions, memories, generated answers, credentials, embedding caches, or backend deployments.
Use real embedding-model outputs with recorded provenance. Do not substitute synthetic examples in the viewer.
Conversation paths connect turns only within one session. Never silently join separate sessions or claim sampled memory overlays are complete.
Show all visible conversation paths as faint gray hairlines by default. Hover is a stronger C0-blue line; selection is thicker C4 purple. Lines themselves must be pickable between dots. Never use black highlights or couple path visibility to marker opacity.
Automatic marker opacity follows local screen density, not zoom alone. Filtered/sparse conversations must remain clearly visible at overview zoom. Keep marker sizes fixed and GPU batching intact. Gray context paths stay soft even in sparse views (at most 45% opacity and 0.5 CSS pixels); hover and selection provide emphasis. Dense overview paths remain hairlines. Depth shading must not make points disappear.
Named t-SNE comparison fits use `--tsne-layout local --perplexity 10` and `--tsne-layout broad --perplexity 50`. Preserve the original fit; record metric, iterations, and exaggeration. Never use conversation or correctness labels to force separation.
Keep local JSON imports and screenshots under ignored `outputs/`. Do not publish or modify the paper without the user's request.
Preserve upstream data terms and template notices. Do not assign a software license without author approval.
Use `data/manifest.json` to check numerical artifact integrity. Generated output belongs under `outputs/`.
