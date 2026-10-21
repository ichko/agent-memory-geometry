# Memory Atlas

Memory Atlas shows many real conversations in one shared embedding space.
Each path joins consecutive turns within one session. Separate sessions are never connected.
Hover a turn to highlight its conversation. Select it to read the source text.

The main view has UMAP/t-SNE/PCA and 2D/3D switches, a history number input, a memory overlay selector, and text search.
t-SNE is available only when genuine precomputed coordinates are present, and only in 2D.
Enter a history number, use the previous/next arrows, or select **Random**. **All** returns to the overview.
Scroll or pinch to zoom around the cursor. Drag to pan. The grid moves with the same camera as the data.
Dot and memory-marker sizes stay fixed in screen pixels while zooming; grid detail fades in gradually.
The plot uses Matplotlib's C0/C1/C3 colors on white: blue conversations, orange memories, and a red question.
Dense-scatter styling uses small borderless circles, not the oversized default `s=36` markers.
Turn area is 5 pt² (about 3 CSS pixels across), memory area is 9 pt² (4 CSS pixels),
with per-marker opacity controlled separately. Point units convert at 96 CSS pixels per inch.
**Auto** opacity follows local on-screen density, including filtering, zoom, pan, and 3D rotation.
Sparse turns approach 90% opacity and memories 95%; crowded overlaps become more transparent.
An isolated conversation is not given the low opacity of the full overview. Marker sizes and colors do not change.
The density field uses at most 8,192 deterministic samples per layer and 24-pixel cells, with bilinear interpolation
to avoid abrupt cell-boundary changes. The GPU samples a small opacity texture; point positions remain batched.
The Canvas fallback uses the same field. This is a display heuristic, not an analytical density estimate.
Slider readouts marked **≈** report sampled mean opacity before depth shading, not a uniform value for every dot.
Uncheck **Auto** to use uniform manual opacity; the sliders start from the estimated visible means.
The legend identifies the active store.
Opacity applies to individual markers, not a faded image of the whole layer. Overlaps can therefore appear denser.
The **Size** control adjusts marker area for both conversations and memories. Click targets extend beyond the visible markers.
All visible conversations have paths from the start. The full overview uses faint gray, quarter-pixel hairlines.
Sparse views keep visible but soft gray strokes, capped at 45% opacity and 0.5 CSS pixels.
Path opacity depends separately on conversation count and zoom, and never becomes zero because dots are faint.
Hover makes the path blue and more prominent at 1.1 CSS pixels. Selection changes it to Matplotlib C4 purple
at 1.7 pixels and full opacity. A narrow white underlay separates these highlights from dense dots.
Selected paths draw above hovered paths. Marker opacity does not hide or fade paths.
Hovering or clicking a line between dots also works, with a 5-pixel hit tolerance.
The segment index is built lazily and refreshed after 3D rotation; it contains no cross-session edges.
Context paths gain opacity when zooming in so they remain visible outside the overview.
Select **Focus this path** to inspect one conversation more closely.
Click empty map space to clear a selection. The canvas also supports arrow keys to select conversations,
plus/minus to zoom, and Escape to close the reader.

In **3D**, drag to orbit, Shift-drag or right-drag to pan, and scroll/pinch to zoom.
This is an orthographic view with fixed-size markers. A faint cube and axis indicator show orientation.
**Depth shading** combines lit markers, camera-depth haze, and stronger opacity toward the camera.
The GPU shades each circular marker with a soft highlight and darker edge; distant points recede without disappearing.
It updates during rotation, keeps the blue/orange categories and fixed marker sizes, and leaves 2D styling unchanged.
Uncheck it in the legend to compare the flat rendering. Hovered and selected paths retain their clear blue/purple colors for readability.
This is only a viewing cue, not a relevance or quality score. Its depth range uses the dataset's mean and
variance along the current camera axis. Depth opacity stays at or above 55% of base point opacity;
background paths retain at least 65% of their base depth weight. Selection is never depth-faded.
Source paths and memory samples rotate together. The same reader, history filters, search, and guided examples work in both views.
3D requires WebGL and real three-axis coordinates in the loaded atlas; it is disabled when these are unavailable.
Background paths and points are GPU-batched so rotation does not rebuild or upload every path on each frame.

## Source–memory inspection

Select **Inspect memories**, or click an orange memory point. The panel uses one history and the active memory store.
If the map shows all histories, it chooses a history containing the selected memory or conversation and states that context explicitly.
With nothing selected, it starts at the first history. Use the main history controls and memory-store selector to switch context.
These changes keep an open inspector available. Closing the panel, pressing Escape, or clicking empty map space exits inspection.

**Connections** ranks the five closest source turns to a memory by cosine similarity in the original embedding space.
The source and memory texts are shown side by side with their score. Number inputs and arrows browse the exported memories.
Dashed orange connections link the memory to those source turns; the active source path is purple.
Select **Focus matches** to frame the linked points. Switching projections or rotating 3D does not change the score or ranking.

**Heatmap** shows every source turn in the history against every exported memory for the active store.
Rows follow session/turn order, with session separators. Columns follow export order and are labeled M1, M2, and so on.
Scroll inside the heatmap to inspect its rows and columns. Hover a cell to highlight that pair on the map;
click it to select the pair and read its texts below. The heatmap supports arrow-key cell navigation.
Colors use the same fixed cosine range, −1 to +1, across histories and systems: white near zero,
dark blue for positive similarity, and gray for negative similarity. The scale is not normalized separately for each store.

Scores come from the real cached analysis vectors, normalized before matrix multiplication and exported to six decimal places.
They do not use PCA, UMAP, or t-SNE coordinates. They do not identify verified extraction provenance,
entailment, actual retrieval, or whether an answer can be produced. Model input truncation also applies to these scores.
The panel states the exported and full store counts. A weakly matched turn in a sampled overlay does not prove that a fact is absent.

Prepare the per-history files without any embedding or generation calls:

```sh
uv run python tools/build_atlas.py --output outputs/atlas --correspondence
```

The atlas gains correspondence metadata and a fingerprint of its embedding provenance and point texts.
Each `outputs/atlas/correspondence/hN.json` records turn IDs, memory IDs, sample counts, and the cosine matrices.
The browser loads only the active history and caches at most four histories. It validates the fingerprint, history,
row order, memory columns, counts, and scores before using them. Original vectors and embedding caches are not served.
When opening an atlas as a local file or using a static host without these sidecars, use **Open similarities**
to select the matching history JSON. It stays in the tab and is not uploaded. Missing scores never fall back to projected distances.

## Guided examples

The **Look closer** buttons resolve three text-checked cases from a compatible loaded atlas:
a sampled summary that omits a number, different mileage values in separate sessions,
and a stored assistant recommendation. Cases are hidden when their required evidence is not present.
No source conversations are bundled with the static viewer.

Click a preset to select its history and overlay, animate the camera, highlight its conversation paths,
and show a short explanation. Numbered rings link the map to source and memory excerpts in the bubble.
Click an excerpt to read the full text; closing the reader returns to the explanation.
**Recenter example** restores the framing after panning. Exiting the example restores the previous view.
Click the active preset again, select **Clear ×**, press Escape, or click empty map space to exit.
The clear button stays available even when an excerpt has opened the reader.
Switching projection preserves the example; manually changing history, store, or search exits it.
Camera animation respects reduced-motion preferences and stops when the user pans or zooms.

These are qualitative examples, not performance estimates. An omitted detail in one sampled memory
does not establish its absence from the full store. Textual correspondence is not verified extraction
provenance. Paths join turns only within their own sessions, including in guided cases.

## Local preview

```sh
python3 tools/serve_atlas.py --data outputs/atlas/atlas.json
```

Open `http://127.0.0.1:8766/`. The server binds only to localhost.
It serves the static viewer from `site/` and the selected ignored atlas separately as `/atlas.json`.
The atlas is not copied into the publishable site directory.
No private repository configuration, model weights, source embedding cache, or benchmark logs are served.

The browser application uses HTML, CSS, and JavaScript with no external runtime dependencies.
Marker artwork is generated by native Canvas 2D. When WebGL is available, the GPU places these same sprites
in batches without recalculating their colors. This avoids a separate JavaScript drawing call per point.
A cached Canvas 2D renderer remains available as a fallback. Both keep the same sizes, palette, and depth-opacity rules.
It has no synthetic fallback, embedding API clients, analytics, remote scripts, or remote fonts.
An atlas opened through **Open atlas** stays in the browser tab and is not uploaded.

For local performance checks, `?profile=1` exposes rolling draw timings in the map element's `data-profile`
attribute. Add `&renderer=canvas` to compare the fallback renderer. These controls do not change the data.

## Build a real-data atlas

Install the optional preprocessing dependencies:

```sh
uv sync --locked --extra explorer
```

Download the public [LongMemEval-small JSON](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) separately.
Then run:

```sh
uv run python tools/build_atlas.py \
  --dataset /absolute/path/to/longmemeval_s_cleaned.json \
  --output outputs/atlas \
  --device cpu
```

Use `--device mps` on a supported Apple GPU or `--device cuda` on a supported NVIDIA GPU.
The builder downloads the public [Redis LangCache Embed v3 Small model](https://huggingface.co/redis/langcache-embed-v3-small) if it is not already cached.
It runs inference locally. No paid embedding API is used.

To overlay saved memory stores, repeat `--run "System label=/absolute/path/to/answers.jsonl"`.
The JSONL needs `question_id` and `memories` fields. Question IDs match the source dataset.
The builder does not export answer text, judgments, service settings, or run metadata.

All source turns are included after stripping empty text. Identical sessions are deduplicated by their ordered role/text pairs.
A session can belong to several benchmark histories. Its path is drawn once in the overview.
The history filter selects the sessions included in one question's source history.
Identical text uses one embedding point, even when it appears in several sessions or stores.

By default, each history/system overlay contains at most 40 uniformly sampled stored memories.
This bounds the export size. Set `--memory-limit` to change it. The export records both the sample limit and full memory count.
The UI labels sampled stores. The sample is not a retrieval ranking and is not the complete store.
Conversation paths are not sampled.

Embeddings are checkpointed by model revision and text under `outputs/atlas/embedding-cache.sqlite`.
Rerunning the same command reuses completed vectors. The cache contains float32 arrays, not executable pickles.
Use `--embed-only` to save `source.json` and `vectors.npy` without projection.
Use `--project-only` to rebuild projections from those artifacts without rerunning the embedder.

## Projection and interpretation

The model produces 384-dimensional analysis embeddings with a 128-token input limit.
Long turns and memories are truncated by the model. The full original text remains available in the reader.
These are not the memory systems' native retrieval vectors.

The maps use the same complete set of embedded turn, memory, and question texts.
UMAP uses cosine neighborhoods after a 64-dimensional PCA preprocessing step for efficient neighborhood search.
Its recorded settings are 20 neighbors, `min_dist=0.12`, 150 epochs, and seed 42.
The PCA comparison view uses the first two axes of that same PCA fit.
Both projections are computed offline. The browser does not refit the map when filtering or selecting points.

To add a separate real 2D t-SNE layout while preserving existing UMAP/PCA coordinates:

```sh
uv run python tools/build_atlas.py --output outputs/atlas --add-tsne
```

This uses [openTSNE](https://opentsne.readthedocs.io/en/stable/api/index.html) 1.0.4 with a shared fit over every cached vector,
PCA to 64 dimensions, Euclidean neighborhoods after PCA, perplexity 30, PCA initialization, seed 42,
250 early-exaggeration iterations and 1,000 normal iterations, approximate NN-descent search, and FFT repulsion.
Normal-phase exaggeration is 1. Use `--perplexity`, `--iterations`, `--exaggeration`, `--tsne-metric`, and `--jobs`
to change recorded choices. It reuses cached vectors; no embedding APIs are called.
Settings and the number of fitted points are recorded in `projections.tsne`. Reload after the fit finishes to enable the button.
No depth coordinates are invented for t-SNE. UMAP and PCA remain available for real 3D.
t-SNE can change apparent separation, gaps, and cluster sizes. These are not evidence of conversation identity or memory quality.

Named comparison fits preserve the original t-SNE coordinates:

```sh
uv run python tools/build_atlas.py --output outputs/atlas --add-tsne --tsne-layout local --perplexity 10
uv run python tools/build_atlas.py --output outputs/atlas --add-tsne --tsne-layout broad --perplexity 50
uv run python tools/build_atlas.py --output outputs/atlas --add-tsne --tsne-layout full --perplexity 30 --tsne-metric cosine --tsne-pca-dimensions 0
```

The **t-SNE fit** buttons switch between available Local, Broad, Full, and Original layouts without changing the source data
or the original-space similarity scores. The first named fit becomes the default unless a different fit is explicitly selected.
The historical Original fit remains the earlier 30-perplexity, 500-iteration, cosine-after-PCA layout;
all settings are shown in tooltips and About. Local and Broad use 1,000 normal iterations and Euclidean-after-PCA neighborhoods.
Lower perplexity emphasizes smaller neighborhoods; higher perplexity shows a broader neighborhood tradeoff.
Full uses cosine neighborhoods in the original embedding dimensions, without PCA reduction for neighborhood search.
Its two-axis PCA initialization is only a starting position for optimization, not a reduction of the neighborhood inputs.
Use `--tsne-default` with a named fit to make it the default. An existing fit can be selected without recomputing it:

```sh
uv run python tools/build_atlas.py --output outputs/atlas --tsne-layout full --tsne-default
```

See the [parameter guide](https://opentsne.readthedocs.io/en/stable/parameters.html) for how settings alter visual structure.
These fits use no conversation labels or correctness labels. More separated-looking clusters are not automatically more faithful.

To add genuine 3D coordinates to an already built atlas, reuse its cached vectors:

```sh
uv run python tools/build_atlas.py --output outputs/atlas --add-3d
```

This preserves existing 2D coordinates. It fits a separate 3D UMAP layout with the same neighborhood settings,
and exports the first three PCA components with their explained variance. It makes no embedding API calls.
The optional `projections3d` metadata and `pca3d`/`umap3d` point arrays record the extra layouts.
Reload the viewer after preprocessing completes. Switching dimensions keeps point identities and text unchanged;
2D and 3D UMAP are separate fits, not a single layout with an invented depth axis.

A path shows source turn order, not a physical trajectory or a measure of reasoning.
Projection can distort distances, densities, and apparent separation.
Memory points sharing a history with a conversation are not verified extractions from that particular session.
Proximity does not prove factual coverage, freshness, entailment, retrieval, or answer quality.
This is an exploratory viewer, not a new predictive result from the paper.

## Checks and publication

```sh
node --test tests/explorer.test.mjs
uv run python -m unittest discover -s tests -v
uv run python tools/check_release.py
```

Keep atlas JSON, source text, vectors, caches, and screenshots under ignored `outputs/`.
Do not publish these artifacts without a separate content and data-permissions review.
The manual GitHub Pages workflow uploads only `site/`. No data is published automatically.
For a static deployment, users can open a separately prepared atlas JSON.
Distributing a default public atlas is a separate publication step.

To publish the viewer, open the repository's **Settings → Pages**, choose **GitHub Actions** as the source,
then open **Actions → Publish Memory Atlas → Run workflow** and select `main`.
The workflow tests the JavaScript and uploads only `site/`. It does not enable Pages or run automatically on push.
The expected project URL is `https://ichko.github.io/agent-memory-geometry/` once deployment succeeds.
Without a separately hosted `atlas.json` and matching `correspondence/` sidecars, the hosted viewer requires
**Open atlas** and **Open similarities**. The localhost preview server is not part of the deployment.
