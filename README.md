# Memory Atlas demo

Standalone static deployment of [The Geometry of Agent Memory](https://github.com/ichko/agent-memory-geometry).
Source code, tests, and the packaging tool are on the `main` branch.

Enable GitHub Pages under **Settings → Pages → Deploy from a branch → demo → /(root)**.
Paper page: https://www.izhv.eu/agent-memory-geometry/
Memory Atlas: https://www.izhv.eu/agent-memory-geometry/atlas.html

This branch contains all 500 LongMemEval-small histories, 18,464 conversation paths, sampled memory overlays,
precomputed UMAP/PCA/t-SNE layouts, real 3D layouts, and lazy-loaded original-space cosine similarities.
The paper landing page loads no map data. Opening the separate Atlas downloads about 110 MB of compressed data;
similarity files load only when inspected.

See [data attribution](data-notice.html) and [file checksums](publication-manifest.json).
Credential-shaped examples and local paths are masked in 30 displayed point texts.
Point identities, coordinates, and similarity scores are unchanged; embeddings precede display redaction.
There are no backend services, API credentials, model weights, raw vectors, or embedding caches in this deployment.

To preview locally, run `python3 -m http.server 8767 --bind 127.0.0.1` from this branch and open http://127.0.0.1:8767/.
The explorer is at `/atlas.html`. The poster link will be added when the authors' poster is available.
