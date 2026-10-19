# Working on this release

Use Python 3.11 or 3.12 with uv. Run `uv sync --locked` for dependencies.

Tests: `uv run python -m unittest discover -s tests -v`.
Numerical checks: `uv run memory-geometry-reproduce --check-paper --n-boot 0 --no-plots`.
Publication check: `uv run python tools/check_release.py`.

Keep historical evaluation choices explicit. Do not silently replace preprocessing and claim exact reproduction.
Do not commit raw conversations, questions, memories, generated answers, credentials, embedding caches, or backend deployments.
Preserve upstream data terms and template notices. Do not assign a software license without author approval.
Use `data/manifest.json` to check numerical artifact integrity. Generated output belongs under `outputs/`.
