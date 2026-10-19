# Release validation

Validated on October 19, 2026, with Python 3.11.12 on macOS arm64.
The standalone environment was created from this repository's public-PyPI lockfile.

- All 17 unit tests passed.
- The source distribution and wheel built successfully.
- The full reproduction command passed its published-result checks with 1,000 bootstrap draws.
- Pooled geometry, difficulty, and controlled-baseline AUROCs matched the reported rounded estimates.
- LoCoMo category scores and the answer-anchor proxy counts matched the paper's descriptive results.
- The publication checker validated every numerical CSV against its field and value allowlist.
- Common credential and private-path checks found no matches in the release or extracted paper text.
- Paper PDF metadata and figure metadata were inspected separately.

The release uses a new Git history. It contains no backend integrations, raw textual logs, or embedding cache.
These checks are not a guarantee against every possible disclosure. Review future changes before publishing them.
