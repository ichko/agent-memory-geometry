# Numerical artifacts

These files contain derived numerical measurements and correctness labels. They contain no source conversation, memory, question, answer, or judge-rationale text.

## LongMemEval

`longmemeval/geometry_*.csv` contains one row per available question–system pair for each of four embedding models.
Each table contains 2,367 rows, 500 question identifiers, five system aliases, binary correctness labels, and 25 geometry features.
`nan` marks an undefined feature. Rows retain their original order for grouped-split reproduction.
The same anonymous question ID identifies a question across systems and tables.
The identifiers are not an anonymization guarantee for arbitrary linked data.

`longmemeval/question_covariates.csv` contains character counts and 64 PCA coordinates from cached question embeddings.
The PCA fit uses all unique questions. Question type comes from the geometry table.
The file preserves the published baseline without redistributing source embeddings.

`longmemeval/confidence.csv` contains verbalized confidence and P(True) scores for the 1,368 rows with those measurements.
Join it to geometry using `question_id` and `strategy`. The `strategy` column means system identity, not a retrieval setting.
One verbalized-confidence value is unavailable and remains NaN.

`longmemeval/answer_anchors.csv` contains answer-to-memory and query-to-anchor distances from the offline proxy analysis.
It contains no reference-answer text. The anchor is the logged memory closest to the reference-answer embedding.
The resulting bins do not establish fact coverage, prompt inclusion, or the cause of an error.

Sources: [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) and [cleaned data](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
Credit the benchmark authors listed in the paper bibliography. Consult the upstream terms before using source data.

## Context selection

`context_selection/longmemeval.csv` contains numeric outcomes for k values 5, 15, 30, and 60.
The answer model is recorded per row. Only k=15 includes both evaluated answer models.
`condition` identifies all, closest-k, farthest-k, native-k, or a random-seed subset.
`n_memories_used` records the actual subset size. It may be smaller than k.
Only scores 0 and 1 enter accuracy summaries. Empty or other scores denote unavailable judgments.
Some selection-log rows omit question type. Those fields remain empty.

`context_selection/locomo.csv` contains the four evaluated conditions for 1,986 questions and k=15.
`rich` means the full-conversation condition. `conversation_id` supports conversation-level grouping.
The summary command preserves the original numerical outcomes. It does not call an answer model or judge.

## LoCoMo nearest-turn analysis

`locomo/nearest_turn.csv` contains 1,986 rows with category, correctness, nearest-turn distance, and grouping identifiers.
The numerical score is computed in original embedding space using text-embedding-3-small.
Categories are 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, and 5=adversarial.

Source: [LoCoMo by Maharana et al.](https://github.com/snap-research/locomo).
LoCoMo uses [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Treat LoCoMo-derived artifacts as subject to those noncommercial terms. No commercial-use permission is implied by this release.
The export changes identifiers and removes all textual payloads. It retains derived scores, categories, and correctness labels.

## Integrity

`manifest.json` records file hashes, sizes, row counts, and field names.
Regenerate it only after reviewing any artifact change:

```sh
uv run python tools/check_release.py --write-manifest
```

The checker validates the allowed CSV fields and values. It also checks common secret patterns and private-path markers.
It is a review aid, not proof that arbitrary future files are safe to publish.
