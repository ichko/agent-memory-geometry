# Evaluation protocol

## Representations

The LongMemEval table contains 2,367 labeled rows across 500 questions and five systems.
The public system aliases are Sem-RAG, Ext-Dense, Ext-Hybrid, Obs-Extract, and Svc-Mem.
Their row counts are 412, 456, 500, 499, and 500.
Incomplete answer or judge logs account for the missing 133 rows. Missingness may not be random.

For each example, the original analysis deduplicated memory strings across answer logs and used their union.
This union does not reconstruct the complete backend store or the actual answer-model prompt.
No result here verifies that an answer-entailing fact reached the prompt.

## Features

The full model uses 25 features:

- Conversation structure: five counts or structure measurements.
- Correspondence: seven turn-to-memory coverage measurements, five memory-dispersion measurements, and two alignment measurements.
- Reachability: six query-proximity measurements.

The correspondence ablation includes dispersion and alignment. The full model also includes conversation structure.
All distances use cosine distance in the original embedding space.
Query top-k uses k=5. Density thresholds use percentiles 25, 10, and 5 of pairwise turn distances.
Coverage and redundancy features use percentiles 10 and 5.
The code uses strict distance comparisons against these thresholds.

The GMM uses diagonal covariance, seed 0, and regularization 0.0001.
BIC selects from one through min(10, number of turns) components.
Effective rank is the exponential entropy of normalized singular values.
Undefined singleton-set measurements remain NaN in the released tables.

## Classifier and splits

Correctness is the target: 1 means correct and 0 means incorrect.
Predicted probabilities rank expected correctness. They are not validated deployment-calibrated probabilities.
The model requires labeled training outcomes. Prediction uses pre-answer features without an additional LLM generation call.

The classifier is scikit-learn logistic regression with L2 regularization, C=1, solver lbfgs, and max_iter=2000.
It uses no class weighting. Five-fold StratifiedGroupKFold uses shuffling and seed 0.
All rows of a question stay in the same fold across systems. Each training fold fits its own scaler.
Per-type and per-system evaluations fit separate models on their subsets.

Historical mean imputation uses each complete evaluation table before splitting.
Five undefined features in 499 Obs-Extract rows produce 2,495 imputed entries.
All-undefined columns become zero.
The difficulty baseline uses question-type one-hot encoding, character count, and question-embedding PCA-64.
PCA was fitted to all unique questions, including held-out covariates, without correctness labels.
The release supplies the resulting numerical PCA coordinates, not the underlying text or embeddings.

These choices match the published analysis. They are not a fully fold-local preprocessing protocol.
New studies should fit imputation and PCA within training folds and reserve an independent final test set.
The release cannot reconstruct fold-local question PCA from the published reduced coordinates alone.

## Controls and uncertainty

System identity means a categorical indicator for the memory system that produced each answer.
The central control compares difficulty plus identity against difficulty plus identity plus geometry.
The estimated pooled AUROC increment is small and not statistically established.

Paired intervals resample complete question groups from fixed out-of-fold predictions.
The default uses 1,000 bootstrap draws and seed 0. Some published pooled baseline intervals used 2,000 draws.
Use `--n-boot 2000` to match that draw count for the corresponding comparison.
Intervals do not include model retraining, exploratory feature selection, or global-preprocessing uncertainty.

Leave-one-system-out evaluation trains on four systems and tests on the fifth.
It uses max_iter=3000. Question identities overlap between training and held-out systems.
The model still uses the historical global imputation described above.
This is a system-holdout check, not a simultaneous unseen-question and unseen-system evaluation.

Selective-answering curves rank fixed out-of-fold correctness probabilities.
They are offline analyses, not results from an independently tuned and tested abstention policy.
The horizontal random-deferral curve shows expected accuracy, not one sampled random trial.

## Context selection and LoCoMo

Context selection holds the logged memory pool fixed and varies the supplied subset.
The selector supports closest-k, farthest-k, random-k, and all memories.
LongMemEval random subsets use five seeds. The summary averages their accuracies and reports population standard deviation.
Native ranking results are included numerically. Native backend retrieval code is not part of this release.
Answer generation and judging are not rerun by the reproduction command.
Selecting relevant context does not establish a causal effect of the 25-feature classifier.

LoCoMo uses 10 conversations and 1,986 questions. Its category score is negative nearest-turn distance.
Category correctness labels come from the full-conversation condition. No classifier is trained for this category table.
Category counts are 841 single-hop, 282 multi-hop, 321 temporal, 96 open-domain, and 446 adversarial.
The category table does not have conversation-bootstrap confidence intervals.
The separate context-selection comparison uses 15 source turns per query.

## Replay limits

Dependency versions match the local numerical validation environment. Small differences can still arise from numerical libraries or platforms.
The paper used API model aliases rather than retained dated snapshots. Exact generation replay is not guaranteed.
The preprint links to this analysis release. Its results and figures are unchanged.
Answer-anchor decomposition remains a proxy analysis, not a verified error taxonomy.
Raw answer-anchor inputs and manual fact-coverage annotations are not part of this release.
