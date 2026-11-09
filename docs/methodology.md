# Evaluation methodology

## Protocol specified before test evaluation

Seed 42; exact duplicate removal; stable ascending Time ordering; approximately 60/20/20 chronological train/validation/test split. Boundaries use the timestamp of each proposed row cutoff, keeping equal timestamps together. No random fallback is selected based on performance. Each partition must contain both classes or training fails. Fraud counts and time boundaries are saved before model fitting. No entity split is possible without entity identifiers. Fraud counts, not total row count alone, determine the effective evidence for detection performance.

Training-only EDA precedes model fitting; raw full-data audit and final test plots are produced after selection is locked. The full source schema and duplicate integrity checks happen before splitting. They do not fit transformations. `StandardScaler` fits only on training data inside pipelines. No resampling is used. Validation and test retain their observed distributions after deduplication.

## Fixed bounded candidates

- Constant training-prior DummyClassifier: no discrimination, deterministic input-order ties for ranking. This is not an economic review model.
- LogisticRegression with balanced class weights, scaled features and 1,000 maximum iterations.
- RandomForest: 120 trees, maximum depth 14, minimum leaf size 2, balanced subsample weights, two worker threads.
- IsolationForest: 150 trees, 2,048 maximum sampled rows per tree, normal-only training rows, scaled features. Score is negative `score_samples`, so higher means more anomalous. Validation labels select its threshold; the detector itself fits without fraudulent examples.

There is no exhaustive hyperparameter search. Algorithm settings are fixed in code; seed, split, alert cap and costs live in `configs/default.json`. There is no refit on validation after threshold selection, so deployment and evaluation use the same fitted score function.

## Selection and decisions

For each candidate, consider every unique validation score as a >= threshold, plus a threshold above the maximum that selects zero alerts. Ties remain together. Minimize `FP * 1 + FN * 100` subject to at most 1% validation alerts; ties favor fewer alerts. These are arbitrary cost units, not Ozon costs or claimed monetary savings. Choose the candidate with the lowest validation cost; AP then name break candidate ties. Save `selection_lock.json` before any test scores. Test all four fixed candidates once for the predefined comparison; only the validation-selected model is deployed. Never switch the winner using the test table.

A validation threshold does not guarantee a future alert-volume cap. Deployment emits threshold recommendations; a queue with a strict review capacity would need a separate ranking/budget stage. Fixed top-K evaluation sorts descending with stable input-order tie breaks and does not select the threshold. Precision@K is captured labeled fraud divided by K; Recall@K is captured labeled fraud divided by all labeled fraud in that partition. K=100 is a partition budget, **not a daily operating capacity**. The sensitivity chart evaluates alternative cost policies and ranked review fractions on validation only.

## Metrics and uncertainty

Average Precision is the recall-increment-weighted average of precision values (`sklearn.metrics.average_precision_score`), not trapezoidal PR area. Precision, recall, F1 and confusion matrices use locked thresholds. ROC-AUC is supplemental: huge numbers of true negatives can obscure alert burden. An always-legitimate classifier attains approximately 99.83% accuracy on the original dataset, making accuracy an inappropriate objective.

Classifier `predict_proba` outputs are **uncalibrated ranking scores**. Class weighting changes the training objective and generally impairs direct probability interpretation. Brier scores and quantile reliability plots diagnose this; no calibration is claimed or fitted. Anomaly scores are not probabilities. Reliability bins can be sparse or misleading with rare events. Neither isolated example errors nor amount distributions establish causal fraud mechanisms or actual loss.

No confidence intervals are claimed: a single short chronological split with clustered events and few positive labels cannot establish stable future performance; naive independent-row bootstrap intervals would overstate certainty if fraud is temporally correlated. All conclusions apply only to this historical holdout. Duplicate removal, prior PCA and unknown label delays remain limitations. Larger out-of-time evaluations and temporal-block uncertainty analysis need more history.

## Production gap

This prototype has no authenticated external exposure, reviewer interface, queue capacity enforcement, label delay modeling, entity histories, drift alarms, feature freshness guarantees, fairness assessment, investigator feedback, throughput SLA or production governance. It does not deny transactions or make authorized fraud determinations. Trusted local joblib artifacts can execute code on load; never accept user-uploaded artifacts. Startup verifies saved prediction probes but is not a cryptographic supply-chain check. The local API binds to localhost; any deployment needs separate access controls and operational review.

## Metric distinctions

| Quantity | Interpretation in this project |
|---|---|
| Precision | TP / (TP + FP) at the locked score threshold |
| Average Precision (AP) | Ranking summary over recall increments across thresholds; different from threshold precision and trapezoidal PR-AUC |
| Recall | TP / all labeled fraud at the locked threshold |
| Recall@K | Fraud in the highest K scores / all labeled fraud, under a separate fixed partition budget |
| ROC-AUC | Ranking across true-positive and false-positive rates, with all negatives in the FPR denominator |
| PR metrics | Focus on retrieved positives and alert purity; strongly affected by prevalence |

High ROC-AUC can coexist with poor precision under extreme imbalance. Good ranking does not imply calibrated probabilities. Validation chooses policies; test only measures the frozen comparison.
