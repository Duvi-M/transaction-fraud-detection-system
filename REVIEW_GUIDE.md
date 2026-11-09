# Review guide

1. Start with `README.md` and `IMPLEMENTATION_REPORT.md`; distinguish measured real results from synthetic tests and unverified container/remote CI execution.
2. Verify `data/provenance.json` against `scripts/acquire.py` and the Kaggle source. Raw CSV and artifacts are ignored. `uv.lock` and `requirements.lock` freeze the dependency resolution.
3. Inspect `src/fraud_detection/data/core.py`: exact schema, nonfinite/negative rejection, label exclusion, duplicate handling, conflicting labels, timestamp grouping, original row lineage and explicit failure for one-class partitions. Compare `reports/real/split.json` and locally regenerated `lineage.csv`.
4. Read `modeling/models.py` for fixed candidate complexity, scaling and anomaly-score sign. Read `modeling/train.py` in order: training-only EDA, fitting, validation policies, selection lock, serialization verification, then test evaluation. The absence of a validation refit is deliberate.
5. Inspect `decisions/policy.py`: >= comparisons, grouped ties, zero-alert option, cost/budget constraint, and exact top-K denominators. A future threshold policy is not a guaranteed-capacity queue.
6. Compare `selection_lock.json` with `results.json`: the winner and thresholds must remain unchanged. Read plots alongside numeric metrics. Sensitivity is validation-only. Locally regenerated `errors.csv` and `error_amount_summary.csv` describe mistakes without invented explanations; individual records are not published.
7. Inspect `evaluation/metrics.py`: AP is sklearn average precision; confusion matrix order is [[TN, FP], [FN, TP]]. Brier scores are diagnostics, not a calibration claim.
8. Inspect `api/app.py`: strict Pydantic input contract, target forbidden, 503 readiness/model errors, verified trusted artifact loading and stable batch ordering. Local artifact verification is not tamper resistance.
9. Read tests for independent known-metric fixtures, timestamp/duplicate disjointness, raw-row lineage, training-only scaler statistics, classifier orientation, anomaly direction, tied budget policy, serialization probes, complete training integration and API error cases.
10. Reproduce with `make install`, `make acquire`, `make lint`, `make test`. To train without overwriting the archived result, run:

```sh
.venv/bin/fraud train --csv data/raw/creditcard.csv --kind real \
  --output reports/reproduction --artifacts artifacts/reproduction
.venv/bin/python scripts/smoke_api.py --artifact artifacts/reproduction/model.joblib
```

Training refuses an output directory that already contains results. A separate rerun remains a repeated test evaluation and must not become a hidden tuning loop. Scrutinize two-day coverage, unknown labels/feature delays, duplicate removal, global upstream PCA, hypothetical costs, small positive counts, uncalibrated outputs, source terms and missing production controls.
