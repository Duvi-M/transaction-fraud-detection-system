# Implementation report

Date: 2026-11-09. Status: locally executed historical-data portfolio prototype. This records the original implementation run. Publication checks are documented separately in [RELEASE_REPORT.md](RELEASE_REPORT.md). No external bank integration, secrets or real accounts were used for modeling.

## Implemented and executed

- Python 3.12.12 `src/` package, fixed primary versions, full transitive `uv.lock` and hashed `requirements.lock`; isolated `.venv`.
- Public original Kaggle acquisition with a reproducible command, SHA-256 verification and provenance record. No Zenodo mirror was used.
- Strict schema validation, raw dtype/null/duplicate/time/distribution audit, class-conditioned plots and four executed DuckDB query outputs.
- Exact duplicate removal, conflicting-label rejection, chronological timestamp-grouped splits, saved raw-row lineage and partition fraud counts. Training-only EDA precedes model selection.
- Constant prior, class-weighted logistic regression, bounded random forest and normal-only IsolationForest; scaler/estimator pipelines and train-only fitting.
- Validation-only threshold and candidate selection using hypothetical costs and a validation alert cap, persisted before test evaluation. Predefined candidates scored on test once; no test-based retuning or winner changes.
- AP, ROC-AUC, precision, recall, F1, confusion matrices, top-K metrics, Brier diagnostics, reliability/PR/anomaly plots, validation cost/budget sensitivity, and error amount summaries.
- Verified serialized selected model, strict FastAPI single/batch scoring, model information and readiness; 503 when artifact unavailable and 422 for invalid inputs.
- Dockerfile, Makefile, CI configuration, dataset/methodology/interview docs and ordered review guide.
- An independent synthetic CLI experiment and fixture-based integration tests, stored separately and never reported as real-data evidence.

## Source and split evidence

Original CSV SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
Original rows: 284,807; fraud labels: 492; duplicates removed: 1,081 (19 positive duplicate rows).
Retained rows: 283,726; retained positive labels: 473.

| Partition | Rows | Fraud labels | Elapsed seconds |
|---|---:|---:|---|
| Train | 170235 | 342 | 0–120392 |
| Validation | 56745 | 57 | 120393–145233 |
| Test | 56746 | 74 | 145234–172792 |

`reports/real/split.json` contains the published partition summary. Lineage and full audit outputs are generated locally and excluded from the public repository. Time boundaries cover unequal durations: fractions are based on row counts, not equal calendar periods.

## Locked real-data result

Selected model: **logistic regression**, validation cost 1,087 versus 1,174 for random forest, 3,789 for IsolationForest, 5,700 for constant prior. Costs FP=1, FN=100 are hypothetical units; cap=1% validation rows. Selected threshold=0.8907583217076319. No recalibration or validation refit was performed.

| Model | Test AP | Test ROC-AUC | Precision | Recall | F1 | Alerts | Test cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant | 0.001304 | 0.500000 | 0 | 0 | 0 | 0 | 7400 |
| Logistic | 0.747459 | 0.981814 | 0.221831 | 0.851351 | 0.351955 | 284 | 1321 |
| Random forest | 0.816616 | 0.946187 | 0.458647 | 0.824324 | 0.589372 | 133 | 1372 |
| IsolationForest | 0.060548 | 0.957485 | 0.095679 | 0.418919 | 0.155779 | 324 | 4593 |

Logistic confusion matrix: [[56451, 221], [11, 63]]. Top-100: 58 fraud labels captured, precision 0.58, recall 0.783784. Random forest captures 60 of 74 at top-100; IsolationForest captures 2. The fixed partition budget is not a daily queue guarantee. Random forest wins test AP, while the deployed model remains the validation-cost winner. Logistic Brier=0.023840 versus constant prior 0.001303 highlights poor probability calibration.

Detailed metrics, policy parameters and package versions are saved in `reports/real/results.json`; the selection-only version is `selection_lock.json`. No production benefit, actual avoided loss or generalization confidence interval is claimed.

## Verification and actual commands

Executed from this project directory:

```sh
uv venv --python 3.12 .venv  # portable equivalent of the local interpreter path
curl -L --fail --max-time 180 https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud -o data/raw/creditcard.zip
unzip -o data/raw/creditcard.zip -d data/raw
uv --no-cache pip install --python .venv/bin/python --reinstall -e '.[dev]'
.venv/bin/python scripts/acquire.py
UV_CACHE_DIR="$PWD/.cache/uv-clean" uv lock
UV_CACHE_DIR="$PWD/.cache/uv-clean" uv export --frozen --extra dev --no-emit-project --format requirements-txt -o requirements.lock
UV_CACHE_DIR="$PWD/.cache/uv-clean" uv sync --frozen --extra dev
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/fraud train --csv data/raw/creditcard.csv --kind real --output reports/real --artifacts artifacts/real
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/pytest -q
make lint
make fixture
make eda
.venv/bin/python scripts/smoke_api.py
.venv/bin/python scripts/smoke_http.py
.venv/bin/python scripts/write_summary.py
uv pip freeze --python .venv/bin/python
```

**Final tests: 20 passed, 0 failed.** `reports/tests.log` contains the result and 246 warnings, primarily upstream Matplotlib/pyparsing deprecations, a Starlette/AnyIO deprecation, and IsolationForest reducing max_samples for the small synthetic test. Warnings are recorded rather than suppressed. Ruff check and format check both pass (`reports/lint.log`). Real-artifact ASGI smoke verifies all four routes and stable batch output (`reports/api_smoke.json`). Real localhost Uvicorn readiness passes (`reports/http_smoke.json`, `reports/uvicorn.log`); this is not a load or latency benchmark. The comparison PNG was visually inspected for readable axes/legends.

## Failures encountered and repaired

1. Initial sandboxed network access could not resolve Kaggle/PyPI hosts. Approved network-enabled commands downloaded the public dataset and installed project dependencies successfully. No dataset blocker remains.
2. An initial broad Ruff invocation traversed the project-local dependency cache. The run was interrupted before model fitting, dependencies were fully reinstalled without cache, explicit exclusions were added, and all final lint commands target only `src tests scripts`. No measured model results originate from that interrupted attempt.
3. First completed test run: 19 passed / 1 failed. A single-vs-batch score comparison required bitwise equality despite a ~4e-23 BLAS roundoff difference. The test now compares scores with rtol=1e-12, atol=1e-15 while preserving exact decision/metadata comparisons. The failed run is retained in `reports/tests_initial_failure.log`. Final run passes all 20 tests. No model choice or real-data evaluation was changed to fix this test.
4. First HTTP smoke could not bind localhost inside the sandbox. The approved local-network retry started the server, verified readiness and shut it down successfully.

## Unverified components and remaining limitations

- Docker CLI exists, but `docker info` failed because its daemon socket did not exist. Image build and container runtime are **unverified**; see `reports/docker_check.log`. No daemon or external infrastructure was started.
- GitHub Actions is authored but **unverified remotely**. No repository was pushed or external workflow triggered.
- No calibration model, uncertainty interval, long-horizon/blocked temporal evaluation, live review queue, authentication, production drift monitoring or throughput benchmark is implemented. These are explicit production gaps, not achieved deliverables.
- Synthetic integration tests establish execution/contract behavior only. The independent fixture run is under `reports/synthetic/`; its intentionally separable signal and different prevalence do not validate historical performance.
- Unknown source label delays, original full-dataset PCA, small positive counts, duplicate ambiguity, hypothetical costs and only two days of data limit the historical inference. API artifact verification is a prediction consistency check, not cryptographic authenticity or cross-platform certification.
- Runtime files, data and dependency caches are local and ignored by Git. The final selected artifact is `artifacts/real/model.joblib`; reproducible training recreates it. Consult the source's usage terms before redistributing data.

The local prototype is complete for the measured scope. Container execution and remote CI remain explicitly unverified; production readiness is not claimed.

## Public-release note

The original execution logs and row-level diagnostic outputs referenced above remain local and are deliberately excluded from Git. The public release contains reviewed aggregate results and plots only. See `RELEASE_REPORT.md` and `reports/release_checks.json` for current verification; the commands and outcomes above describe the original run.
