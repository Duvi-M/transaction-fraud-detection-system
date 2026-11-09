# Public release verification

This release preserves the original experiment. No model settings, feature transformations, split rules, costs, thresholds or saved historical results were changed during release preparation.

## Release changes

- Added a GitHub-renderable Mermaid architecture diagram and embedded existing aggregate comparison chart.
- Corrected fresh-clone training: `make train` writes to `reports/reproduction/` and `artifacts/real/`; `REPORT_DIR` and `ARTIFACT_DIR` can override those destinations.
- Added an artifact argument and numerical score tolerance to the API smoke script; decisions and response metadata remain exact comparisons.
- Added `scripts/verify_results.py` to check saved reports, source hash, temporal partitions, artifact identity and recomputed frozen-winner metrics without refitting or reselection.
- Restricted publication to explicitly reviewed files. Data, archives, artifacts, row-level errors, lineage, local logs and environment paths are excluded. Existing local files were preserved.
- Set read-only repository-content permissions for CI and retained its pinned Python/dependency setup and synthetic-only tests.

## Local verification

Python 3.12.12; dependency installation from `uv.lock` succeeded. Package imports, Ruff lint and formatting passed. **20 tests passed, 0 failed**, including synthetic training integration, serialization verification, API success/error behavior and stable batch ordering. The suite emitted 246 known upstream deprecation/small-fixture warnings, not test failures. Real-artifact ASGI smoke passed all four routes; a temporary localhost Uvicorn server returned successful readiness and was shut down afterward.

An isolated export of the staged source, with a fresh virtual environment and no local dataset or model, also passed locked installation, imports, Ruff and all 20 tests.

Saved metrics match the documented historical table. The source checksum and split summary match; the selected serialized logistic model reproduces its held-out metrics within 1e-12 numerical tolerance. This is a disclosed verification of a previously evaluated holdout, not a new selection experiment. The other candidates' historical metrics were checked in saved reports; those models were not retrained for release.

Docker CLI is installed, but its daemon socket is unavailable. **Docker build and container smoke remain unverified.** The release does not start a system daemon. Remote GitHub Actions status must be read from the repository's Actions page; local success alone is not a remote CI claim.

## Reproduce the checks

```sh
make install
make lint
make test
make acquire
make train
.venv/bin/python scripts/smoke_api.py
.venv/bin/python scripts/smoke_http.py
.venv/bin/python scripts/verify_results.py
```

`verify_results.py` intentionally checks the frozen real artifact against the original reports. It is an optional cross-platform reproducibility diagnostic; training into a new directory never replaces the archived results. Use a new `REPORT_DIR` if a reproduction already exists.

## Publication audit

The repository initially had no commits, branches with history, or remotes. No pre-existing published work needed rewriting. Git identity uses the intended account's GitHub noreply address rather than a personal email. Source acquisition uses Kaggle's public endpoint; its metadata lists “Database: Open Database, Contents: Database Contents.” This release provides attribution and acquisition instructions, not a redistribution of transaction records.

Only small aggregate JSON results, partition counts, cost sensitivity, summary Markdown and three PNG plots are published with source, tests, configuration and documentation. Individual records, data archives, local transaction examples, trained models and logs remain excluded. The staged file list is reviewed before commit and the remote tree is checked after publication.

## Suitability and limitations

Suitable as an internship portfolio demonstration of reproducible offline ML, imbalance-aware evaluation and model serving. It is not evidence of production antifraud performance, calibrated fraud probabilities, bank savings, compliance approval, live investigator outcomes or operational readiness. There are only two historical days and 74 positive test labels; global anonymized PCA and unknown label/feature timing remain material limitations. Docker runtime verification is still outstanding.
