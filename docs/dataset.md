# Dataset and provenance

Original source: [ULB/Worldline Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Public acquisition endpoint: `https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud`.
Run `make acquire`. The downloader verifies SHA-256 against the observed CSV and records provenance in `data/provenance.json`. It does not inspect credentials. If the public endpoint stops working, obtain `creditcard.csv` from the source page and place it in `data/raw/`, then run the verifier. No secrets are required by the application.

The source describes European card transactions over two days in September 2013: 284,807 rows, 492 fraud labels. The alternative Zenodo record 7395559 describes a copy from Kaggle, but was not used: the original public Kaggle download succeeded. Kaggle's public dataset metadata, checked during release, lists **Database: Open Database, Contents: Database Contents** (ODbL/DbCL). See the [source metadata](https://www.kaggle.com/api/v1/datasets/view/mlg-ulb/creditcardfraud) and [Open Data Commons ODbL](https://opendatacommons.org/licenses/odbl/1-0/). This repository provides acquisition instructions and attribution rather than redistributing transaction records; downstream redistribution must follow the source terms. No license for the dataset is inferred from this project's code.

| Column | Type | Meaning / validation |
|---|---|---|
| Time | finite number >= 0 | Elapsed seconds from the first observed transaction |
| V1–V28 | finite number | Anonymized PCA components; original feature meanings unavailable |
| Amount | finite number >= 0 | Transaction amount; no invented currency conversion or loss interpretation |
| Class | 0 or 1 | Historical fraud label; excluded from inference inputs |

Exact columns are required; missing, nonnumeric, nonfinite and extra fields fail validation. Nulls are rejected rather than imputed. Raw audit retains duplicate counts. Exact full-row duplicates are dropped before splitting, keeping the first raw row index as lineage. This prevents inflated evidence and duplicate split contamination, but may remove genuine repeated events; no transaction ID distinguishes duplicates from separate identical transactions. Conflicting labels on identical feature vectors fail explicitly. Equal timestamps stay in one partition.

No customer IDs, merchants, devices, IPs or geographic fields exist here. PCA features cannot be interpreted as economic mechanisms. Label timing, investigation policy, label noise and availability of individual upstream features are unknown. Sorting by observed time does not establish point-in-time availability of features or labels, and prior full-dataset PCA cannot be undone. Locally generated `lineage.csv` uses raw zero-based CSV data-row positions, not real transaction identifiers.

The artificial generator is only an engineering fixture with an intentionally stronger and different fraud signal and prevalence. Its metrics are never estimates of real-data performance.
