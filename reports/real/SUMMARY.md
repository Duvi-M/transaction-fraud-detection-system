# Measured real-data results

Source: `reports/real/results.json`.

| Model | Validation cost | Test AP | Precision | Recall | Alerts | P@100 | R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 5700 | 0.0013 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 |
| logistic | 1087 | 0.7475 | 0.2218 | 0.8514 | 284 | 0.5800 | 0.7838 |
| random_forest | 1174 | 0.8166 | 0.4586 | 0.8243 | 133 | 0.6000 | 0.8108 |
| isolation_forest | 3789 | 0.0605 | 0.0957 | 0.4189 | 324 | 0.0200 | 0.0270 |

Selected on validation only: **logistic**.

Top-100 is a fixed budget for the entire test partition, not a daily budget.
Costs are hypothetical FP=1, FN=100. All outputs are uncalibrated scores.
