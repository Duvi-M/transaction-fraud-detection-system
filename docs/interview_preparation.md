# Interview preparation

1. **Why not accuracy?** Always predicting legitimate achieves about 99.83% raw accuracy while capturing no fraud. Evaluate AP plus precision/recall and actual alert counts.
2. **What is AP?** Precision averaged over increases in recall across ranked score cutoffs. It summarizes ranking across thresholds and depends on prevalence; it is not the trapezoidal area under a PR curve.
3. **What do Precision@K and Recall@K mean?** Fraud captured in K alerts / K, and fraud captured / all fraud in the held-out partition. Stable ties matter. This K is not automatically a daily budget.
4. **Where can leakage happen?** Scaling before splitting, label or duplicate contamination, future features, global PCA preprocessing, model/threshold decisions based on test scores. The project controls the first three where possible but cannot verify anonymized upstream feature timing.
5. **Why temporal validation?** It approximates scoring later observations and reveals temporal shifts. Two days cannot represent seasonality or durable generalization. No entity IDs allow grouped customer evaluation.
6. **Why class weights?** They increase training loss for the rare class without resampling holdouts. They do not make the resulting numbers calibrated fraud probabilities.
7. **Why might IsolationForest lose?** Unusual legitimate observations can be common, and fraud need not be globally isolated in feature space. Supervision can learn class-specific separation. Our detector trains only on training normals and uses labels solely for validation policy selection.
8. **How is the threshold selected?** Enumerate validation score groups and zero alerts; minimize hypothetical FP/FN cost within a validation cap. Save it before test scoring. A future distribution can exceed the cap.
9. **Which model won?** Read the actual `reports/real/results.json` selected_model and metrics. Do not confuse the highest test AP with the model selected on validation cost.
10. **What remains before shipping?** More history, delayed-label evaluation, data contracts, investigator feedback, authenticated service, capacity management, monitoring, audit trails, incident response, bias/governance review and measured latency under load. This dataset proves none of these.
11. **Can we claim savings?** No. The costs are assumed units; missed label counts and amounts are not measured recoverable bank losses.
12. **How can another engineer reproduce it?** Install the locked dependencies with Python 3.12, verify source SHA-256, run the fixed configuration into a new reports directory, inspect the selection lock and lineage, then run tests and the API smoke script.
