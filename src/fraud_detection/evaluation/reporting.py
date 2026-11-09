import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay

from fraud_detection.data.core import audit
from fraud_detection.decisions.policy import select_threshold, top_k


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def eda(frame, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", audit(frame))
    frame.describe().T.to_csv(output / "feature_distribution.csv")
    connection = duckdb.connect()
    connection.register("transactions", frame)
    sql_path = Path(__file__).resolve().parents[3] / "sql/analytics.sql"
    for i, query in enumerate(sql_path.read_text().split(";")):
        if query.strip():
            connection.execute(query).df().to_csv(output / f"sql_{i + 1}.csv", index=False)
    connection.close()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    frame.Class.value_counts().sort_index().plot.bar(ax=axes[0], logy=True)
    axes[0].set(title="Class counts (log scale)", xlabel="Class")
    for label, group in frame.groupby("Class"):
        axes[1].hist(
            np.log1p(group.Amount), bins=45, density=True, alpha=0.5, label=f"Class {label}"
        )
        axes[2].hist(group.Time / 3600, bins=48, density=True, alpha=0.5, label=f"Class {label}")
    axes[1].set(xlabel="log(1 + Amount)", title="Amount distributions")
    axes[2].set(xlabel="Elapsed hours", title="Time distributions")
    axes[1].legend()
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "eda.png", dpi=150)
    plt.close(fig)


def comparisons(y, scores, output, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for model_name, values in scores.items():
        PrecisionRecallDisplay.from_predictions(y, values, name=model_name, ax=axes[0])
        if model_name != "isolation_forest":
            observed, predicted = calibration_curve(y, values, n_bins=10, strategy="quantile")
            axes[1].plot(predicted, observed, "o-", label=model_name)
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set(xlabel="Mean uncalibrated classifier output", ylabel="Observed fraud fraction")
    axes[1].legend()
    values = scores["isolation_forest"]
    for label in [0, 1]:
        axes[2].hist(
            values[np.asarray(y) == label], bins=40, density=True, alpha=0.5, label=f"Class {label}"
        )
    axes[2].set(xlabel="Negative IsolationForest score_samples", title="Higher = more anomalous")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / f"{name}_comparisons.png", dpi=150)
    plt.close(fig)


def sensitivity(y, scores, output):
    # Validation-only alternative policies. Test threshold remains locked.
    costs = [1, 10, 25, 100, 250, 1000]
    budgets = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    policies = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cost in costs:
        policy = select_threshold(y, scores, 1, cost, 0.01)
        policies.append(policy)
    axes[0].plot(costs, [p["alerts"] for p in policies], "o-")
    axes[0].set(xscale="log", xlabel="Hypothetical FN cost (FP = 1)", ylabel="Validation alerts")
    recalls = [top_k(y, scores, int(len(y) * b))["recall_at_k"] for b in budgets]
    axes[1].plot(np.array(budgets) * 100, recalls, "o-")
    axes[1].set(xlabel="Validation review budget (%)", ylabel="Recall at budget")
    fig.tight_layout()
    fig.savefig(output / "validation_sensitivity.png", dpi=150)
    plt.close(fig)
    write_json(
        output / "validation_sensitivity.json",
        {"cost_policies": policies, "budget_fractions": budgets, "recall_at_budget": recalls},
    )
