"""Check frozen real results and the saved winner without refitting or changing selection."""

import json
from pathlib import Path

import numpy as np

from fraud_detection.data.core import fingerprint, read_csv, split
from fraud_detection.evaluation.metrics import evaluate
from fraud_detection.modeling.models import load_artifact, score


def main():
    results = json.loads(Path("reports/real/results.json").read_text())
    lock = json.loads(Path("reports/real/selection_lock.json").read_text())
    for key, value in lock.items():
        assert results[key] == value, f"Selection changed: {key}"
    assert results["selected_model"] == "logistic"
    for name, expected in {
        "logistic": [0.747459, 0.221831, 0.851351],
        "random_forest": [0.816616, 0.458647, 0.824324],
        "isolation_forest": [0.060548, 0.095679, 0.418919],
    }.items():
        actual = [results["test"][name][k] for k in ["average_precision", "precision", "recall"]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0000005)
    csv = Path("data/raw/creditcard.csv")
    assert fingerprint(csv) == results["source_sha256"]
    frame = read_csv(csv)
    assert len(frame) == 284807 and frame.Class.sum() == 492
    cfg = results["config"]
    parts, protocol = split(frame, cfg["train_fraction"], cfg["validation_fraction"])
    assert protocol == results["split"]
    artifact = load_artifact("artifacts/real/model.joblib")
    assert artifact["metadata"] == lock
    name = artifact["name"]
    policy = lock["policies"][name]
    assert artifact["threshold"] == policy["threshold"]
    test = parts["test"]
    measured = evaluate(
        test.Class,
        score(artifact["model"], test, name),
        artifact["threshold"],
        cfg["top_k"],
        policy["fp_cost"],
        policy["fn_cost"],
        probability=True,
    )
    for key, value in measured.items():
        np.testing.assert_allclose(value, results["test"][name][key], rtol=1e-12, atol=1e-12)
    print("Saved comparisons, source hash, split, selection lock and winner metrics verified")
    print("No fitting, model selection or threshold changes performed")


if __name__ == "__main__":
    main()
