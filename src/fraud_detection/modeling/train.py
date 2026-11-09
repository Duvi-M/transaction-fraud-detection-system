import importlib.metadata
from pathlib import Path

import joblib
import pandas as pd

from fraud_detection.data.core import FEATURES, fingerprint, read_csv, split
from fraud_detection.decisions.policy import select_threshold
from fraud_detection.evaluation.metrics import evaluate
from fraud_detection.evaluation.reporting import comparisons, eda, sensitivity, write_json
from fraud_detection.modeling.models import candidates, load_artifact, score


def run(csv_path, output, artifact_dir, config, kind):
    output, artifact_dir = Path(output), Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Prevent accidental repeated test-set peeking in the same experiment directory.
    if (output / "results.json").exists():
        raise ValueError("Results already exist; use a new output path for a disclosed rerun")
    frame = read_csv(csv_path)
    if kind == "real" and (len(frame) != 284807 or frame.Class.sum() != 492):
        raise ValueError("Real-source shape/count differs from documented ULB dataset")
    parts, protocol = split(frame, config["train_fraction"], config["validation_fraction"])
    # Exploratory analysis uses training only. Full raw audit is generated after lock.
    eda(parts["train"], output / "training_eda")
    write_json(output / "split.json", protocol)
    pd.concat(
        [pd.DataFrame({"raw_row_id": p.index, "partition": name}) for name, p in parts.items()]
    ).to_csv(output / "lineage.csv", index=False)
    train, val, test = (parts[n] for n in ["train", "validation", "test"])
    fitted, val_scores, policies, validation = {}, {}, {}, {}
    for name, model in candidates(config["seed"]).items():
        print(f"Training {name} on {kind} data", flush=True)
        subset = train.loc[train.Class == 0] if name == "isolation_forest" else train
        model.fit(subset[FEATURES], subset.Class)
        values = score(model, val, name)
        policy = select_threshold(
            val.Class,
            values,
            config["false_positive_cost"],
            config["false_negative_cost"],
            config["max_alert_fraction"],
        )
        validation[name] = evaluate(
            val.Class,
            values,
            policy["threshold"],
            config["top_k"],
            policy["fp_cost"],
            policy["fn_cost"],
            name != "isolation_forest",
        )
        fitted[name], val_scores[name], policies[name] = model, values, policy
    winner = min(
        validation,
        key=lambda n: (policies[n]["validation_cost"], -validation[n]["average_precision"], n),
    )
    metadata = {
        "dataset_kind": kind,
        "source_sha256": fingerprint(csv_path),
        "config": config,
        "selection_rule": "minimum validation cost within alert cap; AP then name break ties",
        "selected_model": winner,
        "policies": policies,
        "validation": validation,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "joblib", "fastapi", "duckdb"]
        },
        "score_semantics": "uncalibrated ranking score; not an authorized fraud determination",
    }
    # Persist selection before test scoring. No fitting/refitting below this point.
    write_json(output / "selection_lock.json", metadata)
    comparisons(val.Class, val_scores, output, "validation")
    sensitivity(val.Class, val_scores[winner], output)
    probe = val[FEATURES].head(3)
    artifact = {
        "format_version": 1,
        "model": fitted[winner],
        "name": winner,
        "features": FEATURES,
        "threshold": policies[winner]["threshold"],
        "metadata": metadata,
        "verification": {
            "features": probe.to_dict("list"),
            "scores": score(fitted[winner], probe, winner).tolist(),
        },
    }
    joblib.dump(artifact, artifact_dir / "model.joblib")
    load_artifact(artifact_dir / "model.joblib")
    test_scores, test_results = {}, {}
    for name, model in fitted.items():
        values = score(model, test, name)
        policy = policies[name]
        test_scores[name] = values
        test_results[name] = evaluate(
            test.Class,
            values,
            policy["threshold"],
            config["top_k"],
            policy["fp_cost"],
            policy["fn_cost"],
            name != "isolation_forest",
        )
    write_json(output / "results.json", {**metadata, "test": test_results, "split": protocol})
    comparisons(test.Class, test_scores, output, "test")
    eda(frame, output / "raw_eda_after_lock")
    errors = test[["Time", "Amount", "Class"]].copy()
    errors["score"] = test_scores[winner]
    errors["review_recommended"] = errors.score >= policies[winner]["threshold"]
    errors["error_type"] = "correct"
    errors.loc[(errors.Class == 0) & errors.review_recommended, "error_type"] = "false_positive"
    errors.loc[(errors.Class == 1) & ~errors.review_recommended, "error_type"] = "false_negative"
    errors.groupby("error_type").Amount.agg(["count", "mean", "median", "max"]).to_csv(
        output / "error_amount_summary.csv"
    )
    errors.loc[errors.error_type != "correct"].sort_values("score", ascending=False).to_csv(
        output / "errors.csv", index_label="raw_row_id"
    )
    print(f"Locked winner: {winner}; results: {output / 'results.json'}", flush=True)
    return artifact
