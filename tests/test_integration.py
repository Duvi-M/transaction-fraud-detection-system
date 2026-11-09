import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.app import create_app
from fraud_detection.data.core import FEATURES, synthetic
from fraud_detection.modeling.models import load_artifact, score
from fraud_detection.modeling.train import run


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiment")
    csv = root / "fixture.csv"
    synthetic(csv, rows=3000)
    config = {
        "seed": 42,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "false_positive_cost": 1,
        "false_negative_cost": 100,
        "max_alert_fraction": 0.05,
        "top_k": 20,
    }
    artifact = run(csv, root / "reports", root / "artifacts", config, "synthetic")
    return root, artifact


def test_full_pipeline_serialization_and_lock(trained):
    root, artifact = trained
    loaded = load_artifact(root / "artifacts/model.joblib")
    probe = pd.DataFrame(artifact["verification"]["features"])
    np.testing.assert_array_equal(
        score(artifact["model"], probe, artifact["name"]),
        score(loaded["model"], probe, loaded["name"]),
    )
    results = json.loads((root / "reports/results.json").read_text())
    lock = json.loads((root / "reports/selection_lock.json").read_text())
    assert "test" not in lock
    assert results["policies"] == lock["policies"]
    assert results["dataset_kind"] == "synthetic"
    assert (root / "reports/test_comparisons.png").exists()
    artifact["verification"]["scores"][0] += 1
    joblib.dump(artifact, root / "bad.joblib")
    with pytest.raises(ValueError, match="verification"):
        load_artifact(root / "bad.joblib")


def test_api_success_validation_and_batch(trained):
    root, _ = trained
    a = load_artifact(root / "artifacts/model.joblib")
    record = {k: float(v[0]) for k, v in a["verification"]["features"].items()}
    other = {k: float(v[1]) for k, v in a["verification"]["features"].items()}
    with TestClient(create_app(root / "artifacts/model.joblib")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model/info").json()["features"] == FEATURES
        first = client.post("/score", json=record)
        assert first.status_code == 200
        assert first.json()["score"] == pytest.approx(a["verification"]["scores"][0])
        batch = client.post("/score/batch", json={"transactions": [record, other, record]}).json()
        assert batch["predictions"][0] == batch["predictions"][2]
        assert batch["predictions"][0]["score"] == pytest.approx(
            first.json()["score"], rel=1e-12, abs=1e-15
        )
        assert batch["predictions"][0]["review_recommended"] == first.json()["review_recommended"]
        single_other = client.post("/score", json=other).json()
        for key, value in batch["predictions"][1].items():
            if key == "score":
                assert value == pytest.approx(single_other[key], rel=1e-12, abs=1e-15)
            else:
                assert value == single_other[key]
        for bad in [
            {**record, "Class": 1},
            {**record, "V1": "1"},
            {**record, "Amount": -1},
            {**record, "V1": None},
            {k: v for k, v in record.items() if k != "Time"},
        ]:
            assert client.post("/score", json=bad).status_code == 422
        assert client.post("/score/batch", json={"transactions": []}).status_code == 422
        assert (
            client.post("/score/batch", json={"transactions": [record] * 1001}).status_code == 422
        )


def test_missing_and_invalid_model(tmp_path):
    record = dict.fromkeys(FEATURES, 0.0)
    for path in [tmp_path / "missing", tmp_path / "corrupt"]:
        if path.name == "corrupt":
            path.write_text("invalid model")
        with TestClient(create_app(path)) as client:
            assert client.get("/health").status_code == 503
            assert client.get("/model/info").status_code == 503
            assert client.post("/score", json=record).status_code == 503
