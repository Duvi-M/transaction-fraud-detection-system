"""Exercise the real trained artifact with the ASGI application, no socket required."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from fraud_detection.api.app import create_app
from fraud_detection.data.core import FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", default=os.getenv("FRAUD_ARTIFACT", "artifacts/real/model.joblib")
    )
    args = parser.parse_args()
    record = pd.read_csv("data/raw/creditcard.csv", nrows=1)[FEATURES].iloc[0].to_dict()
    results = {}
    with TestClient(create_app(args.artifact)) as client:
        for endpoint in ["/health", "/model/info"]:
            response = client.get(endpoint)
            assert response.status_code == 200, response.text
            results[endpoint] = response.json()
        single = client.post("/score", json=record)
        assert single.status_code == 200, single.text
        batch = client.post("/score/batch", json={"transactions": [record, record]})
        assert batch.status_code == 200, batch.text
        for prediction in batch.json()["predictions"]:
            np.testing.assert_allclose(
                prediction["score"], single.json()["score"], rtol=1e-12, atol=1e-15
            )
            assert {k: v for k, v in prediction.items() if k != "score"} == {
                k: v for k, v in single.json().items() if k != "score"
            }
        assert single.json()["dataset_kind"] == "real"
        results["single"] = single.json()
        results["batch_stable"] = True
    Path("reports/api_smoke.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Real-artifact ASGI smoke passed: health, model info, single, stable batch")


if __name__ == "__main__":
    main()
