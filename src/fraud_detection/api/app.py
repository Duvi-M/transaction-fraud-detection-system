"""Review recommendation API. Only trusted local artifacts may be loaded."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud_detection.data.core import FEATURES
from fraud_detection.modeling.models import load_artifact, score

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", strict=True, allow_inf_nan=False),
    **{
        name: (float, Field(..., ge=0) if name in ["Time", "Amount"] else Field(...))
        for name in FEATURES
    },
)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


def create_app(artifact_path=None):
    path = Path(artifact_path or os.getenv("FRAUD_ARTIFACT", "artifacts/real/model.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.artifact = None
        app.state.error = None
        try:
            app.state.artifact = load_artifact(path)
        except Exception as exc:
            # No internal paths or exception details are sent to clients.
            app.state.error = type(exc).__name__
        yield

    app = FastAPI(title="Transaction review scoring", version="0.1.0", lifespan=lifespan)

    def artifact():
        if app.state.artifact is None:
            raise HTTPException(503, "Verified model unavailable; train a model before scoring")
        return app.state.artifact

    @app.get("/health")
    def health():
        artifact()
        return {"status": "ready", "model_loaded": True}

    @app.get("/model/info")
    def info():
        a = artifact()
        return {
            "model": a["name"],
            "features": a["features"],
            "threshold": a["threshold"],
            "metadata": a["metadata"],
        }

    def predict(records):
        a = artifact()
        values = score(a["model"], pd.DataFrame([r.model_dump() for r in records]), a["name"])
        return [
            {
                "score": float(value),
                "review_recommended": bool(value >= a["threshold"]),
                "score_type": "uncalibrated_ranking_score",
                "model": a["name"],
                "dataset_kind": a["metadata"]["dataset_kind"],
            }
            for value in values
        ]

    @app.post("/score")
    def single(transaction: Transaction):
        return predict([transaction])[0]

    @app.post("/score/batch")
    def batch(payload: Batch):
        return {"predictions": predict(payload.transactions)}

    return app


app = create_app()
