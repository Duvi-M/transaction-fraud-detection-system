import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.data.core import FEATURES


def candidates(seed=42):
    return {
        "constant": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=14,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=2,
                        random_state=seed,
                    ),
                )
            ]
        ),
        "isolation_forest": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    IsolationForest(
                        n_estimators=150, max_samples=2048, n_jobs=2, random_state=seed
                    ),
                ),
            ]
        ),
    }


def score(model, frame, name):
    x = frame[FEATURES]
    if name == "isolation_forest":
        return -model.score_samples(x)  # Larger means more anomalous.
    classes = list(model.classes_)
    return model.predict_proba(x)[:, classes.index(1)]


def load_artifact(path):
    # joblib is executable pickle: load only locally trained, trusted artifacts.
    artifact = joblib.load(path)
    if artifact.get("format_version") != 1 or artifact["features"] != FEATURES:
        raise ValueError("Incompatible model artifact")
    probe = pd.DataFrame(artifact["verification"]["features"])[FEATURES]
    actual = score(artifact["model"], probe, artifact["name"])
    if not np.allclose(actual, artifact["verification"]["scores"], rtol=1e-12, atol=1e-12):
        raise ValueError("Artifact verification scores differ")
    return artifact
