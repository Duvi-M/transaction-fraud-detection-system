"""Strict schema validation, raw auditing and timestamp-grouped lineage splits."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
COLUMNS = [*FEATURES, "Class"]


def validate(frame: pd.DataFrame, labeled: bool = True) -> pd.DataFrame:
    expected = COLUMNS if labeled else FEATURES
    if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
        raise ValueError(f"Expected exactly these columns: {expected}")
    if frame.empty:
        raise ValueError("Empty dataset")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in expected):
        raise ValueError("All columns must have numeric dtypes")
    if not np.isfinite(frame[expected].to_numpy(dtype=float)).all():
        raise ValueError("Null or non-finite values are forbidden")
    if (frame[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time and Amount must be nonnegative")
    if labeled and not frame.Class.isin([0, 1]).all():
        raise ValueError("Class must contain only 0 or 1")
    return frame[expected].copy()


def read_csv(path: str | Path) -> pd.DataFrame:
    return validate(pd.read_csv(path))


def fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "fraud_count": int(frame.Class.sum()),
        "fraud_fraction": float(frame.Class.mean()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "null_counts": frame.isna().sum().to_dict(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "time_monotonic": bool(frame.Time.is_monotonic_increasing),
        "time_min": float(frame.Time.min()),
        "time_max": float(frame.Time.max()),
        "distributions": frame.describe().to_dict(),
    }


def split(
    frame: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> tuple[dict, dict]:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Split fractions must be positive and sum to less than one")
    frame = validate(frame).reset_index(drop=True)
    # Raw row positions survive de-duplication and sorting as lineage identifiers.
    clean = frame.drop_duplicates().sort_values("Time", kind="stable")
    # Contradictory labels for identical features cannot support evaluation.
    if clean.duplicated(FEATURES).any():
        raise ValueError("Conflicting labels for identical feature vectors")
    n = len(clean)
    if n < 10:
        raise ValueError("Too few observations")
    first = float(clean.iloc[int(n * train_fraction)].Time)
    second = float(clean.iloc[int(n * (train_fraction + validation_fraction))].Time)
    partitions = {
        "train": clean.loc[clean.Time < first],
        "validation": clean.loc[(clean.Time >= first) & (clean.Time < second)],
        "test": clean.loc[clean.Time >= second],
    }
    details = {}
    for name, part in partitions.items():
        if part.Class.nunique() != 2:
            raise ValueError(f"{name} lacks both classes; no automatic fallback permitted")
        details[name] = {
            "rows": len(part),
            "fraud_count": int(part.Class.sum()),
            "fraud_fraction": float(part.Class.mean()),
            "time_min": float(part.Time.min()),
            "time_max": float(part.Time.max()),
        }
    return partitions, {"deduplicated_rows": n, "cut_times": [first, second], "partitions": details}


def synthetic(path: Path, seed: int = 42, rows: int = 12000) -> None:
    """Artificial separable fixture, never a substitute for historical evidence."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.02, rows)
    x = rng.normal(size=(rows, 28))
    x[:, :5] += y[:, None] * rng.normal(2.2, 0.5, (rows, 5))
    frame = pd.DataFrame(x, columns=FEATURES[1:-1])
    frame["Time"] = np.arange(rows, dtype=float)
    frame["Amount"] = rng.lognormal(3, 1.2, rows)
    frame["Class"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[COLUMNS].to_csv(path, index=False)
