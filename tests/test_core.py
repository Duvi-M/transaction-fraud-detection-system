import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraud_detection.data.core import FEATURES, read_csv, split, synthetic, validate
from fraud_detection.decisions.policy import select_threshold, top_k
from fraud_detection.evaluation.metrics import evaluate
from fraud_detection.modeling.models import candidates, score


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "fixture.csv"
    synthetic(path, rows=3000)
    return read_csv(path)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "null", "infinite", "label", "string", "negative", "leaked_target"],
)
def test_schema_rejects(frame, mutation):
    bad = frame.copy()
    if mutation == "missing":
        bad = bad.drop(columns="V1")
    elif mutation == "extra":
        bad["customer"] = 0
    elif mutation == "null":
        bad.loc[0, "V1"] = np.nan
    elif mutation == "infinite":
        bad.loc[0, "V1"] = np.inf
    elif mutation == "label":
        bad.loc[0, "Class"] = 2
    elif mutation == "string":
        bad["V1"] = "wrong"
    elif mutation == "negative":
        bad.loc[0, "Amount"] = -1
    with pytest.raises(ValueError):
        validate(bad, labeled=mutation != "leaked_target")


def test_split_lineage_and_duplicate_isolation(frame):
    raw = pd.concat([frame, frame.iloc[[1, 2000]]], ignore_index=True)
    parts, info = split(raw)
    assert info["deduplicated_rows"] == len(frame)
    ids = [set(p.index) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(frame.index)
    assert parts["train"].Time.max() < parts["validation"].Time.min()
    assert parts["validation"].Time.max() < parts["test"].Time.min()
    for part in parts.values():
        pd.testing.assert_frame_equal(part, raw.loc[part.index])


def test_equal_timestamps_not_split(frame):
    frame.Time = (frame.Time // 500) * 500
    parts, _ = split(frame)
    groups = [set(p.Time) for p in parts.values()]
    assert not (groups[0] & groups[1] or groups[1] & groups[2])


def test_conflicting_labels_rejected(frame):
    row = frame.iloc[[0]].copy()
    row.Class = 1 - row.Class
    with pytest.raises(ValueError, match="Conflicting"):
        split(pd.concat([frame, row]))


def test_unusable_split_fails(frame):
    frame.Class = 0
    with pytest.raises(ValueError, match="lacks both classes"):
        split(frame)


def test_preprocessing_train_only(frame):
    parts, _ = split(frame)
    model = candidates()["logistic"]
    train = parts["train"]
    model.fit(train[FEATURES], train.Class)
    np.testing.assert_allclose(model["scale"].mean_, train[FEATURES].mean())
    original = model["scale"].mean_.copy()
    shifted = parts["validation"][FEATURES] + 10000
    score(model, shifted, "logistic")
    np.testing.assert_array_equal(model["scale"].mean_, original)
    transformed = model["scale"].transform(train[FEATURES])
    np.testing.assert_allclose(
        score(model, train, "logistic"), model["model"].predict_proba(transformed)[:, 1]
    )


def test_metrics_known_fixture():
    y = [0, 1, 0, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert average_precision_score(y, scores) == 1.0
    result = evaluate(y, scores, 0.5, k=2)
    assert result["average_precision"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["recall"] == 0.5
    assert evaluate([1, 0, 1], [0.9, 0.8, 0.7], 0.5)["average_precision"] == pytest.approx(5 / 6)


def test_top_k_ties_and_zero():
    assert top_k([1, 0, 1], [0.5, 0.5, 0.1], 1)["caught"] == 1
    assert top_k([1, 0, 1], [0.5, 0.5, 0.1], 0)["precision_at_k"] == 0
    with pytest.raises(ValueError):
        top_k([1], [1], 2)


def test_threshold_budget_ties_and_cost():
    y, scores = [1, 0, 1, 0], np.array([0.9, 0.9, 0.5, 0.1])
    p = select_threshold(y, scores, max_alert_fraction=0.25)
    assert (scores >= p["threshold"]).sum() == 0
    p = select_threshold(y, scores, max_alert_fraction=0.75)
    assert p["threshold"] == 0.5
    p = select_threshold(y, scores, fp_cost=1000, fn_cost=1, max_alert_fraction=1)
    assert p["alerts"] == 0


def test_anomaly_and_positive_orientation(frame):
    model = candidates()["isolation_forest"]
    normal = frame.loc[frame.Class == 0, FEATURES]
    model.fit(normal)
    outlier = normal.head(1).copy() + 100
    assert score(model, outlier, "isolation_forest")[0] > np.median(
        score(model, normal, "isolation_forest")
    )
    np.testing.assert_allclose(
        score(model, normal, "isolation_forest"), -model.score_samples(normal)
    )
    classifier = candidates()["logistic"].fit(frame[FEATURES], frame.Class)
    predictions = score(classifier, frame, "logistic")
    assert predictions[frame.Class == 1].mean() > predictions[frame.Class == 0].mean()
