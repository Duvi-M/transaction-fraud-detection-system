import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection.decisions.policy import top_k


def evaluate(y, scores, threshold, k=100, fp_cost=1, fn_cost=100, probability=False):
    prediction = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "alerts": int(prediction.sum()),
        "alert_fraction": float(prediction.mean()),
        "hypothetical_cost": float(fp * fp_cost + fn * fn_cost),
        **top_k(y, scores, min(k, len(y))),
    }
    if probability:
        result["brier_score"] = float(brier_score_loss(y, scores))
    return result
