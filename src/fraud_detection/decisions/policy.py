"""Ranking ties use input order; threshold ties always stay together."""

import numpy as np


def top_k(y, scores, k: int) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    if not isinstance(k, int) or not 0 <= k <= len(y) or len(y) != len(scores):
        raise ValueError("K must be an integer between zero and the partition size")
    order = np.argsort(-scores, kind="stable")[:k]
    caught = int(y[order].sum())
    return {
        "k": k,
        "caught": caught,
        "precision_at_k": caught / k if k else 0.0,
        "recall_at_k": caught / int(y.sum()) if y.sum() else 0.0,
    }


def select_threshold(y, scores, fp_cost=1.0, fn_cost=100.0, max_alert_fraction=0.01):
    if fp_cost < 0 or fn_cost < 0 or not 0 <= max_alert_fraction <= 1:
        raise ValueError("Invalid costs or alert fraction")
    y, scores = np.asarray(y), np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    # Only complete tie groups are feasible under >= threshold semantics.
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(y) - 1]
    alerts = ends + 1
    tp = np.cumsum(y[order])[ends]
    costs = (alerts - tp) * fp_cost + (y.sum() - tp) * fn_cost
    allowed = alerts <= int(len(y) * max_alert_fraction)
    candidates = [(float(y.sum() * fn_cost), 0, float(np.nextafter(scores.max(), np.inf)))]
    candidates += [
        (float(c), int(a), float(ranked[e]))
        for c, a, e in zip(costs[allowed], alerts[allowed], ends[allowed])
    ]
    cost, alerts, threshold = min(candidates)  # Lower alert volume breaks cost ties.
    return {
        "threshold": threshold,
        "validation_cost": cost,
        "alerts": alerts,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "max_alert_fraction": max_alert_fraction,
    }
