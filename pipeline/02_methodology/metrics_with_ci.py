"""Metryki z bootstrap 95% CI dla F1, accuracy, precision, recall, AUC.

Bootstrap: 1000 resamples z zwracaniem.
Output: dict z mean, std, ci_lo, ci_hi (95%).

Użycie:
    from metrics_with_ci import compute_metrics_with_ci
    res = compute_metrics_with_ci(y_true, y_pred, y_prob, n_bootstrap=1000)
    # res = {"f1": {"mean": 0.814, "ci_lo": 0.79, "ci_hi": 0.83, "std": 0.012}, ...}
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Zwraca {mean, std, ci_lo, ci_hi, n_bootstrap}."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            scores[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            scores[i] = np.nan
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n_bootstrap": 0}
    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lo": float(np.quantile(scores, alpha)),
        "ci_hi": float(np.quantile(scores, 1 - alpha)),
        "n_bootstrap": len(scores),
    }


def compute_metrics_with_ci(
    y_true,
    y_pred,
    y_prob=None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    average: str = "macro",
) -> dict[str, dict[str, float]]:
    """Kompletny zestaw metryk z bootstrap CI dla każdej."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, dict[str, float]] = {}

    def _f1(yt, yp): return f1_score(yt, yp, average=average, zero_division=0)
    def _prec(yt, yp): return precision_score(yt, yp, average=average, zero_division=0)
    def _rec(yt, yp): return recall_score(yt, yp, average=average, zero_division=0)
    def _acc(yt, yp): return accuracy_score(yt, yp)

    out["f1"] = bootstrap_ci(y_true, y_pred, _f1, n_bootstrap, seed)
    out["precision"] = bootstrap_ci(y_true, y_pred, _prec, n_bootstrap, seed)
    out["recall"] = bootstrap_ci(y_true, y_pred, _rec, n_bootstrap, seed)
    out["accuracy"] = bootstrap_ci(y_true, y_pred, _acc, n_bootstrap, seed)

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        try:
            def _auc(yt, yp):
                # yp is class predictions for bootstrap idx; recompute prob for that idx
                # Hack: bootstrap_ci wywoła metric_fn(y_true_resampled, y_pred_resampled)
                # — dla AUC potrzebujemy y_prob_resampled. Robimy własny bootstrap niżej.
                return roc_auc_score(yt, yp)
            # Custom AUC bootstrap (z y_prob)
            rng = np.random.default_rng(seed)
            n = len(y_true)
            scores = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                if len(np.unique(y_true[idx])) < 2:
                    continue
                scores.append(roc_auc_score(y_true[idx], y_prob_pos[idx]))
            if scores:
                scores = np.array(scores)
                alpha = (1 - 0.95) / 2
                out["auc"] = {
                    "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                    "ci_lo": float(np.quantile(scores, alpha)),
                    "ci_hi": float(np.quantile(scores, 1 - alpha)),
                    "n_bootstrap": len(scores),
                }
        except Exception:
            pass

    return out
