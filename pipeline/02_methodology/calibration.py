"""Calibration analysis: Expected Calibration Error (ECE), reliability diagram, temperature scaling.

Reliability diagram pokazuje czy zwracane prawdopodobieństwa odpowiadają rzeczywistym częstościom.
ECE: średnia różnica między confidence a accuracy w binach.
Temperature scaling: T*logits korekta post-hoc (Guo et al. 2017) — przeskalowuje wektor logits
żeby zmniejszyć overconfidence/underconfidence.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


def ece(y_true, y_prob, n_bins: int = 10) -> dict[str, float]:
    """Expected Calibration Error.

    y_prob: probability of positive class (shape [N]) — dla binary classification.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_value = 0.0
    bin_info: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            bin_info.append({"lo": lo, "hi": hi, "n": 0, "acc": 0.0, "conf": 0.0})
            continue
        acc = float(np.mean(y_true[mask] == (y_prob[mask] >= 0.5)))
        conf = float(np.mean(y_prob[mask]))
        n = int(np.sum(mask))
        ece_value += (n / len(y_true)) * abs(acc - conf)
        bin_info.append({"lo": float(lo), "hi": float(hi), "n": n, "acc": acc, "conf": conf})
    return {"ece": float(ece_value), "n_bins": n_bins, "bins": bin_info}


def reliability_diagram(y_true, y_prob, n_bins: int = 10, save_to: str | None = None,
                         title: str = "Reliability diagram") -> None:
    """Plot reliability diagram z ECE w tytule."""
    e = ece(y_true, y_prob, n_bins=n_bins)
    bins = e["bins"]
    mids = [(b["lo"] + b["hi"]) / 2 for b in bins]
    accs = [b["acc"] for b in bins]
    confs = [b["conf"] for b in bins]
    ns = [b["n"] for b in bins]

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / n_bins
    # Gap (acc - conf) jako kolorowane słupki
    for i in range(n_bins):
        if ns[i] == 0:
            continue
        ax.bar(mids[i], accs[i], width=width * 0.9, color="#3b82f6", alpha=0.7,
               edgecolor="black", label="accuracy" if i == 0 else "")
        ax.bar(mids[i], confs[i] - accs[i], bottom=accs[i], width=width * 0.9,
               color="#ef4444", alpha=0.5, edgecolor="black",
               label="confidence-acc gap" if i == 0 else "")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (confidence)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}  (ECE = {e['ece']:.4f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig


def temperature_scale(
    y_true,
    logits,
    init_T: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Post-hoc temperature scaling (Guo et al. 2017).

    Args:
        y_true: shape [N] (int 0/1).
        logits: shape [N, 2] dla binary classification.

    Returns:
        optimal_T, calibrated_probs (shape [N, 2]).
    """
    y_true = np.asarray(y_true).astype(int)
    logits = np.asarray(logits).astype(float)
    if logits.ndim != 2:
        raise ValueError("logits must be shape [N, num_classes]")

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        scaled = logits / T
        # log-softmax stable
        log_probs = scaled - np.log(np.sum(np.exp(scaled - scaled.max(axis=1, keepdims=True)), axis=1, keepdims=True)) - scaled.max(axis=1, keepdims=True)
        return -np.mean(log_probs[np.arange(len(y_true)), y_true])

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    T = float(res.x)
    scaled = logits / T
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return T, probs
