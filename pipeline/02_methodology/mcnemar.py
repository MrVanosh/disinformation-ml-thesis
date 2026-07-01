"""McNemar test dla par predykcji modeli (na tym samym zbiorze testowym).

Użycie:
    from mcnemar import mcnemar_test, mcnemar_matrix

    # 2 modele:
    result = mcnemar_test(y_true, pred_a, pred_b)
    # result = {"chi2": 4.12, "pvalue": 0.042, "significant_at_005": True,
    #           "a_better_count": 23, "b_better_count": 11}

    # N modeli — macierz p-value:
    matrix = mcnemar_matrix(y_true, {"BERT": pred_a, "LLM": pred_b, "SVM": pred_c})
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar


def mcnemar_test(
    y_true,
    pred_a,
    pred_b,
    correction: bool = True,
) -> dict[str, Any]:
    """McNemar dla pary predykcji (binarna klasyfikacja).

    H0: oba modele mają tę samą distribucję błędów.
    H1: jeden z modeli systematycznie różni się od drugiego.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    assert len(y_true) == len(pred_a) == len(pred_b)

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    # Contingency table:
    #     B correct  B wrong
    # A correct  n00      n01
    # A wrong    n10      n11
    n00 = int(np.sum(a_correct & b_correct))
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    n11 = int(np.sum(~a_correct & ~b_correct))

    table = [[n00, n01], [n10, n11]]
    res = sm_mcnemar(table, exact=False, correction=correction)

    return {
        "chi2": float(res.statistic),
        "pvalue": float(res.pvalue),
        "significant_at_005": bool(res.pvalue < 0.05),
        "significant_at_001": bool(res.pvalue < 0.01),
        "a_only_correct": n01,  # A correct, B wrong
        "b_only_correct": n10,  # B correct, A wrong
        "both_correct": n00,
        "both_wrong": n11,
        "table": table,
    }


def mcnemar_matrix(
    y_true,
    predictions_by_model: dict[str, np.ndarray],
    correction: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Pełna macierz p-value dla wszystkich par modeli.

    Returns:
        {model_a: {model_b: {chi2, pvalue, significant_at_005, ...}}}
    """
    models = list(predictions_by_model.keys())
    out: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in models}
    for a, b in combinations(models, 2):
        res = mcnemar_test(y_true, predictions_by_model[a], predictions_by_model[b], correction)
        out[a][b] = res
        # Symetrycznie zapisz (z odwrotnymi a/b labels)
        out[b][a] = {
            **res,
            "a_only_correct": res["b_only_correct"],
            "b_only_correct": res["a_only_correct"],
        }
    return out
