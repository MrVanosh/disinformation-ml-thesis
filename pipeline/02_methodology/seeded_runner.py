"""Wrapper dla multi-seed runów per (dataset × model × variant).

Iteruje seedy {13, 42, 71, 89, 113} i wywołuje user-supplied `run_fn(seed, ...)`
zbierając wyniki w unified schema z bootstrap CI + cost metrics + git commit.

Każdy run zapisuje:
    experiments/results_v2/<dataset>_<model>_<variant>_seed<seed>.json
    experiments/preds_v2/<dataset>_<model>_<variant>_seed<seed>.jsonl

Aggregator (aggregator_v2.py) potem konsoliduje wszystko.

Użycie:
    from seeded_runner import SeededRunner

    runner = SeededRunner(
        dataset="truthseeker", model="bert-base", variant="finetune",
        seeds=[13, 42, 71, 89, 113],
        output_root="experiments/results_v2/",
    )
    for seed in runner.seeds:
        if runner.is_done(seed):
            continue
        y_true, y_pred, y_prob, logits, cost = train_and_eval(seed)  # user-defined
        runner.record(seed, y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                      logits=logits, cost=cost)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from metrics_with_ci import compute_metrics_with_ci
from calibration import ece, temperature_scale

logger = logging.getLogger("seeded_runner")
DEFAULT_SEEDS = (13, 42, 71, 89, 113)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class SeededRunner:
    def __init__(
        self,
        dataset: str,
        model: str,
        variant: str,
        seeds: Sequence[int] = DEFAULT_SEEDS,
        output_root: str = "experiments/results_v2/",
        preds_root: str = "experiments/preds_v2/",
        config_path: str | None = None,
        data_manifest_sha: str | None = None,
    ):
        self.dataset = dataset
        self.model = model
        self.variant = variant
        self.seeds = list(seeds)
        self.output_root = Path(output_root)
        self.preds_root = Path(preds_root)
        self.config_path = config_path
        self.data_manifest_sha = data_manifest_sha
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.preds_root.mkdir(parents=True, exist_ok=True)

    def _path(self, seed: int) -> Path:
        return self.output_root / f"{self.dataset}_{self.model}_{self.variant}_seed{seed}.json"

    def _preds_path(self, seed: int) -> Path:
        return self.preds_root / f"{self.dataset}_{self.model}_{self.variant}_seed{seed}.jsonl"

    def is_done(self, seed: int) -> bool:
        return self._path(seed).exists()

    def record(
        self,
        seed: int,
        y_true,
        y_pred,
        y_prob=None,
        logits=None,
        cost: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Compute metrics with bootstrap CI
        metrics = compute_metrics_with_ci(y_true, y_pred, y_prob=y_prob)

        # Calibration (jeśli y_prob dostępne i binary)
        ece_pre = None
        ece_post = None
        T_opt = None
        if y_prob is not None:
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
                y_prob_pos = y_prob_arr[:, 1]
            else:
                y_prob_pos = y_prob_arr
            ece_pre = ece(y_true, y_prob_pos)["ece"]

            if logits is not None:
                logits_arr = np.asarray(logits)
                try:
                    T_opt, probs_cal = temperature_scale(y_true, logits_arr)
                    ece_post = ece(y_true, probs_cal[:, 1] if probs_cal.ndim == 2 else probs_cal)["ece"]
                except Exception as e:
                    logger.warning("Temperature scaling failed: %s", e)

        # Ścieżka względna do cwd gdy możliwe (lokalnie), inaczej absolutna (Modal: /results poza /data)
        try:
            preds_path_str = str(self._preds_path(seed).relative_to(Path.cwd()))
        except ValueError:
            preds_path_str = str(self._preds_path(seed))
        record = {
            "dataset": self.dataset,
            "model": self.model,
            "variant": self.variant,
            "seed": seed,
            "split": "grouped_v2",
            "config_path": self.config_path,
            "data_manifest_sha": self.data_manifest_sha,
            "metrics": metrics,
            "calibration": {
                "ece_pre": ece_pre,
                "ece_post_temp": ece_post,
                "temperature_T": T_opt,
            },
            "cost": cost or {},
            "preds_path": preds_path_str,
            "git_commit": _git_commit(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "extra": extra or {},
        }

        # Save predictions JSONL
        with self._preds_path(seed).open("w", encoding="utf-8") as fh:
            for i in range(len(y_true)):
                row = {
                    "i": int(i),
                    "y_true": int(y_true[i]),
                    "y_pred": int(y_pred[i]),
                }
                if y_prob is not None:
                    yp = np.asarray(y_prob)
                    row["y_prob"] = yp[i].tolist() if yp.ndim == 2 else float(yp[i])
                fh.write(json.dumps(row) + "\n")

        # Save metrics record
        with self._path(seed).open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
        logger.info("Saved seed %d: F1=%.4f ± %.4f", seed,
                    metrics["f1"]["mean"], metrics["f1"]["std"])
        return self._path(seed)
