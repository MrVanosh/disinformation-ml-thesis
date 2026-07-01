"""Konsoliduje wszystkie results_v2/*.json w jeden CSV + Markdown summary.

Output:
  - pipeline/06_thesis_inputs/all_results_v2.csv (long format z wszystkimi seedami + mean/CI)
  - pipeline/06_thesis_inputs/summary_v2.md (czytelna tabela per dataset × model class)
  - pipeline/06_thesis_inputs/significance_matrices/<dataset>.json (McNemar matrix)

Pozwala figurom F4-F6 (cost-quality, calibration, leakage sensitivity) korzystać
z jednego źródła prawdy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mcnemar import mcnemar_matrix

logger = logging.getLogger("aggregator_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for p in sorted(root.glob("*.json")):
        try:
            with p.open(encoding="utf-8") as fh:
                records.append(json.load(fh))
        except Exception as e:
            logger.warning("Cannot read %s: %s", p, e)
    return records


def _aggregate_by_combo(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Long-format DF: jedna row per seed + jedna row per "mean" (seed=None)."""
    rows = []
    for r in records:
        m = r.get("metrics", {})
        f1 = m.get("f1", {})
        prec = m.get("precision", {})
        rec = m.get("recall", {})
        acc = m.get("accuracy", {})
        auc = m.get("auc", {})
        cal = r.get("calibration", {})
        cost = r.get("cost", {})
        rows.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "variant": r["variant"],
            "seed": r["seed"],
            "f1": f1.get("mean"),
            "f1_std": f1.get("std"),
            "f1_ci_lo": f1.get("ci_lo"),
            "f1_ci_hi": f1.get("ci_hi"),
            "precision": prec.get("mean"),
            "recall": rec.get("mean"),
            "accuracy": acc.get("mean"),
            "auc": auc.get("mean") if auc else None,
            "ece_pre": cal.get("ece_pre"),
            "ece_post_temp": cal.get("ece_post_temp"),
            "temperature_T": cal.get("temperature_T"),
            "train_s": cost.get("elapsed_s") or cost.get("train_s"),
            "ms_per_sample": cost.get("ms_per_sample"),
            "peak_ram_mb": cost.get("peak_ram_mb"),
            "peak_vram_mb": cost.get("peak_vram_mb"),
            "trainable_params": cost.get("trainable_params"),
            "git_commit": r.get("git_commit", ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Per-combo aggregates
    agg_rows = []
    for (ds, mod, var), grp in df.groupby(["dataset", "model", "variant"]):
        if len(grp) < 2:
            # Pojedynczy seed — kopiujemy, ale oznaczamy
            r = grp.iloc[0].to_dict()
            r["seed"] = "mean"
            r["n_seeds"] = 1
            agg_rows.append(r)
            continue
        agg = {
            "dataset": ds, "model": mod, "variant": var, "seed": "mean",
            "n_seeds": len(grp),
        }
        for col in ["f1", "precision", "recall", "accuracy", "auc",
                    "ece_pre", "ece_post_temp", "train_s", "ms_per_sample",
                    "peak_ram_mb", "peak_vram_mb"]:
            vals = grp[col].dropna()
            if len(vals) > 0:
                agg[col] = float(vals.mean())
                agg[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
                agg[f"{col}_min"] = float(vals.min())
                agg[f"{col}_max"] = float(vals.max())
        # CI 95% średniej między seedami (normal approx: mean ± 1.96·std/√n)
        if "f1" in agg and agg.get("f1_std") is not None and len(grp) >= 2:
            import math
            se = agg["f1_std"] / math.sqrt(len(grp))
            agg["f1_ci_lo"] = agg["f1"] - 1.96 * se
            agg["f1_ci_hi"] = agg["f1"] + 1.96 * se
        agg["trainable_params"] = int(grp["trainable_params"].dropna().mean()) if grp["trainable_params"].notna().any() else None
        agg_rows.append(agg)

    return pd.concat([df, pd.DataFrame(agg_rows)], ignore_index=True)


def _compute_significance_matrices(records: list[dict[str, Any]], preds_root: Path
                                    ) -> dict[str, dict[str, dict]]:
    """Per dataset: macierz p-value pomiędzy parami (model, variant) — używa preds seedu 42 jako referencyjnego."""
    out: dict[str, dict[str, dict]] = {}
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)

    for ds, recs in by_ds.items():
        # Bierzemy tylko seed=42 (reference)
        seed42 = [r for r in recs if r["seed"] == 42]
        if len(seed42) < 2:
            continue

        # Wczytaj predykcje wszystkich modeli seed42
        loaded = []  # (key, y_true_arr, y_pred_arr)
        for r in seed42:
            preds_path = Path(r.get("preds_path", ""))
            if not preds_path.is_absolute():
                preds_path = Path(".") / preds_path
            if not preds_path.exists():
                preds_path = preds_root / f"{r['dataset']}_{r['model']}_{r['variant']}_seed42.jsonl"
            if not preds_path.exists():
                continue
            y_true_list, y_pred_list = [], []
            with preds_path.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    y_true_list.append(row["y_true"])
                    y_pred_list.append(row["y_pred"])
            loaded.append((f"{r['model']}/{r['variant']}", np.array(y_true_list), np.array(y_pred_list)))

        if len(loaded) < 2:
            continue
        # McNemar wymaga TEGO SAMEGO test setu — modele ZS samplują (2000) inaczej niż pełne (1926).
        # Bierzemy największą grupę o wspólnej długości test setu jako referencję.
        len_counts = Counter(len(yt) for _, yt, _ in loaded)
        ref_len = len_counts.most_common(1)[0][0]
        ref_group = [(k, yt, yp) for k, yt, yp in loaded if len(yt) == ref_len]
        y_true_ref = ref_group[0][1]
        pred_dict = {k: yp for k, yt, yp in ref_group if np.array_equal(yt, y_true_ref)}
        if len(pred_dict) < 2:
            continue
        if len(pred_dict) < len(loaded):
            logger.info("McNemar %s: %d/%d modeli (wspólny test %d próbek; reszta inny sampling — pominięte)",
                        ds, len(pred_dict), len(loaded), ref_len)
        out[ds] = mcnemar_matrix(y_true_ref, pred_dict)

    return out


def _write_summary_markdown(df: pd.DataFrame, output: Path) -> None:
    means_only = df[df["seed"] == "mean"].copy()
    if means_only.empty:
        output.write_text("# No results yet.\n", encoding="utf-8")
        return

    lines = [
        "# Wyniki eksperymentów (mean ± std z 5 seedów + 95% CI bootstrap)",
        "",
        f"Total combos: {len(means_only)}",
        "",
    ]

    for ds, ds_df in means_only.groupby("dataset"):
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| Model | Wariant | n seeds | F1 mean | F1 std | F1 95% CI | Acc | ECE pre | ECE post | ms/sample |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|")
        for _, row in ds_df.sort_values("f1", ascending=False).iterrows():
            f1 = row["f1"] or 0
            f1_std = row.get("f1_std") or 0
            ci_lo = row.get("f1_ci_lo") or 0
            ci_hi = row.get("f1_ci_hi") or 0
            acc = row.get("accuracy") or 0
            ece_pre = row.get("ece_pre")
            ece_post = row.get("ece_post_temp")
            ms = row.get("ms_per_sample")
            lines.append(
                f"| {row['model']} | {row['variant']} | {row.get('n_seeds', '?')} | "
                f"{f1:.4f} | {f1_std:.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] | "
                f"{acc:.4f} | {format(ece_pre, '.4f') if isinstance(ece_pre, (int, float)) else '-'} | "
                f"{format(ece_post, '.4f') if isinstance(ece_post, (int, float)) else '-'} | "
                f"{format(ms, '.1f') if isinstance(ms, (int, float)) else '-'} |"
            )
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def aggregate(
    results_root: Path = Path("experiments/results_v2/"),
    preds_root: Path = Path("experiments/preds_v2/"),
    output_csv: Path = Path("pipeline/06_thesis_inputs/all_results_v2.csv"),
    output_md: Path = Path("pipeline/06_thesis_inputs/summary_v2.md"),
    output_sig_dir: Path = Path("pipeline/06_thesis_inputs/significance_matrices/"),
) -> pd.DataFrame:
    records = _load_records(results_root)
    logger.info("Loaded %d records from %s", len(records), results_root)

    df = _aggregate_by_combo(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("CSV → %s", output_csv)

    _write_summary_markdown(df, output_md)
    logger.info("Markdown → %s", output_md)

    # McNemar matrices per dataset
    if records:
        sig = _compute_significance_matrices(records, preds_root)
        output_sig_dir.mkdir(parents=True, exist_ok=True)
        for ds, matrix in sig.items():
            (output_sig_dir / f"{ds}.json").write_text(
                json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.info("McNemar matrices → %s", output_sig_dir)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results_v2/")
    parser.add_argument("--preds-root", default="experiments/preds_v2/")
    parser.add_argument("--output-csv", default="pipeline/06_thesis_inputs/all_results_v2.csv")
    parser.add_argument("--output-md", default="pipeline/06_thesis_inputs/summary_v2.md")
    parser.add_argument("--output-sig-dir", default="pipeline/06_thesis_inputs/significance_matrices/")
    args = parser.parse_args()

    aggregate(
        Path(args.results_root), Path(args.preds_root),
        Path(args.output_csv), Path(args.output_md), Path(args.output_sig_dir),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
