"""Generator wszystkich figur dla Sekcji 3 pracy.

Output do `pipeline/06_thesis_inputs/figures/` jako PDF (LaTeX-friendly).

Figury:
  - pareto_cost_quality_<dataset>.pdf  — F1 vs ms/sample
  - seed_variance_<dataset>.pdf        — box plot 5-seedów per model
  - calibration_<best>_<dataset>.pdf   — reliability + ECE
  - leakage_sensitivity_scatter.pdf    — F1_drop vs trainable_params
  - cross_dataset_heatmap.pdf          — transfer matrix
  - cm_<best>_<dataset>.pdf            — confusion matrices
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # pipeline/ (plots.py jest 2 poziomy głęboko)
sys.path.insert(0, str(ROOT / "02_methodology"))
from calibration import reliability_diagram, ece  # noqa: E402

logger = logging.getLogger("plots")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def _short_label(model: str, variant: str) -> str:
    m = (str(model).replace("llama31-", "Llama-").replace("qwen25-", "Qwen-")
         .replace("bert_base", "BERT").replace("distilbert", "DistilBERT")
         .replace("mbert", "mBERT").replace("herbert", "HerBERT")
         .replace("_", " "))
    v = (str(variant).replace("lora_big", "big-LoRA").replace("lora_basic", "basic-LoRA")
         .replace("lora_natural", "nat-LoRA").replace("zs_short", "ZS")
         .replace("finetune", "ft").replace("tfidf", "").replace("text_meta", "").strip())
    return f"{m} {v}".strip()


def pareto_cost_quality(df: pd.DataFrame, dataset: str, output_path: Path) -> None:
    means = df[(df["seed"] == "mean") & (df["dataset"] == dataset)].copy()
    if means.empty or means["ms_per_sample"].isna().all():
        logger.warning("No ms_per_sample for %s — skipping Pareto", dataset)
        return

    from adjustText import adjust_text
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    families = means["model"].apply(_family_for_model)
    colors = {"classical": "#22c55e", "encoder": "#3b82f6", "llm": "#ef4444", "ensemble": "#a855f7"}
    markers = {"classical": "o", "encoder": "s", "llm": "^", "ensemble": "D"}
    fam_pl = {"classical": "klasyczne", "encoder": "enkodery", "llm": "LLM", "ensemble": "ensemble"}

    texts = []
    for fam in colors:
        sub = means[families == fam]
        if sub.empty:
            continue
        ax.scatter(
            sub["ms_per_sample"], sub["f1"],
            s=80, c=colors[fam], marker=markers[fam], alpha=0.85, label=fam_pl[fam],
            edgecolors="black", linewidths=0.5, zorder=3,
        )
        for _, row in sub.iterrows():
            texts.append(ax.text(row["ms_per_sample"], row["f1"],
                                 _short_label(row["model"], row["variant"]), fontsize=7))

    # Pareto front
    sorted_df = means.sort_values("ms_per_sample")
    pareto_pts, best_f1 = [], -1
    for _, row in sorted_df.iterrows():
        if pd.notna(row["f1"]) and row["f1"] > best_f1:
            pareto_pts.append((row["ms_per_sample"], row["f1"]))
            best_f1 = row["f1"]
    if pareto_pts:
        xs, ys = zip(*pareto_pts)
        ax.plot(xs, ys, "k--", alpha=0.4, lw=1, label="front Pareto", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Latencja (ms / próbkę, skala log.)")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Kompromis koszt-jakość — {dataset}")
    # Legenda POZA obszarem wykresu (nie zasłania punktów)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95, title="Rodzina")
    ax.grid(alpha=0.3, which="both")
    adjust_text(texts, ax=ax, only_move={"points": "y", "text": "xy"},
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pareto → %s", output_path)


def _family_for_model(name: str) -> str:
    name = str(name).lower()
    if any(k in name for k in ("logistic", "svm", "random_forest", "xgb")):
        return "classical"
    if any(k in name for k in ("bert", "distil", "roberta", "herbert", "mbert")):
        return "encoder"
    if any(k in name for k in ("llama", "qwen", "mistral")):
        return "llm"
    if "ensemble" in name:
        return "ensemble"
    return "other"


def seed_variance_boxplot(df: pd.DataFrame, dataset: str, output_path: Path) -> None:
    per_seed = df[(df["dataset"] == dataset) & (df["seed"] != "mean")].copy()
    if per_seed.empty:
        return

    per_seed["combo"] = per_seed["model"] + "/" + per_seed["variant"]
    order = per_seed.groupby("combo")["f1"].mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(order) + 1.5)))
    box_data = [per_seed[per_seed["combo"] == c]["f1"].dropna().values for c in order]
    bp = ax.boxplot(box_data, labels=order, vert=False, patch_artist=True,
                     boxprops=dict(facecolor="#e0e7ff", color="#3730a3"),
                     medianprops=dict(color="black"),
                     flierprops=dict(marker="o", markersize=3))
    ax.set_xlabel("Macro F1 (across 5 seeds)")
    ax.set_title(f"Wariancja po seedach — {dataset}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Box plot → %s", output_path)


def calibration_plot(preds_root: Path, dataset: str, model: str, variant: str,
                     seed: int, output_path: Path) -> None:
    preds_path = preds_root / f"{dataset}_{model}_{variant}_seed{seed}.jsonl"
    if not preds_path.exists():
        return
    y_true, y_prob = [], []
    with preds_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if "y_prob" not in r:
                continue
            y_true.append(r["y_true"])
            yp = r["y_prob"]
            y_prob.append(yp[1] if isinstance(yp, list) else yp)
    if not y_prob:
        return
    fig = reliability_diagram(
        np.array(y_true), np.array(y_prob),
        save_to=str(output_path),
        title=f"Calibration — {model}/{variant} on {dataset}",
    )
    plt.close(fig)
    logger.info("Calibration → %s", output_path)


def cross_dataset_heatmap(transfer_csv: Path, output_path: Path) -> None:
    if not transfer_csv.exists():
        logger.warning("No transfer CSV — skipping heatmap")
        return
    df = pd.read_csv(transfer_csv)
    pivot = df.pivot_table(index="train_on", columns="test_on", values="f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.3, vmax=0.95)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Test on")
    ax.set_ylabel("Train on")
    ax.set_title("Cross-dataset transfer (Macro F1)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                         color="black" if 0.55 < v < 0.85 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Heatmap → %s", output_path)


def confusion_matrix_plot(preds_root: Path, dataset: str, model: str, variant: str,
                          seed: int, output_path: Path) -> None:
    preds_path = preds_root / f"{dataset}_{model}_{variant}_seed{seed}.jsonl"
    if not preds_path.exists():
        return
    y_true, y_pred = [], []
    with preds_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            y_true.append(r["y_true"])
            y_pred.append(r["y_pred"])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["not_disinfo", "disinfo"])
    ax.set_yticklabels(["not_disinfo", "disinfo"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"CM — {model}/{variant}\n{dataset}, seed={seed}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                     ha="center", va="center", color="white" if cm_norm[i,j] > 0.5 else "black",
                     fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("CM → %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="pipeline/06_thesis_inputs/all_results_v2.csv")
    parser.add_argument("--preds-root", default="experiments/preds_v2/")
    parser.add_argument("--transfer-csv", default="pipeline/06_thesis_inputs/cross_dataset_matrix.csv")
    parser.add_argument("--output-dir", default="pipeline/06_thesis_inputs/figures/")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    out = Path(args.output_dir)

    for ds in df["dataset"].dropna().unique():
        pareto_cost_quality(df, ds, out / f"pareto_cost_quality_{ds}.pdf")
        seed_variance_boxplot(df, ds, out / f"seed_variance_{ds}.pdf")

        means = df[(df["seed"] == "mean") & (df["dataset"] == ds)]
        if not means.empty:
            best = means.sort_values("f1", ascending=False).iloc[0]
            calibration_plot(
                Path(args.preds_root), ds, str(best["model"]), str(best["variant"]),
                seed=42, output_path=out / f"calibration_{best['model']}_{ds}.pdf",
            )
            confusion_matrix_plot(
                Path(args.preds_root), ds, str(best["model"]), str(best["variant"]),
                seed=42, output_path=out / f"cm_{best['model']}_{ds}.pdf",
            )

    cross_dataset_heatmap(Path(args.transfer_csv), out / "cross_dataset_heatmap.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
