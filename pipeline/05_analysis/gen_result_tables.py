"""Generuje gotowe wiersze LaTeX wyników (Acc, F1 macro mean±std, 95% CI, czas)
z all_results_v2.csv — jedno źródło prawdy dla tabel rozdz. 3 (zero literówek).

Wyjście: pipeline/06_thesis_inputs/result_rows.txt — pogrupowane per dataset×klasa,
każdy wiersz w formacie tabeli thesis: "Nazwa & Acc & F1 & [CI] & czas \\\\".
"""
from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

# Ładne nazwy modeli do tabel
DISP = {
    "lr": "Regresja logistyczna", "svm": "Liniowa SVM", "rf": "Las losowy",
    "distilbert": "DistilBERT", "bert_base": "BERT-base", "mbert": "mBERT",
    "herbert": "HerBERT", "ensemble": "Ensemble (tekst+meta)",
    "llama31-8b": "Llama-3.1-8B", "qwen25-7b": "Qwen-2.5-7B",
    "qwen25-14b": "Qwen-2.5-14B", "llama31-70b": "Llama-3.1-70B",
}
VAR_DISP = {
    "tfidf": "", "finetune": "", "text_meta": "",
    "zs_short": "ZS", "lora_basic": "LoRA basic", "lora_big": "LoRA big",
    "lora_natural_distribution": "LoRA natural",
}


def _fmt(v, d=4):
    return f"{v:.{d}f}".replace(".", ",") if isinstance(v, (int, float)) and pd.notna(v) else "---"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pipeline/06_thesis_inputs/all_results_v2.csv")
    ap.add_argument("--out", default="pipeline/06_thesis_inputs/result_rows.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["seed"] == "mean"].copy()

    lines = []
    for ds in ["liar", "truthseeker", "pl_claims", "euvsdisinfo"]:
        d = df[df["dataset"] == ds]
        if d.empty:
            continue
        lines.append(f"\n{'='*70}\n### {ds}  (n={len(d)} kombinacji)\n{'='*70}")
        d = d.sort_values("f1", ascending=False)
        best_f1 = d["f1"].max()
        for _, r in d.iterrows():
            name = DISP.get(r["model"], r["model"])
            vd = VAR_DISP.get(r["variant"], r["variant"])
            label = f"{name} {vd}".strip()
            f1 = r["f1"]
            f1s = f"{_fmt(f1)}" + (f"$\\pm${_fmt(r.get('f1_std'),3)}" if pd.notna(r.get("f1_std")) and r.get("n_seeds", 1) > 1 else "")
            if pd.notna(f1) and abs(f1 - best_f1) < 1e-9:
                f1s = f"\\textbf{{{f1s}}}"
            ci = (f"[{_fmt(r.get('f1_ci_lo'),3)}, {_fmt(r.get('f1_ci_hi'),3)}]"
                  if pd.notna(r.get("f1_ci_lo")) else "---")
            t = r.get("train_s")
            ts = f"{t:.1f}".replace(".", ",") if pd.notna(t) else "---"
            ms = r.get("ms_per_sample")
            mss = f"{ms:.1f}".replace(".", ",") if pd.notna(ms) else "---"
            lines.append(f"  {label:26s} & {_fmt(r.get('accuracy'))} & {f1s} & {ci} & {ts} \\\\   "
                         f"% n={int(r.get('n_seeds',1))} ms/próbkę={mss}")

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Zapisano {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
