"""Generator CSV szablonu do ręcznej kategoryzacji błędów najlepszego modelu.

Czyta `all_results_v2.csv`, znajduje najlepszy model per dataset, ładuje jego
predykcje (z preds_v2/), losuje 50 FP + 50 FN, zapisuje do CSV z pustą kolumną
'category' do uzupełnienia.

Po ręcznej kategoryzacji, `analyze_taxonomy.py` agreguje wyniki.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_data"))
from leakage_audit import LOADERS  # noqa: E402

logger = logging.getLogger("error_taxonomy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


CATEGORIES = [
    "satyra_ironia",
    "opinia_vs_fakt",
    "polprawda",
    "homonimia_kontekst",
    "krotka_wypowiedz_bez_kontekstu",
    "jezyk_obcy",  # głównie dla EU/PL
    "anotacja_watpliwa",
    "inne",
]


def _find_best_model_per_dataset(results_csv: Path, dataset: str) -> tuple[str, str]:
    df = pd.read_csv(results_csv)
    means = df[(df["seed"] == "mean") & (df["dataset"] == dataset)]
    if means.empty:
        raise RuntimeError(f"No 'mean' rows for {dataset}")
    best = means.sort_values("f1", ascending=False).iloc[0]
    return str(best["model"]), str(best["variant"])


def _load_predictions(preds_root: Path, dataset: str, model: str, variant: str, seed: int = 42) -> pd.DataFrame:
    preds_path = preds_root / f"{dataset}_{model}_{variant}_seed{seed}.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(preds_path)
    rows = []
    with preds_path.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results-csv", default="pipeline/06_thesis_inputs/all_results_v2.csv")
    parser.add_argument("--preds-root", default="experiments/preds_v2/")
    parser.add_argument("--seed", type=int, default=42, help="Seed którego predykcje używamy")
    parser.add_argument("--n-fp", type=int, default=50)
    parser.add_argument("--n-fn", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-text-chars", type=int, default=500)
    args = parser.parse_args()

    model, variant = _find_best_model_per_dataset(Path(args.results_csv), args.dataset)
    logger.info("Best model on %s: %s/%s", args.dataset, model, variant)

    preds = _load_predictions(Path(args.preds_root), args.dataset, model, variant, seed=args.seed)
    logger.info("Loaded %d predictions", len(preds))

    # Load dataset for text
    ds = LOADERS[args.dataset](Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    # Mapowanie i → text — predykcje są w tej samej kolejności co test_df podczas SeededRunner.record
    # Trzeba odzyskać test_ids z split file
    with open(Path("experiments/splits_v2") / f"{args.dataset}_seed{args.seed}.json") as fh:
        split = json.load(fh)
    test_ids = list(split["test_ids"])
    if len(test_ids) != len(preds):
        # możliwe że LLM użył sample_size — wtedy mapping jest tylko częściowy
        logger.warning("test_ids (%d) != preds (%d) — może sampling. Używamy ID z preds[i] jako sequence.",
                       len(test_ids), len(preds))

    id_to_text = dict(zip(df["id"].astype(str), df["text"].astype(str)))

    rng = np.random.default_rng(42)
    preds["i"] = preds["i"].astype(int)
    # Jeśli mamy pełne mapping
    if len(test_ids) == len(preds):
        preds["test_id"] = [test_ids[i] for i in preds["i"]]
        preds["text"] = preds["test_id"].map(id_to_text).fillna("")

    fp_mask = (preds["y_true"] == 0) & (preds["y_pred"] == 1)
    fn_mask = (preds["y_true"] == 1) & (preds["y_pred"] == 0)
    n_fp_avail = int(fp_mask.sum())
    n_fn_avail = int(fn_mask.sum())
    logger.info("FP available: %d (sampling %d), FN available: %d (sampling %d)",
                n_fp_avail, args.n_fp, n_fn_avail, args.n_fn)

    fp_sample = preds[fp_mask].sample(n=min(args.n_fp, n_fp_avail), random_state=42)
    fn_sample = preds[fn_mask].sample(n=min(args.n_fn, n_fn_avail), random_state=43)

    out_df = pd.concat([fp_sample, fn_sample], ignore_index=True)
    out_df["error_type"] = out_df.apply(
        lambda r: "FP (model says disinfo, truth is OK)" if r["y_pred"] == 1 else "FN (model misses disinfo)",
        axis=1,
    )
    if "text" in out_df.columns:
        out_df["text_excerpt"] = out_df["text"].astype(str).str[:args.max_text_chars]
    out_df["model"] = model
    out_df["variant"] = variant
    out_df["dataset"] = args.dataset
    out_df["category"] = ""  # do ręcznego wypełnienia
    out_df["notes"] = ""

    cols = ["dataset", "i", "y_true", "y_pred", "error_type", "model", "variant",
            "text_excerpt", "category", "notes"]
    out_df = out_df[[c for c in cols if c in out_df.columns]]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, encoding="utf-8")
    logger.info("Template → %s", out)
    logger.info("Categories (do uzupełnienia w kolumnie 'category'): %s", CATEGORIES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
