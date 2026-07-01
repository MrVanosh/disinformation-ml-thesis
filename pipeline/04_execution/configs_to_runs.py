"""Generator listy (dataset × model × variant × seed) jobs do uruchomienia.

Output JSONL gdzie każda linia to jeden job:
  {
    "dataset": "truthseeker",
    "model_short": "llama31-8b",
    "variant": "lora_basic",
    "seed": 42,
    "runner": "llm_lora",
    "config_path": "pipeline/03_models/configs/lora_basic.yaml",
    "split_path": "experiments/splits_v2/truthseeker_seed42.json",
    "compute": "local_mps",
    "estimated_minutes": 30,
  }

Konsumowane przez run_local_all.py i run_modal_all.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path

import yaml

logger = logging.getLogger("configs_to_runs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


DATASETS_TO_RUN = ["liar", "truthseeker", "euvsdisinfo", "pl_claims"]

CLASSICAL_CONFIGS = [
    "classical_lr.yaml", "classical_svm.yaml", "classical_rf.yaml",
]
ENCODER_CONFIGS = {
    "liar": ["encoder_distilbert.yaml", "encoder_bert_base.yaml"],
    "truthseeker": ["encoder_distilbert.yaml", "encoder_bert_base.yaml"],
    "euvsdisinfo": ["encoder_mbert.yaml"],
    "pl_claims": ["encoder_mbert.yaml", "encoder_herbert.yaml"],
}

# Wariant C (miks seedów wg kosztu) — liczba seedów per rodzina/wariant.
# Tanie i szybkie → pełne 5 seedów (mocna statystyka); drogie LLM → mniej.
SEED_COUNTS = {
    "classical": 5,
    "encoder": 5,
    "llm_zs": 3,
    "lora_basic": 1,
    "lora_big": 3,
    "lora_natural": 3,
}

ESTIMATED_MIN = {
    "classical": 1,
    "encoder": 15,
    "llm_zs_8b": 25,
    "llm_zs_14b": 35,
    "llm_zs_70b": 60,
    "lora_basic": 30,
    "lora_big": 100,
    "lora_natural": 50,
    "ensemble": 5,
}


def _runs_for_classical(seeds: list[int]) -> list[dict]:
    runs = []
    seeds = seeds[:SEED_COUNTS["classical"]]
    for ds, cfg in product(DATASETS_TO_RUN, CLASSICAL_CONFIGS):
        for seed in seeds:
            runs.append({
                "dataset": ds,
                "model_short": Path(cfg).stem.replace("classical_", ""),
                "variant": "tfidf",
                "seed": seed,
                "runner": "classical",
                "config_path": f"pipeline/03_models/configs/{cfg}",
                "split_path": f"experiments/splits_v2/{ds}_seed{seed}.json",
                "compute": "local_cpu",
                "estimated_minutes": ESTIMATED_MIN["classical"],
                "tier": 1,
            })
    return runs


def _runs_for_encoder(seeds: list[int]) -> list[dict]:
    runs = []
    seeds = seeds[:SEED_COUNTS["encoder"]]
    for ds, configs in ENCODER_CONFIGS.items():
        for cfg in configs:
            for seed in seeds:
                runs.append({
                    "dataset": ds,
                    "model_short": Path(cfg).stem.replace("encoder_", ""),
                    "variant": "finetune",
                    "seed": seed,
                    "runner": "encoder",
                    "config_path": f"pipeline/03_models/configs/{cfg}",
                    "split_path": f"experiments/splits_v2/{ds}_seed{seed}.json",
                    "compute": "local_mps",
                    "estimated_minutes": ESTIMATED_MIN["encoder"],
                    "tier": 1,
                })
    return runs


def _runs_for_llm(seeds: list[int]) -> list[dict]:
    runs = []
    base = Path("pipeline/03_models/configs")

    # ZS — Wariant C: 3 seedy (chyba że model ma własny seeds_override)
    zs_seeds = seeds[:SEED_COUNTS["llm_zs"]]
    with (base / "llm_zs.yaml").open() as fh:
        zs_cfg = yaml.safe_load(fh)
    for model in zs_cfg["models"]:
        m_seeds = model.get("seeds_override", zs_seeds)
        for ds in DATASETS_TO_RUN:
            if model["tier"] == 3 and ds not in ("truthseeker", "euvsdisinfo"):
                continue
            if model["tier"] == 2 and ds == "pl_claims":
                continue
            for seed in m_seeds:
                est = ESTIMATED_MIN.get(f"llm_zs_{model['short_name'].split('-')[-1]}",
                                         ESTIMATED_MIN["llm_zs_8b"])
                runs.append({
                    "dataset": ds, "model_short": model["short_name"],
                    "variant": "zs_short", "seed": seed,
                    "runner": "llm_zs",
                    "config_path": f"pipeline/03_models/configs/llm_zs.yaml",
                    "split_path": f"experiments/splits_v2/{ds}_seed{seed}.json",
                    "compute": model["preferred_compute"],
                    "estimated_minutes": est,
                    "tier": model["tier"],
                })

    # LoRA basic + big + natural — Wariant C: basic 1 seed, big/natural 3 seedy
    for lora_yaml in ["lora_basic.yaml", "lora_big.yaml", "lora_natural.yaml"]:
        with (base / lora_yaml).open() as fh:
            lcfg = yaml.safe_load(fh)
        applies = lcfg.get("applies_to", DATASETS_TO_RUN)
        variant_name = lcfg["variant"]
        # 'natural_distribution' → klucz polityki 'lora_natural'
        policy_key = "lora_natural" if variant_name.startswith("natural") else f"lora_{variant_name}"
        lora_seeds = seeds[:SEED_COUNTS.get(policy_key, 3)]
        for model in lcfg["models"]:
            est_key = f"lora_{lcfg['variant']}"
            for ds in applies:
                for seed in lora_seeds:
                    runs.append({
                        "dataset": ds, "model_short": model["short_name"],
                        "variant": f"lora_{lcfg['variant']}", "seed": seed,
                        "runner": "llm_lora",
                        "config_path": f"pipeline/03_models/configs/{lora_yaml}",
                        "split_path": f"experiments/splits_v2/{ds}_seed{seed}.json",
                        "compute": model.get("preferred_compute", "local_mps"),
                        "estimated_minutes": ESTIMATED_MIN.get(est_key, 50),
                        "tier": model.get("tier", 1),
                    })

    return runs


def _runs_for_cross_dataset(seeds: list[int]) -> list[dict]:
    """Cross-dataset transfer: train na A, eval na B (out-of-domain generalizacja).

    Pary kompatybilne formatem zadania:
      - claim-level:    liar ↔ pl_claims (krótkie tezy + werdykt)
      - document-level: euvsdisinfo ↔ pl_articles (pełne artykuły)
    Modele: mBERT (wielojęzyczny encoder) + najlepszy big-LoRA. 3 seedy (analiza uzupełniająca).
    """
    pairs = [
        ("liar", "pl_claims"), ("pl_claims", "liar"),
        ("euvsdisinfo", "pl_articles"), ("pl_articles", "euvsdisinfo"),
    ]
    cross_seeds = seeds[:3]  # cross-dataset jako analiza uzupełniająca — 3 seedy wystarczą
    runs = []
    for train_ds, eval_ds in pairs:
        for seed in cross_seeds:
            # (a) encoder mBERT
            runs.append({
                "dataset": train_ds, "eval_dataset": eval_ds,
                "model_short": "mbert-base", "variant": f"transfer_{eval_ds}",
                "seed": seed, "runner": "encoder",
                "config_path": "pipeline/03_models/configs/encoder_mbert.yaml",
                "split_path": f"experiments/splits_v2/{train_ds}_seed{seed}.json",
                "eval_split_path": f"experiments/splits_v2/{eval_ds}_seed{seed}.json",
                "compute": "local_mps",
                "estimated_minutes": ESTIMATED_MIN.get("encoder", 15),
                "tier": 1,
            })
            # (b) najlepszy big-LoRA (Qwen 7B — pilotowo najlepszy na claim/TS)
            runs.append({
                "dataset": train_ds, "eval_dataset": eval_ds,
                "model_short": "qwen25-7b", "variant": f"lora_big_transfer_{eval_ds}",
                "seed": seed, "runner": "llm_lora",
                "config_path": "pipeline/03_models/configs/lora_big.yaml",
                "split_path": f"experiments/splits_v2/{train_ds}_seed{seed}.json",
                "eval_split_path": f"experiments/splits_v2/{eval_ds}_seed{seed}.json",
                "compute": "local_mps",
                "estimated_minutes": ESTIMATED_MIN.get("lora_big", 100),
                "tier": 1,
            })
    return runs


def _runs_for_ensemble(seeds: list[int]) -> list[dict]:
    return [{
        "dataset": "truthseeker", "model_short": "ensemble",
        "variant": "text_meta", "seed": seed,
        "runner": "ensemble",
        "config_path": "pipeline/03_models/configs/classical_xgb_metadata.yaml",
        "split_path": f"experiments/splits_v2/truthseeker_seed{seed}.json",
        "compute": "local_cpu",
        "estimated_minutes": ESTIMATED_MIN["ensemble"],
        "tier": 1,
    } for seed in seeds]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default="1,2,3", help="Comma-sep tier IDs to include")
    parser.add_argument("--seeds", default="13,42,71,89,113")
    parser.add_argument("--datasets", default=None,
                        help="Comma-sep filtr zbiorów (np. 'liar,truthseeker,pl_claims' by pominąć EU). Domyślnie wszystkie.")
    parser.add_argument("--include-cross-dataset", action="store_true",
                        help="Dodaj joby cross-dataset transfer (wymaga loaderów pl_claims/pl_articles)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tiers = {int(t) for t in args.tiers.split(",")}
    ds_filter = {d.strip() for d in args.datasets.split(",")} if args.datasets else None

    runs = []
    runs.extend(_runs_for_classical(seeds))
    runs.extend(_runs_for_encoder(seeds))
    runs.extend(_runs_for_llm(seeds))
    runs.extend(_runs_for_ensemble(seeds))
    if args.include_cross_dataset:
        runs.extend(_runs_for_cross_dataset(seeds))

    runs = [r for r in runs if r["tier"] in tiers]
    if ds_filter is not None:
        # zostaw job tylko gdy zbiór treningowy (i ewentualny eval) jest w filtrze
        runs = [r for r in runs
                if r["dataset"] in ds_filter
                and (r.get("eval_dataset") is None or r["eval_dataset"] in ds_filter)]
        logger.info("Po filtrze --datasets %s: %d jobs", sorted(ds_filter), len(runs))
    runs.sort(key=lambda r: (r["estimated_minutes"], r["dataset"], r["model_short"], r["seed"]))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r) + "\n")

    total_min = sum(r["estimated_minutes"] for r in runs)
    logger.info("Generated %d jobs", len(runs))
    logger.info("Estimated total time: %d min (%.1f h)", total_min, total_min / 60)
    by_compute = {}
    for r in runs:
        by_compute.setdefault(r["compute"], []).append(r)
    for c, rs in by_compute.items():
        logger.info("  %s: %d jobs, ~%d min", c, len(rs), sum(r["estimated_minutes"] for r in rs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
