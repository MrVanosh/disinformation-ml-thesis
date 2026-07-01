"""Reproducible GroupShuffleSplit utility — wspólny dla wszystkich faz E/F.

Funkcjonalność:
  - `grouped_split(df, group_col, test_size, seed)` → (train_idx, test_idx).
  - Walidacja: brak overlapu grup, stratifikacja "miękka" (proporcja klas).
  - Dla każdego zbioru produkuje deterministic split files w `experiments/splits_v2/`
    z polami: dataset, seed, train_ids, test_ids, group_overlap_check.

Te splity są jedynym źródłem prawdy dla wszystkich runów w Fazie E.
Aggregator (Faza F) sprawdza że predykcje pochodzą z tych samych split_ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger("grouped_split")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def grouped_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (train_idx, test_idx) z gwarancją braku overlapu grup."""
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' missing")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))

    # Sanity: brak overlapu
    train_groups = set(df.iloc[train_idx][group_col].tolist())
    test_groups = set(df.iloc[test_idx][group_col].tolist())
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Group overlap: {len(overlap)} groups — bug in splitter")

    # Class balance check (soft)
    if "label" in df.columns:
        train_pos = df.iloc[train_idx]["label"].mean()
        test_pos = df.iloc[test_idx]["label"].mean()
        delta = abs(train_pos - test_pos)
        if delta > 0.15:
            logger.warning("Class imbalance between splits: train_pos=%.3f, test_pos=%.3f (Δ=%.3f)",
                           train_pos, test_pos, delta)

    return np.array(train_idx), np.array(test_idx)


def write_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset_name: str,
    group_col: str,
    seed: int,
    output_dir: Path,
    id_column: str = "id",
) -> Path:
    """Zapisuje split do JSON w deterministycznej strukturze."""
    if id_column not in df.columns:
        # Fallback to index
        df = df.reset_index().rename(columns={"index": id_column})

    train_ids = df.iloc[train_idx][id_column].astype(str).tolist()
    test_ids = df.iloc[test_idx][id_column].astype(str).tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}_seed{seed}.json"
    data = {
        "dataset": dataset_name,
        "group_column": group_col,
        "seed": seed,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "split_hash": hashlib.sha256(
            (",".join(sorted(train_ids)) + "|" + ",".join(sorted(test_ids))).encode("utf-8")
        ).hexdigest()[:16],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.info("Split → %s (hash %s)", out_path, data["split_hash"])
    return out_path


def load_split(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=["liar", "truthseeker", "euvsdisinfo",
                                 "pl_corpus", "pl_claims", "pl_articles"])
    parser.add_argument("--seeds", default="13,42,71,89,113")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-dir", default="experiments/splits_v2/")
    args = parser.parse_args()

    # Re-use loaders from leakage_audit
    sys.path.insert(0, str(Path(__file__).parent))
    from leakage_audit import LOADERS
    loader = LOADERS[args.dataset]
    ds = loader(Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    for seed in (int(s) for s in args.seeds.split(",")):
        train_idx, test_idx = grouped_split(
            df, group_col=ds.group_column, test_size=args.test_size, seed=seed,
        )
        write_split(df, train_idx, test_idx, args.dataset, ds.group_column, seed,
                    Path(args.output_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
