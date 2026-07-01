"""Generate RANDOM (non-grouped) stratified splits — for the leakage-audit
contrast table (M1). Same format/test_size as grouped_split.py, but ignores
groups: this is the naive protocol used in much of the literature.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from leakage_audit import LOADERS  # noqa


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", default="13,42,71,89,113")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--output-dir", default="experiments/splits_random/")
    args = p.parse_args()

    ds = LOADERS[args.dataset](Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    ids = df["id"].astype(str).to_numpy()
    y = df["label"].astype(int).to_numpy()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        tr, te = train_test_split(
            np.arange(len(df)), test_size=args.test_size,
            random_state=seed, stratify=y,
        )
        data = {
            "dataset": args.dataset, "group_column": "__RANDOM__", "seed": seed,
            "n_train": len(tr), "n_test": len(te),
            "train_ids": ids[tr].tolist(), "test_ids": ids[te].tolist(),
        }
        fp = out / f"{args.dataset}_seed{seed}.json"
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"random split → {fp} (train={len(tr)}, test={len(te)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
