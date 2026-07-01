"""Generate EU splits grouped by ARTICLE DOMAIN/publisher (M3 domain-leakage test).
Same format as grouped_split.py, but groups by article_domain instead of debunk_id:
no domain appears in both train and test, so a classifier cannot exploit the
publisher 'fingerprint' (rt/sputnik vs bbc/guardian).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from leakage_audit import LOADERS  # noqa


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="13,42,71,89,113")
    p.add_argument("--group-col", default="article_domain")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--output-dir", default="experiments/splits_domain/")
    args = p.parse_args()

    ds = LOADERS["euvsdisinfo"](Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    ids = df["id"].astype(str).to_numpy()
    groups = df[args.group_col].fillna("__UNK__").astype(str).replace("nan", "__UNK__").to_numpy()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        tr, te = next(gss.split(df, groups=groups))
        # sanity: no domain overlap
        overlap = set(groups[tr]) & set(groups[te])
        assert not overlap, f"domain overlap: {len(overlap)}"
        data = {
            "dataset": "euvsdisinfo", "group_column": args.group_col, "seed": seed,
            "n_train": len(tr), "n_test": len(te),
            "train_ids": ids[tr].tolist(), "test_ids": ids[te].tolist(),
        }
        fp = out / f"euvsdisinfo_seed{seed}.json"
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"domain split → {fp} (train={len(tr)}, test={len(te)}, groups_test={len(set(groups[te]))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
