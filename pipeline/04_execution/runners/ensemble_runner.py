"""Ensemble runner — text (TF-IDF + LR) + metadata (XGBoost) → meta-LR stacking.

Tylko TruthSeeker (jedyny zbiór z bogatymi metadanami konta).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_data"))
sys.path.insert(0, str(ROOT / "02_methodology"))

from leakage_audit import LOADERS  # noqa: E402
from seeded_runner import SeededRunner  # noqa: E402
from cost_meter import CostMeter  # noqa: E402

logger = logging.getLogger("ensemble_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


METADATA_FEATURES = [
    "BotScore", "cred", "followers_count", "normalize_influence",
    "favourites_count", "retweets", "Word count",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-root", default="experiments/results_v2/")
    parser.add_argument("--preds-root", default="experiments/preds_v2/")
    args = parser.parse_args()

    ds = LOADERS["truthseeker"](Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    with Path(args.split_file).open(encoding="utf-8") as fh:
        split = json.load(fh)
    train_mask = df["id"].astype(str).isin(set(split["train_ids"]))
    test_mask = df["id"].astype(str).isin(set(split["test_ids"]))

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # Sprawdź czy metadane istnieją
    avail_meta = [c for c in METADATA_FEATURES if c in df.columns]
    if len(avail_meta) < 4:
        logger.error("Insufficient metadata columns in TS dataset (%s)", avail_meta)
        return 1
    logger.info("Using metadata: %s", avail_meta)

    runner = SeededRunner(
        dataset="truthseeker", model="ensemble", variant="text_meta",
        seeds=[args.seed],
        output_root=args.output_root, preds_root=args.preds_root,
    )

    with CostMeter() as cm:
        # OOF predictions na trainie (5-fold) — żeby uniknąć leakage do meta-LR
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        n_train = len(train_df)
        oof_text = np.zeros((n_train, 2))
        oof_meta = np.zeros((n_train, 2))

        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True, min_df=2)
        X_text_all = vec.fit_transform(train_df["text"].fillna("").astype(str))
        # to_numeric(coerce): kilka wierszy TS ma uszkodzone wartości (np. favourites_count
        # = stringowa tablica '[[0 0 ...]]') — zamieniamy na NaN→0, by XGBoost nie wywalał się.
        X_meta_all = train_df[avail_meta].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        y_train_all = train_df["label"].astype(int).values

        for fold_i, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
            X_tr_text = X_text_all[tr_idx]
            X_va_text = X_text_all[va_idx]
            X_tr_meta = X_meta_all[tr_idx]
            X_va_meta = X_meta_all[va_idx]
            y_tr = y_train_all[tr_idx]

            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
            lr.fit(X_tr_text, y_tr)
            oof_text[va_idx] = lr.predict_proba(X_va_text)

            xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 eval_metric="logloss",
                                 random_state=args.seed, n_jobs=-1)
            xgb.fit(X_tr_meta, y_tr)
            oof_meta[va_idx] = xgb.predict_proba(X_va_meta)

        # Meta-LR
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
        meta_lr.fit(np.hstack([oof_text, oof_meta]), y_train_all)
        logger.info("Meta weights: w_text=%.3f, w_meta=%.3f",
                    np.linalg.norm(meta_lr.coef_[0][:2]), np.linalg.norm(meta_lr.coef_[0][2:]))

        # Train final komponenty na full train
        lr_full = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
        lr_full.fit(X_text_all, y_train_all)

        xgb_full = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  eval_metric="logloss",
                                  random_state=args.seed, n_jobs=-1)
        xgb_full.fit(X_meta_all, y_train_all)

        # Predict na test
        X_test_text = vec.transform(test_df["text"].fillna("").astype(str))
        X_test_meta = test_df[avail_meta].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        p_text = lr_full.predict_proba(X_test_text)
        p_meta = xgb_full.predict_proba(X_test_meta)
        ensemble_features = np.hstack([p_text, p_meta])
        y_prob_ensemble = meta_lr.predict_proba(ensemble_features)
        y_pred = np.argmax(y_prob_ensemble, axis=1)

        cm.set_n_samples(len(test_df))

    runner.record(
        seed=args.seed,
        y_true=test_df["label"].astype(int).values,
        y_pred=y_pred,
        y_prob=y_prob_ensemble,
        cost=cm.report(),
        extra={
            "meta_features": avail_meta,
            "meta_weights": meta_lr.coef_[0].tolist(),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
