"""Runner dla klasycznych modeli (LR/SVM/RF) na TF-IDF.

Czyta YAML config + split file + dataset, trenuje model, zapisuje rezultaty
przez SeededRunner do experiments/results_v2/.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Add pipeline modules to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_data"))
sys.path.insert(0, str(ROOT / "02_methodology"))

from leakage_audit import LOADERS  # noqa: E402
from seeded_runner import SeededRunner  # noqa: E402
from cost_meter import CostMeter  # noqa: E402

logger = logging.getLogger("classical_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _clean_text(text: str, lower: bool = True, remove_urls: bool = True,
                remove_mentions: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    if remove_urls:
        text = re.sub(r"https?://\S+", " ", text)
    if remove_mentions:
        text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if lower:
        text = text.lower()
    return text


def _make_vectorizer(cfg: dict) -> TfidfVectorizer:
    v = cfg.get("vectorizer", {})
    return TfidfVectorizer(
        ngram_range=tuple(v.get("ngram_range", [1, 2])),
        max_features=v.get("max_features", 50000),
        sublinear_tf=v.get("sublinear_tf", True),
        min_df=v.get("min_df", 2),
        max_df=v.get("max_df", 0.95),
    )


def _make_classifier(cfg: dict, seed: int):
    c = cfg.get("classifier", {})
    model_name = cfg["model_name"]
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=c.get("C", 1.0),
            penalty=c.get("penalty", "l2"),
            solver=c.get("solver", "liblinear"),
            max_iter=c.get("max_iter", 1000),
            class_weight=c.get("class_weight", "balanced"),
            random_state=seed,
        )
    if model_name == "linear_svm":
        return LinearSVC(
            C=c.get("C", 1.0),
            loss=c.get("loss", "squared_hinge"),
            penalty=c.get("penalty", "l2"),
            max_iter=c.get("max_iter", 5000),
            class_weight=c.get("class_weight", "balanced"),
            dual="auto" if c.get("dual") == "auto" else False,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=c.get("n_estimators", 200),
            max_depth=c.get("max_depth"),
            min_samples_split=c.get("min_samples_split", 2),
            min_samples_leaf=c.get("min_samples_leaf", 1),
            n_jobs=c.get("n_jobs", -1),
            class_weight=c.get("class_weight", "balanced"),
            random_state=seed,
        )
    raise ValueError(f"Unknown classical model: {model_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--data-manifest-sha", default=None)
    parser.add_argument("--run-model", default=None,
                        help="Override nazwy modelu w pliku wyniku (kanon z orkiestratora)")
    parser.add_argument("--run-variant", default=None,
                        help="Override nazwy wariantu w pliku wyniku")
    parser.add_argument("--output-root", default="experiments/results_v2/")
    parser.add_argument("--preds-root", default="experiments/preds_v2/")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Loaded config: %s", cfg["model_name"])

    # Load dataset
    ds = LOADERS[args.dataset](Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    # Load split
    import json
    with Path(args.split_file).open(encoding="utf-8") as fh:
        split = json.load(fh)
    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])
    train_mask = df["id"].astype(str).isin(train_ids)
    test_mask = df["id"].astype(str).isin(test_ids)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        logger.error("Empty train or test split — check ID matching")
        return 1

    pre = cfg.get("preprocessing", {})
    df["text_clean"] = df["text"].astype(str).map(
        lambda t: _clean_text(
            t,
            lower=pre.get("lowercase", True),
            remove_urls=pre.get("remove_urls", True),
            remove_mentions=pre.get("remove_mentions", args.dataset == "truthseeker"),
        )
    )

    X_train_text = df.loc[train_mask, "text_clean"].tolist()
    X_test_text = df.loc[test_mask, "text_clean"].tolist()
    y_train = df.loc[train_mask, "label"].astype(int).values
    y_test = df.loc[test_mask, "label"].astype(int).values

    runner = SeededRunner(
        dataset=args.dataset,
        model=args.run_model or cfg["model_name"],
        variant=args.run_variant or "tfidf",
        seeds=[args.seed],
        output_root=args.output_root,
        preds_root=args.preds_root,
        config_path=args.config,
        data_manifest_sha=args.data_manifest_sha,
    )

    with CostMeter() as cm:
        vec = _make_vectorizer(cfg)
        X_train = vec.fit_transform(X_train_text)
        X_test = vec.transform(X_test_text)

        clf = _make_classifier(cfg, args.seed)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        # Probability for AUC/calibration (LinearSVC nie ma predict_proba domyślnie)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)
        elif hasattr(clf, "decision_function"):
            df_score = clf.decision_function(X_test)
            # Sigmoid as pseudo-prob
            y_prob_pos = 1.0 / (1.0 + np.exp(-df_score))
            y_prob = np.stack([1 - y_prob_pos, y_prob_pos], axis=1)
        else:
            y_prob = None

        cm.set_n_samples(len(y_test))
        cm.set_trainable_params(clf)  # noop dla sklearn

    runner.record(
        seed=args.seed,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        cost=cm.report(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
