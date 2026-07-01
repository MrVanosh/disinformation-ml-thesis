"""Runner dla encoder fine-tuningu (DistilBERT, BERT, mBERT, HerBERT).

Używa Hugging Face Trainer + MPS backend na M4 Pro (lub CUDA na Modal).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_data"))
sys.path.insert(0, str(ROOT / "02_methodology"))

from leakage_audit import LOADERS  # noqa: E402
from seeded_runner import SeededRunner  # noqa: E402
from cost_meter import CostMeter  # noqa: E402

logger = logging.getLogger("encoder_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1": f1}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--eval-dataset", default=None,
                        help="Cross-dataset transfer: trenuj na --dataset, ewaluuj na tym zbiorze")
    parser.add_argument("--eval-split-file", default=None,
                        help="Split zbioru ewaluacyjnego (wymagany gdy --eval-dataset)")
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
    logger.info("Config: %s (%s)", cfg["model_name"], cfg["hf_repo"])

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    ds = LOADERS[args.dataset](Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    with Path(args.split_file).open(encoding="utf-8") as fh:
        split = json.load(fh)
    train_mask = df["id"].astype(str).isin(set(split["train_ids"]))
    test_mask = df["id"].astype(str).isin(set(split["test_ids"]))

    # Eval = test (proste), val carved z train
    train_df = df[train_mask].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    # Cap rozmiaru treningu — duże zbiory (TruthSeeker ~100k z masywną near-dup redundancją:
    # ~1062 tez × ~100 tweetów) trenują się godzinami. 30k stratyfikowane = ample dla BERT,
    # ~5× szybciej, ten sam wniosek. Małe zbiory (liar/pl_claims ~2-3k) są poniżej capu → bez zmian.
    max_train = int(cfg.get("hyperparameters", {}).get("max_train_samples", 30000))
    if len(train_df) > max_train:
        from sklearn.model_selection import train_test_split as _tts
        _, train_df = _tts(train_df, test_size=max_train, stratify=train_df["label"],
                           random_state=args.seed)
        train_df = train_df.reset_index(drop=True)
        logger.info("Train capped: %d → %d (stratified, max_train_samples)", train_mask.sum(), len(train_df))
    n_val = max(50, min(2000, int(len(train_df) * 0.1)))
    val_df = train_df.iloc[:n_val].copy()
    train_df = train_df.iloc[n_val:].copy()

    # Test: in-domain (ten sam zbiór) LUB cross-dataset (inny zbiór ewaluacyjny)
    if args.eval_dataset:
        if not args.eval_split_file:
            raise ValueError("--eval-dataset wymaga --eval-split-file")
        logger.info("CROSS-DATASET: train=%s → eval=%s", args.dataset, args.eval_dataset)
        eds = LOADERS[args.eval_dataset](Path(".").resolve())
        edf = eds.df.reset_index(drop=True)
        if "id" not in edf.columns:
            edf["id"] = edf.index.astype(str)
        with Path(args.eval_split_file).open(encoding="utf-8") as fh:
            esplit = json.load(fh)
        # cross-dataset: bierzemy TEST część zbioru ewaluacyjnego (out-of-domain)
        eval_test_mask = edf["id"].astype(str).isin(set(esplit["test_ids"]))
        test_df = edf[eval_test_mask].copy()
    else:
        test_df = df[test_mask].copy()

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_repo"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_repo"], num_labels=2,
    )

    hp = cfg["hyperparameters"]
    max_length = hp.get("max_length", 128)
    train_ds = TextDataset(train_df["text"].astype(str), train_df["label"].astype(int),
                            tokenizer, max_length)
    val_ds = TextDataset(val_df["text"].astype(str), val_df["label"].astype(int),
                          tokenizer, max_length)
    test_ds = TextDataset(test_df["text"].astype(str), test_df["label"].astype(int),
                           tokenizer, max_length)

    # Device — auto-detect
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    output_dir = Path("experiments/_encoder_train_tmp") / f"{cfg['model_name']}_{args.dataset}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # float() / int() chronią przed YAML 1.1: "2e-5" (bez kropki) parsuje się jako STRING,
    # co wywala optimizer (0.0 <= lr). Koercja czyni runner odpornym na format configu.
    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(hp.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(hp.get("per_device_train_batch_size", 32)),
        per_device_eval_batch_size=int(hp.get("per_device_eval_batch_size", 64)),
        learning_rate=float(hp.get("learning_rate", 2e-5)),
        warmup_ratio=float(hp.get("warmup_ratio", 0.1)),
        weight_decay=float(hp.get("weight_decay", 0.01)),
        max_grad_norm=float(hp.get("gradient_clipping", 1.0)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=hp.get("load_best_model_at_end", True),
        metric_for_best_model=hp.get("metric_for_best", "f1"),
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
        logging_steps=50,
        # transformers 5.x: MPS/CUDA wykrywane automatycznie (usunięto use_mps_device).
        # Wymuszenie CPU tylko gdy brak akceleratora:
        use_cpu=(device == "cpu"),
    )

    callbacks = []
    if hp.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"]))

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    variant = f"transfer_{args.eval_dataset}" if args.eval_dataset else "finetune"
    runner = SeededRunner(
        dataset=args.dataset,
        model=args.run_model or cfg["model_name"],
        variant=args.run_variant or variant,
        seeds=[args.seed],
        output_root=args.output_root,
        preds_root=args.preds_root,
        config_path=args.config,
        data_manifest_sha=args.data_manifest_sha,
    )

    with CostMeter() as cm:
        trainer.train()
        cm.set_trainable_params(model)

        # Predict
        preds_output = trainer.predict(test_ds)
        logits = preds_output.predictions
        y_pred = np.argmax(logits, axis=-1)
        # Softmax → prob
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_prob = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        cm.set_n_samples(len(test_df))

    runner.record(
        seed=args.seed,
        y_true=test_df["label"].astype(int).values,
        y_pred=y_pred,
        y_prob=y_prob,
        logits=logits,
        cost=cm.report(),
        extra={"device": device, "train_dataset": args.dataset,
               "eval_dataset": args.eval_dataset or args.dataset,
               "is_cross_dataset": bool(args.eval_dataset)},
    )

    # Cleanup checkpointów
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
