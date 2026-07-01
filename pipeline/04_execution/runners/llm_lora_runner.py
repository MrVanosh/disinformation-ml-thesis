"""LLM LoRA runner — fine-tune adapter + eval na test split.

Backend:
  - mlx_lm.lora (Apple Silicon) — szybkie lokalnie.
  - transformers + peft (Modal NVIDIA) — fallback dla 70B i big LoRA.

Format danych dla mlx-lm: JSONL z polem 'text' lub {'prompt', 'completion'}.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_data"))
sys.path.insert(0, str(ROOT / "02_methodology"))
sys.path.insert(0, str(ROOT / "03_models"))

from leakage_audit import LOADERS  # noqa: E402
from seeded_runner import SeededRunner  # noqa: E402
from cost_meter import CostMeter  # noqa: E402
from prompts import build_prompt, parse_response  # noqa: E402

logger = logging.getLogger("llm_lora_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _build_mlx_training_data(df, dataset_name: str, prompt_variant: str, output_dir: Path, val_split: float = 0.15):
    """Buduje JSONL kompatybilny z mlx_lm.lora train format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in df.iterrows():
        prompt = build_prompt(dataset_name, prompt_variant, str(row["text"]))
        completion = "TRUE" if int(row["label"]) == 0 else "FALSE"
        if dataset_name in ("euvsdisinfo", "pl_corpus"):
            completion = "TRUSTWORTHY" if int(row["label"]) == 0 else "DISINFORMATION"
        rows.append({"text": prompt + " " + completion})

    n_val = max(50, int(len(rows) * val_split))
    np.random.shuffle(rows)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    with (output_dir / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in train_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (output_dir / "valid.jsonl").open("w", encoding="utf-8") as fh:
        for r in val_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Train: %d, Val: %d → %s", len(train_rows), len(val_rows), output_dir)


def _balance_training_data(df, mode: str, seed: int, target_size: int):
    """Zbalansowanie zbioru treningowego."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    rng = np.random.default_rng(seed)
    if mode == "balanced":
        n_per = target_size // 2
        pos_sample = pos.sample(n=min(n_per, len(pos)), random_state=seed)
        neg_sample = neg.sample(n=min(n_per, len(neg)), random_state=seed + 1)
        return pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:  # natural — keep ratio, just subsample
        if len(df) <= target_size:
            return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return df.sample(n=target_size, random_state=seed).reset_index(drop=True)


def _module_keys(target_modules: list[str]) -> list[str]:
    """Mapuje nazwy projekcji (q_proj…) na klucze modułów mlx_lm (self_attn./mlp.)."""
    attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp = {"gate_proj", "up_proj", "down_proj"}
    keys = []
    for m in target_modules:
        if m in attn:
            keys.append(f"self_attn.{m}")
        elif m in mlp:
            keys.append(f"mlp.{m}")
        else:
            keys.append(m)
    return keys


def _run_mlx_lora(model_repo: str, data_dir: Path, output_adapter_dir: Path, lora_cfg: dict, train_cfg: dict):
    """Wywołuje mlx_lm.lora train przez pełny config YAML (-c).

    KLUCZOWE: rank/alpha/dropout/target_modules mlx_lm czyta TYLKO z configu YAML,
    nie z flag CLI — bez tego LoRA trenuje z domyślnym rank=8 (psuje rozróżnienie
    basic rank16 vs big rank32 w H4/H5). scale = alpha/rank (semantyka LoRA jak w HF).
    """
    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    rank = int(lora_cfg["r"])
    alpha = float(lora_cfg.get("lora_alpha", rank))
    # Klamp num_layers do faktycznej liczby warstw — Qwen2.5-7B ma 28 (Llama-3.1-8B 32),
    # a mlx_lm.lora ERROR jeśli num_layers > warstwy. "all layers" = tyle ile model ma.
    n_layers = int(lora_cfg["num_layers"])
    try:
        from transformers import AutoConfig
        model_layers = getattr(AutoConfig.from_pretrained(model_repo), "num_hidden_layers", n_layers)
        if n_layers > model_layers:
            logger.info("num_layers %d > model %d — klampuję do %d", n_layers, model_layers, model_layers)
            n_layers = model_layers
    except Exception as e:
        logger.warning("Nie odczytano num_hidden_layers (%s) — używam %d", e, n_layers)
    cfg = {
        "model": model_repo,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": n_layers,
        "batch_size": int(train_cfg["batch_size"]),
        "iters": int(train_cfg["num_iters"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "max_seq_length": int(train_cfg["max_seq_length"]),
        "adapter_path": str(output_adapter_dir),
        "save_every": int(train_cfg["save_every"]),
        "steps_per_eval": int(train_cfg.get("save_every", 200)),
        "seed": 42,  # mlx-lm seed; główny seed kontroluje próbkowanie danych
        "lora_parameters": {
            "keys": _module_keys(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
            "rank": rank,
            "scale": alpha / max(1, rank),
            "dropout": float(lora_cfg.get("lora_dropout", 0.0)),
        },
    }
    config_yaml = output_adapter_dir / "mlx_lora_config.yaml"
    config_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # sys.executable = interpreter .venv (unika "python not found")
    cmd = [sys.executable, "-m", "mlx_lm.lora", "-c", str(config_yaml)]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("mlx_lm.lora failed:\n%s", result.stderr[-2000:])
        raise RuntimeError("mlx_lm.lora train failed")
    logger.info("Train output: %s", result.stdout[-2000:])


def _eval_with_adapter_mlx(model_repo: str, adapter_dir: Path, prompts: list, max_tokens: int) -> list[str]:
    """Inference z załadowanym LoRA adapter."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(model_repo, adapter_path=str(adapter_dir))
    sampler = make_sampler(temp=0.0)  # deterministic (greedy)
    responses = []
    for p in prompts:
        r = generate(model, tokenizer, prompt=p, max_tokens=max_tokens,
                     sampler=sampler, verbose=False)
        responses.append(r)
    return responses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--eval-dataset", default=None,
                        help="Cross-dataset transfer: trenuj na --dataset, ewaluuj na tym")
    parser.add_argument("--eval-split-file", default=None)
    parser.add_argument("--prompt-variant", default="short")
    parser.add_argument("--run-model", default=None,
                        help="Override nazwy modelu w pliku wyniku (kanon z orkiestratora)")
    parser.add_argument("--run-variant", default=None,
                        help="Override nazwy wariantu w pliku wyniku")
    parser.add_argument("--output-root", default="experiments/results_v2/")
    parser.add_argument("--preds-root", default="experiments/preds_v2/")
    parser.add_argument("--backend", default="mlx", choices=["mlx", "transformers"])
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    variant = cfg.get("variant", "basic")
    model_cfg = next((m for m in cfg["models"] if m["short_name"] == args.model), None)
    if model_cfg is None:
        logger.error("Model %s not in config", args.model)
        return 1

    # Data
    global pd
    import pandas as pd  # noqa
    ds = LOADERS[args.dataset](Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    with Path(args.split_file).open(encoding="utf-8") as fh:
        split = json.load(fh)
    train_mask = df["id"].astype(str).isin(set(split["train_ids"]))
    train_df = df[train_mask].copy()

    # Test: in-domain LUB cross-dataset (out-of-domain)
    if args.eval_dataset:
        if not args.eval_split_file:
            logger.error("--eval-dataset wymaga --eval-split-file")
            return 1
        logger.info("CROSS-DATASET: train=%s → eval=%s", args.dataset, args.eval_dataset)
        eds = LOADERS[args.eval_dataset](Path(".").resolve())
        edf = eds.df.reset_index(drop=True)
        if "id" not in edf.columns:
            edf["id"] = edf.index.astype(str)
        with Path(args.eval_split_file).open(encoding="utf-8") as fh:
            esplit = json.load(fh)
        test_df = edf[edf["id"].astype(str).isin(set(esplit["test_ids"]))].copy()
    else:
        test_mask = df["id"].astype(str).isin(set(split["test_ids"]))
        test_df = df[test_mask].copy()

    data_cfg = cfg["data_config"]
    train_df = _balance_training_data(
        train_df, mode=data_cfg["class_balance"],
        seed=args.seed, target_size=data_cfg["train_size"],
    )
    logger.info("Train (after balancing): %d, Test: %d", len(train_df), len(test_df))

    # Temp LoRA (dane + adaptery, ~2GB/job) — na T7 jeśli podłączony (lokalny dysk ciasny).
    _tmp_base = Path("/Volumes/T7/AI/lora_tmp") if Path("/Volumes/T7").exists() else Path("experiments")
    job_tag = f"{args.model}_{args.dataset}_{variant}_seed{args.seed}"

    # Build mlx training data
    data_dir = _tmp_base / "_lora_data" / job_tag
    if args.backend == "mlx":
        _build_mlx_training_data(train_df, args.dataset, cfg["prompt_template_id"], data_dir)

    adapter_dir = _tmp_base / "_lora_adapters" / job_tag

    variant_tag = f"lora_{variant}_transfer_{args.eval_dataset}" if args.eval_dataset else f"lora_{variant}"
    runner = SeededRunner(
        dataset=args.dataset,
        model=args.run_model or model_cfg["short_name"],
        variant=args.run_variant or variant_tag,
        seeds=[args.seed],
        output_root=args.output_root,
        preds_root=args.preds_root,
        config_path=args.config,
    )

    with CostMeter() as cm:
        if args.backend == "mlx":
            _run_mlx_lora(
                model_repo=model_cfg.get("mlx_repo") or model_cfg["hf_repo"],
                data_dir=data_dir, output_adapter_dir=adapter_dir,
                lora_cfg=cfg["lora_config"], train_cfg=cfg["training"],
            )

            # Eval
            sample_size = min(2000, len(test_df))
            if len(test_df) > sample_size:
                test_df_sample = test_df.sample(n=sample_size, random_state=args.seed)
            else:
                test_df_sample = test_df

            prompts = [build_prompt(args.dataset, cfg["prompt_template_id"], str(t))
                       for t in test_df_sample["text"]]
            responses = _eval_with_adapter_mlx(
                model_repo=model_cfg.get("mlx_repo") or model_cfg["hf_repo"],
                adapter_dir=adapter_dir, prompts=prompts, max_tokens=8,
            )
            y_pred = [parse_response(r, cfg["prompt_template_id"]) for r in responses]
            y_pred = [p if p is not None else 0 for p in y_pred]
            y_true = test_df_sample["label"].astype(int).values

            cm.set_n_samples(len(y_true))
        else:
            logger.error("transformers+peft backend not yet implemented in this runner — use Modal version")
            return 1

    runner.record(
        seed=args.seed,
        y_true=np.array(y_true), y_pred=np.array(y_pred), y_prob=None,
        cost=cm.report(),
        extra={"backend": args.backend, "n_train": len(train_df),
               "n_eval": len(y_true), "lora_variant": variant,
               "train_dataset": args.dataset,
               "eval_dataset": args.eval_dataset or args.dataset,
               "is_cross_dataset": bool(args.eval_dataset)},
    )

    # Cleanup — usuwamy też adapter (checkpointy ~570MB/job zapychały dysk → mlx crash).
    # Wynik + metryki są w results_v2; adapter nie jest dalej używany.
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(adapter_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
