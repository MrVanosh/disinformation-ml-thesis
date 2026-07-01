"""Runner dla LLM zero-shot inferencji.

Backendy:
  - mlx_lm (Apple Silicon, default lokalnie) — szybkie i optymalizowane na M4 Pro.
  - transformers + bitsandbytes (Modal NVIDIA) — fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

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

logger = logging.getLogger("llm_zs_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _detect_backend() -> str:
    try:
        import mlx.core as mx  # noqa
        return "mlx"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "transformers_cuda"
    except ImportError:
        pass
    return "transformers_cpu"


def _load_model_mlx(repo: str):
    from mlx_lm import load
    logger.info("Loading mlx-lm model: %s", repo)
    model, tokenizer = load(repo)
    return model, tokenizer


def _generate_mlx(model, tokenizer, prompt: str, max_tokens: int, temperature: float,
                  top_p: float = 1.0):
    # mlx_lm 0.31.x: temperatura przez sampler (usunięto kwarg temp= z generate()).
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temperature, top_p=top_p)
    text = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, sampler=sampler, verbose=False,
    )
    return text


def _load_model_transformers(hf_repo: str, quantize: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = None
    if quantize:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16,
    )
    return model, tokenizer


def _generate_transformers(model, tokenizer, prompt: str, max_tokens: int, temperature: float):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="short_name z config models[].short_name")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--prompt-variant", default="short", choices=["short", "cot"])
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Override sample size z configu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "transformers"])
    parser.add_argument("--run-model", default=None,
                        help="Override nazwy modelu w pliku wyniku (kanon z orkiestratora)")
    parser.add_argument("--run-variant", default=None,
                        help="Override nazwy wariantu w pliku wyniku")
    parser.add_argument("--output-root", default="experiments/results_v2/")
    parser.add_argument("--preds-root", default="experiments/preds_v2/")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Znajdź model w configu
    model_cfg = next((m for m in cfg["models"] if m["short_name"] == args.model), None)
    if model_cfg is None:
        logger.error("Model %s not in config", args.model)
        return 1

    # Backend
    backend = args.backend if args.backend != "auto" else _detect_backend()
    logger.info("Backend: %s, model: %s", backend, model_cfg["short_name"])

    # Sample size
    sample_size = args.sample_size or cfg["sample_sizes"].get(args.dataset)
    if model_cfg.get("sample_size_override", {}).get(args.dataset):
        sample_size = model_cfg["sample_size_override"][args.dataset]
    logger.info("Sample size: %s", sample_size)

    # Data
    ds = LOADERS[args.dataset](Path(".").resolve())
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    with Path(args.split_file).open(encoding="utf-8") as fh:
        split = json.load(fh)
    test_df = df[df["id"].astype(str).isin(set(split["test_ids"]))].copy()

    # Stratified sample
    if sample_size and len(test_df) > sample_size:
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(
            test_df, test_size=sample_size,
            stratify=test_df["label"] if test_df["label"].nunique() > 1 else None,
            random_state=args.seed,
        )

    if args.dry_run and len(test_df) > 100:
        test_df = test_df.head(100)

    logger.info("Test set size: %d", len(test_df))

    # Load model
    if backend == "mlx":
        model, tokenizer = _load_model_mlx(model_cfg.get("mlx_repo") or model_cfg["hf_repo"])
    else:
        model, tokenizer = _load_model_transformers(model_cfg["hf_repo"], quantize=True)

    gen = cfg["generation"]
    max_tokens = gen.get("max_tokens" if args.prompt_variant == "short" else "cot_max_tokens", 8)
    temperature = gen.get("temperature", 0.0)

    runner = SeededRunner(
        dataset=args.dataset,
        model=args.run_model or model_cfg["short_name"],
        variant=args.run_variant or f"zs_{args.prompt_variant}",
        seeds=[args.seed],
        output_root=args.output_root,
        preds_root=args.preds_root,
        config_path=args.config,
    )

    y_true_list, y_pred_list, raw_responses = [], [], []
    n_unparsed = 0

    with CostMeter() as cm:
        for i, (_, row) in enumerate(test_df.iterrows()):
            prompt = build_prompt(args.dataset, args.prompt_variant, str(row["text"]))
            if backend == "mlx":
                resp = _generate_mlx(model, tokenizer, prompt, max_tokens, temperature)
            else:
                resp = _generate_transformers(model, tokenizer, prompt, max_tokens, temperature)
            label = parse_response(resp, args.prompt_variant)
            if label is None:
                # Failsafe: większościowa klasa (zwykle 1=disinfo dla EU/PL)
                n_unparsed += 1
                label = 1 if args.dataset in ("euvsdisinfo", "pl_corpus") else 0
            y_true_list.append(int(row["label"]))
            y_pred_list.append(int(label))
            raw_responses.append(resp[:200])

            if (i + 1) % 100 == 0:
                logger.info("Progress %d/%d (unparsed=%d)", i + 1, len(test_df), n_unparsed)

        cm.set_n_samples(len(test_df))

    runner.record(
        seed=args.seed,
        y_true=np.array(y_true_list),
        y_pred=np.array(y_pred_list),
        y_prob=None,  # ZS bez prob (mlx-lm nie zwraca prosto logits)
        cost=cm.report(),
        extra={"backend": backend, "n_unparsed": n_unparsed,
               "sample_size": len(test_df), "prompt_variant": args.prompt_variant},
    )

    # Zapisz raw responses dla manualnej inspekcji
    raw_path = Path(args.preds_root) / f"{args.dataset}_{model_cfg['short_name']}_zs_{args.prompt_variant}_seed{args.seed}_raw.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as fh:
        for i, (yt, yp, r) in enumerate(zip(y_true_list, y_pred_list, raw_responses)):
            fh.write(json.dumps({"i": i, "y_true": yt, "y_pred": yp, "raw_response": r}, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
