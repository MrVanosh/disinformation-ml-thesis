"""Modal app — runners dla zadań niemieszczących się lokalnie:
  - LLM 70B ZS (Llama 3.1 70B Instruct, 4-bit, H100 80GB)
  - LLM big LoRA fallback (gdy lokalnie wolne, A100 80GB)

Uruchamiać:
    modal run pipeline/04_execution/modal_app.py::llm_70b_zs_all
    modal run pipeline/04_execution/modal_app.py::big_lora_remote --model llama31-8b --dataset truthseeker --seed 42
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("umcs-disinfo-detection")

# Image definicja
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Sprawdzony, kompatybilny stack QLoRA (transformers 5.x łamie peft → set_submodule):
        "torch==2.4.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "accelerate==1.1.1",
        "bitsandbytes==0.44.1",
        "sentencepiece>=0.2.0",
        "scikit-learn>=1.5.0",
        "datasets>=2.20.0",
        "pyyaml>=6.0.1",
        "scipy>=1.14.0",
        "pandas>=2.2.0",
        "datasketch>=1.6.4",
        "statsmodels>=0.14.0",
        "matplotlib>=3.9.0",
    )
    .env({"HF_HOME": "/models", "HF_HUB_ENABLE_HF_TRANSFER": "0"})  # cache modeli na volume
    .add_local_dir("pipeline", remote_path="/app/pipeline")  # cały kod
)

# Volumes
data_vol = modal.Volume.from_name("disinfo-data", create_if_missing=True)
models_vol = modal.Volume.from_name("disinfo-models", create_if_missing=True)
results_vol = modal.Volume.from_name("disinfo-results", create_if_missing=True)

VOLUMES = {
    "/data": data_vol,
    "/models": models_vol,
    "/results": results_vol,
}

# Secrets
hf_secret = modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])


@app.function(
    image=image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=60 * 60 * 4,  # 4h
    memory=131_072,  # 128 GB RAM (host)
)
def llm_70b_zs_one(
    model_short: str = "llama31-70b",
    dataset: str = "truthseeker",
    seed: int = 42,
    sample_size: int | None = None,
    prompt_variant: str = "short",
) -> dict:
    """Single (model, dataset, seed) zero-shot inference dla 70B na H100."""
    import sys
    sys.path.insert(0, "/app/pipeline/01_data")
    sys.path.insert(0, "/app/pipeline/02_methodology")
    sys.path.insert(0, "/app/pipeline/03_models")
    sys.path.insert(0, "/app/pipeline/04_execution/runners")

    # Reuse local runner code w środowisku Modal
    import json
    import yaml
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from leakage_audit import LOADERS
    from seeded_runner import SeededRunner
    from cost_meter import CostMeter
    from prompts import build_prompt, parse_response

    # Load config
    with open("/app/pipeline/03_models/configs/llm_zs.yaml") as fh:
        cfg = yaml.safe_load(fh)
    model_cfg = next(m for m in cfg["models"] if m["short_name"] == model_short)
    sample_size = sample_size or model_cfg.get("sample_size_override", {}).get(dataset) \
        or cfg["sample_sizes"][dataset]

    # Load data (z Modal volume /data/datasets/)
    os.chdir("/data")
    ds = LOADERS[dataset](Path("."))
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    split_path = Path("/data/experiments/splits_v2") / f"{dataset}_seed{seed}.json"
    with split_path.open() as fh:
        split = json.load(fh)
    test_df = df[df["id"].astype(str).isin(set(split["test_ids"]))].copy()

    # Stratified sample
    if sample_size and len(test_df) > sample_size:
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(
            test_df, test_size=sample_size,
            stratify=test_df["label"] if test_df["label"].nunique() > 1 else None,
            random_state=seed,
        )

    # Model
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_repo"], token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_repo"], quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16,
        token=os.environ["HF_TOKEN"],
    )

    y_true_list, y_pred_list = [], []
    n_unparsed = 0
    with CostMeter() as cm:
        for _, row in test_df.iterrows():
            prompt = build_prompt(dataset, prompt_variant, str(row["text"]))
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            label = parse_response(resp, prompt_variant)
            if label is None:
                n_unparsed += 1
                label = 1 if dataset in ("euvsdisinfo", "pl_corpus") else 0
            y_true_list.append(int(row["label"]))
            y_pred_list.append(int(label))
        cm.set_n_samples(len(test_df))

    runner = SeededRunner(
        dataset=dataset, model=model_short, variant="zs_short",
        seeds=[seed],
        output_root="/results/results_v2/",
        preds_root="/results/preds_v2/",
        config_path="pipeline/03_models/configs/llm_zs.yaml",
    )
    rec_path = runner.record(
        seed=seed,
        y_true=np.array(y_true_list),
        y_pred=np.array(y_pred_list),
        cost=cm.report(),
        extra={"backend": "transformers", "n_unparsed": n_unparsed, "gpu": "H100",
               "sample_size": len(test_df)},
    )
    results_vol.commit()
    return {"path": str(rec_path), "f1_n_unparsed": n_unparsed, "n_samples": len(test_df)}


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret],
              timeout=60 * 60 * 4, memory=131_072)
def big_lora_remote(model_short: str, dataset: str, seed: int,
                    config_name: str = "lora_big", eval_dataset: str | None = None,
                    eval_sample: int = 2000) -> dict:
    """Remote big LoRA (QLoRA transformers+peft na CUDA).

    Mirror llm_lora_runner.py, ale na NVIDIA (H100/H200/B200 zależnie od rozmiaru modelu).
    Zapisuje wynik pod TĄ SAMĄ nazwą co lokalnie (model_short, variant=lora_big[_transfer_X]),
    by `--skip-existing` lokalnej kolejki pominął te joby (zero podwójnego liczenia).
    """
    import sys, json, time
    for p in ("01_data", "02_methodology", "03_models", "04_execution/runners"):
        sys.path.insert(0, f"/app/pipeline/{p}")

    import yaml
    import numpy as np
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, Trainer, DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset as HFDataset

    from leakage_audit import LOADERS
    from seeded_runner import SeededRunner
    from cost_meter import CostMeter
    from prompts import build_prompt, parse_response

    # ── Config ──
    with open(f"/app/pipeline/03_models/configs/{config_name}.yaml") as fh:
        cfg = yaml.safe_load(fh)
    model_cfg = next(m for m in cfg["models"] if m["short_name"] == model_short)
    lora_c, train_c, data_c = cfg["lora_config"], cfg["training"], cfg["data_config"]
    tmpl = cfg["prompt_template_id"]
    # cuda_repo = ungated mirror do POBRANIA (te same wagi); hf_repo zostaje oficjalne do cytowania
    repo = model_cfg.get("cuda_repo") or model_cfg["hf_repo"]

    # ── Dane + split ──
    os.chdir("/data")
    ds = LOADERS[dataset](Path("."))
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    with open(f"/data/experiments/splits_v2/{dataset}_seed{seed}.json") as fh:
        split = json.load(fh)
    train_df = df[df["id"].astype(str).isin(set(split["train_ids"]))].copy()
    if eval_dataset:
        eds = LOADERS[eval_dataset](Path("."))
        edf = eds.df.reset_index(drop=True)
        if "id" not in edf.columns:
            edf["id"] = edf.index.astype(str)
        with open(f"/data/experiments/splits_v2/{eval_dataset}_seed{seed}.json") as fh:
            esplit = json.load(fh)
        test_df = edf[edf["id"].astype(str).isin(set(esplit["test_ids"]))].copy()
    else:
        test_df = df[df["id"].astype(str).isin(set(split["test_ids"]))].copy()

    # ── Balansowanie treningu (mirror _balance_training_data) ──
    target = int(data_c["train_size"])
    if data_c["class_balance"] == "balanced":
        pos = train_df[train_df["label"] == 1]; neg = train_df[train_df["label"] == 0]
        n = target // 2
        train_df = (pos.sample(n=min(n, len(pos)), random_state=seed)
                    .pipe(lambda a: __import__("pandas").concat(
                        [a, neg.sample(n=min(n, len(neg)), random_state=seed + 1)]))
                    .sample(frac=1.0, random_state=seed).reset_index(drop=True))
    else:
        train_df = train_df.sample(n=min(target, len(train_df)), random_state=seed).reset_index(drop=True)

    # ── Teksty treningowe (prompt + completion, spójne z mlx) ──
    def _completion(lbl):
        if dataset in ("euvsdisinfo", "pl_corpus", "pl_articles"):
            return "TRUSTWORTHY" if int(lbl) == 0 else "DISINFORMATION"
        return "TRUE" if int(lbl) == 0 else "FALSE"
    train_texts = [build_prompt(dataset, tmpl, str(r["text"])) + " " + _completion(r["label"])
                   for _, r in train_df.iterrows()]

    # ── Model 4-bit + tokenizer ──
    tok = AutoTokenizer.from_pretrained(repo, token=os.environ["HF_TOKEN"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, token=os.environ["HF_TOKEN"])
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=int(lora_c["r"]), lora_alpha=int(lora_c["lora_alpha"]),
        lora_dropout=float(lora_c.get("lora_dropout", 0.05)), bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_c.get("target_modules", ["q_proj", "v_proj"]))
    model = get_peft_model(model, peft_cfg)
    model.enable_input_require_grads()  # wymagane dla gradient checkpointing + PEFT
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Tokenizacja + Trainer ──
    max_len = int(train_c["max_seq_length"])
    hf_ds = HFDataset.from_dict({"text": train_texts}).map(
        lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True,
        remove_columns=["text"])
    targs = TrainingArguments(
        output_dir="/tmp/lora_out", max_steps=int(train_c["num_iters"]),
        per_device_train_batch_size=int(train_c["batch_size"]),
        gradient_accumulation_steps=int(train_c.get("grad_accum", 8)),
        learning_rate=float(train_c["learning_rate"]),
        warmup_steps=int(train_c.get("warmup_steps", 100)),
        weight_decay=float(train_c.get("weight_decay", 0.0)),
        max_grad_norm=float(train_c.get("gradient_clipping", 1.0)),
        lr_scheduler_type="cosine", bf16=True, logging_steps=50,
        gradient_checkpointing=True,  # kluczowe by 70B zmieściło się w VRAM
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no", report_to=[], optim="paged_adamw_8bit", seed=seed)
    torch.cuda.reset_peak_memory_stats()
    with CostMeter() as cm:
        trainer = Trainer(model=model, args=targs, train_dataset=hf_ds,
                          data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        model.config.use_cache = False
        trainer.train()

        # Insurance: zapisz adapter na volume ZARAZ po treningu — jeśli eval padnie/timeout,
        # wytrenowany adapter nie przepada (można go potem doewaluować osobno, tanio).
        try:
            adapter_save = f"/results/adapters/{dataset}_{model_short}_lora_big_seed{seed}"
            model.save_pretrained(adapter_save)
            results_vol.commit()
        except Exception as _e:
            print(f"(adapter save skipped: {_e})")

        # ── Eval (batched generation, left-pad) ──
        model.config.use_cache = True
        model.eval()
        tok.padding_side = "left"
        n_eval = min(eval_sample, len(test_df))
        test_sample = (test_df.sample(n=n_eval, random_state=seed) if len(test_df) > n_eval else test_df)
        prompts = [build_prompt(dataset, tmpl, str(t)) for t in test_sample["text"]]
        y_pred, n_unparsed = [], 0
        BS = 16
        for i in range(0, len(prompts), BS):
            batch = prompts[i:i + BS]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            for j in range(len(batch)):
                gen = out[j][enc["input_ids"].shape[1]:]
                lbl = parse_response(tok.decode(gen, skip_special_tokens=True), tmpl)
                if lbl is None:
                    n_unparsed += 1
                    lbl = 1 if dataset in ("euvsdisinfo", "pl_corpus", "pl_articles") else 0
                y_pred.append(int(lbl))
        y_true = test_sample["label"].astype(int).values
        cm.set_n_samples(len(y_true))

    gpu_name = torch.cuda.get_device_name(0)
    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    variant_tag = f"lora_big_transfer_{eval_dataset}" if eval_dataset else "lora_big"
    runner = SeededRunner(
        dataset=dataset, model=model_short, variant=variant_tag, seeds=[seed],
        output_root="/results/results_v2/", preds_root="/results/preds_v2/",
        config_path=f"pipeline/03_models/configs/{config_name}.yaml")
    rec = runner.record(
        seed=seed, y_true=np.array(y_true), y_pred=np.array(y_pred), y_prob=None,
        cost={**cm.report(), "gpu": gpu_name, "peak_vram_mb": round(peak_vram, 1),
              "trainable_params": int(trainable)},
        extra={"backend": "transformers_cuda", "gpu": gpu_name, "n_train": len(train_df),
               "n_eval": int(len(y_true)), "n_unparsed": n_unparsed, "lora_variant": "big",
               "train_dataset": dataset, "eval_dataset": eval_dataset or dataset,
               "is_cross_dataset": bool(eval_dataset), "peak_vram_mb": round(peak_vram, 1)})
    results_vol.commit()
    return {"path": str(rec), "model": model_short, "dataset": dataset, "seed": seed,
            "gpu": gpu_name, "peak_vram_mb": round(peak_vram, 1), "n_unparsed": n_unparsed,
            "trainable_params": int(trainable)}


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 60 * 2)
def prefetch_models(repos: list[str]) -> dict:
    """CPU-only: pobiera modele do /models (volume) RAZ, by równoległe joby GPU
    nie ściągały tych samych wag po wielokroć (oszczędza GPU-sekundy)."""
    from huggingface_hub import snapshot_download
    out = {}
    for r in repos:
        p = snapshot_download(r, token=os.environ["HF_TOKEN"],
                              ignore_patterns=["*.pth", "*.gguf", "original/*"])
        out[r] = p
        print(f"  ✓ {r} → {p}")
    models_vol.commit()
    return out


@app.local_entrypoint()
def big_lora_smoke():
    """Walidacja pipeline'u QLoRA na CUDA — 1 tani job (config lora_smoke, A10G ~$0.10)."""
    print("Prefetch Qwen-7B (ungated)...")
    prefetch_models.remote(["Qwen/Qwen2.5-7B-Instruct"])
    print("Smoke big_lora_remote na A10G...")
    r = big_lora_remote.with_options(gpu="A10G").remote(
        "qwen25-7b", "liar", 13, config_name="lora_smoke")
    print("WYNIK:", r)


@app.local_entrypoint()
def big_lora_matrix(seeds: str = "13,42,71", datasets: str = "liar,truthseeker",
                    include_70b: bool = False):
    """big-LoRA 7-8B (Llama+Qwen × zbiory × seedy) na H100 RÓWNOLEGLE.
    --seeds 13 --datasets euvsdisinfo → EU big-LoRA. Domyślnie liar+truthseeker, 3 seedy.
    include_70b=True dodaje 1-seed test Llama-70B na H200 (config lora_big_70b)."""
    seed_list = [int(s) for s in str(seeds).split(",")]
    ds_list = [d.strip() for d in str(datasets).split(",")]
    print(f"Prefetch modeli 7-8B do /models... (seedy: {seed_list}, zbiory: {ds_list})")
    prefetch_models.remote(["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"])

    jobs = [(m, d, s) for m in ("llama31-8b", "qwen25-7b")
            for d in ds_list for s in seed_list]
    print(f"Spawn {len(jobs)} jobów big-LoRA na H100 (równolegle)...")
    handles = [(j, big_lora_remote.spawn(*j)) for j in jobs]

    if include_70b:
        print("Prefetch + spawn 70B (H200)...")
        prefetch_models.remote(["meta-llama/Llama-3.1-70B-Instruct"])
        h = big_lora_remote.with_options(gpu="H200").spawn(
            "llama31-70b", "truthseeker", 13, config_name="lora_big_70b")
        handles.append((("llama31-70b", "truthseeker", 13), h))

    ok = fail = 0
    for j, h in handles:
        try:
            r = h.get()
            ok += 1
            print(f"  ✅ {j} → vram={r.get('peak_vram_mb')}MB unparsed={r.get('n_unparsed')}")
        except Exception as e:
            fail += 1
            print(f"  ❌ {j} FAILED: {e}")
    print(f"\nGotowe: OK={ok} FAIL={fail}. Wyniki na volume disinfo-results → pobierz: "
          "modal volume get disinfo-results /results_v2 ./experiments/results_v2")


@app.local_entrypoint()
def big_lora_70b(seeds: str = "13", dataset: str = "truthseeker", gpu: str = "A100-80GB"):
    """70B big-LoRA na Modalu (nie mieści się lokalnie na 24GB M4 Pro).
    --seeds 13,42 dla wielu seedów; --gpu H200 jeśli A100-80GB OOM. Odpalać z --detach."""
    seed_list = [int(s) for s in str(seeds).split(",")]
    print(f"Prefetch 70B (~140GB, raz na volume)... seedy={seed_list} GPU={gpu}")
    prefetch_models.remote(["meta-llama/Llama-3.1-70B-Instruct"])
    handles = [(s, big_lora_remote.with_options(gpu=gpu, timeout=8 * 3600).spawn(
        "llama31-70b", dataset, s, config_name="lora_big_70b", eval_sample=800)) for s in seed_list]
    for s, h in handles:
        try:
            print(f"✅ 70B {dataset} seed{s} →", h.get())
        except Exception as e:
            print(f"❌ 70B {dataset} seed{s}: {e}")


@app.local_entrypoint()
def zs_matrix(models: str = "llama31-8b,qwen25-7b,qwen25-14b",
              datasets: str = "euvsdisinfo", seeds: str = "13,42,71",
              gpu: str = "A100-80GB"):
    """ZS (zero-shot, inferencja) dla małych/średnich LLM na Modalu — offload z Maca.
    Domyślnie EU × {8B,7B,14B} × 3 seedy. Tańszy GPU (inferencja, bez treningu)."""
    m_list = [m.strip() for m in models.split(",")]
    d_list = [d.strip() for d in datasets.split(",")]
    s_list = [int(s) for s in seeds.split(",")]
    repo_map = {"llama31-8b": "meta-llama/Llama-3.1-8B-Instruct",
                "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
                "qwen25-14b": "Qwen/Qwen2.5-14B-Instruct"}
    print(f"Prefetch ZS modeli: {m_list}")
    prefetch_models.remote([repo_map[m] for m in m_list if m in repo_map])
    fn = llm_70b_zs_one.with_options(gpu=gpu, timeout=2 * 3600)
    handles = [((m, d, s), fn.spawn(model_short=m, dataset=d, seed=s))
               for m in m_list for d in d_list for s in s_list]
    ok = fail = 0
    for (m, d, s), h in handles:
        try:
            h.get(); ok += 1
            print(f"  ✅ ZS {m}/{d} seed{s}")
        except Exception as e:
            fail += 1
            print(f"  ❌ ZS {m}/{d} seed{s}: {e}")
    print(f"\nZS gotowe: OK={ok} FAIL={fail}. Pobierz: "
          "modal volume get disinfo-results /results_v2 ./experiments/results_v2")


@app.local_entrypoint()
def llm_70b_zs_all():
    """Uruchom 70B ZS dla 3 seedów × 2 datasety (TS, EU) — total ~$30-40."""
    seeds = [13, 42, 71]
    datasets = ["truthseeker", "euvsdisinfo"]
    results = []
    for ds in datasets:
        for s in seeds:
            print(f"Running 70B ZS: {ds} seed={s}")
            r = llm_70b_zs_one.remote(model_short="llama31-70b", dataset=ds, seed=s)
            results.append((ds, s, r))
            print(f"  → {r}")
    print("\nDone. Results:")
    for ds, s, r in results:
        print(f"  {ds} seed={s}: {r}")
