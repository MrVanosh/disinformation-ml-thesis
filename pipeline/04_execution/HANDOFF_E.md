# HANDOFF Faza E — Execution (matrix runów ~270 z 3-tierowym lineupem)

## Cel

Wykonać pełną matrycę eksperymentów i deponować wyniki do `experiments/results_v2/` +
predykcje do `experiments/preds_v2/` w schemie konsumowanym przez aggregator (Faza C).

3-tierowy lineup zatwierdzony:
- **Tier 1** (lokalnie M4 Pro, must-have): Llama 3.1 8B, Qwen 2.5 7B.
- **Tier 2** (lokalnie, stretch): Qwen 2.5 14B (ZS + basic LoRA, bez big LoRA).
- **Tier 3** (Modal H100, headline): Llama 3.1 70B (ZS only, 3 seedy, 2 datasety).

## Run matrix (finalna)

| Dataset | Klasyczne | Encoder | LLM Tier1 (ZS+bL+bigL) | LLM Tier2 (ZS+bL) | LLM Tier3 (ZS only) | Seedy | Suma |
|---|---|---|---|---|---|---|---|
| LIAR | 3 (LR/SVM/RF) | 2 (DistilBERT/BERT) | 2 × 3 = 6 | 1 × 2 = 2 | – | 5 | 65 |
| TruthSeeker (grouped) | 3 + 1 ensemble | 2 | 2 × 3 = 6 | 1 × 2 = 2 | 1 (Llama 70B) | 5/3 | 73 |
| EUvsDisinfo (grouped) | 3 | 2 (mBERT + 1) | 2 × 3 = 6 | 1 × 2 = 2 | 1 (Llama 70B) | 5/3 | 73 |
| PL corpus | 2 (LR/SVM) | 2 (mBERT/HerBERT) | 1 × 1 (Llama 8B ZS) | – | – | 5 | 30 |
| **+ LoRA-natural** | – | – | 2 (Llama/Qwen on LIAR) | – | – | 5 | 10 |
| **Razem** | | | | | | | **~250** |

(Liczby przybliżone, kilka kombinacji wykluczonych — np. HerBERT tylko na PL, ensemble tylko na TS.)

## Pliki dostarczane w tym HANDOFF

| Plik | Cel |
|---|---|
| `04_execution/runners/classical_runner.py` | Klasyczne ML (TF-IDF + LR/SVM/RF/XGBoost meta) |
| `04_execution/runners/encoder_runner.py` | Encoder fine-tune (HuggingFace Trainer) |
| `04_execution/runners/llm_zs_runner.py` | LLM zero-shot (mlx-lm lokalnie, transformers fallback) |
| `04_execution/runners/llm_lora_runner.py` | LLM LoRA (mlx-lm fine-tune lokalnie, PEFT na Modal) |
| `04_execution/runners/ensemble_runner.py` | Stacking text+metadata (TruthSeeker only) |
| `04_execution/modal_app.py` | Modal app definicja (volumes, secrets, GPU decorators) |
| `04_execution/run_local_all.py` | Scheduler dla lokalnych runów (Tier 1+2) |
| `04_execution/run_modal_all.py` | Scheduler dla Modal runów (Tier 3 + big LoRA fallback) |
| `04_execution/configs_to_runs.py` | Generator listy (dataset × model × variant × seed) jobs |

## Kolejność wykonania (lokalny CC)

### Step E1: Pre-flight checks

```bash
cd /Users/mbasarab/Workspace/PERSONAL/UMCS/magisterka
source .venv/bin/activate
git pull origin main  # świeże pipeline/ z cloud CC

# Verify splits istnieją (z Fazy B)
ls experiments/splits_v2/  # Oczekiwane: liar_seed{13,42,71,89,113}.json itd.

# Verify manifest
test -f datasets/MANIFEST.json && echo OK
```

### Step E2: Smoke test każdego runnera (1 seed, 1 dataset)

```bash
# Najszybszy — klasyczny LR na LIAR
python pipeline/04_execution/runners/classical_runner.py \
    --config pipeline/03_models/configs/classical_lr.yaml \
    --dataset liar --seed 42 --split-file experiments/splits_v2/liar_seed42.json
# Oczekiwane: ~30s, F1 ~0.55, plik experiments/results_v2/liar_logistic_regression_tfidf_seed42.json

# Encoder — DistilBERT na LIAR
python pipeline/04_execution/runners/encoder_runner.py \
    --config pipeline/03_models/configs/encoder_distilbert.yaml \
    --dataset liar --seed 42 --split-file experiments/splits_v2/liar_seed42.json
# Oczekiwane: ~10 min na MPS, F1 ~0.62

# LLM ZS — Llama 8B na 100 LIAR samples (test)
python pipeline/04_execution/runners/llm_zs_runner.py \
    --config pipeline/03_models/configs/llm_zs.yaml \
    --model llama31-8b \
    --dataset liar --seed 42 --split-file experiments/splits_v2/liar_seed42.json \
    --sample-size 100 --dry-run
# Oczekiwane: ~2 min na MPS, F1 ~0.55
```

Jeśli każdy smoke test przechodzi → kontynuuj E3.

### Step E3: Full local matrix (Tier 1 + 2)

```bash
# Generuje listę wszystkich (dataset, model, variant, seed) lokalnych jobs
python pipeline/04_execution/configs_to_runs.py \
    --tiers 1,2 \
    --output pipeline/04_execution/run_matrix_local.jsonl

# Wykonuje sekwencyjnie (z resume jeśli się przerwie)
python pipeline/04_execution/run_local_all.py \
    --matrix pipeline/04_execution/run_matrix_local.jsonl \
    --skip-existing \
    --report pipeline/04_execution/REPORT_E_local.md
```

Estymowany czas:
- Klasyczne: ~1h łącznie (60 runs × <1 min).
- Encoder: ~15h (90 runs × ~10 min).
- LLM ZS: ~10h (30 runs × ~20 min).
- LLM basic LoRA: ~12h (30 runs × ~25 min).
- LLM big LoRA: ~50h ← **najdłuższy etap**, jedno full weekend.

**Sumarycznie ~80-90h compute lokalnego.** Warto rozbić na noce:
- noc 1: klasyczne + encoder (~15h).
- noc 2-3: LLM ZS + basic LoRA (~25h).
- noc 4-5-6: LLM big LoRA (~50h).

**Alternatywa szybsza**: big LoRA z M4 Pro przenieść na Modal H100 (~$40 łącznie, ~10h wall-clock).

### Step E4: Modal Tier 3 (Llama 70B ZS)

```bash
# Setup (jednorazowo)
modal token new
modal volume create disinfo-data
modal volume create disinfo-models
modal volume create disinfo-results

# Wgraj splits + dane na Modal
modal volume put disinfo-data experiments/splits_v2/
modal volume put disinfo-data datasets/

# Run
modal run pipeline/04_execution/modal_app.py::llm_70b_zs_all
```

Estymowany koszt: ~$30-40 (3 seedy × 2 datasety × ~1.5h H100 × $3.95).

Jeśli wynik 70B ZS jest słaby (poniżej 8B big LoRA), to *kontrybucja naukowa* — argument za specjalizacją.

### Step E5: Aggregate

```bash
python pipeline/02_methodology/aggregator_v2.py
# Wynik:
#   pipeline/06_thesis_inputs/all_results_v2.csv
#   pipeline/06_thesis_inputs/summary_v2.md
#   pipeline/06_thesis_inputs/significance_matrices/*.json
```

### Step E6: Reproducibility manifest

```bash
python pipeline/02_methodology/reproducibility.py
# Wynik: pipeline/06_thesis_inputs/REPRODUCIBILITY.md + pip_freeze.txt
```

## Sanity checks po Fazie E

- [ ] `wc -l pipeline/06_thesis_inputs/all_results_v2.csv` ≥ 250 (long format).
- [ ] Każda kombinacja (dataset, model, variant) ma 5 lub 3 (Tier 3) seedy.
- [ ] `pipeline/06_thesis_inputs/significance_matrices/*.json` istnieje dla 3 datasetów.
- [ ] Brak `null` w kolumnach `f1` (każdy run produkuje metric).

## Co commit'ować

```bash
# Tylko thesis_inputs (CSV/MD), nie surowe results
git add pipeline/06_thesis_inputs/
git add pipeline/04_execution/REPORT_E_local.md
git add pipeline/04_execution/REPORT_E_modal.md
git commit -m "Phase E: full experiment matrix complete"
git push origin main
```

Surowe wyniki w `experiments/results_v2/` zostają lokalnie (gitignored).

## Następny krok

Faza F (analysis): error taxonomy, ablations, cross-dataset generalization, Pareto plots.
