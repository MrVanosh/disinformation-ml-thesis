# HANDOFF Faza F — Scientific analysis

## Cel

Z surowych wyników Fazy E zrobić **analitykę naukową** wprost do wstawienia w
Sekcję 3 pracy (z numerami, figurami, tabelami):

1. **Error taxonomy** (per dataset): 100 ręcznie sklasyfikowanych błędów najlepszego modelu.
2. **Ablation studies** (3 ablacje na headline modelu).
3. **Cross-dataset generalization** (transfer matrix).
4. **Cost-quality Pareto plot**.
5. **Calibration plots** (reliability + ECE pre/post temp scaling).
6. **Leakage sensitivity scatter** (F1_drop vs trainable_params).

## Pliki dostarczane

| Plik | Cel |
|---|---|
| `05_analysis/error_taxonomy_template.py` | Generator szablonów CSV do ręcznej kategoryzacji 100 błędów per dataset |
| `05_analysis/ablations.py` | Wykonuje ablacje (rank, depth, metadata) — wymaga dodatkowych runów |
| `05_analysis/cross_dataset_transfer.py` | Train na A → test na B (wszystkie kombinacje) |
| `05_analysis/plots.py` | Cost-quality Pareto + calibration + leakage scatter + seed variance |
| `05_analysis/build_thesis_inputs.py` | Top-level orchestrator — uruchamia wszystko i deponuje do 06_thesis_inputs/ |

## Kolejność wykonania

### Step F1: Error taxonomy (manualna część, ~3-4h pracy autora)

```bash
python pipeline/05_analysis/error_taxonomy_template.py \
    --dataset truthseeker --n-samples 100 \
    --output pipeline/06_thesis_inputs/error_taxonomy_truthseeker.csv

# Powtórz dla liar, euvsdisinfo, pl_corpus
```

Skrypt wybiera **najlepszy model per dataset** (z all_results_v2.csv), losuje 50 FP + 50 FN
i zapisuje do CSV z kolumnami:
  - `i, text, y_true, y_pred, model, category` (puste do uzupełnienia ręcznego).

Autor (lub lokalny CC) wypełnia kolumnę `category` jedną z:
  - satyra_ironia
  - opinia_vs_fakt
  - polprawda
  - homonimia_kontekst
  - krotka_wypowiedz_bez_kontekstu
  - jezyk_obcy
  - anotacja_watpliwa
  - inne

Po wypełnieniu — agregator zlicza i generuje tabelę do pracy.

### Step F2: Cross-dataset transfer

```bash
python pipeline/05_analysis/cross_dataset_transfer.py \
    --train-on truthseeker --test-on liar,euvsdisinfo,pl_corpus \
    --model llama31-8b --variant lora_big \
    --output pipeline/06_thesis_inputs/cross_dataset_matrix.csv
```

Powtarza dla 4 (train_on) × 4 (test_on) = 16 kombinacji × 2 modeli headline (LLM big LoRA + mBERT)
= 32 dodatkowych runów (każdy ~5-15 min). Estymacja: ~6h compute.

### Step F3: Ablations (3 dodatkowe runy na headline modelu)

```bash
# Wpływ rangi LoRA
python pipeline/05_analysis/ablations.py --type lora_rank \
    --base llama31-8b --dataset truthseeker --seeds 13,42,71

# Wpływ głębokości adaptacji
python pipeline/05_analysis/ablations.py --type lora_depth \
    --base llama31-8b --dataset truthseeker --seeds 13,42,71

# Wpływ metadanych w ensemble
python pipeline/05_analysis/ablations.py --type ensemble_metadata \
    --dataset truthseeker --seeds 13,42,71
```

Każda ablacja: 3-5 dodatkowych runów. Sumarycznie ~6h compute.

### Step F4: Plots

```bash
python pipeline/05_analysis/plots.py \
    --results pipeline/06_thesis_inputs/all_results_v2.csv \
    --output-dir pipeline/06_thesis_inputs/figures/
```

Generuje (per dataset gdzie sensowne):
- `pareto_cost_quality_<dataset>.pdf` — F1 vs ms/sample, Pareto front zaznaczony.
- `calibration_<best>_<dataset>.pdf` — reliability + ECE pre/post.
- `leakage_sensitivity_scatter.pdf` — F1_drop (random→grouped) vs log10(trainable_params).
- `seed_variance_<dataset>.pdf` — box plot per model 5-seedów.
- `cross_dataset_heatmap.pdf` — transfer matrix (z F2).
- `cm_<best>_<dataset>.pdf` — confusion matrices (najlepszy model).

### Step F5: Build thesis inputs

```bash
python pipeline/05_analysis/build_thesis_inputs.py
# Wynik:
#  - tabele LaTeX (.tex) gotowe do \input{} w pracy
#  - figury PDF skopiowane do fig_v2/
#  - bibtex entries dla nowych cytowań
```

## Co commit'ować

```bash
git add pipeline/06_thesis_inputs/
git add pipeline/05_analysis/REPORT_F.md
git commit -m "Phase F: scientific analysis + plots + thesis inputs"
git push origin main
```

## Następny krok

Faza G — thesis rewrite. Wszystkie liczby i figury są już gotowe; pozostaje wstawić je w LaTeX.
