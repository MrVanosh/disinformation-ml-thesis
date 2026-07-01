# HANDOFF Faza D — Model lineup design

## Cel

Tight, uzasadniony lineup modeli z konkretnymi konfiguracjami YAML do każdego runa.
Każdy model w pracy ma odpowiedź na pytanie "dlaczego ten?".

## Final lineup (zatwierdzony w Fazie A)

### Klasyczne (TF-IDF, sklearn)

| Model | Config | Uzasadnienie |
|---|---|---|
| Logistic Regression | C=1.0, L2, max_iter=1000 | Najprostszy baseline z interpretowalnymi wagami |
| Linear SVM | C=1.0, L2 | Historyczny SOTA dla klasyfikacji TF-IDF |
| Random Forest | 200 trees, max_depth=None | Nieliniowy baseline, ważność cech (Gini) |
| XGBoost (TS metadata) | 300 trees, max_depth=6, lr=0.1 | Komponent ensemble — tylko na TruthSeeker |

### Encoder fine-tuning

| Model | Param | Dla zbiorów |
|---|---|---|
| DistilBERT-base-uncased | 66M | LIAR, TS |
| BERT-base-uncased | 110M | LIAR, TS |
| mBERT-base-multilingual-cased | 178M | EU (wielojęzyczne) |
| HerBERT-base-cased | 124M | PL corpus |

### LLM (mlx-lm lokalnie, transformers+QLoRA na Modal)

| Model | Warianty | Konfiguracje |
|---|---|---|
| Llama 3.1 8B Instruct (4-bit) | ZS, basic LoRA, big LoRA | configs/llama31_*.yaml |
| Qwen 2.5 7B Instruct (4-bit) | ZS, basic LoRA, big LoRA | configs/qwen25_*.yaml |

### Dodatkowe ablacje (na LIAR, dla H4 — kalibracja klas LoRA)

- LoRA-natural (rozkład 73/27 zamiast 50/50) — Llama, Qwen.

## Pliki dostarczane w tym HANDOFF

| Plik | Cel |
|---|---|
| `03_models/configs/classical_lr.yaml` | Logistic Regression config |
| `03_models/configs/classical_svm.yaml` | Linear SVM config |
| `03_models/configs/classical_rf.yaml` | Random Forest config |
| `03_models/configs/classical_xgb_metadata.yaml` | XGBoost na metadanych TS |
| `03_models/configs/encoder_distilbert.yaml` | DistilBERT fine-tune |
| `03_models/configs/encoder_bert_base.yaml` | BERT-base fine-tune |
| `03_models/configs/encoder_mbert.yaml` | mBERT-multilingual fine-tune |
| `03_models/configs/encoder_herbert.yaml` | HerBERT-base fine-tune |
| `03_models/configs/llm_zs.yaml` | Wspólny config zero-shot (prompt warianty) |
| `03_models/configs/lora_basic.yaml` | LoRA basic (8 warstw, 400 iter, r=16) |
| `03_models/configs/lora_big.yaml` | LoRA big (32 warstwy, 1500 iter, r=32) |
| `03_models/configs/lora_natural.yaml` | LoRA z natural distribution (LIAR 73/27) |
| `03_models/prompts.py` | Templates promptów dla LLM ZS |

## Kolejność wykonania (lokalny CC)

Faza D jest **tylko deklaratywna** — wystarczy zwalidować że YAML się parsuje:

```bash
python -c "
import yaml
from pathlib import Path
for p in Path('pipeline/03_models/configs/').glob('*.yaml'):
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    print(p.name, '→', list(cfg.keys()))
"
```

```bash
# Smoke test prompt builder
python -c "from pipeline.03_models.prompts import build_prompt; print(build_prompt('truthseeker', 'short', 'Ukraine NATO membership'))"
```

## Sanity checks

- [ ] Każdy YAML w `configs/` ma pola: `model_name`, `task`, `hyperparameters`, `seeds`, `metadata`.
- [ ] `prompts.py` ma funkcje dla 3 datasetów × 2 warianty (short binary + CoT).
- [ ] Konfigi LoRA mają `target_modules` i `r` zgodne z planem.

## Co commit'ować

```bash
git add pipeline/03_models/REPORT_D.md
git commit -m "Phase D: model lineup configs"
git push origin main
```

Konfigi YAML i `prompts.py` już są w repo.
