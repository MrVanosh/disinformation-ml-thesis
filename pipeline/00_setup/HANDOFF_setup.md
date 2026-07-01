# HANDOFF — Setup środowiska

## Cel

Przygotowanie środowiska Python + Modal + tokenów + struktury katalogów, tak żeby kolejne fazy (B-F) miały deterministyczne miejsce do pracy.

## Pre-requisites (po stronie autora, NIE Claude Code)

- macOS Apple Silicon (M4 Pro) z 24 GB unified RAM, ≥100 GB wolnego dysku.
- Konto Modal (https://modal.com), token w środowisku.
- Konto DiffBot z aktywnym tokenem API (https://app.diffbot.com).
- Konto Hugging Face z access tokenem (dla pobierania Llama 3.1 8B — gated).
- `homebrew` z `git`, `pyenv` (opcjonalnie), `make`.

## Kroki dla lokalnego Claude Code

### Step 1: Verify environment

```bash
cd /Users/mbasarab/Workspace/PERSONAL/UMCS/magisterka
git pull origin main
python3 --version  # oczekiwane: 3.11.x lub 3.12.x
pip --version
```

### Step 2: Setup venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Step 3: Install dependencies

```bash
pip install -r pipeline/00_setup/requirements.txt
```

Jeśli `mlx-lm` nie zainstaluje się (Linux box) — pominąć, działa tylko na Apple Silicon. Na Modal big LoRA użyje `transformers + bitsandbytes` zamiast mlx.

### Step 4: Setup .env

```bash
cp pipeline/00_setup/.env.template .env
# Otwórz .env i wpisz prawdziwe tokeny:
# DIFFBOT_TOKEN=<twój_token>
# HF_TOKEN=hf_xxx
# MODAL_TOKEN_ID=ak-xxx
# MODAL_TOKEN_SECRET=as-xxx
```

### Step 5: Login do Modal

```bash
modal token new  # otwiera przeglądarkę, autoryzacja
modal volume list  # weryfikacja czy działa
```

### Step 6: Utworzenie shared volumes na Modal

```bash
modal volume create disinfo-data
modal volume create disinfo-models
modal volume create disinfo-results
```

### Step 7: Verify HuggingFace access do Llama 3.1

```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.model_info('meta-llama/Llama-3.1-8B-Instruct').gated)"
# Oczekiwane: False (czyli twój token ma już accepted license)
```

Jeśli `Gated repo` error — wejdź na https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, zaakceptuj license i ponów.

### Step 8: Snapshot manifestu obecnego stanu eksperymentów

```bash
mkdir -p pipeline/00_setup/snapshots
ls -la datasets/ > pipeline/00_setup/snapshots/datasets_initial.txt
ls -la experiments/ > pipeline/00_setup/snapshots/experiments_initial.txt
find experiments/ -name "*.json" -path "*/results*/*" | head -50 > pipeline/00_setup/snapshots/results_initial.txt
du -sh datasets/* > pipeline/00_setup/snapshots/dataset_sizes.txt
```

## Sanity checks (przed przejściem do Fazy B)

- [ ] `python -c "import torch; print(torch.backends.mps.is_available())"` → `True`
- [ ] `python -c "import mlx_lm; print('OK')"` → `OK` (na Apple Silicon)
- [ ] `python -c "import modal; print(modal.__version__)"` → `0.6x` lub nowsze
- [ ] `python -c "import os; print(bool(os.getenv('DIFFBOT_TOKEN')))"` (po `source .env` lub `dotenv`) → `True`
- [ ] `modal volume list` zawiera `disinfo-data`, `disinfo-models`, `disinfo-results`
- [ ] `git status` czysty (poza .env który jest gitignored)

## Co commit'ować

```bash
git add pipeline/00_setup/snapshots/
git commit -m "setup: snapshot initial state of datasets/experiments"
git push origin main
```

Po commit'cie cloud CC zobaczy snapshoty i może lepiej zaplanować Fazę B (wie dokładnie co już jest, czego nie ma).

## Następny krok

Otwórz `pipeline/01_data/HANDOFF_B.md`.
