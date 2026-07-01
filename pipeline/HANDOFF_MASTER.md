# HANDOFF MASTER — instrukcja wykonawcza dla lokalnego Claude Code

To jest **plik startowy dla lokalnego CC** uruchamianego u~autora w~katalogu
`/Users/mbasarab/Workspace/PERSONAL/UMCS/magisterka`. Cloud CC przygotował
kompletny pakiet kodu, configów, instrukcji i~tekstów LaTeX --- lokalny CC
wykonuje fazy B--E (compute), F (analizy), G.compile (LaTeX build), H (pakiet konkursowy).

## Status pakietu (zatwierdzony przez autora)

| Faza | Status | Co lokalny CC ma zrobić |
|---|---|---|
| A. Strategic alignment | ✅ Zaakceptowana | Nic --- decyzje w `scratchpad/faza-a/teza_hipotezy.md` |
| B. Data audit + DiffBot + PL scraping | 📦 Kod gotowy, czeka wykonanie | Wykonać `pipeline/01_data/HANDOFF_B.md` |
| C. Methodology utilities | 📦 Kod gotowy | Smoke testy z `pipeline/02_methodology/HANDOFF_C.md` |
| D. Model configs | 📦 YAML configi gotowe | Sanity check |
| E. Execution matrix | 📦 Runnery + Modal app gotowe | Wykonać `pipeline/04_execution/HANDOFF_E.md` |
| F. Scientific analysis | 📦 Skrypty + szablony gotowe | Wykonać po~E: `pipeline/05_analysis/HANDOFF_F.md` |
| G. Thesis text | ✅ Nowe wersje `src/*.tex` + `refs.bib` gotowe | Build PDF + uzupełnić tabele/figury z~`pipeline/06_thesis_inputs/` |
| Obrona-ready | ⏳ Po F | Final PDF do~promotora **deadline: 8.07.2026** |
| H. Konkurs Rejewski | ⏳ Po obronie | Szablony w~`pipeline/competition_templates/`; deadline 30.11.2026 |

## Krytyczne deadliny

- **8 lipca 2026**: PDF gotowy do~oddania promotorowi (bufor 1 tydzień na~recenzję).
- **15 lipca 2026**: obrona magisterska.
- **30 listopada 2026**: zgłoszenie konkursu Rejewskiego (Faza~H).

## Pierwsza akcja: pull cloud commitów

```bash
cd /Users/mbasarab/Workspace/PERSONAL/UMCS/magisterka
git pull origin main
# Powinien pojawić się katalog `pipeline/` (4160+ linii kodu)
# Oraz zmodyfikowane src/*.tex
```

## Krok 1: Setup środowiska (15 min)

```bash
# Patrz: pipeline/00_setup/HANDOFF_setup.md
python3 -m venv .venv
source .venv/bin/activate
pip install -r pipeline/00_setup/requirements.txt
cp pipeline/00_setup/.env.template .env
# → wypełnij DIFFBOT_TOKEN, HF_TOKEN, MODAL tokens
modal token new
```

## Krok 2: Faza B (8-12h pracy + 4-6h API/scraping wall-clock)

```bash
# Audyt + DiffBot fill + PL scraping (Demagog/OKO/CEDMO + EU PL subset)
# Patrz pełna instrukcja: pipeline/01_data/HANDOFF_B.md

# 1) initial audit obecnych zbiorów
for ds in liar truthseeker euvsdisinfo; do
    python pipeline/01_data/leakage_audit.py \
        --dataset $ds \
        --output pipeline/06_thesis_inputs/audit/${ds}_initial.md
done

# 2) DiffBot fill EU (4-6h, ~$30-60 API cost)
python pipeline/01_data/diffbot_scrape.py \
    --input datasets/euvsdisinfo/errors.jsonl \
    --output datasets/euvsdisinfo/scraped_diffbot.jsonl \
    --max-calls 6000 --rate-limit-per-min 60

# 3) PL corpus
python pipeline/01_data/pl_demagog_scrape.py \
    --output datasets/pl_extra/demagog.jsonl --since-date 2022-01-01
python pipeline/01_data/pl_okopress_scrape.py \
    --output datasets/pl_extra/okopress.jsonl --since-date 2022-01-01
python pipeline/01_data/pl_cedmo_export.py \
    --output datasets/pl_extra/cedmo.jsonl --language pl

# 4) EU PL subset + merge
python pipeline/01_data/pl_eu_subset_extract.py \
    --input datasets/euvsdisinfo/scraped.jsonl,datasets/euvsdisinfo/scraped_diffbot.jsonl \
    --output datasets/pl_extra/eu_pl_subset.jsonl

python pipeline/01_data/build_pl_corpus.py \
    --sources datasets/pl_extra/demagog.jsonl,datasets/pl_extra/okopress.jsonl,datasets/pl_extra/cedmo.jsonl,datasets/pl_extra/eu_pl_subset.jsonl \
    --output datasets/pl_extra/corpus_pl.jsonl \
    --report pipeline/06_thesis_inputs/pl_corpus_stats.md

# 5) re-audit po dopełnieniu (final raporty do pracy)
for ds in liar truthseeker euvsdisinfo pl_corpus; do
    python pipeline/01_data/leakage_audit.py \
        --dataset $ds --split-mode grouped \
        --output pipeline/06_thesis_inputs/audit/${ds}_final.md
done

# 6) manifest danych
python pipeline/01_data/manifest.py \
    --root datasets/ --output datasets/MANIFEST.md

# 7) splity grupowe — generuj dla 5 seedów
for ds in liar truthseeker euvsdisinfo pl_corpus; do
    python pipeline/01_data/grouped_split.py \
        --dataset $ds --seeds 13,42,71,89,113 \
        --output-dir experiments/splits_v2/
done
```

**Co commit:**
```bash
# update .gitignore aby MANIFEST.md był versjonowany
echo "!datasets/MANIFEST.md" >> .gitignore
git add datasets/MANIFEST.md
git add pipeline/06_thesis_inputs/audit/ pipeline/06_thesis_inputs/pl_corpus_stats.md
git commit -m "phase B: data audit + DiffBot fill + PL corpus"
git push
```

## Krok 3: Faza C smoke tests (5 min)

```bash
# Patrz pipeline/02_methodology/HANDOFF_C.md
python -c "
import sys; sys.path.insert(0, 'pipeline/02_methodology')
from metrics_with_ci import bootstrap_ci
from mcnemar import mcnemar_test
from calibration import ece, temperature_scale
from cost_meter import CostMeter
print('All methodology modules OK')
"
```

## Krok 4: Faza E (compute heavy, wall-clock 1-7 dni)

**Lokalnie M4 Pro (Tier 1+2)** — może zająć kilka nocy:

```bash
# Patrz pipeline/04_execution/HANDOFF_E.md

# generuj run matrix (lokalne tylko)
python pipeline/04_execution/configs_to_runs.py \
    --tiers 1,2 --output pipeline/04_execution/run_matrix_local.jsonl

# uruchom (resume-safe; sprawdza existing)
python pipeline/04_execution/run_local_all.py \
    --matrix pipeline/04_execution/run_matrix_local.jsonl \
    --skip-existing \
    --report pipeline/04_execution/REPORT_E_local.md
```

**Modal (Tier 3, ~$30-40)**:

```bash
# 1) wgraj dane na Modal volume
modal volume put disinfo-data experiments/splits_v2/
modal volume put disinfo-data datasets/

# 2) uruchom 70B ZS na TS+EU × 3 seedy
modal run pipeline/04_execution/modal_app.py::llm_70b_zs_all

# 3) ściągnij wyniki
modal volume get disinfo-results /tmp/disinfo-results-modal
cp -r /tmp/disinfo-results-modal/results_v2/* experiments/results_v2/
cp -r /tmp/disinfo-results-modal/preds_v2/* experiments/preds_v2/
```

## Krok 5: Faza F — analizy + figury (1-2 dni)

```bash
# Patrz pipeline/05_analysis/HANDOFF_F.md

# 1) aggregate wszystkie runy w jeden CSV + summary
python pipeline/02_methodology/aggregator_v2.py

# 2) generuj wszystkie figury
python pipeline/05_analysis/plots.py \
    --results pipeline/06_thesis_inputs/all_results_v2.csv \
    --output-dir pipeline/06_thesis_inputs/figures/

# 3) error taxonomy templates (do ręcznej kategoryzacji)
for ds in liar truthseeker euvsdisinfo pl_corpus; do
    python pipeline/05_analysis/error_taxonomy_template.py \
        --dataset $ds --n-fp 50 --n-fn 50 \
        --output pipeline/06_thesis_inputs/error_taxonomy_${ds}.csv
done
# Po ręcznym wypełnieniu kolumny 'category' w CSV — agreguj
# (skrypt analyze_taxonomy.py do zaprojektowania na podstawie wzorca z plots.py)

# 4) reprodukowalność
python pipeline/02_methodology/reproducibility.py
```

## Krok 6: Build PDF pracy (30 min)

```bash
# najpierw wstaw wartości z all_results_v2.csv do src/rozdzial_3_1.tex
# (tabele klasyczne, encoder, LLM — wartości są aktualnie pilotowe;
# zaktualizuj liczbami z `pipeline/06_thesis_inputs/summary_v2.md`)

# Build
pdflatex -interaction=nonstopmode thesis.tex
biber thesis
pdflatex -interaction=nonstopmode thesis.tex
pdflatex -interaction=nonstopmode thesis.tex

# Sprawdzenia jakości
grep -nE "TODO|FIXME|zostan[ąa] dodane|w kolejnej iteracji" src/*.tex || echo "No TODO ✓"
grep -nE "Undefined references" thesis.log
grep -nE "Reference .* undefined" thesis.log

# Manualne sprawdzenie PDF:
open thesis.pdf
# - spis treści zawiera 3 rozdziały + wstep + wnioski
# - tabele rozdz. 3 mają wartości (nie ---)
# - wszystkie figury są obecne
# - brak orphan'ów stron
```

## Krok 7: Wysyłka do promotora

Email do prof. Wójcika (~~lub aktualnego promotora UMCS~~) z załączonym `thesis.pdf`,
prośbą o~recenzję i~potwierdzenie terminu obrony 15.07.2026.

## Krok 8: Faza H (po obronie 15.07, do 30.11)

```bash
# Patrz pipeline/competition_templates/HANDOFF_H.md (do utworzenia)
# - przeczytaj regulaminy (Rejewski PDF + ABW DOCX)
# - wypełnij formularze
# - cover letter (cover_letter_pl.tex + cover_letter_en.tex)
# - oświadczenie autora + zgoda promotora
# - submit do konkurs.cyber@mon.gov.pl
```

## Tips dla lokalnego CC

- **Każdy duży krok (B, E, F)** — generuj REPORT_<faza>.md z czasem wykonania, błędami, nuance'ami.
- **Push tylko gdy potwierdzone z autorem** dla destruktywnych akcji.
- **Modal budget**: monitoruj koszty. Twardy limit: $150 (autor potwierdził budget $30 + $30 drugie konto = $60; zostaw bufor).
- **Failed runs**: nie kontynuować jeśli >3 kolejnych falie — pauza i diagnoza.
- **Komunikuj progress z autorem co 24h** (krótki update: co zrobione, co dalej, czy potrzeba decyzji).

## Pliki referencyjne (kontekst dla decyzji)

- `scratchpad/faza-a/teza_hipotezy.md` — zaakceptowana teza/hipotezy/threat model.
- `pipeline/<faza>/HANDOFF_<faza>.md` — instrukcja konkretnej fazy.
- `pipeline/03_models/configs/*.yaml` — konfigi modeli (źródło prawdy dla hyperparametrów).
- `~/.claude/plans/here-is-a-draft-polymorphic-journal.md` — pełen plan strategiczny.
