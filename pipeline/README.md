# Pipeline: handoff dla lokalnego Claude Code

Ten katalog zawiera **kompletny pakiet skryptów, konfigów i instrukcji** do wykonania pełnej iteracji eksperymentów + zwrócenia artefaktów wejściowych dla rozdziałów pracy (tabele, figury, JSON-y).

**Model pracy:**

```
┌────────────────────────┐                     ┌─────────────────────┐
│  Cloud Claude Code     │                     │ Lokalny Claude Code │
│  (planuje, pisze       │  ── git push/pull ──│ (wykonuje na M4 Pro │
│   skrypty, LaTeX)      │                     │  + Modal H100)      │
└────────────────────────┘                     └─────────────────────┘
         │                                              │
         │   produkuje pipeline/*.py, *.yaml, *.md      │
         │   produkuje src/*.tex                        │
         ├──────────────────────────────────────────────┤
                                                        │
                                                        │ konsumuje:
                                                        │ - pipeline/HANDOFF_<faza>.md
                                                        │ - pipeline/<faza>/*.py
                                                        │ - .env z DIFFBOT_TOKEN, HF_TOKEN
                                                        │
                                                        │ produkuje:
                                                        │ - datasets/...jsonl   (gitignored)
                                                        │ - experiments/results_v2/...json
                                                        │ - pipeline/06_thesis_inputs/*.csv, *.png
                                                        │ - audit reports do scratchpad
                                                        │
                                                        │ commit ← tylko pipeline/06_thesis_inputs/*
                                                        │ push  ← cloud CC widzi wyniki
```

## Status faz (live dashboard)

| Faza | Plik wykonawczy | Status | Wynik dla pracy |
|---|---|---|---|
| A | (decyzje, brak kodu) | ✅ akceptowana | teza/hipotezy/threat model (`scratchpad/faza-a/teza_hipotezy.md`) |
| B | `pipeline/01_data/HANDOFF_B.md` | 📦 kod gotowy, czeka wykonanie | manifest danych, audyt leakage, scrapy DiffBot + PL |
| C | `pipeline/02_methodology/HANDOFF_C.md` | 📦 kod gotowy, czeka wykonanie | runner multi-seed, agregator z CI, McNemar, calibration |
| D | `pipeline/03_models/HANDOFF_D.md` | 📦 kod gotowy, czeka wykonanie | YAML configs, lineup decisions |
| E | `pipeline/04_execution/HANDOFF_E.md` | 🔧 generowane | matrix runów ~235 |
| F | `pipeline/05_analysis/HANDOFF_F.md` | ⏳ czeka | taxonomy + plots + ablations |
| G | `src/*.tex` rewrite | ⏳ czeka | finalna praca |

(Aktualizowane przy każdym commit'cie cloud CC.)

## Konwencja interakcji cloud ↔ lokalny CC

1. **Cloud CC** pisze `pipeline/<faza>/HANDOFF_<faza>.md` zawierający:
   - Cel fazy (1 akapit).
   - Lista plików kodu w `pipeline/<faza>/` z opisem każdego.
   - Krok-po-kroku instrukcja dla lokalnego CC (`step 1: ...`, `step 2: ...`).
   - Oczekiwane outputy (jakie pliki, gdzie, w jakim formacie).
   - Sanity checks (`assert X`, `head -5 file.jsonl`).
   - Co commit'ować z powrotem (tylko `pipeline/06_thesis_inputs/*` + opcjonalnie meta-pliki).

2. **Lokalny CC** odczytuje HANDOFF, wykonuje kroki, raportuje wynik w `pipeline/<faza>/REPORT_<faza>.md` (jakie liczby, jakie problemy, ile czasu zajęło), commit'uje pliki wymienione w sekcji "co commit'ować".

3. **Cloud CC** po pull-u czyta REPORT, dostosowuje plany dalszych faz, pisze następny HANDOFF.

## Zasady bezpieczeństwa

- **Sekretów (DIFFBOT_TOKEN, HF_TOKEN, MODAL_TOKEN) NIE commit'uj.** `.env` jest gitignored. Templates w `pipeline/00_setup/.env.template`.
- **Wyników surowych (datasets, modele) NIE commit'uj.** Mieszczą się w `datasets/` i `experiments/` które są gitignored.
- **Commit'uj tylko**: `pipeline/06_thesis_inputs/*` (tabele CSV + figury PNG dla pracy), `pipeline/<faza>/REPORT_<faza>.md` (raporty wykonania).
- **Modal cost monitoring**: po każdej fazie z compute lokalny CC raportuje wydane $$ w REPORT. Twardy limit per faza: $50 bez consultacji z autorem.

## Środowisko techniczne (założenia)

- Lokalna maszyna: Apple M4 Pro, 24 GB unified RAM, macOS.
- Python ≥ 3.11; venv lub conda env.
- Modal CLI zalogowany (`modal token new`).
- DiffBot API token w `.env` jako `DIFFBOT_TOKEN`.
- Hugging Face token w `.env` jako `HF_TOKEN` (do pobierania modeli LLM gated).
- Pełna instalacja zależności: `pip install -r pipeline/00_setup/requirements.txt`.

## Pierwsza akcja dla lokalnego CC

Otwórz `pipeline/00_setup/HANDOFF_setup.md` i wykonaj kroki 1-N. Następnie przejdź do `pipeline/01_data/HANDOFF_B.md`.
