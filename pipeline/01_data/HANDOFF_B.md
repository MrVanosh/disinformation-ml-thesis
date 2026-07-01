# HANDOFF Faza B — Data audit + DiffBot fill + PL corpus

## Cel fazy

Zbudować **rygorystycznie audytowaną** wersję każdego z czterech zbiorów danych (LIAR, TruthSeeker, EUvsDisinfo full, polski korpus PL), z:
- pełnym multi-level audytem leakage (group, temporal, source, near-duplicate, label-consistency),
- uzupełnieniem EUvsDisinfo o ~5700 dodatkowych URLi via DiffBot (12 480 → ~17 000),
- polskim korpusem ≥500 etykietowanych dokumentów z Demagog/OKO/CEDMO + polskiego podzbioru EUvsDisinfo,
- manifestem `datasets/MANIFEST.md` z SHA-256 każdego pliku.

**Wszystkie te kroki należą do Sekcji 2 pracy ("Metodologia"), nie do Sekcji 3 ("Wyniki")** — to są działania, które zostały podjęte by zapewnić rzetelność, nie wnioski naukowe.

## Pliki dostarczane w tym HANDOFF

| Plik | Cel |
|---|---|
| `01_data/leakage_audit.py` | Multi-level audit (5 poziomów leakage), wynik → markdown report |
| `01_data/diffbot_scrape.py` | Klient DiffBot, batch scraping URLi z `errors.jsonl` |
| `01_data/pl_demagog_scrape.py` | RSS-based scrape Demagog.org.pl |
| `01_data/pl_okopress_scrape.py` | RSS scrape kategorii fact-check OKO.press |
| `01_data/pl_cedmo_export.py` | Pobranie eksportu CEDMO (jeśli dostępny via Google Fact Check API) |
| `01_data/pl_eu_subset_extract.py` | Filtr PL artykułów z EUvsDisinfo (po trafilatura + DiffBot) |
| `01_data/build_pl_corpus.py` | Merge + deduplikacja + etykietowanie polskiego korpusu |
| `01_data/manifest.py` | Generator `datasets/MANIFEST.md` z SHA-256 i metadanymi |
| `01_data/grouped_split.py` | Refaktor obecnego GroupShuffleSplit do reproducible utility |

## Kolejność wykonania (lokalny CC)

### Step B1: Initial audit obecnych zbiorów (przed jakąkolwiek modyfikacją)

```bash
cd /Users/mbasarab/Workspace/PERSONAL/UMCS/magisterka
source .venv/bin/activate

python pipeline/01_data/leakage_audit.py \
    --dataset liar \
    --output pipeline/06_thesis_inputs/audit/liar_initial.md

python pipeline/01_data/leakage_audit.py \
    --dataset truthseeker \
    --output pipeline/06_thesis_inputs/audit/truthseeker_initial.md

python pipeline/01_data/leakage_audit.py \
    --dataset euvsdisinfo \
    --output pipeline/06_thesis_inputs/audit/euvsdisinfo_initial.md
```

**Oczekiwane wyjście** (per dataset, w markdown):
- Tabela poziomów leakage (statement, debunk, temporal, source, near-duplicate, label-consistency).
- Heatmap overlap train↔test (zapisany jako PNG obok markdown).
- Rekomendacja protokołu splitu.

Jeśli istnieje sprzeczność z bieżącymi splitami w `experiments/results_grouped/` — w raporcie wprost o tym napisać, na końcu kroku poinformować autora (Q1 do uzgodnienia: czy używamy istniejących grouped splits czy regenerujemy).

### Step B2: DiffBot scrape EUvsDisinfo errors

```bash
python pipeline/01_data/diffbot_scrape.py \
    --input datasets/euvsdisinfo/errors.jsonl \
    --output datasets/euvsdisinfo/scraped_diffbot.jsonl \
    --batch-size 50 \
    --max-calls 6000 \
    --rate-limit-per-min 60
```

Twardy limit: 6000 calls (bufor poniżej 10k trial). Skrypt loguje koszt aktualny po każdej batch'y.

**Sanity check po zakończeniu:**
```bash
wc -l datasets/euvsdisinfo/scraped_diffbot.jsonl
# Oczekiwane: 4500-5700 (część URLi się nie uda nawet z DiffBot — 404, robots)

python -c "
import json
with open('datasets/euvsdisinfo/scraped_diffbot.jsonl') as f:
    rows = [json.loads(l) for l in f]
print('Total:', len(rows))
print('PL:', sum(1 for r in rows if r.get('lang')=='pl'))
print('RU:', sum(1 for r in rows if r.get('lang')=='ru'))
print('UA:', sum(1 for r in rows if r.get('lang')=='uk'))
"
```

### Step B3: PL corpus — Demagog + OKO + CEDMO

Demagog (RSS):
```bash
python pipeline/01_data/pl_demagog_scrape.py \
    --output datasets/pl_extra/demagog.jsonl \
    --since-date 2022-01-01
```

OKO.press fact-checks (RSS):
```bash
python pipeline/01_data/pl_okopress_scrape.py \
    --output datasets/pl_extra/okopress.jsonl \
    --since-date 2022-01-01
```

CEDMO (Google Fact Check API — wymaga GOOGLE_FACTCHECK_API_KEY w .env, opcjonalne):
```bash
python pipeline/01_data/pl_cedmo_export.py \
    --output datasets/pl_extra/cedmo.jsonl \
    --language pl
```

**Oczekiwany efekt łączny:** ≥400 artykułów z polskiego ekosystemu fact-check (sami autorzy Demagog publikują ~2-4 fact-checki dziennie od lat).

### Step B4: PL subset z EUvsDisinfo

```bash
python pipeline/01_data/pl_eu_subset_extract.py \
    --input datasets/euvsdisinfo/scraped.jsonl,datasets/euvsdisinfo/scraped_diffbot.jsonl \
    --output datasets/pl_extra/eu_pl_subset.jsonl
```

### Step B5: Build PL corpus (merge + dedup + label normalization)

```bash
python pipeline/01_data/build_pl_corpus.py \
    --sources datasets/pl_extra/demagog.jsonl,datasets/pl_extra/okopress.jsonl,datasets/pl_extra/cedmo.jsonl,datasets/pl_extra/eu_pl_subset.jsonl \
    --output datasets/pl_extra/corpus_pl.jsonl \
    --min-length 50 \
    --dedup-threshold 0.90 \
    --report pipeline/06_thesis_inputs/pl_corpus_stats.md
```

**Sanity check:**
```bash
python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('datasets/pl_extra/corpus_pl.jsonl')]
print('Total:', len(rows))
print('Labels:', Counter(r['label'] for r in rows))
print('Sources:', Counter(r['source'] for r in rows))
print('Avg len:', sum(len(r['text']) for r in rows)/len(rows))
"
```

**Decyzja awaryjna**: jeśli total < 500 (cel must-have), wykonać `build_pl_corpus.py --augment-translation` (mBART translation z RU/EN→PL na 200 wybranych przykładów), z jawnym oznaczeniem `synthetic_pl=true` w manifeście.

### Step B6: Re-audit po dopełnieniu

Powtórz Step B1 na zaktualizowanych datasetach. Te raporty są **finalnymi** raportami audytu i wejdą do Sekcji 2 pracy.

```bash
for ds in liar truthseeker euvsdisinfo pl_corpus; do
    python pipeline/01_data/leakage_audit.py \
        --dataset $ds \
        --output pipeline/06_thesis_inputs/audit/${ds}_final.md
done
```

### Step B7: Manifest danych

```bash
python pipeline/01_data/manifest.py \
    --root datasets/ \
    --output datasets/MANIFEST.md
```

`MANIFEST.md` zawiera dla każdego pliku: SHA-256, rozmiar, data ostatniej modyfikacji, źródło (URL/cytat), licencja.

### Step B8: Grouped split utility

Refaktoryzacja istniejącego split kodu z `experiments/utils/data.py` (jeśli istnieje) do `pipeline/01_data/grouped_split.py` — czysta, zreproduktowana, z testami.

```bash
pytest pipeline/01_data/test_grouped_split.py -v
```

## Outputy do commit'a

```bash
git add pipeline/06_thesis_inputs/audit/
git add pipeline/06_thesis_inputs/pl_corpus_stats.md
git add datasets/MANIFEST.md   # wymaga zmiany .gitignore — patrz uwaga
git add pipeline/01_data/REPORT_B.md
git commit -m "Phase B: data audit + DiffBot fill + PL corpus"
git push origin main
```

**Uwaga gitignore**: `datasets/` jest gitignored, ale `datasets/MANIFEST.md` chcemy versjonować. Dodaj wyjątek do `.gitignore`:
```
datasets/
!datasets/MANIFEST.md
```

## Raport końcowy

W `pipeline/01_data/REPORT_B.md` zapisać:
- Liczby per dataset przed/po (LIAR niezmieniony, TS niezmieniony, EU 12480 → X, PL 0 → Y).
- Wynik audytu leakage (rozmiar wycieku per poziom).
- Koszt DiffBot (calls + USD).
- Wykryte problemy (np. "license CEDMO niezgodna z research use, pominęto").
- Czas wykonania per krok.
- Pytania do autora przed Fazą C (np. "PL corpus jest 320, czy używamy translation augmentation?").

## Pytania potencjalne do cloud CC (zapisz w REPORT)

- Q1: Czy używamy istniejących grouped splits z `experiments/results_grouped/` czy regenerujemy z `grouped_split.py`?
- Q2: Jeśli scraping CEDMO/OKO/Demagog nie da 500 PL — czy włączamy translation augmentation, czy akceptujemy mniejszy korpus?
- Q3: Czy DiffBot dla wszystkich `errors.jsonl` (6000+ calls) czy tylko PL/RU/UA/DE (oszczędność ~$$$ trial)?
