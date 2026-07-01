# Audyt wycieku danych — euvsdisinfo (random split, seed=42)

- **N total**: 14,350
- **N train**: 12,197
- **N test**: 2,153

## Wyniki audytu (5 poziomów)

| Poziom | Status | Metryka | Wartość |
|---|---|---|---|
| 1. Group overlap | WYCIEK | % grup testowych obecnych też w treningu | 61.80% (1134/1835) |
| 2. Temporal leakage | uwaga | % próbek testowych z datą < max(train) | 100.0% (train_max=2023-07-29, test_min=2015-11-19) |
| 3. Source domain leakage | wysoki | % próbek testowych z domeny obecnej w trainie | 88.3% (wspólnych domen: 479/718) |
| 4. Near-duplicate leakage | WYCIEK | % próbek testowych z MinHash sim ≥ 0.85 do dowolnej w train (sample 5000) | 5.48% (~5.5% ekstrap.) |
| 5. Label consistency in group | problem | % grup z więcej niż 1 unikalną etykietą | 19.31% (1466/7590) |

## Rekomendacje

- Zastosuj `GroupShuffleSplit` po kolumnie `group_key` — obecny split zawiera 1134 grup nakładających się.
- Większość testowych próbek ma datę wcześniejszą niż max(train) — rozważ `TimeSeriesSplit` lub jawne odcięcie po dacie.
- ≥5% próbek testowych ma near-duplicate w trainie (Jaccard ≥ 0.85). Wykonaj dedup MinHash przed splittingiem.
- 1466 grup ma mieszane etykiety — sprawdź czy stosujemy 'majority label per group' lub czy faktycznie istnieją sprzeczności w danych źródłowych.
