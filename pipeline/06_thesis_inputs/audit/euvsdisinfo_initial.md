# Audyt wycieku danych — euvsdisinfo (random split, seed=42)

- **N total**: 12,123
- **N train**: 10,304
- **N test**: 1,819

## Wyniki audytu (5 poziomów)

| Poziom | Status | Metryka | Wartość |
|---|---|---|---|
| 1. Group overlap | WYCIEK | % grup testowych obecnych też w treningu | 58.98% (926/1570) |
| 2. Temporal leakage | uwaga | % próbek testowych z datą < max(train) | 100.0% (train_max=2023-12-07, test_min=2015-02-12) |
| 3. Source domain leakage | wysoki | % próbek testowych z domeny obecnej w trainie | 88.3% (wspólnych domen: 401/605) |
| 4. Near-duplicate leakage | WYCIEK | % próbek testowych z MinHash sim ≥ 0.85 do dowolnej w train (sample 5000) | 6.87% (~6.9% ekstrap.) |
| 5. Label consistency in group | problem | % grup z więcej niż 1 unikalną etykietą | 16.86% (1121/6650) |

## Rekomendacje

- Zastosuj `GroupShuffleSplit` po kolumnie `group_key` — obecny split zawiera 926 grup nakładających się.
- Większość testowych próbek ma datę wcześniejszą niż max(train) — rozważ `TimeSeriesSplit` lub jawne odcięcie po dacie.
- ≥5% próbek testowych ma near-duplicate w trainie (Jaccard ≥ 0.85). Wykonaj dedup MinHash przed splittingiem.
- 1121 grup ma mieszane etykiety — sprawdź czy stosujemy 'majority label per group' lub czy faktycznie istnieją sprzeczności w danych źródłowych.
