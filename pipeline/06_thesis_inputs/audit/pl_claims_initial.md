# Audyt wycieku danych — pl_claims (random split, seed=42)

- **N total**: 3,011
- **N train**: 2,559
- **N test**: 452

## Wyniki audytu (5 poziomów)

| Poziom | Status | Metryka | Wartość |
|---|---|---|---|
| 1. Group overlap | OK | % grup testowych obecnych też w treningu | 0.00% (0/452) |
| 2. Temporal leakage | N/A | brak prawidłowych dat | - |
| 3. Source domain leakage | wysoki | % próbek testowych z domeny obecnej w trainie | 100.0% (wspólnych domen: 1/1) |
| 4. Near-duplicate leakage | OK | % próbek testowych z MinHash sim ≥ 0.85 do dowolnej w train (sample 5000) | 0.00% (~0.0% ekstrap.) |
| 5. Label consistency in group | OK | % grup z więcej niż 1 unikalną etykietą | 0.00% (0/3011) |

## Rekomendacje

- Niemal wszystkie domeny testowe są obecne w trainie — model może uczyć się 'rozpoznawania domeny' zamiast 'rozpoznawania dezinformacji'. Rozważ split po domenach (DomainAwareSplit).
