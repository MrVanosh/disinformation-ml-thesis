# Audyt wycieku danych — liar (random split, seed=42)

- **N total**: 12,836
- **N train**: 10,910
- **N test**: 1,926

## Wyniki audytu (5 poziomów)

| Poziom | Status | Metryka | Wartość |
|---|---|---|---|
| 1. Group overlap | OK | % grup testowych obecnych też w treningu | 0.00% (0/1926) |
| 2. Temporal leakage | N/A | brak daty publikacji | - |
| 3. Source domain leakage | N/A | brak URL | - |
| 4. Near-duplicate leakage | OK | % próbek testowych z MinHash sim ≥ 0.85 do dowolnej w train (sample 5000) | 0.21% (~0.2% ekstrap.) |
| 5. Label consistency in group | OK | % grup z więcej niż 1 unikalną etykietą | 0.00% (0/12836) |
