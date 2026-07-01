# Audyt wycieku danych — truthseeker (random split, seed=42)

- **N total**: 134,203
- **N train**: 114,072
- **N test**: 20,131

## Wyniki audytu (5 poziomów)

| Poziom | Status | Metryka | Wartość |
|---|---|---|---|
| 1. Group overlap | WYCIEK | % grup testowych obecnych też w treningu | 99.23% (902/909) |
| 2. Temporal leakage | N/A | brak daty publikacji | - |
| 3. Source domain leakage | N/A | brak URL | - |
| 4. Near-duplicate leakage | ostrzeżenie | % próbek testowych z MinHash sim ≥ 0.85 do dowolnej w train (sample 5000) | 2.42% (~2.4% ekstrap.) |
| 5. Label consistency in group | problem | % grup z więcej niż 1 unikalną etykietą | 63.50% (675/1063) |

## Rekomendacje

- Zastosuj `GroupShuffleSplit` po kolumnie `group_key` — obecny split zawiera 902 grup nakładających się.
- 675 grup ma mieszane etykiety — sprawdź czy stosujemy 'majority label per group' lub czy faktycznie istnieją sprzeczności w danych źródłowych.
