# HANDOFF Faza C — Methodology hardening

## Cel

Wyposażyć repo eksperymentalne w narzędzia metodologiczne wymagane do rzetelnej publikacji:
- multi-seed runner (5 seedów) z deterministyką,
- bootstrap CI 95% dla F1/accuracy,
- McNemar test pomiędzy parami modeli (na tych samych predykcjach),
- calibration analysis (ECE + reliability diagram + temperature scaling),
- cost-quality reporting (latency, peak memory, trainable params),
- reproducibility manifest (wersje bibliotek, hash konfigów, hash danych).

Wszystkie te elementy są warunkiem konkursowym ("samodzielność badawcza, innowacyjność")
i muszą być w pełni opisane w Sekcji 2 pracy.

## Pliki dostarczane w tym HANDOFF

| Plik | Cel |
|---|---|
| `02_methodology/seeded_runner.py` | Wrapper który dla danego (model, dataset, variant) iteruje seedy i zapisuje rekordy do `experiments/results_v2/` |
| `02_methodology/metrics_with_ci.py` | Bootstrap 95% CI dla F1/accuracy/precision/recall (1000 resamples) |
| `02_methodology/mcnemar.py` | McNemar test dla par predykcji + matrix p-value |
| `02_methodology/calibration.py` | ECE + reliability diagram + temperature scaling |
| `02_methodology/cost_meter.py` | Context manager mierzący czas/peak memory/trainable params |
| `02_methodology/aggregator_v2.py` | Konsoliduje wszystkie `results_v2/*.json` w `all_results_v2.csv` i `summary_v2.md` |
| `02_methodology/reproducibility.py` | Generator `REPRODUCIBILITY.md` (pip freeze, git hash, data hash, hardware) |

## Kolejność wykonania (lokalny CC)

Te skrypty są **library functions** wywoływane z runnerów per-modelowych w Fazie E.
W Fazie C lokalny CC tylko **testuje że importy działają i smoke-testy przechodzą**:

```bash
cd /Users/mbasarab/Workspace/PERSONAL/UMCS/magisterka
source .venv/bin/activate

# Smoke test każdego modułu
python -c "from pipeline.02_methodology.seeded_runner import SeededRunner; print('OK')"
python -c "from pipeline.02_methodology.metrics_with_ci import bootstrap_ci; print('OK')"
python -c "from pipeline.02_methodology.mcnemar import mcnemar_matrix; print('OK')"
python -c "from pipeline.02_methodology.calibration import ece, temperature_scale; print('OK')"
python -c "from pipeline.02_methodology.cost_meter import CostMeter; print('OK')"
python -c "from pipeline.02_methodology.aggregator_v2 import aggregate; print('OK')"
```

(Jeśli `pipeline.02_methodology` jako moduł nie zadziała ze względu na cyfrę na początku
nazwy katalogu — utwórz symlink lub przemianuj na `methodology` przy imports, albo
dodaj `sys.path.insert(0, "pipeline/02_methodology")`.)

```bash
# Unit testy
pytest pipeline/02_methodology/tests/ -v
```

## Sanity checks

- [ ] `bootstrap_ci([0.81, 0.82, 0.83, 0.80, 0.84])` zwraca CI o szerokości <0.05.
- [ ] `mcnemar_matrix` dla dwóch identycznych prediction lists zwraca p=1.0.
- [ ] `ece` dla calibrated predictions (uniformly distributed) jest <0.05.
- [ ] `CostMeter` mierzy >0 sekund i >0 MB peak.
- [ ] `aggregator_v2.aggregate` dla pustego katalogu zwraca pusty DataFrame bez błędu.

## Co commit'ować

```bash
git add pipeline/02_methodology/REPORT_C.md
git commit -m "Phase C: methodology hardening utilities"
git push origin main
```

Skrypty `pipeline/02_methodology/*.py` już są w repo (od cloud CC).

## Następny krok

Po zakończeniu Fazy B (wykonaniu) — przejście do Fazy D (`pipeline/03_models/HANDOFF_D.md`).
