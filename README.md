# Disinformation detection — thesis code & artifacts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21112976.svg)](https://doi.org/10.5281/zenodo.21112976)

Companion code and experimental artifacts for the M.Sc. thesis *"Wykorzystanie uczenia
maszynowego do klasyfikacji treści dezinformacyjnych"* (UMCS Lublin). This repository
contains **only code, configurations and result artifacts — it does not include the thesis
text.**

Archived release (permanent DOI): **[10.5281/zenodo.21112976](https://doi.org/10.5281/zenodo.21112976)**

## What's here

- `pipeline/01_data/` — dataset loaders, five-level data-leakage audit, split generation
  (grouped / random / domain-grouped).
- `pipeline/02_methodology/` — aggregation, McNemar significance testing, calibration.
- `pipeline/03_models/configs/` — model configs (classical, encoders, LoRA basic/big/natural, ZS).
- `pipeline/04_execution/runners/` — training/eval runners (classical, encoder, LLM ZS, LoRA, ensemble) + Modal cloud app.
- `pipeline/05_analysis/` — plots, error extraction, result tables.
- `pipeline/06_thesis_inputs/` — aggregated results (`all_results_v2.csv`, `summary_v2.md`),
  McNemar significance matrices, figures.
- `experiments/results_*` — per-run metrics (grouped, random, domain-grouped splits).
- `experiments/splits_*` — the exact train/test id-lists used (reproducibility).

## Methodology highlights

- **Data-leakage audit + grouped splits** (by shared source statement / debunk_id). On
  TruthSeeker, correcting leakage drops macro F1 by ~18–22 pts (SVM 0.94 → 0.72).
- **Domain-leakage test** on EUvsDisinfo: grouping by publisher drops F1 by ~14 pts
  (0.95 → 0.80) — much of the "high" EU score is publisher fingerprint.
- Multi-seed runs, bootstrap / t-Student confidence intervals, McNemar tests, calibration
  (temperature scaling), full reproducibility manifest.

## Data availability

- **EUvsDisinfo** raw article text is **not** redistributed (licensing); the repo includes
  the scrape/reconstruction scripts and the label/URL lists (via the dataset's public repo).
- **Polish corpus** (Demagog.org.pl fact-check claims): released as claim identifiers / URLs
  and labels, respecting source terms.
- Large prediction dumps (`experiments/preds_*`) are omitted; they are regenerable from the
  runners + splits.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # see thesis REPRODUCIBILITY manifest for pinned versions
python pipeline/01_data/grouped_split.py --dataset truthseeker
python pipeline/04_execution/run_local_all.py   # or individual runners
python pipeline/02_methodology/aggregator_v2.py
```

## License

Code released under the MIT License. Dataset artifacts follow their respective source licenses.
