"""Scheduler dla lokalnych runów (Tier 1+2).

Czyta matrix JSONL, dla każdego joba sprawdza czy wynik już istnieje (skip-existing),
wywołuje odpowiedni runner przez subprocess, loguje progress + czas + cost.

Wyjście:
  - REPORT_E_local.md (timeline runów, błędy, sumaryczne CI)
  - experiments/results_v2/...json (per run, przez SeededRunner)

Reżim awaryjny: po N kolejnych failach automatycznie pauza i logowanie do REPORT.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ładuje .env (m.in. HF_HUB_CACHE → modele na dysku zewnętrznym T7).
# subprocess.run() bez env= dziedziczy os.environ, więc runnery dostają HF_HUB_CACHE.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("run_local_all")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


RUNNER_SCRIPT = {
    "classical": "pipeline/04_execution/runners/classical_runner.py",
    "encoder": "pipeline/04_execution/runners/encoder_runner.py",
    "llm_zs": "pipeline/04_execution/runners/llm_zs_runner.py",
    "llm_lora": "pipeline/04_execution/runners/llm_lora_runner.py",
    "ensemble": "pipeline/04_execution/runners/ensemble_runner.py",
}


def _build_command(job: dict) -> list[str]:
    runner = job["runner"]
    script = RUNNER_SCRIPT[runner]
    # sys.executable = ten sam interpreter (.venv) co orchestrator — unika "python not found"
    cmd = [sys.executable, script, "--seed", str(job["seed"]),
           "--split-file", job["split_path"]]

    if runner == "ensemble":
        return cmd  # ensemble nie wymaga config (hardcoded)

    cmd.extend(["--config", job["config_path"], "--dataset", job["dataset"]])
    # Kanoniczna nazwa wyniku = z joba (single source of truth dla skip-existing + agregatora)
    cmd.extend(["--run-model", job["model_short"], "--run-variant", job["variant"]])
    if runner in ("llm_zs", "llm_lora"):
        cmd.extend(["--model", job["model_short"]])
    # Cross-dataset transfer (encoder + llm_lora)
    if job.get("eval_dataset") and runner in ("encoder", "llm_lora"):
        cmd.extend(["--eval-dataset", job["eval_dataset"],
                    "--eval-split-file", job["eval_split_path"]])
    return cmd


def _result_path(job: dict) -> Path:
    """Ścieżka oczekiwanego pliku wyniku — do skip-existing check."""
    return Path("experiments/results_v2/") / \
        f"{job['dataset']}_{job['model_short']}_{job['variant']}_seed{job['seed']}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="JSONL z jobami")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-consecutive-fails", type=int, default=3)
    parser.add_argument("--filter-compute", default=None,
                        help="Tylko jobs o compute=<X> (np. local_mps)")
    parser.add_argument("--report", default="pipeline/04_execution/REPORT_E_local.md")
    args = parser.parse_args()

    jobs = []
    with Path(args.matrix).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    logger.info("Loaded %d jobs", len(jobs))

    if args.filter_compute:
        jobs = [j for j in jobs if j["compute"] == args.filter_compute]
        logger.info("After compute filter %s: %d jobs", args.filter_compute, len(jobs))

    # Filter local only (skip modal_h100 etc)
    jobs = [j for j in jobs if j["compute"].startswith("local")]
    logger.info("Local-only jobs: %d", len(jobs))

    todo = jobs
    if args.skip_existing:
        todo = [j for j in jobs if not _result_path(j).exists()]
        logger.info("After skip-existing: %d todo", len(todo))

    report_lines = [
        f"# REPORT Faza E (local) — start {datetime.utcnow().isoformat()}Z",
        "",
        f"Jobs total: {len(jobs)}, todo: {len(todo)}",
        "",
        "| Time | Dataset | Model | Variant | Seed | Status | Duration (s) |",
        "|---|---|---|---|---|---|---|",
    ]

    n_ok = n_fail = n_skip = 0
    consecutive_fails = 0
    t_start = time.time()

    for i, job in enumerate(todo):
        if _result_path(job).exists() and args.skip_existing:
            n_skip += 1
            continue

        cmd = _build_command(job)
        logger.info("[%d/%d] %s/%s/%s seed=%d (~%d min)",
                    i + 1, len(todo), job["dataset"], job["model_short"], job["variant"],
                    job["seed"], job["estimated_minutes"])
        t0 = time.time()
        ts = datetime.utcnow().isoformat()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=job["estimated_minutes"] * 60 * 3)  # 3x bufor
            duration = time.time() - t0
            if result.returncode == 0:
                n_ok += 1
                consecutive_fails = 0
                report_lines.append(f"| {ts} | {job['dataset']} | {job['model_short']} | "
                                     f"{job['variant']} | {job['seed']} | ✅ OK | {duration:.1f} |")
                logger.info("  → OK (%.1fs)", duration)
            else:
                n_fail += 1
                consecutive_fails += 1
                report_lines.append(f"| {ts} | {job['dataset']} | {job['model_short']} | "
                                     f"{job['variant']} | {job['seed']} | ❌ FAIL | {duration:.1f} |")
                logger.error("  → FAIL (rc=%d): %s", result.returncode, result.stderr[-500:])
                if consecutive_fails >= args.max_consecutive_fails:
                    logger.error("Too many consecutive fails (%d) — stopping", consecutive_fails)
                    report_lines.append(f"\n**STOPPED after {consecutive_fails} consecutive fails**\n")
                    break
        except subprocess.TimeoutExpired:
            n_fail += 1
            consecutive_fails += 1
            report_lines.append(f"| {ts} | {job['dataset']} | {job['model_short']} | "
                                 f"{job['variant']} | {job['seed']} | ⏱ TIMEOUT | - |")
            logger.error("  → TIMEOUT")

    total_min = (time.time() - t_start) / 60
    report_lines.append("")
    report_lines.append(f"**Summary**: OK={n_ok}, FAIL={n_fail}, SKIP={n_skip}, "
                         f"total wall-clock {total_min:.1f} min")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Report → %s", args.report)
    logger.info("Done. OK=%d FAIL=%d SKIP=%d (%.1f min)", n_ok, n_fail, n_skip, total_min)
    return 0


if __name__ == "__main__":
    sys.exit(main())
