#!/usr/bin/env bash
# Wznawia pracę w tle po uśpieniu/przeniesieniu laptopa.
# Oba procesy są resume-safe (--skip-existing / --resume) — nic nie liczy się dwa razy.
set -e
cd "$(dirname "$0")/.."
export $(grep -E "HF_HUB_CACHE|HF_TOKEN|DIFFBOT_TOKEN" .env | xargs) 2>/dev/null

echo "▶ Wznawiam nocny run Fazy E (pomija gotowe)..."
nohup .venv/bin/python pipeline/04_execution/run_local_all.py \
  --matrix pipeline/04_execution/matrix_E_ready.jsonl \
  --skip-existing --max-consecutive-fails 5 \
  --report pipeline/04_execution/REPORT_E_overnight.md \
  > pipeline/04_execution/overnight.log 2>&1 &
echo "  PID: $!"

echo "▶ Wznawiam DiffBot EU fill (pomija pobrane)..."
nohup .venv/bin/python pipeline/01_data/diffbot_scrape.py \
  --input datasets/euvsdisinfo/errors_alive.jsonl \
  --output datasets/euvsdisinfo/scraped_diffbot.jsonl \
  --max-calls 4000 --cost-per-call-usd 0 --budget-usd 1 \
  --rate-limit-per-min 40 --resume \
  > datasets/euvsdisinfo/diffbot_resume.log 2>&1 &
echo "  PID: $!"

echo ""
echo "✅ Oba procesy wznowione. Postęp: ls experiments/results_v2/ | wc -l (cel 128)"
