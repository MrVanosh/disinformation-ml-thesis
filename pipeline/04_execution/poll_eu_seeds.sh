#!/bin/bash
cd /Users/mbasarab/Workspace/PERSONAL/UMCS/magisterka
for i in $(seq 1 60); do
  N=$(.venv/bin/modal volume ls disinfo-results /results_v2 2>/dev/null | grep -E "euvsdisinfo.*lora_big.*seed(42|71)" | grep -c euvsdisinfo)
  echo "[$(date +%H:%M)] poll $i: EU big-LoRA seed42/71 = $N / cel 4"
  if [ "$N" -ge 4 ]; then echo "=== 4 NOWE SEEDY GOTOWE ==="; break; fi
  sleep 120
done
echo "=== poll koniec ($N/4) ==="
