#!/bin/bash
cd /Users/mbasarab/Workspace/PERSONAL/UMCS/magisterka
for i in $(seq 1 90); do
  LS=$(.venv/bin/modal volume ls disinfo-results /results_v2 2>/dev/null)
  N=$(echo "$LS" | grep -c "euvsdisinfo")
  ZS=$(echo "$LS" | grep -E "euvsdisinfo.*(zs|zeroshot)" | grep -c "euvsdisinfo")
  LORA=$(echo "$LS" | grep -E "euvsdisinfo.*(lora_big|big)" | grep -c "euvsdisinfo")
  echo "[$(date +%H:%M)] poll $i: EU results=$N (ZS~$ZS, LoRA~$LORA) / cel 11"
  if [ "$N" -ge 11 ]; then echo "=== WSZYSTKIE 11 EU GOTOWE ==="; break; fi
  sleep 120
done
echo "=== poll koniec ($N/11) ==="
