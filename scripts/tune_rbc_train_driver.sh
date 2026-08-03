#!/usr/bin/env bash
set -uo pipefail
export ENERGYPLUS_PATH=${ENERGYPLUS_PATH:-/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0}
export PYTHONUNBUFFERED=1
cd ~/code_sync/maitrise/b2b-transfer-experiments
mkdir -p runs/rbc_tuning/train
PY=vendor/morel/.venv/bin/python
NT=${NT:-25}; NB=${NB:-5}; PAR=${PAR:-4}
run_one() {
  local bt="$1" idx="$2"
  $PY scripts/tune_rbc_comfort.py --building-type "$bt" --index "$idx" \
      --split train --n-trials "$NT" > "runs/rbc_tuning/train/${bt}_${idx}.log" 2>&1
  echo "[$(date +%H:%M)] done $bt $idx: $(grep -oE 'tuned=[-0-9.]+' runs/rbc_tuning/train/${bt}_${idx}.log | tail -1)" >> runs/rbc_tuning/train/driver.log
}
running=0
for bt in RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall; do
  for ((idx=0; idx<NB; idx++)); do
    run_one "$bt" "$idx" &
    running=$((running+1))
    if [ "$running" -ge "$PAR" ]; then wait -n; running=$((running-1)); fi
  done
done
wait
echo "ALL_TUNING_DONE" >> runs/rbc_tuning/train/driver.log
