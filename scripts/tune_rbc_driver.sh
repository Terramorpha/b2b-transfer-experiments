#!/usr/bin/env bash
set -uo pipefail
export ENERGYPLUS_PATH=${ENERGYPLUS_PATH:-/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0}
export PYTHONUNBUFFERED=1
cd ~/code_sync/maitrise/b2b-transfer-experiments
PY=vendor/morel/.venv/bin/python
NT=${NT:-25}
for bt in RestaurantFastFood RetailStandalone OfficeSmall; do
  echo "===== [$(date +%H:%M)] TUNE $bt index 0 ($NT trials) =====" | tee -a runs/rbc_tuning/driver.log
  $PY scripts/tune_rbc_comfort.py --building-type "$bt" --index 0 --n-trials "$NT" \
      > "runs/rbc_tuning/tune_${bt}.log" 2>&1
  echo "===== [$(date +%H:%M)] $bt done =====" | tee -a runs/rbc_tuning/driver.log
  grep -E "tuned full-year|no-deadband RBC|specialization gain" "runs/rbc_tuning/tune_${bt}.log" | tee -a runs/rbc_tuning/driver.log
done
echo "===== TUNING SWEEP COMPLETE =====" | tee -a runs/rbc_tuning/driver.log
