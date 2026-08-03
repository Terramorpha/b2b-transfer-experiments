#!/usr/bin/env bash
# Sequential ModuMorph PERSISTENT full-year sweep (16GB RAM / 4 cores -> one seed
# at a time). For each seed: train on the exact Amorpheus Part-II protocol
# (persistent, full year, from scratch) then full-year eval on the test split.
# Resumable via per-seed .done markers. Usage:
#   SEEDS="0 1 2" bash scripts/sweep_modumorph_persistent.sh
set -uo pipefail
export ENERGYPLUS_PATH=${ENERGYPLUS_PATH:-/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0}
export PYTHONUNBUFFERED=1
cd ~/code_sync/maitrise/b2b-transfer-experiments
PY=vendor/morel/.venv/bin/python
SEEDS=${SEEDS:-"0 1 2"}
OUT=runs/transfer4_modumorph_persistent
mkdir -p "$OUT"
RES="$OUT/fullyear_results.csv"
[ -f "$RES" ] || echo "seed,total_wins" > "$RES"

for s in $SEEDS; do
  ck="$OUT/model_s${s}.eqx"
  if [ ! -f "$OUT/train_s${s}.done" ]; then
    echo "===== [$(date '+%m-%d %H:%M')] TRAIN seed $s (persistent, full year, ~6h) =====" | tee -a "$OUT/sweep.log"
    $PY scripts/train_allactive_modumorph.py --seed "$s" --out "$OUT" \
        > "$OUT/train_s${s}.log" 2>&1 && touch "$OUT/train_s${s}.done"
  fi
  if [ ! -f "$ck" ]; then echo "seed $s: NO CHECKPOINT — train failed, skipping eval" | tee -a "$OUT/sweep.log"; continue; fi
  echo "===== [$(date '+%m-%d %H:%M')] FULL-YEAR EVAL seed $s =====" | tee -a "$OUT/sweep.log"
  $PY scripts/eval_fullyear_panel_modumorph.py --checkpoint "$ck" \
      --tag "modumorph_persistent_s${s}" --n-workers 4 \
      > "$OUT/eval_fullyear_s${s}.log" 2>&1
  tot=$(grep "TOTAL wins" "$OUT/eval_fullyear_s${s}.log" | grep -oE "[0-9]+/20" | head -1)
  echo "$s,${tot:-NA}" >> "$RES"
  echo "===== [$(date '+%m-%d %H:%M')] seed $s DONE -> full-year ${tot:-NA} =====" | tee -a "$OUT/sweep.log"
done
echo "===== SWEEP COMPLETE =====" | tee -a "$OUT/sweep.log"
cat "$RES" | tee -a "$OUT/sweep.log"
