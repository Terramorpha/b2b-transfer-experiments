#!/usr/bin/env bash
# Sequential ModuMorph transfer sweep: train + eval each seed (4-core box, one
# run at a time). Usage:
#   ITERS=700 SEEDS="0 1 2 3 4" bash scripts/sweep_modumorph.sh
# Results (per-seed test-split win counts vs RBC) are appended to
# runs/transfer_modumorph_sweep/results.csv as each seed finishes.
set -uo pipefail
export ENERGYPLUS_PATH=${ENERGYPLUS_PATH:-/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0}
export PYTHONUNBUFFERED=1
cd ~/code_sync/maitrise/b2b-transfer-experiments
PY=vendor/morel/.venv/bin/python
ITERS=${ITERS:-700}
SEEDS=${SEEDS:-"0 1 2 3 4"}
OUT=runs/transfer_modumorph_sweep
mkdir -p "$OUT"
RES="$OUT/results.csv"
[ -f "$RES" ] || echo "seed,iters,total_wins" > "$RES"

for s in $SEEDS; do
  echo "===== [$(date +%H:%M)] TRAIN seed $s ($ITERS iters) =====" | tee -a "$OUT/sweep.log"
  $PY scripts/train_transfer_modumorph.py --seed "$s" --n-workers 4 \
      --total-iterations "$ITERS" --out "$OUT/train_s${s}" \
      > "$OUT/train_s${s}.log" 2>&1
  ck="$OUT/train_s${s}/model_s${s}.eqx"
  if [ ! -f "$ck" ]; then echo "seed $s: NO CHECKPOINT — train failed" | tee -a "$OUT/sweep.log"; continue; fi
  echo "===== [$(date +%H:%M)] EVAL seed $s =====" | tee -a "$OUT/sweep.log"
  $PY scripts/eval_transfer_panel_modumorph.py --checkpoint "$ck" \
      > "$OUT/eval_s${s}.log" 2>&1
  tot=$(grep "TOTAL wins" "$OUT/eval_s${s}.log" | grep -oE "[0-9]+/20" | head -1)
  echo "$s,$ITERS,${tot:-NA}" >> "$RES"
  echo "===== [$(date +%H:%M)] seed $s DONE -> ${tot:-NA} =====" | tee -a "$OUT/sweep.log"
done
echo "===== SWEEP COMPLETE =====" | tee -a "$OUT/sweep.log"
cat "$RES" | tee -a "$OUT/sweep.log"
