#!/usr/bin/env bash
# Specialist oracles: train one policy PER TEST BUILDING, on that building itself.
#
# Purpose: an UPPER BOUND on achievable gain over RBC. A specialist may train on the
# very building it is scored on -- that is deliberate, it is an oracle, not a
# generalization result. It answers "how much is even on the table here?", which
# bounds what any transfer policy could hope to achieve.
#
# Warm-started from the BC clone: a from-scratch specialist would hit the same
# under-actuation failure (2.3% of fan range) and would understate the ceiling.
#
# Usage:  bash scripts/run_specialists.sh [CONCURRENCY] [YEARS]
set -uo pipefail
cd "$(dirname "$0")/.."

CONC="${1:-3}"
YEARS="${2:-1}"
INIT="runs/transfer4_bc_long/model_s0.eqx"
LOGDIR="${HOME}/.claude/jobs/5c069ce0/tmp/specialists"
mkdir -p "$LOGDIR"

TYPES=(RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall)
IDXS=(0 1 2 3 4)

run_one() {
  local bt="$1" idx="$2"
  local out="runs/specialist_${bt}_${idx}"
  local log="${LOGDIR}/${bt}_${idx}.log"
  [ -f "${out}/model_s0.eqx" ] && { echo "  skip ${bt}_${idx} (exists)"; return 0; }
  guix shell energyplus python python-numpy python-pandas python-pytorch ty uv -- bash -c "
    export ENERGYPLUS_PATH=/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0
    export WANDB_MODE=offline MPLBACKEND=Agg
    vendor/morel/.venv/bin/python -u scripts/train_allactive.py \
      --building-types ${bt} --split test --indices ${idx} --n-envs 1 \
      --task task_occ_e0 --years ${YEARS} \
      --lr 2e-5 --ent-coef 0.0 --gamma 0.99 --seed 0 \
      --init-checkpoint ${INIT} --out ${out}" > "$log" 2>&1
  echo "  done ${bt}_${idx} (exit $?)"
}

echo "[specialists] ${#TYPES[@]} types x ${#IDXS[@]} buildings = $(( ${#TYPES[@]} * ${#IDXS[@]} )) runs"
echo "[specialists] concurrency ${CONC}, ${YEARS} year(s) each, warm-started from ${INIT}"

n=0
for idx in "${IDXS[@]}"; do        # idx-major: one building per type first
  for bt in "${TYPES[@]}"; do
    run_one "$bt" "$idx" &
    n=$((n+1))
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
  done
done
wait
echo "SPECIALISTS_DONE ($n launched)"
