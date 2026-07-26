#!/usr/bin/env bash
# True per-building oracles: a FRESH MLP PPO trained + evaluated on each building.
#
# Unlike run_specialists.sh (Amorpheus, warm-started from the BC clone), these pay
# NO transfer tax and use NO clone: plain gym Box obs/action, vanilla MLP actor +
# critic, PPO from scratch. This measures the real ceiling ("how much is on the
# table for a dedicated policy") rather than "how much a morphology-agnostic net
# conditioned on one building can reach". Trains on the TEST building it is scored
# on -- deliberately an oracle, not a generalization result.
#
# Usage:  bash scripts/run_mlp_specialists.sh [CONCURRENCY] [STEPS]
set -uo pipefail
cd "$(dirname "$0")/.."

CONC="${1:-4}"          # 4 cores -> 4 concurrent EnergyPlus sims is the useful max
STEPS="${2:-1000000}"
EVAL_EVERY="${3:-0}"     # 0 = full-year eval only at the end (per-update curve still logged)
LOGDIR="${HOME}/.claude/jobs/5c069ce0/tmp/mlp_specialists"
mkdir -p "$LOGDIR"

TYPES=(RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall)
IDXS=(0 1 2 3 4)

run_one() {
  local bt="$1" idx="$2"
  local out="runs/mlp_specialist_${bt}_${idx}"
  local log="${LOGDIR}/${bt}_${idx}.log"
  # skip if already done OR in progress (dir exists) -> won't duplicate the pilot
  [ -d "${out}" ] && { echo "  skip ${bt}_${idx} (dir exists)"; return 0; }
  guix shell energyplus python python-numpy python-pandas python-pytorch ty uv -- bash -c "
    export ENERGYPLUS_PATH=/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0
    export WANDB_MODE=online MPLBACKEND=Agg
    # cap each process to one JAX/XLA thread so N runs share N cores instead of
    # each spawning a full thread pool and thrashing (the MLP is tiny anyway).
    export OMP_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    vendor/morel/.venv/bin/python -u scripts/train_mlp_specialist.py \
      --building-type ${bt} --index ${idx} --split test \
      --steps ${STEPS} --n-steps 2048 --epochs 10 --minibatch 64 \
      --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 --clip 0.2 --ent-coef 0.0 \
      --seed 0 --eval-every ${EVAL_EVERY} --out ${out}" > "$log" 2>&1
  echo "  done ${bt}_${idx} (exit $?)"
}

echo "[mlp-oracles] ${#TYPES[@]} types x ${#IDXS[@]} buildings = $(( ${#TYPES[@]} * ${#IDXS[@]} )) runs"
echo "[mlp-oracles] concurrency ${CONC}, ${STEPS} steps each, fresh MLP PPO (no clone, no morphology)"

n=0
for idx in "${IDXS[@]}"; do        # idx-major: one building per type first
  for bt in "${TYPES[@]}"; do
    run_one "$bt" "$idx" &
    n=$((n+1))
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
  done
done
wait
echo "MLP_SPECIALISTS_DONE ($n launched)"
