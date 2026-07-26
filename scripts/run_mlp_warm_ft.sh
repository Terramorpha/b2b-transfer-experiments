#!/usr/bin/env bash
# Warm fine-tune ONLY, from existing BC clones (runs/mlp_bc_*), to a chosen step
# budget + output prefix. Reuses the clones fit by run_mlp_warm.sh, so this is the
# "longer-budget ceiling" pass (e.g. 1.5M steps ~= 732 updates, matching the
# Amorpheus specialists' 730). Distinct prefix keeps it separate from the 500k run.
#
# Usage:  bash scripts/run_mlp_warm_ft.sh [CONCURRENCY] [STEPS] [OUT_PREFIX]
set -uo pipefail
cd "$(dirname "$0")/.."

CONC="${1:-4}"; STEPS="${2:-1500000}"; PREFIX="${3:-mlp_warm1p5m}"
TYPES=(RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall)
IDXS=(0 1 2 3 4)
LOGDIR="${HOME}/.claude/jobs/5c069ce0/tmp/${PREFIX}"; mkdir -p "$LOGDIR"
EP=/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0

ft_one() {
  local bt="$1" idx="$2"
  [ -d "runs/${PREFIX}_${bt}_${idx}" ] && { echo "  skip ${bt}_${idx} (dir exists)"; return; }
  [ -f "runs/mlp_bc_${bt}_${idx}/model_s0.eqx" ] || { echo "  NO CLONE ${bt}_${idx}"; return; }
  guix shell energyplus python python-numpy python-pandas python-pytorch ty uv -- bash -c "
    export ENERGYPLUS_PATH=$EP WANDB_MODE=online MPLBACKEND=Agg OMP_NUM_THREADS=1
    export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    vendor/morel/.venv/bin/python -u scripts/train_mlp_specialist.py \
      --building-type ${bt} --index ${idx} --split test --steps ${STEPS} \
      --lr 2e-5 --ent-coef 0 --eval-every 0 \
      --init-checkpoint runs/mlp_bc_${bt}_${idx}/model_s0.eqx \
      --out runs/${PREFIX}_${bt}_${idx}" > "$LOGDIR/${bt}_${idx}.log" 2>&1
  echo "  ${bt}_${idx} done (exit $?)"
}

echo "[warm-ft] ${STEPS} steps (~$(( STEPS / 2048 )) updates), prefix ${PREFIX}, ${CONC}-wide"
for idx in "${IDXS[@]}"; do for bt in "${TYPES[@]}"; do
  ft_one "$bt" "$idx" &
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
done; done
wait
echo "WARM_FT_DONE ${PREFIX}"
