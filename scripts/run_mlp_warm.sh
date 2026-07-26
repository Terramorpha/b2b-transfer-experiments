#!/usr/bin/env bash
# BC-warm-start MLP oracles: clone RBC into a fresh per-building MLP, then PPO
# fine-tune (ent=0, low LR). The architecture-clean analogue of the Amorpheus
# warm-started specialist -- no morphology, no shared universe, plain gym MLP.
#
# Three phases, each with skip logic so it can coexist with a running pilot:
#   1. collect RBC demos in the gym action space (bc_collect_mlp)
#   2. clone RBC into a fresh per-building MLP (bc_fit_mlp)         [skip if clone exists]
#   3. PPO fine-tune from the clone (train_mlp_specialist --init)  [skip if warm dir exists]
#
# Usage:  bash scripts/run_mlp_warm.sh [FT_CONCURRENCY] [STEPS] [COLLECT_WORKERS]
set -uo pipefail
cd "$(dirname "$0")/.."

FT_CONC="${1:-4}"; STEPS="${2:-500000}"; COLLECT_WORKERS="${3:-3}"
TYPES=(RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall)
IDXS=(0 1 2 3 4)
LOGDIR="${HOME}/.claude/jobs/5c069ce0/tmp/mlp_warm"; mkdir -p "$LOGDIR"
EP=/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0

guixrun() {  # $1 = python command
  guix shell energyplus python python-numpy python-pandas python-pytorch ty uv -- bash -c "
    export ENERGYPLUS_PATH=$EP WANDB_MODE=online MPLBACKEND=Agg OMP_NUM_THREADS=1
    export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    $1"
}

echo "=== PHASE 1: collect RBC demos (all 20; existing overwritten harmlessly) ==="
guixrun "vendor/morel/.venv/bin/python -u scripts/bc_collect_mlp.py --split test \
  --n-buildings 5 --n-workers ${COLLECT_WORKERS}" > "$LOGDIR/collect.log" 2>&1
echo "  collect done"

echo "=== PHASE 2: fit clones (skip existing) ==="
for bt in "${TYPES[@]}"; do for idx in "${IDXS[@]}"; do
  [ -f "runs/mlp_bc_${bt}_${idx}/model_s0.eqx" ] && { echo "  skip fit ${bt}_${idx}"; continue; }
  guixrun "vendor/morel/.venv/bin/python -u scripts/bc_fit_mlp.py \
    --building-type ${bt} --index ${idx}" > "$LOGDIR/fit_${bt}_${idx}.log" 2>&1
  echo "  fit ${bt}_${idx} done"
done; done

echo "=== PHASE 3: warm fine-tune (skip existing dir), ${FT_CONC}-wide ==="
ft_one() {
  local bt="$1" idx="$2"
  [ -d "runs/mlp_warm_${bt}_${idx}" ] && { echo "  skip ft ${bt}_${idx} (dir exists)"; return; }
  guixrun "vendor/morel/.venv/bin/python -u scripts/train_mlp_specialist.py \
    --building-type ${bt} --index ${idx} --split test --steps ${STEPS} \
    --lr 2e-5 --ent-coef 0 --eval-every 0 \
    --init-checkpoint runs/mlp_bc_${bt}_${idx}/model_s0.eqx \
    --out runs/mlp_warm_${bt}_${idx}" > "$LOGDIR/ft_${bt}_${idx}.log" 2>&1
  echo "  ft ${bt}_${idx} done (exit $?)"
}
for idx in "${IDXS[@]}"; do for bt in "${TYPES[@]}"; do
  ft_one "$bt" "$idx" &
  while [ "$(jobs -rp | wc -l)" -ge "$FT_CONC" ]; do sleep 10; done
done; done
wait
echo "MLP_WARM_DONE"
