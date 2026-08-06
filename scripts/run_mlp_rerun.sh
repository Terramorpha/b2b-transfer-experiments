#!/bin/bash
# RERUN of the MLP specialist ceiling AFTER the obs-normalizer bug fix in
# train_mlp_specialist.py (previously norm.update() was fed NORMALIZED obs; now it
# updates on RAW obs, and warm runs use --freeze-norm). bc_fit_mlp.py was verified
# correct (updates the normalizer on raw demo obs), so the existing runs/mlp_bc_*
# clones + their obs_norm.npz are REUSED unchanged.
#
# Writes to *_fix_* dirs so the old (buggy) runs are preserved for comparison.
#   from-scratch -> runs/mlp_specialist_fix_<bt>_<idx>   (fixed trainer, norm updates on raw)
#   BC-warm 1.5M -> runs/mlp_warm1p5m_fix_<bt>_<idx>     (init = existing mlp_bc clone, --freeze-norm)
#
# 4 cores => 4-wide job pool. Dependency-checked (skips a cell whose out dir exists).
# NOT launched automatically -- this is queued; run it when the box is free:
#   nohup bash scripts/run_mlp_rerun.sh > runs/mlp_rerun.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

CONC="${1:-4}"
FS_STEPS="${2:-1000000}"     # from-scratch steps (matches run_mlp_specialists.sh)
WARM_STEPS="${3:-1500000}"   # warm-ft steps (matches run_mlp_warm_ft.sh 1.5M)
EP=/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0
LOGDIR="${HOME}/.claude/jobs/5c069ce0/tmp/mlp_rerun"; mkdir -p "$LOGDIR"
TYPES=(RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall)
IDXS=(0 1 2 3 4)

guixpy() {  # $1 = python command string
  guix shell energyplus python python-numpy python-pandas python-pytorch ty uv -- bash -c "
    export ENERGYPLUS_PATH=$EP WANDB_MODE=online MPLBACKEND=Agg OMP_NUM_THREADS=1
    export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    $1"
}

fromscratch_one() {
  local bt="$1" idx="$2" out="runs/mlp_specialist_fix_${bt}_${idx}"
  [ -d "$out" ] && { echo "  skip fromscratch ${bt}_${idx} (exists)"; return; }
  guixpy "vendor/morel/.venv/bin/python -u scripts/train_mlp_specialist.py \
    --building-type ${bt} --index ${idx} --split test \
    --steps ${FS_STEPS} --n-steps 2048 --epochs 10 --minibatch 64 \
    --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 --clip 0.2 --ent-coef 0.0 \
    --seed 0 --eval-every 0 --out ${out}" > "$LOGDIR/fs_${bt}_${idx}.log" 2>&1
  echo "  fromscratch ${bt}_${idx} done (exit $?)"
}

warm_one() {
  local bt="$1" idx="$2" out="runs/mlp_warm1p5m_fix_${bt}_${idx}"
  local clone="runs/mlp_bc_${bt}_${idx}/model_s0.eqx"
  [ -d "$out" ] && { echo "  skip warm ${bt}_${idx} (exists)"; return; }
  [ -f "$clone" ] || { echo "  NO CLONE ${bt}_${idx} -> skip warm"; return; }
  guixpy "vendor/morel/.venv/bin/python -u scripts/train_mlp_specialist.py \
    --building-type ${bt} --index ${idx} --split test --steps ${WARM_STEPS} \
    --lr 2e-5 --ent-coef 0 --eval-every 0 --freeze-norm \
    --init-checkpoint ${clone} --out ${out}" > "$LOGDIR/warm_${bt}_${idx}.log" 2>&1
  echo "  warm ${bt}_${idx} done (exit $?)"
}

echo "[mlp-rerun] from-scratch (${FS_STEPS} steps) then warm-1.5M (--freeze-norm), ${CONC}-wide"
for phase in fromscratch warm; do
  echo "=== PHASE: ${phase} ==="
  for idx in "${IDXS[@]}"; do for bt in "${TYPES[@]}"; do
    "${phase}_one" "$bt" "$idx" &
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 10; done
  done; done
  wait
done
echo "MLP_RERUN_DONE"
