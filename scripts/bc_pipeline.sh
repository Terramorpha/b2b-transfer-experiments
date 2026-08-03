#!/usr/bin/env bash
set -uo pipefail
export ENERGYPLUS_PATH=${ENERGYPLUS_PATH:-/gnu/store/99mrp2na1mh6a4z1rrxirk36dsy6wrkp-energyplus-25.1.0}
export PYTHONUNBUFFERED=1
cd ~/code_sync/maitrise/b2b-transfer-experiments
PY=vendor/morel/.venv/bin/python
LOG=runs/rbc_tuning/pipeline.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

cfgs=(data/rbc_tuned_configs/*_train_*.json)
say "STAGE collect (${#cfgs[@]} tuned configs)"
$PY scripts/bc_collect_tuned.py --n-buildings 5 --n-workers 4 > runs/rbc_tuning/bc_collect.log 2>&1
if [[ "$(<runs/rbc_tuning/bc_collect.log)" != *BC_COLLECT_DONE* ]]; then say "COLLECT FAILED"; exit 1; fi
demos=(data/bc_demos_tuned/*.npz); say "collect done: ${#demos[@]} demo files"

say "STAGE bc_fit"
WANDB_MODE=offline $PY scripts/bc_fit.py --demos data/bc_demos_tuned \
    --out runs/transfer4_bc_tuned/model_s0.eqx > runs/rbc_tuning/bc_fit.log 2>&1
if [[ ! -f runs/transfer4_bc_tuned/model_s0.eqx ]]; then say "BC_FIT FAILED"; exit 1; fi
say "bc_fit done -> runs/transfer4_bc_tuned/model_s0.eqx"

say "STAGE eval BC-clone full-year on test"
$PY scripts/eval_fullyear_panel.py --checkpoint runs/transfer4_bc_tuned/model_s0.eqx \
    --tag bc_tuned > runs/rbc_tuning/bc_eval.log 2>&1
say "BC-clone eval done (see bc_eval.log)"
say "PIPELINE_READY_FOR_WARMSTART"
