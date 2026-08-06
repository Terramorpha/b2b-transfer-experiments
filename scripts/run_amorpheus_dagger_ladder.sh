#!/bin/bash
# Amorpheus DAgger pipeline: DAgger-clone the per-building-tuned RBCs into a single
# Amorpheus policy (all 4 types, comfort task_occ_e0), then PPO-warm from the final
# DAgger iteration -- the warmstart_unitary_tuned pipeline but DAgger instead of
# naive BC. Sequential, dependency-checked. wandb ONLINE (do not set WANDB_MODE).
#
#   STEP 1: dagger_loop_amorpheus (it0=BC, it1, it2=DAgger) -> runs/dagger_e0_amorpheus
#   STEP 2: PPO warm from model_it2 (--ent-coef 0, --years 2, matching
#           warmstart_unitary_tuned) -> runs/dagger_e0_amorpheus_warm
#
# Master log: runs/amorpheus_dagger_ladder.log

set -u
cd /home/terramorpha/code_sync/maitrise/b2b-transfer-experiments

PROF=/home/terramorpha/.anonymous-profiles/dc686a98b74e01c7352bae596e2181c4/profile
export GUIX_PROFILE="$PROF"; . "$PROF/etc/profile"
export GUIX_LOAD_PROFILE="$PROF"
export ENERGYPLUS_PATH="$PROF"
# wandb ONLINE (scripts default to online; do NOT export WANDB_MODE=offline)
VPY=vendor/morel/.venv/bin/python

mkdir -p runs
MASTER=runs/amorpheus_dagger_ladder.log
log()    { printf '[%(%F %T)T] %s\n' -1 "$1" >> "$MASTER"; }
banner() { printf '\n======================================================\n' >> "$MASTER"
           printf '[%(%F %T)T] %s\n' -1 "$1" >> "$MASTER"
           printf '======================================================\n' >> "$MASTER"; }
have()   { [[ -s "$1" ]]; }

TASK=task_occ_e0
OUT=runs/dagger_e0_amorpheus
IT2=$OUT/model_it2.eqx
WARM=runs/dagger_e0_amorpheus_warm/model_s0.eqx

log "AMORPHEUS DAgger ladder START (task=$TASK, all 4 types x5, 3 iterations)"

# ---- STEP 1: DAgger loop (BC=it0, it1, DAgger=it2) --------------------------
banner "STEP 1  Amorpheus DAgger loop"
if have "$OUT/model_it0.eqx"; then
  log "STEP 1 skip: $OUT/model_it0.eqx already present (loop already ran)"
else
  log "STEP 1 run: dagger_loop_amorpheus.py (log runs/dagger_e0_amorpheus_loop.log)"
  "$VPY" scripts/dagger_loop_amorpheus.py --iterations 3 \
    --building-types RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall \
    --n-buildings 5 --task "$TASK" --out "$OUT" --n-workers 3 \
    > runs/dagger_e0_amorpheus_loop.log 2>&1
  log "STEP 1 rc=$? ; it0=$(have "$OUT/model_it0.eqx" && echo yes || echo NO) it2=$(have "$IT2" && echo yes || echo NO)"
fi

# ---- STEP 2: PPO warm from DAgger it2 (matches warmstart_unitary_tuned) -----
banner "STEP 2  PPO warm from DAgger it2 (ent-coef 0, 2 years)"
if have "$WARM"; then
  log "STEP 2 skip: $WARM present"
elif ! have "$IT2"; then
  log "STEP 2 SKIP: required init $IT2 missing (STEP 1 did not produce DAgger-2 checkpoint)"
else
  log "STEP 2 run: train_allactive.py warm from $IT2 (log runs/train_dagger_e0_amorpheus_warm.log)"
  "$VPY" scripts/train_allactive.py --task "$TASK" \
    --building-types RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall \
    --n-envs 5 --years 2 --ent-coef 0 --seed 0 \
    --init-checkpoint "$IT2" --out runs/dagger_e0_amorpheus_warm \
    > runs/train_dagger_e0_amorpheus_warm.log 2>&1
  log "STEP 2 rc=$? ; out=$(have "$WARM" && echo yes || echo NO)"
fi

banner "AMORPHEUS DAgger ladder DONE"
log "AMORPHEUS_DAGGER_LADDER_DONE"
