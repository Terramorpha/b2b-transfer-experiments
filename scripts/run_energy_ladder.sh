#!/bin/bash
# Energy-aware (task_occ_emed, w_E=1.0) experiment ladder -- SEQUENTIAL and
# dependency-checked. Same protocol as the comfort (task_occ_e0) run; the only
# knob changed is --task task_occ_emed. 4 cores => nothing runs in parallel;
# each cell runs only if its required input exists and its own output does not.
# A failed training cell (e.g. OOM mid-run) does NOT abort the ladder -- the
# driver logs it and continues to the independent cells.
#
# Order (per coordinator):
#   1. DAgger loop (emed)          -> runs/dagger_emed/model_it{0,1,2}.eqx
#   2. train from scratch          -> runs/emed_fromscratch/model_s0.eqx
#   3. train BC-warm  (init it0)   -> runs/emed_bcwarm/model_s0.eqx
#   4. train DAgger-warm (init it2)-> runs/emed_daggerwarm/model_s0.eqx
#   5. eval all 5 checkpoints on emed vs data/rbc_emed_fullyear.json
#
# Master log: runs/energy_ladder.log ; each sub-step its own runs/*.log

set -u
cd /home/terramorpha/code_sync/maitrise/b2b-transfer-experiments

PROF=/home/terramorpha/.anonymous-profiles/dc686a98b74e01c7352bae596e2181c4/profile
export GUIX_PROFILE="$PROF"; . "$PROF/etc/profile"
export GUIX_LOAD_PROFILE="$PROF"
export ENERGYPLUS_PATH="$PROF"
export WANDB_MODE=offline
VPY=vendor/morel/.venv/bin/python

mkdir -p runs
MASTER=runs/energy_ladder.log
DEMO_TASK=task_occ_emed

log()    { printf '[%(%F %T)T] %s\n' -1 "$1" >> "$MASTER"; }
banner() { printf '\n======================================================\n' >> "$MASTER"
           printf '[%(%F %T)T] %s\n' -1 "$1" >> "$MASTER"
           printf '======================================================\n' >> "$MASTER"; }
have()   { [[ -s "$1" ]]; }   # file exists and is non-empty

log "ENERGY LADDER START (task=$DEMO_TASK, variant=hn, seed=0)"

# ---- STEP 1: DAgger loop (BC=it0, DAgger-1=it1, DAgger-2=it2) ----------------
banner "STEP 1  DAgger loop (emed)"
IT0=runs/dagger_emed/model_it0.eqx
IT2=runs/dagger_emed/model_it2.eqx
if have "$IT0"; then
  log "STEP 1 skip: $IT0 already present (loop already ran)"
else
  log "STEP 1 run: dagger_loop.py --task $DEMO_TASK (log runs/dagger_emed_loop.log)"
  "$VPY" scripts/dagger_loop.py --task "$DEMO_TASK" --iterations 3 \
    --building-types RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall \
    --n-buildings 5 --variant hn --out runs/dagger_emed --n-workers 3 \
    > runs/dagger_emed_loop.log 2>&1
  log "STEP 1 rc=$? ; it0=$(have "$IT0" && echo yes || echo NO) it2=$(have "$IT2" && echo yes || echo NO)"
fi

# ---- STEP 2: train from scratch ---------------------------------------------
banner "STEP 2  train from scratch (emed)"
FS=runs/emed_fromscratch/model_s0.eqx
if have "$FS"; then
  log "STEP 2 skip: $FS present"
else
  log "STEP 2 run: train_allactive_modumorph.py from scratch (log runs/train_emed_fromscratch.log)"
  "$VPY" scripts/train_allactive_modumorph.py --task "$DEMO_TASK" --variant hn \
    --seed 0 --out runs/emed_fromscratch \
    > runs/train_emed_fromscratch.log 2>&1
  log "STEP 2 rc=$? ; out=$(have "$FS" && echo yes || echo NO)"
fi

# ---- STEP 3: BC-warm PPO (init = DAgger it0) --------------------------------
banner "STEP 3  BC-warm PPO (init it0, ent-coef 0)"
BCW=runs/emed_bcwarm/model_s0.eqx
if have "$BCW"; then
  log "STEP 3 skip: $BCW present"
elif ! have "$IT0"; then
  log "STEP 3 SKIP: required init $IT0 missing (STEP 1 did not produce BC checkpoint)"
else
  log "STEP 3 run: PPO warm-start from $IT0 (log runs/train_emed_bcwarm.log)"
  "$VPY" scripts/train_allactive_modumorph.py --task "$DEMO_TASK" --variant hn \
    --seed 0 --init-checkpoint "$IT0" --ent-coef 0 --out runs/emed_bcwarm \
    > runs/train_emed_bcwarm.log 2>&1
  log "STEP 3 rc=$? ; out=$(have "$BCW" && echo yes || echo NO)"
fi

# ---- STEP 4: DAgger-warm PPO (init = DAgger it2) ----------------------------
banner "STEP 4  DAgger-warm PPO (init it2, ent-coef 0)"
DGW=runs/emed_daggerwarm/model_s0.eqx
if have "$DGW"; then
  log "STEP 4 skip: $DGW present"
elif ! have "$IT2"; then
  log "STEP 4 SKIP: required init $IT2 missing (STEP 1 did not produce DAgger-2 checkpoint)"
else
  log "STEP 4 run: PPO warm-start from $IT2 (log runs/train_emed_daggerwarm.log)"
  "$VPY" scripts/train_allactive_modumorph.py --task "$DEMO_TASK" --variant hn \
    --seed 0 --init-checkpoint "$IT2" --ent-coef 0 --out runs/emed_daggerwarm \
    > runs/train_emed_daggerwarm.log 2>&1
  log "STEP 4 rc=$? ; out=$(have "$DGW" && echo yes || echo NO)"
fi

# ---- ensure the energy RBC baseline exists before the eval phase ------------
banner "STEP 5  eval phase (vs data/rbc_emed_fullyear.json)"
BASE=data/rbc_emed_fullyear.json
if ! have "$BASE"; then
  log "baseline $BASE missing -> computing it now (log runs/rbc_emed_baseline.log)"
  "$VPY" scripts/compute_rbc_emed_baseline.py --task "$DEMO_TASK" --n-workers 3 \
    --out "$BASE" > runs/rbc_emed_baseline.log 2>&1
  log "baseline rc=$? ; out=$(have "$BASE" && echo yes || echo NO)"
fi

eval_ckpt() {  # eval_ckpt <ckpt> <tag>
  local ckpt="$1" tag="$2"
  local out="data/${tag}_fullyear_eval.csv" elog="runs/eval_${tag}.log"
  if ! have "$ckpt"; then log "EVAL $tag SKIP: checkpoint $ckpt missing"; return 0; fi
  if ! have "$BASE"; then log "EVAL $tag SKIP: baseline $BASE missing"; return 0; fi
  if have "$out";     then log "EVAL $tag skip: $out present"; return 0; fi
  log "EVAL $tag run: $ckpt (log $elog)"
  "$VPY" scripts/eval_fullyear_panel_modumorph.py --task "$DEMO_TASK" --variant hn \
    --checkpoint "$ckpt" --tag "$tag" --baselines "$BASE" > "$elog" 2>&1
  log "EVAL $tag rc=$? ; csv=$(have "$out" && echo yes || echo NO)"
}

eval_ckpt runs/dagger_emed/model_it0.eqx       emed_it0_BC
eval_ckpt runs/dagger_emed/model_it2.eqx       emed_it2_DAgger
eval_ckpt runs/emed_fromscratch/model_s0.eqx   emed_fromscratch
eval_ckpt runs/emed_bcwarm/model_s0.eqx        emed_bcwarm
eval_ckpt runs/emed_daggerwarm/model_s0.eqx    emed_daggerwarm

banner "ENERGY LADDER DONE"
log "ENERGY_LADDER_DONE"
