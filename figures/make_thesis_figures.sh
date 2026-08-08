#!/bin/bash
# Regenerate every repo-side figure referenced by the mémoire
# (~/org-roam/20260608113729-memoire_de_maitrise.org) from committed data.
# One command, no hidden state:  bash figures/make_thesis_figures.sh
#
# Covered here (b2b-transfer-experiments figures):
#   figures_out_fullyear/fullyear_port_preview.png   from-scratch Amorpheus, full year
#       !! REBUILT from transfer4_final (horizon-verified 105120-step eval).
#       The 2026-07-20 original was built the same day as transfer4_mid — a
#       40 000-step eval later found non-comparable. NEVER use transfer4_mid.
#   figures_out_bcwarm/bcwarm_port_preview.png       BC-warm Amorpheus, full year, test
#   figures_out/policy_vs_rbc_actions_preview.png    SAT-vs-setpoint trace diagnostic
#   figures_out/fan_speed_preview.png                fan-speed trace diagnostic
#   figures_out/tuned_vs_amorpheus_port_preview.png  BC-warm vs default+tuned RBC (test)
#   figures_out/bcwarm_train_vs_tuned_port_preview.png  BC-warm vs tuned oracles (train)
#   figures_out/dagger_y1_port_preview.png           streaming DAgger y1 (test)
#   figures_out/dagger_y2_port_preview.png           (skipped until its eval CSV exists)
#
# NOT covered:
#   ../figures/morphology_*/panel{1,3}_*.png  -> per-dir panel1_building.py +
#       ../figures/panel3_generic.py (heavier deps; run in their own dirs)
#   figures_out_warm1p5m/warm1p5m_port_preview.png -> BLOCKED: built from the
#       MLP runs invalidated by the obs-normalizer bug (see EXPERIMENT_PLAN.md);
#       regenerate only after scripts/run_mlp_rerun.sh produces the *_fix_* runs.
#   generated/*.svg in the org file -> org-babel dot block (C-c C-c in Emacs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=vendor/morel/.venv/bin/python
export MPLBACKEND=Agg PYTHONPATH=figures

echo "== derived inputs (data/_*.csv|json) from primary sources =="
$PY - <<'EOF'
import csv, json
TYPES = ["RestaurantFastFood", "RetailStandalone", "OfficeMedium", "OfficeSmall"]
def typ(b):
    for t in TYPES:
        if b.startswith(t): return t

# default RBC on TEST (per building), from the canonical harness json
default = json.load(open("data/rbc_fullyear_ourharness.json"))
with open("data/_default_baseline_fullyear.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["building_type", "building_id", "baseline_return"])
    for bid, v in default.items():
        if typ(bid): w.writerow([typ(bid), bid, v])

# paper's BO-tuned RBC on TEST (ceiling overlay)
bo = {r["building_id"]: float(r["episode_return"])
      for r in csv.DictReader(open("data/pertype_bo_fullyear_eval.csv"))}
json.dump(bo, open("data/_tuned_ceiling.json", "w"))

# per-building tuned oracles on TRAIN (baseline for the train-side figure)
tuned = json.load(open("data/tuned_train_baseline.json"))
with open("data/_tuned_train_baseline.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["building_type", "building_id", "baseline_return"])
    for bid, v in tuned.items():
        if typ(bid): w.writerow([typ(bid), bid, v])

# BC-warm eval with guaranteed building_type column
rows = list(csv.DictReader(open("data/transfer4_bcwarm_fullyear_eval.csv")))
with open("data/_amo_bcwarm_eval.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["building_type", "building_id", "episode_return"])
    for r in rows:
        bid = r["building_id"]
        w.writerow([r.get("building_type") or typ(bid), bid, r["episode_return"]])
print("derived inputs written")
EOF

port() { $PY figures/build_transfer_port_figure.py "$@"; }

echo "== fullyear (from-scratch, HORIZON-VERIFIED transfer4_final) =="
port --eval-csv data/transfer4_final_fullyear_eval.csv \
     --baseline-csv data/_default_baseline_fullyear.csv \
     --ours-label "Ours" --outdir figures_out_fullyear --prefix fullyear

echo "== bcwarm (test) =="
port --eval-csv data/_amo_bcwarm_eval.csv \
     --baseline-csv data/_default_baseline_fullyear.csv \
     --ours-label "Ours" --outdir figures_out_bcwarm --prefix bcwarm

echo "== tuned_vs_amorpheus (test, tuned-RBC overlay) =="
port --eval-csv data/_amo_bcwarm_eval.csv \
     --baseline-csv data/_default_baseline_fullyear.csv \
     --ceiling-json data/_tuned_ceiling.json \
     --ours-label "Amorpheus (BC-warm)" --ceiling-label "Tuned RBC" \
     --reference-label "Default RBC" --prefix tuned_vs_amorpheus

echo "== bcwarm_train_vs_tuned (train, vs per-building oracles) =="
port --eval-csv data/bcwarm_train_fullyear_eval.csv \
     --baseline-csv data/_tuned_train_baseline.csv \
     --ours-label "Amorpheus BC-warm" --reference-label "per-building tuned RBC" \
     --prefix bcwarm_train_vs_tuned

echo "== dagger y1 (test) =="
port --eval-csv data/dagger_stream_y1_test_fullyear_eval.csv \
     --baseline-csv data/_default_baseline_fullyear.csv \
     --ceiling-json data/_tuned_ceiling.json \
     --ours-label "Streaming DAgger (year 1)" --ceiling-label "Tuned RBC" \
     --reference-label "Default RBC" --prefix dagger_y1

if [ -s data/dagger_stream_y2_test_fullyear_eval.csv ]; then
  echo "== dagger y2 (test) =="
  port --eval-csv data/dagger_stream_y2_test_fullyear_eval.csv \
       --baseline-csv data/_default_baseline_fullyear.csv \
       --ceiling-json data/_tuned_ceiling.json \
       --ours-label "Streaming DAgger (year 2)" --ceiling-label "Tuned RBC" \
       --reference-label "Default RBC" --prefix dagger_y2
else
  echo "== dagger y2: SKIPPED (eval CSV not present yet) =="
fi

if [ -s data/spe_y2_test_fullyear_eval.csv ]; then
  echo "== SPE-y2 vs plain-y2 apples-to-apples (comfort, test) =="
  $PY - <<'EOF'
import csv, json
d = {r["building_id"]: float(r["episode_return"])
     for r in csv.DictReader(open("data/dagger_stream_y2_test_fullyear_eval.csv"))}
json.dump(d, open("data/_plain_y2_overlay.json", "w"))
EOF
  port --eval-csv data/spe_y2_test_fullyear_eval.csv \
       --baseline-csv data/_default_baseline_fullyear.csv \
       --ours-color "#c2439b" \
       --ceiling-json data/_plain_y2_overlay.json \
       --ceiling-label "DAgger y2 (plain)" --ceiling-color "#2a78d6" \
       --ceiling2-json data/_tuned_ceiling.json \
       --ceiling2-label "Tuned RBC" --ceiling2-color "#008300" \
       --ours-label "DAgger y2 + spectral PE" \
       --reference-label "Default RBC" --prefix spe_vs_plain_y2
fi

echo "== dagger ppoft (comfort fine-tune, test) =="
port --eval-csv data/dagger_ppoft_test_fullyear_eval.csv \
     --baseline-csv data/_default_baseline_fullyear.csv \
     --ceiling-json data/_tuned_ceiling.json \
     --ours-label "DAgger + PPO fine-tune" --ceiling-label "Tuned RBC" \
     --reference-label "Default RBC" --prefix dagger_ppoft

echo "== comfort summary (all methods vs tuned RBC, one panel) =="
$PY figures/build_comfort_summary_figure.py

echo "== trace diagnostics (policy vs RBC actions / fan speed) =="
$PY figures/build_policy_vs_rbc_actions_figure.py
$PY figures/build_fan_speed_figure.py

if [ -s data/emedtuned_rbc_test_fullyear_eval.csv ]; then
  echo "== emed ppoft vs ENERGY-tuned RBC (test) =="
  $PY - <<'EOF'
import csv, json
bo = {r["building_id"]: float(r["episode_return"])
      for r in csv.DictReader(open("data/emedtuned_rbc_test_fullyear_eval.csv"))}
json.dump(bo, open("data/_emedtuned_ceiling.json", "w"))
d = json.load(open("data/rbc_emed_fullyear.json"))
TYPES = ["RestaurantFastFood", "RetailStandalone", "OfficeMedium", "OfficeSmall"]
with open("data/_default_baseline_emed.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["building_type", "building_id", "baseline_return"])
    for bid, v in d.items():
        for t in TYPES:
            if bid.startswith(t): w.writerow([t, bid, v])
EOF
  # (emed_ppoft figure DROPPED 2026-08-07 by user decision: complicated
  #  4-stage lineage, superseded results; the anchoring finding lives in text.
  #  Its eval CSV data/emed_ppoft_test_fullyear_eval.csv stays for the record.)

  if [ -s data/emed_fromscratch_test_fullyear_eval.csv ]; then
    echo "== from-scratch PPO vs energy-tuned RBC (test) =="
    port --eval-csv data/emed_fromscratch_test_fullyear_eval.csv \
         --baseline-csv data/_default_baseline_emed.csv \
         --ceiling-json data/_emedtuned_ceiling.json \
         --ours-label "From-scratch PPO (2.6 y)" --ceiling-label "Energy-tuned RBC" \
         --reference-label "Default RBC" --prefix emed_fromscratch_vs_tuned
  fi

  if [ -s data/emed_fromscratch_5y_test_fullyear_eval.csv ]; then
    echo "== from-scratch PPO continuation (5.1y) vs energy-tuned RBC (test) =="
    port --eval-csv data/emed_fromscratch_5y_test_fullyear_eval.csv \
         --baseline-csv data/_default_baseline_emed.csv \
         --ceiling-json data/_emedtuned_ceiling.json \
         --ours-label "From-scratch PPO (5.1 y)" --ceiling-label "Energy-tuned RBC" \
         --reference-label "Default RBC" --prefix emed_fromscratch_5y_vs_tuned
  fi

  if [ -s data/emed_fromscratch_5y_test_fullyear_eval.csv ]; then
    echo "== from-scratch 2.6y + 5.1y + tuned RBC, single figure =="
    $PY - <<'EOF'
import csv, json
d = {r["building_id"]: float(r["episode_return"])
     for r in csv.DictReader(open("data/emed_fromscratch_test_fullyear_eval.csv"))}
json.dump(d, open("data/_fromscratch_2p6y_overlay.json", "w"))
EOF
    port --eval-csv data/emed_fromscratch_5y_test_fullyear_eval.csv \
         --baseline-csv data/_default_baseline_emed.csv \
         --ceiling-json data/_fromscratch_2p6y_overlay.json \
         --ceiling-label "From-scratch PPO (2.6 y)" --ceiling-color "#c2439b" \
         --ceiling2-json data/_emedtuned_ceiling.json \
         --ceiling2-label "Energy-tuned RBC" --ceiling2-color "#008300" \
         --ours-label "From-scratch PPO (5.1 y)" \
         --reference-label "Default RBC" --prefix emed_fromscratch_2p6y_vs_5y
  fi

  echo "== energy-tuned RBC vs default RBC (energy-section opener) =="
  port --eval-csv data/emedtuned_rbc_test_fullyear_eval.csv \
       --baseline-csv data/_default_baseline_emed.csv \
       --ours-label "Energy-tuned RBC" --reference-label "Default RBC" \
       --prefix emedtuned_rbc
fi

if [ -s data/branch_compare_OfficeMedium_test0.npz ]; then
  echo "== branched-rollout mechanism figures (OfficeMedium energy dominance) =="
  # data from a real branched rollout (same initial state, each controller
  # on-policy in its own branch):
  #   $PY scripts/rollout_branch_compare.py --building-type OfficeMedium \
  #       --index 0 --steps 2016 --with-default-rbc
  $PY figures/build_branch_compare_figure.py
fi

echo "== vav_supply symmetry scatter (loops bit-identical along a rollout) =="
# data/supply_symmetry_rollout.npz comes from a real rollout:
#   $PY scripts/check_supply_symmetry.py --steps 1000 \
#       --checkpoint runs/dagger_stream_e0_amorpheus_y2/model_s0.eqx \
#       --dump data/supply_symmetry_rollout.npz
$PY figures/build_supply_symmetry_figure.py
if [ -s data/supply_symmetry_rollout_spe.npz ]; then
  # companion: same rollout protocol on the TRAINED SPE checkpoint -- off-diagonal
  $PY figures/build_supply_symmetry_figure.py \
      data/supply_symmetry_rollout_spe.npz supply_symmetry_scatter_spe
fi

echo "ALL_THESIS_FIGURES_DONE"
