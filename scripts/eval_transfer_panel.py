"""Eval a checkpoint across all 4 building types on held-out test buildings vs RBC.

    python scripts/eval_transfer_panel.py --checkpoint runs/transfer4_persistent/model_s0.eqx

Rolls out the policy for CHUNK steps on 5 test-split buildings per type and compares
to the precomputed RBC table (data/baseline_chunk.csv), which was measured on the
same chunk. Prints per-building rows and a per-type summary.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import numpy as np
import pandas as pd

from building2building.data.registry import get_registry
from morel_amorpheus import amorpheus_policy
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, load_model
from morel.morphology import trivial_morphology
from evaluate import _b2b_factory_impl

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
TASK, RP, CHUNK, N = "task_occ_e0", "full_year", 672, 5


def policy_return(ckpt, bt, idx):
    env, source = _b2b_factory_impl(bt, idx, "test", TASK, RP)
    lens = AMORPHEUS_B2B_BRIDGE.apply(trivial_morphology(source))
    sp = amorpheus_policy(load_model(ckpt, d_model=64, n_heads=4, n_layers=3))(
        AMORPHEUS_B2B_BRIDGE.apply(source))
    raw, _ = env.reset()
    ret = 0.0
    for _t in range(CHUNK):
        tca, tpa = sp(*lens.split_observation(source.split_observation(raw)))
        a = np.asarray(source.join_actions(lens.join_actions((tca, tpa))), dtype=np.float64)
        raw, r, term, trunc, _ = env.step(a)
        ret += float(r)
        if term or trunc:
            break
    env.close()
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(REPO, args.checkpoint)

    base = pd.read_csv(os.path.join(REPO, "data", "baseline_chunk.csv"))
    rbc = {r.building_id: r.baseline_return for r in base.itertuples()}
    reg = get_registry()

    summary = {}
    for bt in TYPES:
        print(f"\n=== {bt} (test) ===")
        rows, wins = [], 0
        for idx in range(N):
            bid = reg.get_building_by_index(bt, "test", idx).building_id
            p = policy_return(ckpt, bt, idx)
            b = rbc.get(bid, float("nan"))
            w = p > b
            wins += bool(w)
            rows.append((p, b))
            print(f"  {bid:26s} policy {p:9.2f}  RBC {b:9.2f}  {'POLICY' if w else 'rbc'}")
        pm = float(np.mean([r[0] for r in rows])); bm = float(np.mean([r[1] for r in rows]))
        summary[bt] = (pm, bm, wins)
        print(f"  -> mean policy {pm:.2f} vs RBC {bm:.2f} | beats RBC on {wins}/{N}")

    print("\n=== SUMMARY (mean over 5 test buildings) ===")
    tot = 0
    for bt, (pm, bm, w) in summary.items():
        tot += w
        print(f"  {bt:22s} policy {pm:9.2f}  RBC {bm:9.2f}  wins {w}/{N}")
    print(f"  TOTAL wins: {tot}/{len(TYPES)*N}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
