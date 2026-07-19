"""Eval an Amorpheus checkpoint on the OfficeSmall train+test panel vs RBC.

    python scripts/eval_officesmall_panel.py --checkpoint runs/officesmall_allactive/model_s0.eqx

Rolls out policy and RBC (UnitaryHvac) over the 672-step chunk on 5 train and 5
held-out test buildings; reports per-building return and how many the policy wins.
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

import building2building as b2b
from baselines.controllers import UnitaryHvacPolicy
from building2building.data.registry import get_registry
from morel_amorpheus import amorpheus_policy
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, load_model
from morel.morphology import trivial_morphology
from evaluate import _b2b_factory_impl

TASK, RP, CHUNK, N = "task_occ_e0", "full_year", 672, 5


def policy_return(ckpt, split, idx):
    env, source = _b2b_factory_impl("OfficeSmall", idx, split, TASK, RP)
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


def rbc_return(split, idx):
    env = b2b.make_env("OfficeSmall", split=split, index=idx, task=TASK, run_period=RP)
    pol = UnitaryHvacPolicy(); pol.bind_env(env)
    raw, _ = env.reset(); pol.reset()
    ret = 0.0
    for _t in range(CHUNK):
        act, _ = pol.predict(raw)
        raw, r, term, trunc, _ = env.step(np.asarray(act, dtype=np.float32))
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

    reg = get_registry()
    for split in ("train", "test"):
        print(f"\n=== OfficeSmall {split.upper()} ===")
        wins = 0
        for idx in range(N):
            bid = reg.get_building_by_index("OfficeSmall", split, idx).building_id
            p = policy_return(ckpt, split, idx); rb = rbc_return(split, idx)
            w = p > rb; wins += w
            print(f"  {bid:22s} policy {p:8.2f}  RBC {rb:8.2f}  {'POLICY' if w else 'rbc'}")
        print(f"  -> policy beats RBC on {wins}/{N} {split} buildings")


if __name__ == "__main__":
    main()
