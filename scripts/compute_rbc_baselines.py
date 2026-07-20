"""Compute RBC episode returns for a building pool, save as JSON.

Uses the correct reactive controller per type (AirLoopPolicy for OfficeMedium/VAV,
UnitaryHvacPolicy otherwise -- same dispatch as baseline_chunk.py).

    python scripts/compute_rbc_baselines.py --split train --n 5 \
        --building-types RetailStandalone RestaurantFastFood OfficeMedium OfficeSmall \
        --out data/transfer4_train_rbc_baseline.json

Output {"<type>_<idx>": return, ..., "episode_return": pool_mean} feeds
train_transfer_port.py --baseline-json (constant wandb reference lines) and
annotates status tables with the baseline each building must beat.
"""
import argparse
import json
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
from baselines.controllers import AirLoopPolicy, UnitaryHvacPolicy

TASK, RP, CHUNK = "task_occ_e0", "full_year", 672


def _controller(building_type: str):
    return AirLoopPolicy() if building_type == "OfficeMedium" else UnitaryHvacPolicy()


def rbc_return(bt, split, idx):
    env = b2b.make_env(bt, split=split, index=idx, task=TASK, run_period=RP)
    pol = _controller(bt); pol.bind_env(env)
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
    ap.add_argument("--building-types", nargs="+",
                    default=["RetailStandalone", "RestaurantFastFood",
                             "OfficeMedium", "OfficeSmall"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="data/transfer4_train_rbc_baseline.json")
    args = ap.parse_args()

    base = {}
    for bt in args.building_types:
        for idx in range(args.n):
            k = f"{bt}_{idx}"
            base[k] = rbc_return(bt, args.split, idx)
            print(f"  {k}: {base[k]:.2f}", flush=True)
    base["episode_return"] = float(np.mean([v for k, v in base.items()]))

    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(base, f, indent=2)
    print(f"[done] pool mean {base['episode_return']:.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
