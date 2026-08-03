"""Full-year eval of a NO-DEADBAND (comfort-optimized) unitary RBC on the test
split, to measure how much of the default RBC baseline's comfort slack comes from
its deadband/band.

Only the unitary types (Retail, Restaurant, OfficeSmall) are relevant: the VAV
default (OfficeMedium) already has deadband=0, so there is nothing to remove.

The default unitary controller carries a fixed +/-1 C setpoint band (heat 20 /
cool 22 around a 21 C target), a +/-0.5 C demand_deadband, and idles the fan
inside the band (nearest_setpoint). This variant collapses all three: band ->
+/-0.1 C, demand_deadband -> 0.05, fan active for any deviation (center_of_band).
Every other gain is left at default so we isolate the *deadband* effect, not a
re-tune.

    python scripts/eval_fullyear_rbc_nodeadband.py --n-workers 2
"""
import argparse
import json
import multiprocessing as mp
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
from baselines.controllers import UnitaryHvacPolicy, UnitaryHvacConfig
from building2building.data.registry import get_registry

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeSmall"]
TASK, RP, N, MAX_STEPS = "task_occ_e0", "full_year", 5, 106000


def _nodeadband_cfg() -> UnitaryHvacConfig:
    return UnitaryHvacConfig(
        heating_setpoint_c=20.9,   # collapse the +/-1 C band around the 21 C target
        cooling_setpoint_c=21.1,   # (constructor requires cooling > heating)
        demand_deadband=0.05,      # near-zero Trim-and-Respond deadband
        fan_error_mode="center_of_band",   # fan active for ANY deviation from target
    )


def rbc_return(bt, idx):
    env = b2b.make_env(bt, split="test", index=idx, task=TASK, run_period=RP)
    pol = UnitaryHvacPolicy(_nodeadband_cfg())
    pol.bind_env(env)
    raw, _ = env.reset()
    pol.reset()
    ret, steps = 0.0, 0
    for _t in range(MAX_STEPS):
        act, _ = pol.predict(raw)
        raw, r, term, trunc, _ = env.step(np.asarray(act, dtype=np.float32))
        ret += float(r)
        steps += 1
        if term or trunc:
            break
    env.close()
    return ret, steps


def _worker(args):
    bt, idx = args
    r, s = rbc_return(bt, idx)
    return bt, idx, r, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--out", default="data/rbc_fullyear_nodeadband.json")
    args = ap.parse_args()

    reg = get_registry()
    tasks = [(bt, idx) for idx in range(N) for bt in TYPES]
    ids = {(bt, idx): reg.get_building_by_index(bt, "test", idx).building_id
           for bt in TYPES for idx in range(N)}

    out = {}
    print(f"[eval] {len(tasks)} full-year no-deadband RBC rollouts | "
          f"{args.n_workers} workers", flush=True)
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for k, (bt, idx, r, s) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            bid = ids[(bt, idx)]
            out[bid] = r
            print(f"  [{k:2d}/{len(tasks)}] {bid:26s} return {r:10.1f}  ({s} steps)",
                  flush=True)

    path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {path}\nEVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
