"""Full-year DEFAULT-RBC baseline on the TEST split, keyed by building_id -> return.

The energy-aware (task_occ_emed) analogue of data/rbc_fullyear_ourharness.json
(which is the same default RBC on task_occ_e0). Uses the BARE reactive controller
per type -- no tuned config -- exactly the dispatch in compute_rbc_baselines.py
(AirLoopPolicy for OfficeMedium/VAV, UnitaryHvacPolicy otherwise), rolled one
complete EnergyPlus year per building. Output format matches
rbc_fullyear_ourharness.json ({building_id: return}) so
eval_fullyear_panel_modumorph.py --baselines can compare energy policies to it.

    python scripts/compute_rbc_emed_baseline.py --n-workers 3
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
from baselines.controllers import AirLoopPolicy, UnitaryHvacPolicy
from building2building.data.registry import get_registry

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
RP, N, MAX_STEPS = "full_year", 5, 106000  # full year = 105_120 steps


def _controller(building_type: str):
    """Bare default reactive controller (no tuned config) -- same dispatch as
    compute_rbc_baselines.py / baseline_chunk.py."""
    return AirLoopPolicy() if building_type == "OfficeMedium" else UnitaryHvacPolicy()


def rbc_return(bt, idx, task):
    env = b2b.make_env(bt, split="test", index=idx, task=task, run_period=RP)
    pol = _controller(bt); pol.bind_env(env)
    raw, _ = env.reset(); pol.reset()
    ret, steps = 0.0, 0
    for _t in range(MAX_STEPS):
        act, _ = pol.predict(raw)
        raw, r, term, trunc, _ = env.step(np.asarray(act, dtype=np.float32))
        ret += float(r); steps += 1
        if term or trunc:
            break
    env.close()
    return ret, steps


def _worker(args):
    bt, idx, task = args
    r, s = rbc_return(bt, idx, task)
    return bt, idx, r, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_occ_emed",
                    help="b2b task preset (default energy task_occ_emed)")
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--out", default="data/rbc_emed_fullyear.json")
    args = ap.parse_args()

    reg = get_registry()
    tasks = [(bt, idx, args.task) for idx in range(N) for bt in TYPES]
    ids = {(bt, idx): reg.get_building_by_index(bt, "test", idx).building_id
           for bt in TYPES for idx in range(N)}

    out = {}
    print(f"[rbc-baseline] task={args.task} | {len(tasks)} full-year default-RBC "
          f"rollouts | {args.n_workers} workers", flush=True)
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for k, (bt, idx, r, s) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            bid = ids[(bt, idx)]
            out[bid] = r
            print(f"  [{k:2d}/{len(tasks)}] {bid:26s} return {r:10.1f}  ({s} steps)",
                  flush=True)

    path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {path}\nRBC_EMED_DONE", flush=True)


if __name__ == "__main__":
    main()
