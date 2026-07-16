"""RBC baseline on the SAME first-672-step chunk the transfer policies trained
on (cold start), for a fair in-distribution comparison. Per-HVAC dispatch:
UnitaryHvacPolicy for PSZ types, AirLoopPolicy for OfficeMedium (VAV). Writes
baseline_chunk.csv (per-building baseline_return over the chunk).
"""
from __future__ import annotations

import os
import sys

# The `baselines` package lives at the b2b repo root, not inside the installed
# wheel -> take it from the pinned building2building submodule.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401  pins the b2b dataset revision before any env
import multiprocessing as mp

import numpy as np
import pandas as pd

import building2building as b2b
from building2building.data.registry import get_registry
from baselines.controllers import AirLoopPolicy, UnitaryHvacPolicy

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
TASK = "task_occ_e0"
RUN_PERIOD = "full_year"
N_TEST = 5
CHUNK = 672
OUT = os.path.join(REPO, "data")


def _controller(building_type: str):
    return AirLoopPolicy() if building_type == "OfficeMedium" else UnitaryHvacPolicy()


def _job(args):
    building_type, index, bid = args
    try:
        env = b2b.make_env(building_type, split="test", index=index,
                           task=TASK, run_period=RUN_PERIOD)
        pol = _controller(building_type)
        pol.bind_env(env)
        obs, _ = env.reset()
        pol.reset()
        total = 0.0
        for _t in range(CHUNK):
            action, _ = pol.predict(obs)
            obs, reward, term, trunc, _ = env.step(np.asarray(action, dtype=np.float32))
            total += float(reward)
            if term or trunc:
                break
        env.close()
        return building_type, bid, total, None
    except Exception as e:  # noqa: BLE001
        return building_type, bid, None, f"{type(e).__name__}: {e}"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    reg = get_registry()
    jobs = []
    for t in TYPES:
        for idx in range(N_TEST):
            bid = reg.get_building_by_index(t, "test", idx).building_id
            jobs.append((t, idx, bid))
    print(f"RBC baseline on first {CHUNK} steps, {len(jobs)} buildings", flush=True)
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(processes=4) as p:
        for bt, bid, ret, err in p.imap_unordered(_job, jobs):
            if err is None:
                rows.append({"building_type": bt, "building_id": bid, "baseline_return": ret})
                print(f"  {bt} {bid}: {ret:.1f}", flush=True)
            else:
                print(f"  FAIL {bt} {bid}: {err}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/baseline_chunk.csv", index=False)
    print("BASELINE_CHUNK_DONE", flush=True)


if __name__ == "__main__":
    main()
