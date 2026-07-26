"""Collect RBC demonstrations in the PLAIN GYM action space (for the MLP clone).

Unlike bc_collect.py (which pseudo-inverts RBC into Amorpheus's per-node normalized
space), the MLP acts directly on the env's flat Box obs/action. So we just run the
correct reactive controller and record, per step: the raw gym observation, the RBC
action it takes (which goes straight into env.step), and the reward (for the critic
returns-to-go). One npz per (test) building.

    python scripts/bc_collect_mlp.py --split test --n-buildings 5 --n-workers 4
"""
import argparse
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("vendor/morel", "vendor/Building2Building", "scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import numpy as np

import building2building as b2b
from baselines.controllers import AirLoopPolicy, UnitaryHvacPolicy

TASK, RP = "task_occ_e0", "full_year"
YEAR_STEPS = 105_120
OUTDIR = os.path.join(REPO, "data", "bc_demos_mlp")


def _controller(bt):
    return AirLoopPolicy() if bt == "OfficeMedium" else UnitaryHvacPolicy()


def collect_one(args):
    bt, idx, split, steps, stride = args
    env = b2b.make_env(bt, split=split, index=idx, task=TASK, run_period=RP)
    lo = np.asarray(env.action_space.low, np.float32)
    hi = np.asarray(env.action_space.high, np.float32)
    pol = _controller(bt); pol.bind_env(env)
    raw, _ = env.reset(); pol.reset()
    obs_t, act_t, rew = [], [], []
    for t in range(steps):
        keep = (t % stride) == 0
        a, _ = pol.predict(raw)
        a = np.asarray(a, dtype=np.float32)
        if keep:
            obs_t.append(np.asarray(raw, dtype=np.float32))
            act_t.append(a)
        raw, r, term, trunc, _ = env.step(a)
        rew.append(float(r))
        if term or trunc:
            break
    env.close()
    obs = np.stack(obs_t).astype(np.float32)
    act = np.stack(act_t).astype(np.float32)
    out = os.path.join(OUTDIR, f"{bt}_{idx}.npz")
    np.savez_compressed(out, obs=obs, act=act, rew=np.asarray(rew, np.float64),
                        stride=stride, lo=lo, hi=hi)
    return bt, idx, obs.shape[0], obs.shape[1], act.shape[1], len(rew)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-types", nargs="+",
                    default=["RetailStandalone", "RestaurantFastFood",
                             "OfficeMedium", "OfficeSmall"])
    ap.add_argument("--n-buildings", type=int, default=5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--steps", type=int, default=YEAR_STEPS)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n-workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    tasks = [(bt, i, args.split, args.steps, args.stride)
             for i in range(args.n_buildings) for bt in args.building_types]
    print(f"[collect] {len(tasks)} buildings x {args.steps} steps (stride "
          f"{args.stride}) split={args.split} | {args.n_workers} workers", flush=True)
    with mp.get_context("spawn").Pool(args.n_workers) as pool:
        for k, (bt, idx, n, od, ad, nr) in enumerate(
                pool.imap_unordered(collect_one, tasks), 1):
            print(f"  [{k:2d}/{len(tasks)}] {bt}_{idx}: {n} samples, obs {od}, "
                  f"act {ad}, {nr} steps", flush=True)
    print("BC_COLLECT_MLP_DONE")


if __name__ == "__main__":
    main()
