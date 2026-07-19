"""Compute RBC episode returns for the OfficeSmall train pool, save as JSON.

Produces data/officesmall_train_rbc_baseline.json = {"OfficeSmall_<idx>": return, ...,
"episode_return": pool_mean}. Feed it to train_transfer_port.py --baseline-json so
each per-building plot gets a constant RBC reference line on wandb.
"""
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
from baselines.controllers import UnitaryHvacPolicy

TASK, RP, CHUNK, POOL = "task_occ_e0", "full_year", 672, 10
OUT = os.path.join(REPO, "data", "officesmall_train_rbc_baseline.json")


def rbc_return(idx):
    env = b2b.make_env("OfficeSmall", split="train", index=idx, task=TASK, run_period=RP)
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
    base = {}
    for idx in range(POOL):
        base[f"OfficeSmall_{idx}"] = rbc_return(idx)
        print(f"  OfficeSmall_{idx}: {base[f'OfficeSmall_{idx}']:.2f}", flush=True)
    base["episode_return"] = float(np.mean([base[f"OfficeSmall_{i}"] for i in range(POOL)]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(base, f, indent=2)
    print(f"[done] pool mean {base['episode_return']:.2f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
