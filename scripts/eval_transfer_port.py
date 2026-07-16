"""Held-out eval of the ported transfer policies (current stack).

For each of the 3 seed checkpoints, evaluate on 5 held-out (test-split)
buildings per type, full-year episode, task_occ_e0 — the same task the models
trained on. Scores against the published RBC baseline table
(baseline_returns.csv) via attach_baseline_scores. Writes per-(seed,building)
returns to transfer_port_eval.csv for the figure.
"""
from __future__ import annotations

import functools
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# morel ships some helpers as repo-root modules (evaluate.py, run_training.py)
# that are not part of the installed package -> take them from the pinned
# morel submodule.
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))

import _pin_dataset  # noqa: F401  pins the b2b dataset revision before any env
import pandas as pd

from building2building.data.registry import get_registry

from morel.evaluation import Run, run_evaluation
from morel_amorpheus import amorpheus_policy
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, load_model

from evaluate import _b2b_factory_impl, attach_baseline_scores

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
SEEDS = {s: os.path.join(REPO, "checkpoints", f"model_s{s}.eqx") for s in (0, 1, 2)}
TASK = "task_occ_e0"
RUN_PERIOD = "full_year"
N_TEST = 5
# Evaluate on exactly the chunk the cold-start policy trained on: the first
# n_steps (=672) of the episode from a fresh reset. The baseline is evaluated
# on the SAME chunk (see baseline_chunk.py) — a fair, in-distribution compare.
CHUNK = 672
D_MODEL, N_HEADS, N_LAYERS = 64, 4, 3
OUT = os.path.join(REPO, "data")                       # transfer_port_eval.csv (figure input)
STORE = os.path.join(REPO, "runs", "transfer_port_eval_store")  # eval cache (gitignored)
N_WORKERS = 4


def _policy_factory(ckpt: str):
    return amorpheus_policy(
        load_model(ckpt, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
    )


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(STORE, exist_ok=True)
    reg = get_registry()
    frames = []
    for seed, ckpt in SEEDS.items():
        runs = []
        for t in TYPES:
            for idx in range(N_TEST):
                bid = reg.get_building_by_index(t, "test", idx).building_id
                runs.append(Run(
                    metadata={"building_type": t, "split": "test", "index": idx,
                              "building_id": bid, "task": TASK,
                              "run_period": RUN_PERIOD, "seed": seed},
                    factory=functools.partial(
                        _b2b_factory_impl, t, idx, "test", TASK, RUN_PERIOD),
                ))
        print(f"[seed {seed}] evaluating {len(runs)} held-out buildings ...", flush=True)
        df = run_evaluation(
            runs, morphism=AMORPHEUS_B2B_BRIDGE,
            policy_factory=functools.partial(_policy_factory, ckpt),
            max_steps=CHUNK, store_dir=f"{STORE}/store_s{seed}", n_workers=N_WORKERS,
        )
        df["seed"] = seed
        frames.append(df)
        print(f"[seed {seed}] done, mean return {df['episode_return'].mean():.1f}", flush=True)

    res = pd.concat(frames, ignore_index=True)
    # Baseline is computed on the same chunk by baseline_chunk.py (full-year
    # baseline_returns.csv would be a length/regime mismatch), so no
    # attach_baseline_scores here.
    res.to_csv(f"{OUT}/transfer_port_eval.csv", index=False)
    print(f"[done] wrote {OUT}/transfer_port_eval.csv ({len(res)} rows)", flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
