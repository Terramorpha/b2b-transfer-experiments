"""Full-year held-out eval across all 4 building types vs the full-year RBC baseline.

Unlike eval_transfer_panel.py (which scores the first 672-step chunk = the year's
hardest week, Jan 1-7), this rolls the FULL YEAR so the numbers are annual averages.
Scores against building2building/scores/baseline_returns.csv (task_occ_e0, full_year).

    python scripts/eval_fullyear_panel.py --checkpoint runs/transfer4_persistent_mid/model_s0.eqx

Writes data/<tag>_fullyear_eval.csv and data/baseline_fullyear.csv for the figure.
"""
import argparse
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
import pandas as pd

from building2building.data.registry import get_registry
from morel_amorpheus import amorpheus_policy
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, load_model
from morel.morphology import trivial_morphology
from evaluate import _b2b_factory_impl

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
TASK, RP, N, MAX_STEPS = "task_occ_e0", "full_year", 5, 106000  # full year = 105_120 steps
BASELINE_TABLE = os.path.join(
    REPO, "vendor", "Building2Building", "building2building", "scores",
    "baseline_returns.csv")


def _worker(args):
    """Spawn-pool worker: one full-year rollout in its own process (EnergyPlus-safe).
    Top-level + picklable so it survives spawn."""
    ckpt, bt, idx = args
    ret, steps = policy_return(ckpt, bt, idx)
    return bt, idx, ret, steps


def policy_return(ckpt, bt, idx):
    env, source = _b2b_factory_impl(bt, idx, "test", TASK, RP)
    lens = AMORPHEUS_B2B_BRIDGE.apply(trivial_morphology(source))
    sp = amorpheus_policy(load_model(ckpt, d_model=64, n_heads=4, n_layers=3))(
        AMORPHEUS_B2B_BRIDGE.apply(source))
    raw, _ = env.reset()
    ret, steps = 0.0, 0
    for _t in range(MAX_STEPS):
        tca, tpa = sp(*lens.split_observation(source.split_observation(raw)))
        a = np.asarray(source.join_actions(lens.join_actions((tca, tpa))), dtype=np.float64)
        raw, r, term, trunc, _ = env.step(a)
        ret += float(r); steps += 1
        if term or trunc:
            break
    env.close()
    return ret, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="transfer4_mid")
    ap.add_argument("--n-workers", type=int, default=3,
                    help="parallel rollout processes (one building each)")
    args = ap.parse_args()
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(REPO, args.checkpoint)

    b = pd.read_csv(BASELINE_TABLE)
    b = b[(b.task == TASK) & (b.run_period == RP)]
    base = dict(zip(b.building_id, b.reward_mean))
    reg = get_registry()

    # idx-major ordering => the first wave is one building of EACH type, so a
    # first cross-type read arrives early instead of after all of Retail.
    tasks = [(ckpt, bt, idx) for idx in range(N) for bt in TYPES]
    ids = {(bt, idx): reg.get_building_by_index(bt, "test", idx).building_id
           for bt in TYPES for idx in range(N)}

    rows, brows = [], []
    print(f"[eval] {len(tasks)} full-year rollouts | {args.n_workers} parallel "
          f"processes | one-per-type first", flush=True)
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for k, (bt, idx, p, steps) in enumerate(
                pool.imap_unordered(_worker, tasks), 1):
            bid = ids[(bt, idx)]
            rb = base.get(bid, float("nan"))
            rows.append({"building_type": bt, "building_id": bid,
                         "episode_return": p, "n_steps": steps, "seed": 0})
            brows.append({"building_type": bt, "building_id": bid,
                          "baseline_return": rb})
            print(f"  [{k:2d}/{len(tasks)}] {bid:26s} policy {p:10.1f}  "
                  f"RBC {rb:10.1f}  ({steps} steps)  "
                  f"{'POLICY' if p > rb else 'rbc'}", flush=True)

    df = pd.DataFrame(rows); bdf = pd.DataFrame(brows)
    out = os.path.join(REPO, "data", f"{args.tag}_fullyear_eval.csv")
    bout = os.path.join(REPO, "data", "baseline_fullyear.csv")
    df.to_csv(out, index=False); bdf.to_csv(bout, index=False)

    print("\n=== SUMMARY (full year) ===")
    tot = 0
    for bt in TYPES:
        s = df[df.building_type == bt]; sb = bdf[bdf.building_type == bt]
        w = int((s.episode_return.to_numpy() > sb.baseline_return.to_numpy()).sum())
        tot += w
        norm = (s.episode_return.to_numpy() / sb.baseline_return.to_numpy()).mean()
        print(f"  {bt:22s} policy {s.episode_return.mean():10.1f}  "
              f"RBC {sb.baseline_return.mean():10.1f}  norm {norm:.3f}  wins {w}/{N}")
    print(f"  TOTAL wins: {tot}/{len(TYPES)*N}")
    print(f"[done] {out}\nEVAL_DONE")


if __name__ == "__main__":
    main()
