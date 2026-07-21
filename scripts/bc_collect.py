"""Phase A of BC-from-RBC: collect demonstrations across all building types.

One parallel pass per building recording, at every step:
  - the per-node observation the network sees
  - the TARGET action in the policy's own (normalized) action space, obtained by
    probing the affine target->physical map and pseudo-inverting RBC's command
  - the reward (so returns-to-go can pretrain the critic from the same data)

Uses the correct reactive controller per type (AirLoopPolicy for OfficeMedium/VAV,
UnitaryHvacPolicy otherwise) -- including the fixed VAV controller that tracks the
task's dynamic setpoint.

Each worker writes its own .npz (large arrays never cross the Pool boundary).
Phase B (bc_fit.py) fits a fresh net to these.
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

from baselines.controllers import AirLoopPolicy, UnitaryHvacPolicy
from morel import Unit
from morel.morphology import trivial_morphology
from morel_amorpheus.policy import decode_flat_action, encode_local_obs
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, make_model
from evaluate import _b2b_factory_impl

TASK, RP = "task_occ_e0", "full_year"
YEAR_STEPS = 105_120
OUTDIR = os.path.join(REPO, "data", "bc_demos")


def _controller(bt):
    return AirLoopPolicy() if bt == "OfficeMedium" else UnitaryHvacPolicy()


def collect_one(args):
    bt, idx, steps, stride = args
    out = os.path.join(OUTDIR, f"{bt}_{idx}.npz")
    env, source = _b2b_factory_impl(bt, idx, "train", TASK, RP)
    m = AMORPHEUS_B2B_BRIDGE.apply(source)
    lens = AMORPHEUS_B2B_BRIDGE.apply(trivial_morphology(source))
    cond = make_model(d_model=64, n_heads=4, n_layers=3, seed=0).condition(m)
    nids, aws = cond.node_ids, cond.act_widths
    D = int(sum(aws))

    # Probe the affine target->physical action map, then pseudo-invert it.
    def join_flat(f):
        dd = decode_flat_action(m, nids, aws, np.asarray(f, np.float32))
        return np.asarray(source.join_actions(lens.join_actions((Unit, dd))),
                          dtype=np.float64)
    c = join_flat(np.zeros(D))
    A = np.stack([join_flat(np.eye(D)[j]) - c for j in range(D)], axis=1)
    A_pinv = np.linalg.pinv(A)

    pol = _controller(bt); pol.bind_env(env)
    raw, _ = env.reset(); pol.reset()
    obs_t, act_t, rew = [], [], []
    for t in range(steps):
        keep = (t % stride) == 0
        if keep:
            _c, per_node = lens.split_observation(source.split_observation(raw))
            obs_t.append([np.asarray(o, dtype=np.float32)
                          for o in encode_local_obs(m, nids, per_node)])
        a, _ = pol.predict(raw)
        a = np.asarray(a, dtype=np.float64)
        if keep:
            act_t.append(a)
        raw, r, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32))
        rew.append(float(r))
        if term or trunc:
            break
    env.close()

    a_arr = np.stack(act_t)
    f_target = np.clip((a_arr - c) @ A_pinv.T, -1.0, 1.0).astype(np.float32)
    n_nodes = len(obs_t[0])
    obs_stack = [np.stack([o[i] for o in obs_t]).astype(np.float32)
                 for i in range(n_nodes)]
    rew = np.asarray(rew, dtype=np.float64)

    np.savez_compressed(
        out, f_target=f_target, rew=rew, stride=stride,
        recon_err=float(np.abs((f_target @ A.T) + c - a_arr).mean()),
        **{f"obs{i}": o for i, o in enumerate(obs_stack)})
    return bt, idx, len(f_target), len(rew), float(np.abs((f_target @ A.T) + c - a_arr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-types", nargs="+",
                    default=["RetailStandalone", "RestaurantFastFood",
                             "OfficeMedium", "OfficeSmall"])
    ap.add_argument("--n-buildings", type=int, default=5)
    ap.add_argument("--steps", type=int, default=YEAR_STEPS,
                    help="steps per building (default: one full year)")
    ap.add_argument("--stride", type=int, default=4,
                    help="keep every Nth sample for fitting (rewards kept at full rate)")
    ap.add_argument("--n-workers", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    tasks = [(bt, i, args.steps, args.stride)
             for i in range(args.n_buildings) for bt in args.building_types]
    print(f"[collect] {len(tasks)} buildings x {args.steps} steps "
          f"(stride {args.stride}) | {args.n_workers} workers", flush=True)
    with mp.get_context("spawn").Pool(args.n_workers) as pool:
        for k, (bt, idx, n_s, n_r, err) in enumerate(
                pool.imap_unordered(collect_one, tasks), 1):
            print(f"  [{k:2d}/{len(tasks)}] {bt}_{idx}: {n_s} samples, {n_r} steps, "
                  f"recon_err {err:.4f}", flush=True)
    print("BC_COLLECT_DONE")


if __name__ == "__main__":
    main()
