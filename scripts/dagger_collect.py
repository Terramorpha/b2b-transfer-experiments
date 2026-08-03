"""DAgger (Dataset Aggregation) collection for a ModuMorph student <- RBC expert.

Shadow-mode DAgger: at every step the (optional) ModuMorph student proposes an
action AND the RBC expert is queried (which also advances the RBC's internal
state -- PI integrators, T&R setpoint -- on the student-induced observation
stream). A per-step Bernoulli(beta) coin decides who actually drives the env:
with prob beta the expert acts, else the student acts. The LABEL recorded is
ALWAYS the expert's action (inverted into the policy's normalized action space
via the same affine A/A_pinv probe used by bc_collect_tuned).

beta = 1.0 with no student == pure-expert collection == plain behavior cloning
(this is DAgger iteration 0). Later iterations pass a student checkpoint and a
smaller beta so the dataset aggregates expert labels on states the student
actually visits.

Each worker writes one npz per building, tagged with the iteration so successive
iterations ADD files (never overwrite): {bt}_{idx}_it{iteration}.npz. The npz
format is identical to bc_collect_tuned (f_target, rew, stride, recon_err,
obs{i}) so dagger_fit / bc_fit can consume the aggregated directory directly.
"""
import argparse
import multiprocessing as mp
import os
import sys
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import numpy as np

import json
from baselines.controllers import (
    AirLoopPolicy, UnitaryHvacPolicy, AirLoopConfig, UnitaryHvacConfig)
from morel import Unit
from morel.morphology import trivial_morphology
from morel_amorpheus import amorpheus_policy  # model-agnostic conditioned-policy wrapper
from morel_amorpheus.policy import decode_flat_action, encode_local_obs
from morel_b2b_modumorph import MODUMORPH_B2B_BRIDGE, make_model, load_model
from evaluate import _b2b_factory_impl

TASK, RP = "task_occ_e0", "full_year"
YEAR_STEPS = 105_120
CFG_DIR = os.path.join(REPO, "data", "rbc_tuned_configs")

_NONE = {"", "none", "None", "null"}


def _controller(bt, idx):
    """Load the per-(train-)building Optuna-tuned RBC config for this building.

    The returned policy's .predict(obs) MUTATES its internal state each call --
    essential for shadow mode: querying it every step keeps its integrators /
    setpoint tracker aligned with the (possibly student-induced) obs stream."""
    p = os.path.join(CFG_DIR, f"{bt}_train_{idx}.json")
    d = json.load(open(p))
    params = d["params"]
    if d["is_vav"]:
        return AirLoopPolicy(AirLoopConfig(**params))
    return UnitaryHvacPolicy(UnitaryHvacConfig(**params))


def collect_one(args):
    (bt, idx, steps, stride, beta, iteration, student_ckpt, variant,
     out_dir, task) = args
    out = os.path.join(out_dir, f"{bt}_{idx}_it{iteration}.npz")

    env, source = _b2b_factory_impl(bt, idx, "train", task, RP)
    # ModuMorph morphology (attributes stay in the hypernetwork channel) + its
    # trivial-morphology lens, exactly mirroring the modumorph eval path.
    m = MODUMORPH_B2B_BRIDGE.apply(source)
    lens = MODUMORPH_B2B_BRIDGE.apply(trivial_morphology(source))
    # node_ids / act_widths come from a MODUMORPH conditioned model (NOT amorpheus).
    cond0 = make_model(variant=variant, d_model=64, d_context=32,
                       n_heads=4, n_layers=3, seed=0).condition(m)
    nids, aws = cond0.node_ids, cond0.act_widths
    D = int(sum(aws))

    # Probe the affine target->physical action map, then pseudo-invert it. This
    # is the SAME probe as bc_collect_tuned; join is (Unit, per_node_actions).
    def join_flat(f):
        dd = decode_flat_action(m, nids, aws, np.asarray(f, np.float32))
        return np.asarray(source.join_actions(lens.join_actions((Unit, dd))),
                          dtype=np.float64)
    c = join_flat(np.zeros(D))
    A = np.stack([join_flat(np.eye(D)[j]) - c for j in range(D)], axis=1)
    A_pinv = np.linalg.pinv(A)

    # Optional student policy (ModuMorph). amorpheus_policy is model-agnostic;
    # sp(common, per_node) -> (Unit, per_node_actions), just like the eval script.
    have_student = student_ckpt is not None and student_ckpt not in _NONE
    sp = None
    if have_student:
        model = load_model(student_ckpt, variant=variant, d_model=64,
                           d_context=32, n_heads=4, n_layers=3)
        sp = amorpheus_policy(model)(m)

    # Seeded, per-(building, iteration) RNG for the Bernoulli coin -- no global
    # randomness, so runs are reproducible.
    seed = zlib.crc32(f"{bt}_{idx}_it{iteration}".encode()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    pol = _controller(bt, idx); pol.bind_env(env)
    raw, _ = env.reset(); pol.reset()
    obs_t, act_t, rew = [], [], []
    n_expert = 0
    for t in range(steps):
        keep = (t % stride) == 0

        # (1) student action (physical) if we have one
        a_student = None
        if have_student:
            tca, tpa = sp(*lens.split_observation(source.split_observation(raw)))
            a_student = np.asarray(
                source.join_actions(lens.join_actions((tca, tpa))),
                dtype=np.float64)

        # (2) ALWAYS query the expert: advances RBC shadow state AND is the label.
        a_rbc, _ = pol.predict(raw)
        a_rbc = np.asarray(a_rbc, dtype=np.float64)

        # (3) per-step Bernoulli(beta): expert drives w.p. beta, else student.
        use_expert = (not have_student) or (rng.random() < beta)
        a_exec = a_rbc if use_expert else a_student
        if use_expert:
            n_expert += 1

        # (4) record obs the net sees + the EXPERT action as the label
        if keep:
            _cm, per_node = lens.split_observation(source.split_observation(raw))
            obs_t.append([np.asarray(o, dtype=np.float32)
                          for o in encode_local_obs(m, nids, per_node)])
            act_t.append(a_rbc)

        # (5) step env with whichever action was chosen
        raw, r, term, trunc, _ = env.step(np.asarray(a_exec, dtype=np.float32))
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
    recon_err = float(np.abs((f_target @ A.T) + c - a_arr).mean())

    np.savez_compressed(
        out, f_target=f_target, rew=rew, stride=stride,
        recon_err=recon_err,
        **{f"obs{i}": o for i, o in enumerate(obs_stack)})
    return bt, idx, len(f_target), len(rew), recon_err, n_expert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-checkpoint", default="none",
                    help="ModuMorph .eqx student; 'none' => pure-expert (beta=1)")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="per-step prob the EXPERT drives the env (in [0,1])")
    ap.add_argument("--variant", default="hn", choices=["hn", "faithful", "blind"])
    ap.add_argument("--task", default=TASK,
                    help="b2b task preset (e.g. task_occ_e0 comfort, task_occ_emed energy)")
    ap.add_argument("--building-types", nargs="+",
                    default=["RetailStandalone", "RestaurantFastFood",
                             "OfficeMedium", "OfficeSmall"])
    ap.add_argument("--n-buildings", type=int, default=5)
    ap.add_argument("--steps", type=int, default=YEAR_STEPS,
                    help="steps per building (default: one full year)")
    ap.add_argument("--stride", type=int, default=4,
                    help="keep every Nth sample for fitting (rewards kept at full rate)")
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--out-dir", required=True,
                    help="aggregated demo dir (iterations ADD files here)")
    ap.add_argument("--iteration", type=int, default=0,
                    help="iteration tag used in output filenames")
    args = ap.parse_args()

    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(REPO, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    ckpt = args.student_checkpoint
    if ckpt not in _NONE and not os.path.isabs(ckpt):
        ckpt = os.path.join(REPO, ckpt)

    tasks = [(bt, i, args.steps, args.stride, args.beta, args.iteration,
              ckpt, args.variant, out_dir, args.task)
             for i in range(args.n_buildings) for bt in args.building_types]
    who = "pure-expert" if ckpt in _NONE else f"student={os.path.basename(ckpt)}"
    print(f"[dagger-collect] task={args.task} it{args.iteration} beta={args.beta} {who} | "
          f"{len(tasks)} buildings x {args.steps} steps (stride {args.stride}) | "
          f"{args.n_workers} workers -> {out_dir}", flush=True)
    with mp.get_context("spawn").Pool(args.n_workers) as pool:
        for k, (bt, idx, n_s, n_r, err, n_exp) in enumerate(
                pool.imap_unordered(collect_one, tasks), 1):
            frac = n_exp / max(1, n_r)
            print(f"  [{k:2d}/{len(tasks)}] {bt}_{idx}_it{args.iteration}: "
                  f"{n_s} samples, {n_r} steps, recon_err {err:.4f}, "
                  f"expert-drove {frac:.2f}", flush=True)
    print("DAGGER_COLLECT_DONE")


if __name__ == "__main__":
    main()
