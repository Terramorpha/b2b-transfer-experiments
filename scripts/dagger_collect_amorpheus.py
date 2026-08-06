"""AMORPHEUS DAgger collection: shadow-mode DAgger with a per-building-tuned RBC
teacher, cloning into a single Amorpheus policy.

The Amorpheus counterpart of dagger_collect.py (which is ModuMorph). Identical
DAgger logic -- shadow expert (RBC steps every step on the student-induced obs),
per-step Bernoulli(beta) execution, expert action ALWAYS the label, A/A_pinv
label-inversion -- but on the AMORPHEUS_B2B_BRIDGE (attributes inlined into the
observation) with morel_amorpheus make_model/load_model. Teacher = the same
per-building Optuna-tuned RBC used by bc_collect_tuned
(data/rbc_tuned_configs/{bt}_train_{idx}.json).

beta = 1.0 with no student == pure-expert == plain BC (iteration 0). Writes one
npz per (building, iteration), tagged {bt}_{idx}_it{iteration}.npz, same format
as bc_collect_tuned / dagger_collect so dagger_fit_amorpheus can aggregate them.
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
# ONLINE by default so the collect's rollout reward + per-step imitation error are
# logged; still overridable (WANDB_MODE=offline/disabled in the env wins).
os.environ.setdefault("WANDB_MODE", "online")

import _pin_dataset  # noqa: F401
import numpy as np
import wandb

import json
from baselines.controllers import (
    AirLoopPolicy, UnitaryHvacPolicy, AirLoopConfig, UnitaryHvacConfig)
from morel import Unit
from morel.morphology import trivial_morphology
from morel_amorpheus import amorpheus_policy  # model-agnostic conditioned-policy wrapper
from morel_amorpheus.policy import decode_flat_action, encode_local_obs
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, make_model, load_model
from evaluate import _b2b_factory_impl

TASK, RP = "task_occ_e0", "full_year"
YEAR_STEPS = 105_120
CFG_DIR = os.path.join(REPO, "data", "rbc_tuned_configs")

_NONE = {"", "none", "None", "null"}


def _controller(bt, idx):
    """Per-(train-)building Optuna-tuned RBC teacher (same as bc_collect_tuned).
    .predict(obs) MUTATES its internal state each call -- essential for shadow
    mode (its integrators/setpoint track the student-induced obs stream)."""
    p = os.path.join(CFG_DIR, f"{bt}_train_{idx}.json")
    d = json.load(open(p))
    params = d["params"]
    if d["is_vav"]:
        return AirLoopPolicy(AirLoopConfig(**params))
    return UnitaryHvacPolicy(UnitaryHvacConfig(**params))


def collect_one(args):
    (bt, idx, steps, stride, beta, iteration, student_ckpt, out_dir, task) = args
    out = os.path.join(out_dir, f"{bt}_{idx}_it{iteration}.npz")

    env, source = _b2b_factory_impl(bt, idx, "train", task, RP)
    # Amorpheus morphology (attributes inlined into the observation) + its lens.
    m = AMORPHEUS_B2B_BRIDGE.apply(source)
    lens = AMORPHEUS_B2B_BRIDGE.apply(trivial_morphology(source))
    # node_ids / act_widths from the AMORPHEUS conditioned model.
    cond0 = make_model(d_model=64, n_heads=4, n_layers=3, seed=0).condition(m)
    nids, aws = cond0.node_ids, cond0.act_widths
    D = int(sum(aws))

    # Probe the affine target->physical action map, then pseudo-invert it (same
    # A_pinv inversion as bc_collect_tuned); join is (Unit, per_node_actions).
    def join_flat(f):
        dd = decode_flat_action(m, nids, aws, np.asarray(f, np.float32))
        return np.asarray(source.join_actions(lens.join_actions((Unit, dd))),
                          dtype=np.float64)
    c = join_flat(np.zeros(D))
    A = np.stack([join_flat(np.eye(D)[j]) - c for j in range(D)], axis=1)
    A_pinv = np.linalg.pinv(A)

    # Optional student policy (Amorpheus). amorpheus_policy is model-agnostic;
    # sp(common, per_node) -> (Unit, per_node_actions).
    have_student = student_ckpt is not None and student_ckpt not in _NONE
    sp = None
    if have_student:
        model = load_model(student_ckpt, d_model=64, n_heads=4, n_layers=3)
        sp = amorpheus_policy(model)(m)

    # Seeded, per-(building, iteration) RNG for the Bernoulli coin (reproducible).
    seed = zlib.crc32(f"amorph_{bt}_{idx}_it{iteration}".encode()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    pol = _controller(bt, idx); pol.bind_env(env)
    raw, _ = env.reset(); pol.reset()
    obs_t, act_t, rew = [], [], []
    n_expert = 0
    # per-step imitation error (student vs RBC label, NORMALIZED action space) and
    # downsampled reward/imitation curves for wandb.
    imit_errs = []
    curve_steps, curve_reward, curve_imit = [], [], []
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

        # per-step imitation MSE in NORMALIZED action space (both inverted through
        # the same A_pinv the collector built) -> comparable to the fit's action_mse.
        if have_student:
            f_student = np.clip((a_student - c) @ A_pinv.T, -1.0, 1.0)
            f_rbc_norm = np.clip((a_rbc - c) @ A_pinv.T, -1.0, 1.0)
            imit_t = float(np.mean((f_student - f_rbc_norm) ** 2))
            imit_errs.append(imit_t)
        else:
            imit_t = float("nan")

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

        # downsampled time series (every 500 steps) of reward + imitation error
        if (t % 500) == 0:
            curve_steps.append(t)
            curve_reward.append(float(r))
            curve_imit.append(imit_t)

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
    return {
        "bt": bt, "idx": idx, "n_samples": len(f_target), "n_steps": len(rew),
        "recon_err": recon_err, "n_expert": n_expert,
        "rollout_return": float(rew.sum()),
        "rollout_mean_reward": float(rew.mean()),
        "imitation_mse": (float(np.mean(imit_errs)) if imit_errs else float("nan")),
        "curve_steps": curve_steps, "curve_reward": curve_reward,
        "curve_imit": curve_imit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-checkpoint", default="none",
                    help="Amorpheus .eqx student; 'none' => pure-expert (beta=1)")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="per-step prob the EXPERT drives the env (in [0,1])")
    ap.add_argument("--task", default=TASK,
                    help="b2b task preset (comfort task_occ_e0, energy task_occ_emed)")
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
              ckpt, out_dir, args.task)
             for i in range(args.n_buildings) for bt in args.building_types]
    who = "pure-expert" if ckpt in _NONE else f"student={os.path.basename(ckpt)}"
    # wandb run for this iteration's collect (do NOT init inside spawn workers --
    # they return metrics; the parent logs them here).
    run = wandb.init(
        project="morel-b2b-transfer-port",
        name=f"dagger-collect-amorpheus-it{args.iteration}-b{args.beta:g}",
        tags=["dagger", "amorpheus", "collect", f"it{args.iteration}"],
        config={"phase": "collect", "arch": "amorpheus",
                "iteration": args.iteration, "beta": args.beta,
                "task": args.task, "n_buildings": args.n_buildings,
                "building_types": args.building_types, "steps": args.steps,
                "stride": args.stride},
    )
    print(f"[dagger-collect-amorpheus] task={args.task} it{args.iteration} "
          f"beta={args.beta} {who} | {len(tasks)} buildings x {args.steps} steps "
          f"(stride {args.stride}) | {args.n_workers} workers -> {out_dir} | "
          f"wandb {run.url}", flush=True)

    returns, mrewards, imits, curve_rows = [], [], [], []
    with mp.get_context("spawn").Pool(args.n_workers) as pool:
        for k, res in enumerate(pool.imap_unordered(collect_one, tasks), 1):
            bt, idx = res["bt"], res["idx"]
            frac = res["n_expert"] / max(1, res["n_steps"])
            has_imit = res["imitation_mse"] == res["imitation_mse"]  # not NaN
            print(f"  [{k:2d}/{len(tasks)}] {bt}_{idx}_it{args.iteration}: "
                  f"{res['n_samples']} samples, {res['n_steps']} steps, "
                  f"recon_err {res['recon_err']:.4f}, expert-drove {frac:.2f}, "
                  f"return {res['rollout_return']:.1f}" +
                  (f", imit_mse {res['imitation_mse']:.5f}" if has_imit else ""),
                  flush=True)
            logd = {
                "collect/rollout_return": res["rollout_return"],
                "collect/rollout_mean_reward": res["rollout_mean_reward"],
                "collect/recon_err": res["recon_err"],
                "collect/expert_frac": frac,
                "iteration": args.iteration, "beta": args.beta,
            }
            if has_imit:
                logd["collect/imitation_mse"] = res["imitation_mse"]
                imits.append(res["imitation_mse"])
            wandb.log(logd, step=k)
            returns.append(res["rollout_return"])
            mrewards.append(res["rollout_mean_reward"])
            for s, rw, im in zip(res["curve_steps"], res["curve_reward"],
                                 res["curve_imit"]):
                curve_rows.append([f"{bt}_{idx}", args.iteration, s, rw, im])

    wandb.run.summary["collect/mean_rollout_return"] = float(np.mean(returns))
    wandb.run.summary["collect/mean_rollout_reward"] = float(np.mean(mrewards))
    if imits:
        wandb.run.summary["collect/mean_imitation_mse"] = float(np.mean(imits))
    tbl = wandb.Table(columns=["building", "iteration", "step", "reward", "imit_mse"],
                      data=curve_rows)
    wandb.log({"collect/curves": tbl})
    wandb.finish()
    print("DAGGER_COLLECT_DONE")


if __name__ == "__main__":
    main()
