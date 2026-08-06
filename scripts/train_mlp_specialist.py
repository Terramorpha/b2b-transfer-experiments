"""Per-building MLP PPO oracle: train + evaluate on ONE building.

Deliberately NOT Amorpheus. No morphology, no per-node tokens, no shared universe
-- just the building's plain gym `Box` obs/action and a vanilla MLP actor+critic.
Two modes:
  * from scratch (default): random init + a random-action obs-norm warmup.
  * warm-started (--init-checkpoint): load a BC clone (bc_fit_mlp.py) + its saved
    obs-norm, then PPO fine-tune (run with --ent-coef 0 --lr 2e-5, like the
    Amorpheus warm-start, so the fine-tune refines the clone instead of erasing it).

Policy: state-independent-std Gaussian, tanh-squashed to the action bounds. Value
uses GAE; truncation at year-end bootstraps. Model/PPO live in mlp_ppo.py (shared
with the BC clone so a cloned checkpoint deserialises here).

    # from scratch
    python scripts/train_mlp_specialist.py --building-type OfficeMedium --index 0 \
        --split test --steps 500000 --out runs/mlp_specialist_OfficeMedium_0
    # BC-warm-started fine-tune
    python scripts/train_mlp_specialist.py --building-type OfficeMedium --index 0 \
        --split test --steps 500000 --lr 2e-5 --ent-coef 0 \
        --init-checkpoint runs/mlp_bc_OfficeMedium_0/model_s0.eqx \
        --out runs/mlp_warm_OfficeMedium_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("vendor/morel", "vendor/Building2Building", "scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.environ.setdefault("WANDB_MODE", "online")

import _pin_dataset  # noqa: F401
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

import building2building as b2b

from mlp_ppo import (ActorCritic, RunningNorm, act, value, mean_action,
                     compute_gae, make_update)

TASK, RP = "task_occ_e0", "full_year"
YEAR_STEPS = 105_120


# ------------------------------------------------------------------- env helpers
def _reset(env):
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r


def _step(env, a):
    out = env.step(a)
    if len(out) == 5:  # gymnasium: obs, rew, terminated, truncated, info
        obs, rew, term, trunc, info = out
        return obs, float(rew), bool(term), bool(trunc), info
    obs, rew, done, info = out  # legacy gym
    return obs, float(rew), bool(done), False, info


def _to_env(squashed, lo, hi):
    return lo + 0.5 * (np.asarray(squashed) + 1.0) * (hi - lo)


def evaluate(env, model, norm, lo, hi, max_steps):
    """One deterministic full-year rollout; returns summed reward."""
    obs = _reset(env); total = 0.0; steps = 0
    while True:
        a01 = np.asarray(mean_action(model, jnp.asarray(norm.norm(obs[None])[0],
                                                         dtype=jnp.float32)))
        obs, rew, term, trunc, _ = _step(env, _to_env(a01, lo, hi))
        total += rew; steps += 1
        if term or trunc or steps >= max_steps:
            return total, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-type", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-checkpoint", default=None,
                    help="BC clone to warm-start from; its obs_norm.npz is loaded too")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="eval (full year) every N updates; 0 = only at the end")
    ap.add_argument("--freeze-norm", action="store_true",
                    help="do NOT update the obs-normalizer during training; use the "
                         "seeded/demo-derived stats as-is. Recommended for warm-start.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    env = b2b.make_env(args.building_type, split=args.split, index=args.index,
                       task=args.task, run_period=RP)
    from building2building.data.registry import get_registry
    bid = get_registry().get_building_by_index(
        args.building_type, args.split, args.index).building_id
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    lo = np.asarray(env.action_space.low, np.float32)
    hi = np.asarray(env.action_space.high, np.float32)

    warm = args.init_checkpoint is not None
    run = wandb.init(project="morel-b2b-transfer-port",
                     name=f"mlp-{'warm' if warm else 'oracle'}-{bid}-s{args.seed}",
                     config={"algorithm": "mlp_ppo_warm" if warm else "mlp_ppo_oracle",
                             "building_id": bid, "obs_dim": obs_dim,
                             "act_dim": act_dim, **vars(args)})
    print(f"[setup] {bid}  obs {obs_dim}  act {act_dim}  "
          f"{'WARM' if warm else 'scratch'}  wandb {run.url}", flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, mk = jax.random.split(key)
    norm = RunningNorm(obs_dim)
    if warm:
        ckpt = (args.init_checkpoint if os.path.isabs(args.init_checkpoint)
                else os.path.join(REPO, args.init_checkpoint))
        skel = ActorCritic(obs_dim, act_dim, tuple(args.hidden), mk)
        model = eqx.tree_deserialise_leaves(ckpt, skel)
        nz = np.load(os.path.join(os.path.dirname(ckpt), "obs_norm.npz"))
        norm.mean = nz["mean"].astype(np.float64)
        norm.var = nz["var"].astype(np.float64)
        norm.count = float(nz["count"]) if "count" in nz.files else 1e5
        print(f"[warm] loaded clone {ckpt} + obs-norm (count {norm.count:.0f})",
              flush=True)
    else:
        model = ActorCritic(obs_dim, act_dim, tuple(args.hidden), mk)

    n_updates = args.steps // args.n_steps
    grad_steps = n_updates * args.epochs * max(1, args.n_steps // args.minibatch)
    # linear LR decay to 0 -> standard PPO stabilizer for long single-env runs
    sched = optax.linear_schedule(args.lr, 0.0, grad_steps)
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    update = make_update(opt)
    print(f"[protocol] {args.steps} steps / n_steps {args.n_steps} = {n_updates} "
          f"updates | year={YEAR_STEPS} steps", flush=True)

    obs = _reset(env)
    if not warm:
        # seed the normalizer with a random-action rollout so the first policy
        # update never sees raw (unnormalized) EnergyPlus obs. (Warm start already
        # has the clone's demo-based obs-norm loaded.)
        w = []
        wrng = np.random.default_rng(args.seed + 12345)
        for _ in range(min(args.n_steps * 2, 4096)):
            w.append(obs.astype(np.float32))
            a = wrng.uniform(-1, 1, act_dim)
            obs, _, term, trunc, _ = _step(env, _to_env(a, lo, hi))
            if term or trunc:
                obs = _reset(env)
        norm.update(np.asarray(w, np.float32))
        print(f"[warmup] normalizer seeded on {len(w)} random-policy steps", flush=True)
        obs = _reset(env)

    global_step = 0
    for upd in range(n_updates):
        O, O_raw, A_raw, LP, R, V, EP, BOOT = [], [], [], [], [], [], [], []
        for _ in range(args.n_steps):
            raw_obs = obs.astype(np.float32)            # RAW obs (for the normalizer)
            on = norm.norm(raw_obs[None])[0].astype(np.float32)
            key, sk = jax.random.split(key)
            a01, raw, logp, val = act(model, jnp.asarray(on), sk)
            nobs, rew, term, trunc, _ = _step(env, _to_env(a01, lo, hi))
            O.append(on); O_raw.append(raw_obs)          # network sees `on`; norm updates on RAW
            A_raw.append(np.asarray(raw))  # store the RAW action sample verbatim
            LP.append(float(logp)); R.append(rew); V.append(float(val))
            end = term or trunc
            EP.append(1.0 if end else 0.0)
            if end:
                bv = 0.0 if term else float(value(model, jnp.asarray(
                    norm.norm(nobs[None])[0], dtype=jnp.float32)))
                BOOT.append(bv); obs = _reset(env)
            else:
                BOOT.append(0.0); obs = nobs
            global_step += 1
        last_val = float(value(model, jnp.asarray(norm.norm(obs[None])[0],
                                                  dtype=jnp.float32)))
        O = np.asarray(O, np.float32)
        if not args.freeze_norm:
            norm.update(np.asarray(O_raw, np.float32))  # FIX: update on RAW obs, not normalized
        adv, ret = compute_gae(np.asarray(R, np.float32), np.asarray(V, np.float32),
                               np.asarray(EP), np.asarray(BOOT, np.float32),
                               last_val, args.gamma, args.gae_lambda)
        O = jnp.asarray(O); A_raw = jnp.asarray(np.asarray(A_raw), jnp.float32)
        LP = jnp.asarray(np.asarray(LP), jnp.float32)
        ADV = jnp.asarray(adv); RET = jnp.asarray(ret)

        idx = np.arange(args.n_steps)
        rng = np.random.default_rng(args.seed + upd)
        for _ in range(args.epochs):
            rng.shuffle(idx)
            for s in range(0, args.n_steps, args.minibatch):
                mb = idx[s:s + args.minibatch]
                model, opt_state, loss, (pg, vf, ent) = update(
                    model, opt_state, O[mb], A_raw[mb], LP[mb], ADV[mb], RET[mb],
                    args.clip, args.ent_coef, args.vf_coef)
        wandb.log({"train/mean_reward_per_step": float(np.mean(R)),
                   "train/rollout_return": float(np.sum(R)),
                   "loss/policy": float(pg), "loss/value": float(vf),
                   "loss/entropy": float(ent),
                   "policy/std": float(np.mean(np.exp(np.asarray(model.log_std)))),
                   "update": upd}, step=global_step)
        if upd % 10 == 0 or upd == n_updates - 1:
            print(f"  upd {upd:4d}  step {global_step:8d}  "
                  f"roll_ret {np.sum(R):10.1f}  r/step {np.mean(R):.4f}  "
                  f"std {float(np.mean(np.exp(np.asarray(model.log_std)))):.3f}",
                  flush=True)
        if args.eval_every and (upd + 1) % args.eval_every == 0:
            er, es = evaluate(env, model, norm, lo, hi, YEAR_STEPS + 10)
            wandb.log({"eval/fullyear_return": er}, step=global_step)
            print(f"    [eval] full-year return {er:.1f} ({es} steps)", flush=True)

    ret_final, steps = evaluate(env, model, norm, lo, hi, YEAR_STEPS + 10)
    wandb.log({"eval/fullyear_return": ret_final}, step=global_step)
    eqx.tree_serialise_leaves(os.path.join(out, "model_s0.eqx"), model)
    np.savez(os.path.join(out, "obs_norm.npz"),
             mean=norm.mean, var=norm.var, count=norm.count)
    json.dump({"building_id": bid, "return": ret_final, "steps": steps,
               "n_updates": n_updates, "warm": warm},
              open(os.path.join(out, "eval.json"), "w"), indent=2)
    env.close(); wandb.finish()
    print(f"[done] {bid}  full-year return {ret_final:.1f}\n"
          f"MLP_SPECIALIST_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
