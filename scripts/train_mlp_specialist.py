"""True per-building oracle: a fresh MLP PPO trained + evaluated on ONE building.

Deliberately NOT Amorpheus. No morphology, no per-node tokens, no BC warm-start,
no shared universe -- just the building's plain gym `Box` obs/action space and a
vanilla MLP actor + MLP critic trained from scratch with standard PPO. This
answers "how much is achievable on this building by a dedicated policy that pays
no transfer tax", i.e. the real ceiling for `expert - RBC`.

Policy: state-independent-std Gaussian, tanh-squashed to the action bounds
(the plain PPO-continuous default). Observations are running-normalized (essential
for a raw-EnergyPlus MLP). Value uses GAE; truncation at year-end bootstraps.

    python scripts/train_mlp_specialist.py --building-type OfficeMedium --index 0 \
        --split test --steps 1000000 --out runs/mlp_specialist_OfficeMedium_0
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

TASK, RP = "task_occ_e0", "full_year"


# --------------------------------------------------------------------------- net
class MLP(eqx.Module):
    layers: list

    def __init__(self, sizes, key):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(a, b, key=k)
                       for a, b, k in zip(sizes[:-1], sizes[1:], keys)]

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = jax.nn.tanh(lin(x))
        return self.layers[-1](x)


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # keep std in [~0.007, ~7.4] -> no runaway


class ActorCritic(eqx.Module):
    actor: MLP
    critic: MLP
    log_std: jax.Array

    def __init__(self, obs_dim, act_dim, hidden, key):
        ka, kc = jax.random.split(key)
        self.actor = MLP([obs_dim, *hidden, act_dim], ka)
        self.critic = MLP([obs_dim, *hidden, 1], kc)
        self.log_std = jnp.full((act_dim,), -0.5)  # start at std~0.6 (was 1.0)


def _gauss_logprob(raw, mean, log_std):
    # log N(raw; mean, exp(log_std)^2), summed over action dims
    var = jnp.exp(2.0 * log_std)
    return jnp.sum(-0.5 * ((raw - mean) ** 2 / var) - log_std
                   - 0.5 * jnp.log(2.0 * jnp.pi))


def _squash_logdet(raw):
    # log|d tanh / d raw| correction, summed. The subsequent affine [-1,1]->[lo,hi]
    # map is a constant Jacobian -> cancels in the PPO ratio, so it is omitted.
    return jnp.sum(jnp.log(1.0 - jnp.tanh(raw) ** 2 + 1e-6))


@eqx.filter_jit
def _act(model, obs_n, key):
    """Sample: return squashed action in [-1,1], the RAW pre-tanh sample (stored
    verbatim for the update -- reconstructing it via arctanh clips saturated
    actions and corrupts the PPO ratio), its log-prob, and the value."""
    ls = jnp.clip(model.log_std, LOG_STD_MIN, LOG_STD_MAX)
    mean = model.actor(obs_n)
    raw = mean + jnp.exp(ls) * jax.random.normal(key, mean.shape)
    logp = _gauss_logprob(raw, mean, ls) - _squash_logdet(raw)
    return jnp.tanh(raw), raw, logp, model.critic(obs_n)[0]


@eqx.filter_jit
def _value(model, obs_n):
    return model.critic(obs_n)[0]


@eqx.filter_jit
def _mean_action(model, obs_n):
    return jnp.tanh(model.actor(obs_n))  # deterministic eval action in [-1,1]


# ----------------------------------------------------------------- obs normalizer
class RunningNorm:
    """Welford running mean/var over observations (numpy, CPU-side)."""

    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x):
        b_mean = x.mean(0); b_var = x.var(0); b_n = x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m_a = self.var * self.count
        m_b = b_var * b_n
        self.var = (m_a + m_b + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def norm(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)


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


# --------------------------------------------------------------------------- GAE
def compute_gae(rew, val, ep_end, boot_val, last_val, gamma, lam):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        if ep_end[t]:                       # episode boundary at t
            nextv = boot_val[t]             # 0 if terminated, V(last) if truncated
            delta = rew[t] + gamma * nextv - val[t]
            gae = delta                     # do not propagate across the boundary
        else:
            nextv = last_val if t == T - 1 else val[t + 1]
            delta = rew[t] + gamma * nextv - val[t]
            gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + val


# --------------------------------------------------------------------- PPO update
def make_update(opt):
    @eqx.filter_jit
    def update(model, opt_state, obs, act_raw, old_logp, adv, ret, clip, ent_c, vf_c):
        def loss_fn(m):
            ls = jnp.clip(m.log_std, LOG_STD_MIN, LOG_STD_MAX)
            def per(o, ar):
                mean = m.actor(o)
                logp = _gauss_logprob(ar, mean, ls) - _squash_logdet(ar)
                v = m.critic(o)[0]
                ent = jnp.sum(ls + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))
                return logp, v, ent
            logp, v, ent = jax.vmap(per)(obs, act_raw)
            ratio = jnp.exp(logp - old_logp)
            a = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -jnp.mean(jnp.minimum(ratio * a,
                                       jnp.clip(ratio, 1 - clip, 1 + clip) * a))
            vf = jnp.mean((v - ret) ** 2)
            entropy = jnp.mean(ent)
            return pg + vf_c * vf - ent_c * entropy, (pg, vf, entropy)
        (loss, aux), g = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        upd, opt_state = opt.update(g, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, upd)
        return model, opt_state, loss, aux
    return update


def evaluate(env, model, norm, lo, hi, max_steps):
    """One deterministic full-year rollout; returns summed reward."""
    obs = _reset(env); total = 0.0; steps = 0
    while True:
        a01 = np.asarray(_mean_action(model, jnp.asarray(norm.norm(obs[None])[0],
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
    ap.add_argument("--eval-every", type=int, default=0,
                    help="eval (full year) every N updates; 0 = only at the end")
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

    run = wandb.init(project="morel-b2b-transfer-port",
                     name=f"mlp-oracle-{bid}-s{args.seed}",
                     config={"algorithm": "mlp_ppo_oracle", "building_id": bid,
                             "obs_dim": obs_dim, "act_dim": act_dim, **vars(args)})
    print(f"[setup] {bid}  obs {obs_dim}  act {act_dim}  wandb {run.url}", flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, mk = jax.random.split(key)
    model = ActorCritic(obs_dim, act_dim, tuple(args.hidden), mk)
    n_updates = args.steps // args.n_steps
    grad_steps = n_updates * args.epochs * max(1, args.n_steps // args.minibatch)
    # linear LR decay to 0 -> standard PPO stabilizer for long single-env runs
    sched = optax.linear_schedule(args.lr, 0.0, grad_steps)
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    update = make_update(opt)
    norm = RunningNorm(obs_dim)
    print(f"[protocol] {args.steps} steps / n_steps {args.n_steps} = {n_updates} "
          f"updates | year={105120} steps", flush=True)

    # --- warm up the obs normalizer with a random-action rollout so the very
    #     first policy update never sees raw (unnormalized) EnergyPlus obs ---
    obs = _reset(env); warm = []
    wrng = np.random.default_rng(args.seed + 12345)
    for _ in range(min(args.n_steps * 2, 4096)):
        warm.append(obs.astype(np.float32))
        a = wrng.uniform(-1, 1, act_dim)
        obs, _, term, trunc, _ = _step(env, _to_env(a, lo, hi))
        if term or trunc:
            obs = _reset(env)
    norm.update(np.asarray(warm, np.float32))
    print(f"[warmup] normalizer seeded on {len(warm)} random-policy steps", flush=True)
    obs = _reset(env)
    global_step = 0
    for upd in range(n_updates):
        O, A_raw, LP, R, V, EP, BOOT = [], [], [], [], [], [], []
        for _ in range(args.n_steps):
            on = norm.norm(obs[None])[0].astype(np.float32)
            key, sk = jax.random.split(key)
            a01, raw, logp, val = _act(model, jnp.asarray(on), sk)
            nobs, rew, term, trunc, _ = _step(env, _to_env(a01, lo, hi))
            O.append(on); A_raw.append(np.asarray(raw))  # store the RAW sample verbatim
            LP.append(float(logp)); R.append(rew); V.append(float(val))
            end = term or trunc
            EP.append(1.0 if end else 0.0)
            # bootstrap value at a boundary: 0 if truly terminal, else V(last obs)
            if end:
                bv = 0.0 if term else float(_value(model, jnp.asarray(
                    norm.norm(nobs[None])[0], dtype=jnp.float32)))
                BOOT.append(bv); obs = _reset(env)
            else:
                BOOT.append(0.0); obs = nobs
            global_step += 1
        last_val = float(_value(model, jnp.asarray(norm.norm(obs[None])[0],
                                                   dtype=jnp.float32)))
        O = np.asarray(O, np.float32); norm.update(O)  # normalizer sees this batch
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
            er, es = evaluate(env, model, norm, lo, hi, 105120 + 10)
            wandb.log({"eval/fullyear_return": er}, step=global_step)
            print(f"    [eval] full-year return {er:.1f} ({es} steps)", flush=True)

    ret_final, steps = evaluate(env, model, norm, lo, hi, 105120 + 10)
    wandb.log({"eval/fullyear_return": ret_final}, step=global_step)
    eqx.tree_serialise_leaves(os.path.join(out, "model_s0.eqx"), model)
    np.savez(os.path.join(out, "obs_norm.npz"), mean=norm.mean, var=norm.var)
    json.dump({"building_id": bid, "return": ret_final, "steps": steps,
               "n_updates": n_updates},
              open(os.path.join(out, "eval.json"), "w"), indent=2)
    env.close(); wandb.finish()
    print(f"[done] {bid}  full-year return {ret_final:.1f}\n"
          f"MLP_SPECIALIST_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
