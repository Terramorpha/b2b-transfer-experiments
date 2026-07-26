"""BC-clone RBC into a fresh per-building MLP (pure supervised; no EnergyPlus).

Two phases, mirroring bc_fit.py but for the flat-gym MLP:
  1. POLICY -- fit the squashed actor output tanh(actor(obs_n)) to RBC's action,
     mapped into [-1,1] by the same affine [lo,hi]->[-1,1] the trainer uses.
  2. CRITIC -- fit the value head to discounted returns-to-go.
Observations are running-normalized; the (mean,var) are saved so the PPO
fine-tune seeds its normalizer identically. Separate actor/critic nets, so each
phase's gradient touches only its own head.

    python scripts/bc_fit_mlp.py --building-type OfficeMedium --index 0
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "online")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from mlp_ppo import ActorCritic, RunningNorm

DEMOS = os.path.join(REPO, "data", "bc_demos_mlp")


def returns_to_go(rew, gamma):
    G = np.zeros_like(rew); acc = 0.0
    for t in range(len(rew) - 1, -1, -1):
        acc = rew[t] + gamma * acc; G[t] = acc
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-type", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--policy-steps", type=int, default=4000)
    ap.add_argument("--value-steps", type=int, default=2000)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(REPO, "runs",
                                   f"mlp_bc_{args.building_type}_{args.index}")
    out = out if os.path.isabs(out) else os.path.join(REPO, out)
    os.makedirs(out, exist_ok=True)

    d = np.load(os.path.join(DEMOS, f"{args.building_type}_{args.index}.npz"))
    obs = d["obs"].astype(np.float32)
    act = d["act"].astype(np.float32)
    lo, hi = d["lo"].astype(np.float32), d["hi"].astype(np.float32)
    stride = int(d["stride"])
    G = returns_to_go(d["rew"].astype(np.float64), args.gamma)[::stride][:obs.shape[0]]
    G = G.astype(np.float32)
    N, obs_dim = obs.shape
    act_dim = act.shape[1]

    run = wandb.init(project="morel-b2b-transfer-port",
                     name=f"mlp-bc-{args.building_type}-{args.index}-s{args.seed}",
                     config={"algorithm": "mlp_bc_clone",
                             "building": f"{args.building_type}_{args.index}",
                             "obs_dim": obs_dim, "act_dim": act_dim, "N": N,
                             **vars(args)})
    print(f"[setup] {args.building_type}_{args.index}  N={N} obs={obs_dim} "
          f"act={act_dim}  wandb {run.url}", flush=True)

    # obs normalizer (saved for the fine-tune to reuse verbatim)
    norm = RunningNorm(obs_dim); norm.update(obs)
    obs_n = jnp.asarray(norm.norm(obs), jnp.float32)
    # RBC action -> squashed [-1,1] target (same affine the trainer inverts)
    s_tar = 2.0 * (act - lo) / np.maximum(hi - lo, 1e-8) - 1.0
    s_tar = jnp.asarray(np.clip(s_tar, -1 + 1e-4, 1 - 1e-4), jnp.float32)
    G_j = jnp.asarray(G)

    key = jax.random.PRNGKey(args.seed)
    key, mk = jax.random.split(key)
    model = ActorCritic(obs_dim, act_dim, tuple(args.hidden), mk)
    rng = np.random.default_rng(args.seed)

    @eqx.filter_jit
    def pol_step(m, opt_state, ob, s, opt):
        def loss(mm):
            pred = jax.vmap(lambda o: jnp.tanh(mm.actor(o)))(ob)
            return jnp.mean((pred - s) ** 2)
        v, g = eqx.filter_value_and_grad(loss)(m)
        upd, opt_state = opt.update(g, opt_state, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, upd), opt_state, v

    @eqx.filter_jit
    def val_step(m, opt_state, ob, g_, opt):
        def loss(mm):
            pred = jax.vmap(lambda o: mm.critic(o)[0])(ob)
            return jnp.mean((pred - g_) ** 2)
        v, g = eqx.filter_value_and_grad(loss)(m)
        upd, opt_state = opt.update(g, opt_state, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, upd), opt_state, v

    def sample(n):
        return jnp.asarray(rng.integers(0, N, size=min(n, N)))

    # --- phase 1: policy ---
    opt = optax.adam(args.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    print(f"[policy] {args.policy_steps} steps", flush=True)
    for step in range(args.policy_steps):
        i = sample(args.minibatch)
        model, opt_state, v = pol_step(model, opt_state, obs_n[i], s_tar[i], opt)
        if step % 500 == 0 or step == args.policy_steps - 1:
            full = float(jnp.mean((jax.vmap(lambda o: jnp.tanh(model.actor(o)))(obs_n)
                                   - s_tar) ** 2))
            wandb.log({"bc/action_mse": full, "phase": 0}, step=step)
            print(f"  step {step:5d}  action MSE {full:.5f}", flush=True)

    # --- phase 2: critic ---
    optv = optax.adam(1e-3)
    optv_state = optv.init(eqx.filter(model, eqx.is_array))
    print(f"[critic] {args.value_steps} steps", flush=True)
    for step in range(args.value_steps):
        i = sample(args.minibatch)
        model, optv_state, v = val_step(model, optv_state, obs_n[i], G_j[i], optv)
        if step % 500 == 0 or step == args.value_steps - 1:
            full = float(jnp.mean((jax.vmap(lambda o: model.critic(o)[0])(obs_n)
                                   - G_j) ** 2))
            wandb.log({"bc/value_mse": full, "phase": 1},
                      step=args.policy_steps + step)
            print(f"  step {step:5d}  value MSE {full:.3f}", flush=True)

    eqx.tree_serialise_leaves(os.path.join(out, "model_s0.eqx"), model)
    np.savez(os.path.join(out, "obs_norm.npz"),
             mean=norm.mean, var=norm.var, count=norm.count)
    wandb.finish()
    print(f"[done] {out}\nMLP_BC_FIT_DONE", flush=True)


if __name__ == "__main__":
    main()
