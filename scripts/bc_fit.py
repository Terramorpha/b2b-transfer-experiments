"""Phase B of BC-from-RBC: fit a fresh net to the collected demonstrations.

Two phases, deliberately separate:
  1. POLICY  -- fit the Beta mean to RBC's target actions (MSE).
  2. CRITIC  -- fit the value head to discounted returns-to-go, POLICY FROZEN
                (grad-masked), so the value fit cannot distort the trunk.

Both matter downstream: warm-starting PPO with a random critic destroyed the cloned
policy in an earlier attempt (garbage early advantages), and PPO must then be run with
ent_coef=0 or the entropy bonus blows the imitation apart.

Minibatched so memory stays bounded regardless of demo size.

    python scripts/bc_fit.py --out runs/transfer4_bc/model_s0.eqx
"""
import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
# Default to online: all training runs are logged. Override with
# WANDB_MODE=offline (sync later) or WANDB_MODE=disabled (smoke tests).
os.environ.setdefault("WANDB_MODE", "online")

import _pin_dataset  # noqa: F401
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, make_model, save_model
from evaluate import _b2b_factory_impl

DEMOS = os.path.join(REPO, "data", "bc_demos")
TASK, RP = "task_occ_e0", "full_year"


def returns_to_go(rew, gamma):
    G = np.zeros_like(rew)
    acc = 0.0
    for t in range(len(rew) - 1, -1, -1):
        acc = rew[t] + gamma * acc
        G[t] = acc
    return G


def load_all(gamma, stride_default=4):
    """Rebuild each building's morphology (needed to condition the model) and pair it
    with its stored demonstrations."""
    data = []
    for path in sorted(glob.glob(os.path.join(DEMOS, "*.npz"))):
        name = os.path.basename(path)[:-4]
        bt, idx = name.rsplit("_", 1)
        d = np.load(path)
        env, source = _b2b_factory_impl(bt, int(idx), "train", TASK, RP)
        m = AMORPHEUS_B2B_BRIDGE.apply(source)
        env.close()
        obs = tuple(jnp.asarray(d[k]) for k in sorted(
            (k for k in d.files if k.startswith("obs")),
            key=lambda s: int(s[3:])))
        ftar = jnp.asarray(d["f_target"])
        stride = int(d["stride"]) if "stride" in d.files else stride_default
        G = returns_to_go(np.asarray(d["rew"]), gamma)[::stride][:ftar.shape[0]]
        data.append((name, m, obs, ftar, jnp.asarray(G, dtype=jnp.float32)))
        print(f"  loaded {name:26s} {ftar.shape[0]:6d} samples, "
              f"{len(obs)} nodes, act_dim {ftar.shape[1]}", flush=True)
    return data


def make_policy_grad(m):
    def loss(model, obs_mb, ftar_mb):
        cond = model.condition(m)
        def one(lo):
            d, _ = cond(lo)
            return d.mean
        return jnp.mean((jax.vmap(one)(obs_mb) - ftar_mb) ** 2)
    return eqx.filter_jit(eqx.filter_value_and_grad(loss))


def make_value_grad(m):
    def loss(model, obs_mb, g_mb):
        cond = model.condition(m)
        def one(lo):
            _d, v = cond(lo)
            return v
        return jnp.mean((jax.vmap(one)(obs_mb) - g_mb) ** 2)
    return eqx.filter_jit(eqx.filter_value_and_grad(loss))


def make_policy_loss(m):
    def loss(model, obs_mb, ftar_mb):
        cond = model.condition(m)
        def one(lo):
            d, _ = cond(lo)
            return d.mean
        return jnp.mean((jax.vmap(one)(obs_mb) - ftar_mb) ** 2)
    return eqx.filter_jit(loss)


def make_value_loss(m):
    def loss(model, obs_mb, g_mb):
        cond = model.condition(m)
        def one(lo):
            _d, v = cond(lo)
            return v
        return jnp.mean((jax.vmap(one)(obs_mb) - g_mb) ** 2)
    return eqx.filter_jit(loss)


def eval_all(loss_fns, data, model, which, n=512):
    """Mean loss over ALL buildings on a FIXED evenly-spaced eval slice.

    Training rotates buildings (b = step % n_buildings), so printing that step's loss
    compares different buildings with different return scales -- a meaningless curve.
    This gives a comparable number, plus the per-building spread.
    """
    per = []
    for lf, (_name, _m, obs, ftar, G) in zip(loss_fns, data):
        y = ftar if which == "policy" else G
        step = max(1, y.shape[0] // n)
        i = jnp.arange(0, y.shape[0], step)[:n]
        per.append(float(lf(model, tuple(o[i] for o in obs), y[i])))
    return float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/transfer4_bc/model_s0.eqx")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--policy-steps", type=int, default=4000)
    ap.add_argument("--value-steps", type=int, default=2000)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--init", default=None,
                    help="start from this checkpoint (e.g. to extend critic training)")
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)

    run = wandb.init(
        project="morel-b2b-transfer-port",
        name=f"bc-fit-{os.path.basename(os.path.dirname(out))}-s{args.seed}",
        config={"algorithm": "behavior_cloning", **vars(args)},
    )
    print(f"[setup] wandb: {run.url}", flush=True)

    print("[load] rebuilding morphologies + loading demos", flush=True)
    data = load_all(args.gamma)
    pol_grads = [make_policy_grad(m) for _n, m, *_r in data]
    val_grads = [make_value_grad(m) for _n, m, *_r in data]
    pol_losses = [make_policy_loss(m) for _n, m, *_r in data]
    val_losses = [make_value_loss(m) for _n, m, *_r in data]
    names = [n for n, *_r in data]

    if args.init is not None:
        from morel_b2b_amorpheus import load_model
        init = args.init if os.path.isabs(args.init) else os.path.join(REPO, args.init)
        model = load_model(init, d_model=64, n_heads=4, n_layers=3)
        print(f"[init] resuming from {init}", flush=True)
    else:
        model = make_model(d_model=64, n_heads=4, n_layers=3, seed=args.seed)
    opt = optax.adam(args.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(args.seed)

    def sample(obs, ftar_or_g, n):
        i = rng.integers(0, ftar_or_g.shape[0], size=n)
        i = jnp.asarray(i)
        return tuple(o[i] for o in obs), ftar_or_g[i]

    print(f"[policy] {args.policy_steps} steps, minibatch {args.minibatch}", flush=True)
    for step in range(args.policy_steps):
        b = step % len(data)
        _n, _m, obs, ftar, _G = data[b]
        mb_obs, mb_y = sample(obs, ftar, args.minibatch)
        v, g = pol_grads[b](model, mb_obs, mb_y)
        upd, opt_state = opt.update(g, opt_state)
        model = eqx.apply_updates(model, upd)
        if step % 1000 == 0 or step == args.policy_steps - 1:
            mean, per = eval_all(pol_losses, data, model, "policy")
            worst = names[int(np.argmax(per))]
            wandb.log({"bc/action_mse": mean, "bc/action_mse_best": min(per),
                       "bc/action_mse_worst": max(per), "phase": 0}, step=step)
            print(f"  step {step:6d}  action MSE (all bldgs) {mean:.5f}  "
                  f"[best {min(per):.4f} worst {max(per):.4f} @ {worst}]", flush=True)

    # critic: value head only (policy frozen), so the value fit cannot move the trunk
    opt_v = optax.adam(1e-3)
    opt_v_state = opt_v.init(eqx.filter(model, eqx.is_array))
    print(f"[critic] {args.value_steps} steps, policy frozen", flush=True)
    for step in range(args.value_steps):
        b = step % len(data)
        _n, _m, obs, _f, G = data[b]
        mb_obs, mb_y = sample(obs, G, args.minibatch)
        v, g = val_grads[b](model, mb_obs, mb_y)
        zeroed = jax.tree_util.tree_map(
            lambda x: x * 0.0 if eqx.is_inexact_array(x) else x, g)
        g_v = eqx.tree_at(lambda t: [t.value1, t.value2], zeroed,
                          replace=[g.value1, g.value2])
        upd, opt_v_state = opt_v.update(g_v, opt_v_state)
        model = eqx.apply_updates(model, upd)
        if step % 1000 == 0 or step == args.value_steps - 1:
            mean, per = eval_all(val_losses, data, model, "value")
            # offset the step so the critic phase continues past the policy phase
            wandb.log({"bc/value_mse": mean, "bc/value_mse_best": min(per),
                       "bc/value_mse_worst": max(per), "phase": 1},
                      step=args.policy_steps + step)
            print(f"  step {step:6d}  value MSE (all bldgs) {mean:.3f}  "
                  f"[best {min(per):.2f} worst {max(per):.2f}]", flush=True)

    save_model(model, out)
    wandb.finish()
    print(f"[done] {out}\nBC_FIT_DONE", flush=True)


if __name__ == "__main__":
    main()
