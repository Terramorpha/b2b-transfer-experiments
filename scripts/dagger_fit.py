"""DAgger refit: fit a fresh ModuMorph net to ALL accumulated DAgger demos.

Mirrors bc_fit.py's two deliberately-separate phases, but against the ModuMorph
model + MODUMORPH_B2B_BRIDGE instead of Amorpheus:
  1. POLICY  -- fit the Beta mean to the expert's target actions (MSE).
  2. CRITIC  -- fit the value head to discounted returns-to-go, POLICY FROZEN
                (grad-masked) so the value fit cannot distort the trunk.

The conditioning morphology comes from MODUMORPH_B2B_BRIDGE.apply(source), and
node_ids/act_widths therefore come from the ModuMorph model's .condition(m).
Reads EVERY npz under --demos (the aggregated DAgger dir), so successive DAgger
iterations train on the union of all iterations' data.

    python scripts/dagger_fit.py --demos data/dagger_demos --out runs/dagger/model_it0.eqx
"""
import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "online")

import _pin_dataset  # noqa: F401
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from morel_b2b_modumorph import (
    MODUMORPH_B2B_BRIDGE, make_model, load_model, save_model)
from evaluate import _b2b_factory_impl

TASK, RP = "task_occ_e0", "full_year"


def returns_to_go(rew, gamma):
    G = np.zeros_like(rew)
    acc = 0.0
    for t in range(len(rew) - 1, -1, -1):
        acc = rew[t] + gamma * acc
        G[t] = acc
    return G


def load_all(demos_dir, variant, gamma, task=TASK, stride_default=4):
    """Rebuild each building's ModuMorph morphology (needed to condition the
    model) and pair it with its stored demonstrations. A building can appear in
    several npz (one per DAgger iteration) -- each is its own training buffer,
    keyed by the file's basename, so the whole aggregated dataset is used."""
    data = []
    # cache morphologies by (bt, idx): EnergyPlus env construction is expensive
    # and the same building recurs across iteration-tagged files.
    m_cache = {}
    for path in sorted(glob.glob(os.path.join(demos_dir, "*.npz"))):
        name = os.path.basename(path)[:-4]
        # filenames look like "{bt}_{idx}" (bc) or "{bt}_{idx}_it{iteration}".
        core = name
        if "_it" in core:
            core = core[:core.rindex("_it")]
        bt, idx = core.rsplit("_", 1)
        key = (bt, int(idx))
        if key not in m_cache:
            env, source = _b2b_factory_impl(bt, int(idx), "train", task, RP)
            m_cache[key] = MODUMORPH_B2B_BRIDGE.apply(source)
            env.close()
        m = m_cache[key]
        d = np.load(path)
        obs = tuple(jnp.asarray(d[k]) for k in sorted(
            (k for k in d.files if k.startswith("obs")),
            key=lambda s: int(s[3:])))
        ftar = jnp.asarray(d["f_target"])
        stride = int(d["stride"]) if "stride" in d.files else stride_default
        G = returns_to_go(np.asarray(d["rew"]), gamma)[::stride][:ftar.shape[0]]
        data.append((name, m, obs, ftar, jnp.asarray(G, dtype=jnp.float32)))
        print(f"  loaded {name:30s} {ftar.shape[0]:6d} samples, "
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
    """Mean loss over ALL buffers on a FIXED evenly-spaced eval slice."""
    per = []
    for lf, (_name, _m, obs, ftar, G) in zip(loss_fns, data):
        y = ftar if which == "policy" else G
        step = max(1, y.shape[0] // n)
        i = jnp.arange(0, y.shape[0], step)[:n]
        per.append(float(lf(model, tuple(o[i] for o in obs), y[i])))
    return float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True,
                    help="aggregated DAgger demo dir (all *.npz are used)")
    ap.add_argument("--out", required=True, help="output ModuMorph .eqx")
    ap.add_argument("--variant", default="hn", choices=["hn", "faithful", "blind"])
    ap.add_argument("--task", default=TASK,
                    help="b2b task preset used to rebuild morphologies (must match collect)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--policy-steps", type=int, default=4000)
    ap.add_argument("--value-steps", type=int, default=2000)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--init", default=None,
                    help="start from this ModuMorph checkpoint")
    args = ap.parse_args()

    demos = (args.demos if os.path.isabs(args.demos)
             else os.path.join(REPO, args.demos))
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)

    run = wandb.init(
        project="morel-b2b-transfer-port",
        name=f"dagger-fit-{os.path.basename(out)[:-4]}-s{args.seed}",
        config={"algorithm": "dagger_bc", "arch": "modumorph", **vars(args)},
    )
    print(f"[setup] wandb: {run.url}", flush=True)

    print(f"[load] rebuilding morphologies + loading demos from {demos}", flush=True)
    data = load_all(demos, args.variant, args.gamma, task=args.task)
    if not data:
        raise SystemExit(f"no *.npz demos found under {demos}")
    # Memoize the jit'd fns by morphology OBJECT. The aggregated DAgger dir holds
    # several iteration-tagged files per building, all sharing one cached `m`
    # (load_all caches by (bt,idx)). Building a fresh eqx.filter_jit per FILE made
    # the compiled-executable count grow with iterations: it0=20 files ok,
    # it1=40 ok, it2=60 -> the critic phase's second compilation wave stacked on
    # the resident policy cache and OOM'd the 16GB box. Keying by id(m) caps the
    # count at the number of UNIQUE buildings, constant across iterations.
    def _memo(make):
        cache: dict = {}
        def get(m):
            k = id(m)
            if k not in cache:
                cache[k] = make(m)
            return cache[k]
        return get
    _pg, _vg = _memo(make_policy_grad), _memo(make_value_grad)
    _pl, _vl = _memo(make_policy_loss), _memo(make_value_loss)
    pol_grads = [_pg(m) for _n, m, *_r in data]
    val_grads = [_vg(m) for _n, m, *_r in data]
    pol_losses = [_pl(m) for _n, m, *_r in data]
    val_losses = [_vl(m) for _n, m, *_r in data]
    names = [n for n, *_r in data]

    if args.init is not None:
        init = args.init if os.path.isabs(args.init) else os.path.join(REPO, args.init)
        model = load_model(init, variant=args.variant, d_model=64, d_context=32,
                           n_heads=4, n_layers=3)
        print(f"[init] resuming from {init}", flush=True)
    else:
        model = make_model(variant=args.variant, d_model=64, d_context=32,
                           n_heads=4, n_layers=3, seed=args.seed)
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
            wandb.log({"dagger/action_mse": mean, "dagger/action_mse_best": min(per),
                       "dagger/action_mse_worst": max(per), "phase": 0}, step=step)
            print(f"  step {step:6d}  action MSE (all bldgs) {mean:.5f}  "
                  f"[best {min(per):.4f} worst {max(per):.4f} @ {worst}]", flush=True)

    # critic: value head only (policy frozen)
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
            wandb.log({"dagger/value_mse": mean, "dagger/value_mse_best": min(per),
                       "dagger/value_mse_worst": max(per), "phase": 1},
                      step=args.policy_steps + step)
            print(f"  step {step:6d}  value MSE (all bldgs) {mean:.3f}  "
                  f"[best {min(per):.2f} worst {max(per):.2f}]", flush=True)

    save_model(model, out)
    wandb.finish()
    print(f"[done] {out}\nDAGGER_FIT_DONE", flush=True)


if __name__ == "__main__":
    main()
