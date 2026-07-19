"""Step 2 of the OfficeSmall warm-start recipe: pretrain the critic.

The RBC clone (officesmall_rbc_clone.py) fit only the POLICY; the value head is
still at random init, so PPO's first updates compute advantages from a garbage
critic and knock the good policy out of its basin. Here we roll RBC out again for
per-step rewards, compute discounted returns-to-go, and fit ONLY the value head to
them (policy frozen via grad-masking).

Reads runs/officesmall_rbc_bc, writes runs/officesmall_rbc_bcv. Next: train_allactive.py.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from baselines.controllers import UnitaryHvacPolicy
from morel.morphology import trivial_morphology
from morel_amorpheus.policy import encode_local_obs
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, load_model, save_model
from evaluate import _b2b_factory_impl

IN = os.path.join(REPO, "runs", "officesmall_rbc_bc", "model_s0.eqx")
OUT = os.path.join(REPO, "runs", "officesmall_rbc_bcv", "model_s0.eqx")
TASK, RP, CHUNK, POOL, GAMMA = "task_occ_e0", "full_year", 672, 10, 0.99
EPOCHS, LR = 150, 1e-3


def returns_to_go(rews):
    G = np.zeros_like(rews)
    acc = 0.0
    for t in range(len(rews) - 1, -1, -1):
        acc = rews[t] + GAMMA * acc
        G[t] = acc
    return G


def collect(model):
    data = []
    for idx in range(POOL):
        env, source = _b2b_factory_impl("OfficeSmall", idx, "train", TASK, RP)
        m = AMORPHEUS_B2B_BRIDGE.apply(source)
        lens = AMORPHEUS_B2B_BRIDGE.apply(trivial_morphology(source))
        cond = model.condition(m)
        pol = UnitaryHvacPolicy(); pol.bind_env(env)
        raw, _ = env.reset(); pol.reset()
        obs_traj, rews = [], []
        for _t in range(CHUNK):
            _c, per_node = lens.split_observation(source.split_observation(raw))
            obs_traj.append([np.asarray(o) for o in encode_local_obs(m, cond.node_ids, per_node)])
            act, _ = pol.predict(raw)
            raw, r, term, trunc, _ = env.step(np.asarray(act, dtype=np.float32))
            rews.append(float(r))
            if term or trunc:
                break
        env.close()
        G = returns_to_go(np.asarray(rews))
        n_nodes = len(obs_traj[0])
        obs_stack = tuple(jnp.asarray(np.stack([o[i] for o in obs_traj]))
                          for i in range(n_nodes))
        data.append((m, obs_stack, jnp.asarray(G)))
        print(f"  idx {idx}: {len(rews)} steps, return-to-go[0]={G[0]:.2f}", flush=True)
    return data


def make_grad_fn(m):
    def vloss(student, obs_stack, G):
        cond = student.condition(m)
        def one(lo):
            _d, v = cond(lo)
            return v
        vals = jax.vmap(one)(obs_stack)
        return jnp.mean((vals - G) ** 2)
    return eqx.filter_jit(eqx.filter_value_and_grad(vloss))


def main():
    student = load_model(IN, d_model=64, n_heads=4, n_layers=3)
    print("[collect] re-rolling RBC for rewards/returns", flush=True)
    data = collect(student)
    grad_fns = [make_grad_fn(m) for (m, *_r) in data]

    opt = optax.adam(LR)
    opt_state = opt.init(eqx.filter(student, eqx.is_array))
    print(f"[critic] fitting value head only ({EPOCHS} epochs), policy frozen", flush=True)
    for epoch in range(EPOCHS):
        tot = 0.0
        for gfn, (_m, obs_stack, G) in zip(grad_fns, data):
            val, grads = gfn(student, obs_stack, G)
            zeroed = jax.tree_util.tree_map(
                lambda x: x * 0.0 if eqx.is_inexact_array(x) else x, grads)
            grads_v = eqx.tree_at(lambda t: [t.value1, t.value2], zeroed,
                                  replace=[grads.value1, grads.value2])
            updates, opt_state = opt.update(grads_v, opt_state)
            student = eqx.apply_updates(student, updates)
            tot += float(val)
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:3d}  value MSE {tot/len(data):.4f}", flush=True)

    save_model(student, OUT)
    print(f"[done] policy+critic warm-start -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
