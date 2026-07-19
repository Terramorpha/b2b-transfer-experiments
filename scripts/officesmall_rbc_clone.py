"""Step 1 of the OfficeSmall warm-start recipe: behavior-clone RBC into a fresh net.

RBC acts in physical action space; the policy emits a target (normalized [-1,1])
action mapped to physical by a fixed affine. We probe that affine per building
(join at f=0 and each basis vector), pseudo-invert it to turn RBC's actions into
target actions, then fit a fresh net's Beta *mean* to them by MSE.

Produces runs/officesmall_rbc_bc/model_s0.eqx. Next: officesmall_value_pretrain.py.
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
from morel import Unit
from morel.morphology import trivial_morphology
from morel_amorpheus.policy import decode_flat_action, encode_local_obs
from morel_b2b_amorpheus import AMORPHEUS_B2B_BRIDGE, make_model, save_model
from evaluate import _b2b_factory_impl

OUT = os.path.join(REPO, "runs", "officesmall_rbc_bc", "model_s0.eqx")
TASK, RP, CHUNK, POOL = "task_occ_e0", "full_year", 672, 10
EPOCHS, LR, STUDENT_SEED = 200, 3e-4, 999


def collect():
    schema_model = make_model(d_model=64, n_heads=4, n_layers=3, seed=0)
    data = []
    for idx in range(POOL):
        env, source = _b2b_factory_impl("OfficeSmall", idx, "train", TASK, RP)
        m = AMORPHEUS_B2B_BRIDGE.apply(source)
        lens = AMORPHEUS_B2B_BRIDGE.apply(trivial_morphology(source))
        cond = schema_model.condition(m)
        nids, aws = cond.node_ids, cond.act_widths
        D = int(sum(aws))

        def join_flat(f):
            dd = decode_flat_action(m, nids, aws, np.asarray(f, np.float32))
            return np.asarray(source.join_actions(lens.join_actions((Unit, dd))),
                              dtype=np.float64)
        c = join_flat(np.zeros(D))
        A = np.stack([join_flat(np.eye(D)[j]) - c for j in range(D)], axis=1)
        A_pinv = np.linalg.pinv(A)

        pol = UnitaryHvacPolicy(); pol.bind_env(env)
        raw, _ = env.reset(); pol.reset()
        obs_traj, a_traj = [], []
        for _t in range(CHUNK):
            _c, per_node = lens.split_observation(source.split_observation(raw))
            obs_traj.append([np.asarray(o) for o in encode_local_obs(m, nids, per_node)])
            act, _ = pol.predict(raw)
            a_traj.append(np.asarray(act, dtype=np.float64))
            raw, _r, term, trunc, _ = env.step(np.asarray(act, dtype=np.float32))
            if term or trunc:
                break
        env.close()
        a_arr = np.stack(a_traj)
        f_target = np.clip((a_arr - c) @ A_pinv.T, -1.0, 1.0)
        n_nodes = len(obs_traj[0])
        obs_stack = tuple(jnp.asarray(np.stack([o[i] for o in obs_traj]))
                          for i in range(n_nodes))
        data.append((m, obs_stack, jnp.asarray(f_target)))
        recon = (f_target @ A.T) + c
        print(f"  idx {idx}: {len(obs_traj)} steps, D={D}, "
              f"recon_err {np.abs(recon - a_arr).mean():.3f}", flush=True)
    return data


def make_grad_fn(m):
    def loss(student, obs_stack, ftar):
        cond = student.condition(m)
        def one(lo):
            d, _ = cond(lo)
            return d.mean
        means = jax.vmap(one)(obs_stack)
        return jnp.mean((means - ftar) ** 2)
    return eqx.filter_jit(eqx.filter_value_and_grad(loss))


def main():
    print("[collect] rolling out RBC across the OfficeSmall train pool", flush=True)
    data = collect()
    grad_fns = [make_grad_fn(m) for (m, *_r) in data]

    student = make_model(d_model=64, n_heads=4, n_layers=3, seed=STUDENT_SEED)
    opt = optax.adam(LR)
    opt_state = opt.init(eqx.filter(student, eqx.is_array))
    print(f"[imitate] {EPOCHS} epochs, fresh seed {STUDENT_SEED}", flush=True)
    for epoch in range(EPOCHS):
        tot = 0.0
        for gfn, (_m, obs_stack, ftar) in zip(grad_fns, data):
            val, grads = gfn(student, obs_stack, ftar)
            updates, opt_state = opt.update(grads, opt_state)
            student = eqx.apply_updates(student, updates)
            tot += float(val)
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:3d}  mean action MSE {tot/len(data):.4f}", flush=True)

    save_model(student, OUT)
    print(f"[done] student -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
