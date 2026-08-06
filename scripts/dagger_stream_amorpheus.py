"""STREAMING DAgger: Amorpheus student <- per-building-tuned RBC teacher, with
updates WITHIN episodes.

Why streaming: an EnergyPlus full-year episode is ~105k steps, so batch DAgger
rolls a FROZEN policy for a whole year before a single gradient step. Here the
envs are PERSISTENT (built once, stepped round-robin, auto-reset on done, exactly
the all-active persistent-env protocol of train_allactive*/ppo.train_ppo) and
every `--n-steps` env-steps we pause and run `--updates-per-cycle` supervised SGD
steps against a bounded REPLAY BUFFER.

DAgger, not online BC: each SGD cycle samples from the WHOLE accumulated buffer
(all past visited states), not just the freshly collected slice.

Per env step:
  1. student action (jitted per-morphology act fn; the model is an ARG so updates
     need no recompile)
  2. ALWAYS call that env's own RBC .predict(raw) -- advances the shadow teacher's
     internal state (PI integrators / T&R setpoint) on the student-induced
     observation stream AND is the label
  3. execute a per-step Bernoulli(beta) mixture: expert w.p. beta, else student
  4. push (per-node obs, RBC label inverted via the SAME A/A_pinv probe as
     dagger_collect_amorpheus, reward) into that building's FIFO buffer

beta decays CONTINUOUSLY: beta = 0.5 ** (global_step / half_life), starting at 1.0
(pure expert) -- the streaming analogue of the batch loop's beta_i = 0.5**i.

Teachers are PER ENV and persist alongside their env (their .predict MUTATES
state, so one shared instance would corrupt every building's integrators).

    python scripts/dagger_stream_amorpheus.py --n-envs 5 --total-steps 2000000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
# ONLINE by default (the user watches reward + imitation gap live); overridable.
os.environ.setdefault("WANDB_MODE", "online")

import _pin_dataset  # noqa: F401
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

import json
from baselines.controllers import (
    AirLoopPolicy, UnitaryHvacPolicy, AirLoopConfig, UnitaryHvacConfig)
from morel import Unit
from morel.morphology import trivial_morphology
from morel_amorpheus.policy import decode_flat_action, encode_local_obs
from morel_b2b_amorpheus import (
    AMORPHEUS_B2B_BRIDGE, AMORPHEUS_B2B_SPE_BRIDGE, AMORPHEUS_B2B_SPE_UNIVERSE,
    make_model, load_model, save_model)
from evaluate import _b2b_factory_impl

TASK, RP = "task_occ_e0", "full_year"
YEAR_STEPS = 105_120
CFG_DIR = os.path.join(REPO, "data", "rbc_tuned_configs")


def _controller(bt, idx):
    """Per-(train-)building Optuna-tuned RBC teacher (same as bc_collect_tuned /
    dagger_collect_amorpheus). ONE INSTANCE PER ENV: .predict MUTATES internal
    state, so instances must persist alongside their env and never be shared."""
    p = os.path.join(CFG_DIR, f"{bt}_train_{idx}.json")
    d = json.load(open(p))
    params = d["params"]
    if d["is_vav"]:
        return AirLoopPolicy(AirLoopConfig(**params))
    return UnitaryHvacPolicy(UnitaryHvacConfig(**params))


class Buffer:
    """Bounded FIFO replay for ONE building: per-node obs + normalized-space RBC
    labels + rewards. Contiguous in time (FIFO evicts the oldest), so discounted
    returns-to-go can be computed over the window."""

    def __init__(self, n_nodes: int, cap: int):
        self.cap = cap
        self.obs = [[] for _ in range(n_nodes)]
        self.ftar: list[np.ndarray] = []
        self.rew: list[float] = []

    def push(self, obs_nodes, ftar, rew):
        for i, o in enumerate(obs_nodes):
            self.obs[i].append(o)
        self.ftar.append(ftar)
        self.rew.append(rew)
        if len(self.ftar) > self.cap:
            k = len(self.ftar) - self.cap
            for i in range(len(self.obs)):
                del self.obs[i][:k]
            del self.ftar[:k]
            del self.rew[:k]

    def __len__(self):
        return len(self.ftar)

    def arrays(self):
        obs = tuple(jnp.asarray(np.stack(o)) for o in self.obs)
        return obs, jnp.asarray(np.stack(self.ftar))

    def returns_to_go(self, gamma, bootstrap=0.0):
        """Discounted RTG over the buffer window, bootstrapped at the tail by the
        critic's current value (the streaming analogue of a full-episode RTG)."""
        r = np.asarray(self.rew, dtype=np.float64)
        G = np.zeros_like(r)
        acc = float(bootstrap)
        for t in range(len(r) - 1, -1, -1):
            acc = r[t] + gamma * acc
            G[t] = acc
        return G.astype(np.float32)


def make_act_fn(m):
    """Jitted (policy mean, value) for ONE morphology. `m` is closed over
    (static); `model` is an ARGUMENT, so parameter updates need no recompile."""
    @eqx.filter_jit
    def act(model, local_obs):
        cond = model.condition(m)
        d, v = cond(local_obs)
        return d.mean, v
    return act


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


BRIDGES = {"plain": (AMORPHEUS_B2B_BRIDGE, None),
           "spe":   (AMORPHEUS_B2B_SPE_BRIDGE, AMORPHEUS_B2B_SPE_UNIVERSE)}


class Slot:
    """One PERSISTENT env + its morphology, action-inversion probe, its OWN RBC
    teacher, and its replay buffer. Mirrors ppo._Slot but keeps the RAW obs (the
    RBC consumes raw observations) and the teacher."""

    def __init__(self, bt, idx, task, model, cap, bridge=AMORPHEUS_B2B_BRIDGE):
        self.bt, self.idx, self.label = bt, idx, f"{bt}_{idx}"
        self.env, self.source = _b2b_factory_impl(bt, idx, "train", task, RP)
        self.m = bridge.apply(self.source)
        self.lens = bridge.apply(trivial_morphology(self.source))
        cond = model.condition(self.m)
        self.nids, self.aws = cond.node_ids, cond.act_widths
        D = int(sum(self.aws))

        # Affine target->physical action probe, pseudo-inverted (identical to
        # dagger_collect_amorpheus): labels live in the policy's normalized space.
        def join_flat(f):
            dd = decode_flat_action(self.m, self.nids, self.aws,
                                    np.asarray(f, np.float32))
            return np.asarray(
                self.source.join_actions(self.lens.join_actions((Unit, dd))),
                dtype=np.float64)
        self.join_flat = join_flat
        self.c = join_flat(np.zeros(D))
        self.A = np.stack([join_flat(np.eye(D)[j]) - self.c for j in range(D)], axis=1)
        self.A_pinv = np.linalg.pinv(self.A)

        self.rbc = _controller(bt, idx)      # PERSISTENT teacher for this env
        self.rbc.bind_env(self.env)
        self.raw = None
        self.buf = Buffer(len(self.nids), cap)
        self.ep_return = 0.0
        self.cycle_rewards: list[float] = []
        self.cycle_imit: list[float] = []
        self.last_ep_return = float("nan")

    def ensure_reset(self):
        if self.raw is None:
            self.raw, _ = self.env.reset()
            self.rbc.reset()
            self.ep_return = 0.0

    def local_obs(self):
        _c, per_node = self.lens.split_observation(
            self.source.split_observation(self.raw))
        return encode_local_obs(self.m, self.nids, per_node)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-types", nargs="+",
                    default=["RetailStandalone", "RestaurantFastFood",
                             "OfficeMedium", "OfficeSmall"])
    ap.add_argument("--n-envs", type=int, default=5, help="buildings PER TYPE")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit building indices; default range(n_envs)")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--total-steps", type=int, default=2_000_000,
                    help="total env steps across ALL envs")
    ap.add_argument("--n-steps", type=int, default=512,
                    help="env steps PER ENV between SGD cycles")
    ap.add_argument("--updates-per-cycle", type=int, default=32,
                    help="policy SGD steps per cycle (critic gets the same count)")
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--buffer-cap", type=int, default=100_000,
                    help="TOTAL replay capacity across buildings (split evenly)")
    ap.add_argument("--beta-half-life", type=float, default=200_000,
                    help="global steps for beta to halve (beta = 0.5**(step/hl))")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--out", default="runs/dagger_stream_e0_amorpheus")
    ap.add_argument("--bridge", default="plain", choices=["plain", "spe"],
                    help="spe = compose spectral_pe in front of the Amorpheus "
                         "bridge (breaks the vav_supply symmetry; the one-"
                         "morphism ablation). Checkpoints are NOT weight-"
                         "compatible across bridges.")
    ap.add_argument("--critic-only", action="store_true",
                    help="ON-POLICY CRITIC LEARNING (policy evaluation): freeze the "
                         "actor, always execute the student, never query the RBC "
                         "teacher, skip all policy SGD; train ONLY the value heads on "
                         "returns-to-go. Use to calibrate the critic to a new reward "
                         "(e.g. task_occ_emed) before a PPO fine-tune.")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="save every N cycles")
    ap.add_argument("--snapshot-every", type=int, default=50,
                    help="also keep a numbered snapshot every N cycles")
    args = ap.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"model_s{args.seed}.eqx")

    idxs = args.indices if args.indices else list(range(args.n_envs))
    pairs = [(bt, i) for bt in args.building_types for i in idxs]
    cap_each = max(1000, args.buffer_cap // max(1, len(pairs)))

    bridge, universe = BRIDGES[args.bridge]
    if args.init_checkpoint:
        init = (args.init_checkpoint if os.path.isabs(args.init_checkpoint)
                else os.path.join(REPO, args.init_checkpoint))
        model = load_model(init, d_model=64, n_heads=4, n_layers=3, universe=universe)
        print(f"[setup] warm-starting from {init}", flush=True)
    else:
        model = make_model(d_model=64, n_heads=4, n_layers=3, seed=args.seed,
                           universe=universe)

    run = wandb.init(
        project="morel-b2b-transfer-port",
        name=f"dagger-stream-amorpheus-{args.task}-n{len(pairs)}-s{args.seed}",
        tags=["dagger", "amorpheus", "streaming"],
        config={"algorithm": "dagger_streaming", "arch": "amorpheus",
                "loop": "persistent_all_active", "buffer_cap_each": cap_each,
                **vars(args)},
    )
    print(f"[setup] wandb: {run.url}", flush=True)

    print(f"[setup] building {len(pairs)} persistent envs "
          f"(buffer cap {cap_each}/building)", flush=True)
    slots = [Slot(bt, i, args.task, model, cap_each, bridge=bridge) for bt, i in pairs]
    print(f"[setup] {len(slots)} envs ready", flush=True)

    # memoize jitted fns by morphology OBJECT (the dagger_fit OOM fix): one set
    # per UNIQUE building, constant regardless of how long we stream.
    def _memo(make):
        cache = {}
        def get(m):
            k = id(m)
            if k not in cache:
                cache[k] = make(m)
            return cache[k]
        return get
    _act, _pg, _vg = _memo(make_act_fn), _memo(make_policy_grad), _memo(make_value_grad)

    opt = optax.adam(args.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    opt_v = optax.adam(args.value_lr)
    opt_v_state = opt_v.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(args.seed)

    steps_per_cycle = args.n_steps * len(slots)
    n_cycles = max(1, args.total_steps // steps_per_cycle)
    print(f"[protocol] {len(slots)} persistent envs | {args.n_steps} steps/env/cycle "
          f"| {n_cycles} cycles | {args.updates_per_cycle} SGD steps/cycle | "
          f"beta half-life {args.beta_half_life:g}", flush=True)

    global_step = 0
    t0 = time.time()
    for cycle in range(n_cycles):
        beta = 0.5 ** (global_step / args.beta_half_life)
        for s in slots:
            s.cycle_rewards.clear(); s.cycle_imit.clear()

        # ---------------- streaming rollout (persistent envs, round-robin) -----
        n_expert = 0
        for s in slots:
            s.ensure_reset()
            act = _act(s.m)
            for _t in range(args.n_steps):
                obs_np = [np.asarray(o, np.float32) for o in s.local_obs()]
                f_student, _v = act(model, tuple(jnp.asarray(o) for o in obs_np))
                f_student = np.asarray(f_student, dtype=np.float64)

                if args.critic_only:
                    # pure policy evaluation: student always drives, no teacher,
                    # labels unused (zeros) -- only (obs, reward) matter for V.
                    a_exec = s.join_flat(f_student)
                    s.buf.push(obs_np, np.zeros_like(f_student, dtype=np.float32), 0.0)
                else:
                    # ALWAYS query the teacher: advances shadow state AND is the label
                    a_rbc, _ = s.rbc.predict(s.raw)
                    a_rbc = np.asarray(a_rbc, dtype=np.float64)
                    f_rbc = np.clip((a_rbc - s.c) @ s.A_pinv.T, -1.0, 1.0)

                    # true DAgger gap on the CURRENTLY VISITED state
                    s.cycle_imit.append(float(np.mean((f_student - f_rbc) ** 2)))

                    # per-step Bernoulli(beta) execution mixture
                    use_expert = rng.random() < beta
                    if use_expert:
                        a_exec = a_rbc; n_expert += 1
                    else:
                        a_exec = s.join_flat(f_student)

                    # push BEFORE stepping (obs/label are for the current state)
                    s.buf.push(obs_np, f_rbc.astype(np.float32), 0.0)

                raw, r, term, trunc, _ = s.env.step(
                    np.asarray(a_exec, dtype=np.float32))
                r = float(r)
                s.buf.rew[-1] = r          # reward for the transition just taken
                s.cycle_rewards.append(r)
                s.ep_return += r
                global_step += 1
                if term or trunc:
                    s.last_ep_return = s.ep_return
                    s.raw = None
                    s.ensure_reset()       # persistent env: auto-reset, keep going
                else:
                    s.raw = raw

        # ---------------- SGD cycle: sample the WHOLE buffer -------------------
        # rotate buildings across SGD steps (per-building minibatches keep jit
        # shapes stable, exactly as dagger_fit does).
        ready = [s for s in slots if len(s.buf) >= 8]
        pol_losses, val_losses = [], []
        if ready:
            cached = {}
            for s in ready:
                obs, ftar = s.buf.arrays()
                cached[s.label] = (obs, ftar)

            if not args.critic_only:
                for u in range(args.updates_per_cycle):
                    s = ready[u % len(ready)]
                    obs, ftar = cached[s.label]
                    i = jnp.asarray(rng.integers(0, ftar.shape[0], size=args.minibatch))
                    mb = (tuple(o[i] for o in obs), ftar[i])
                    v, g = _pg(s.m)(model, *mb)
                    upd, opt_state = opt.update(g, opt_state)
                    model = eqx.apply_updates(model, upd)
                    pol_losses.append(float(v))

            # critic on returns-to-go, POLICY FROZEN (grad-masked), as dagger_fit
            for u in range(args.updates_per_cycle):
                s = ready[u % len(ready)]
                obs, _ftar = cached[s.label]
                G = jnp.asarray(s.buf.returns_to_go(args.gamma))
                i = jnp.asarray(rng.integers(0, G.shape[0], size=args.minibatch))
                v, g = _vg(s.m)(model, tuple(o[i] for o in obs), G[i])
                zeroed = jax.tree_util.tree_map(
                    lambda x: x * 0.0 if eqx.is_inexact_array(x) else x, g)
                g_v = eqx.tree_at(lambda t: [t.value1, t.value2], zeroed,
                                  replace=[g.value1, g.value2])
                upd, opt_v_state = opt_v.update(g_v, opt_v_state)
                model = eqx.apply_updates(model, upd)
                val_losses.append(float(v))

        # ---------------- metrics -------------------------------------------
        buf_total = sum(len(s.buf) for s in slots)
        all_rew = [r for s in slots for r in s.cycle_rewards]
        all_imit = [e for s in slots for e in s.cycle_imit]
        logd = {
            "stream/beta": beta,
            "stream/buffer_size": buf_total,
            "stream/expert_frac": n_expert / max(1, steps_per_cycle),
            "stream/global_step": global_step,
            "rollout/mean_reward": float(np.mean(all_rew)),
            "rollout/return_per_cycle": float(np.sum(all_rew) / len(slots)),
            "dagger/imitation_mse": float(np.mean(all_imit)) if all_imit else float("nan"),
            "cycle": cycle,
        }
        if pol_losses:
            logd["train/policy_loss"] = float(np.mean(pol_losses))
        if val_losses:
            logd["train/value_loss"] = float(np.mean(val_losses))
        for s in slots:
            logd[f"rollout/mean_reward/{s.label}"] = float(np.mean(s.cycle_rewards))
            if s.cycle_imit:
                logd[f"dagger/imitation_mse/{s.label}"] = float(np.mean(s.cycle_imit))
            if s.last_ep_return == s.last_ep_return:
                logd[f"rollout/episode_return/{s.label}"] = s.last_ep_return
        wandb.log(logd, step=global_step)

        print(f"[cycle {cycle:4d}] step {global_step:8d}  beta {beta:.3f}  "
              f"buf {buf_total:6d}  imit_mse {logd['dagger/imitation_mse']:.5f}  "
              f"reward {logd['rollout/mean_reward']:+.4f}  "
              f"pol {logd.get('train/policy_loss', float('nan')):.5f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

        if (cycle + 1) % args.checkpoint_every == 0 or cycle == n_cycles - 1:
            save_model(model, ckpt)
            if args.snapshot_every > 0 and (cycle + 1) % args.snapshot_every == 0:
                root, ext = os.path.splitext(ckpt)
                save_model(model, f"{root}_c{cycle + 1}{ext}")
            print(f"  [ckpt] {ckpt}", flush=True)

    save_model(model, ckpt)
    for s in slots:
        s.close()
    wandb.finish()
    print(f"[done] {ckpt}\nDAGGER_STREAM_DONE", flush=True)


if __name__ == "__main__":
    main()
