"""Wall-clock benchmark of the persistent all-active PPO driver (train_allactive.py).

Reproduces the setup of the fig-1 from-scratch comfort run (wandb t3x54yyn on the
old 4-core box: 20 envs = 4 types x 5 train buildings, task_occ_e0, lr 5e-5,
ent 0.01, gamma 0.98) for a handful of updates and prints per-update wall time
plus env-step throughput, so machines can be compared.

NOTE: the driver steps its envs SEQUENTIALLY in one process (round-robin over the
20 persistent EnergyPlus threads, one policy forward per env per step), so this
mostly measures single-core speed + the JAX update phase, not core count.

    python scripts/bench_allactive.py --updates 8

Old-box reference (host jiv, 4 cores): ~55 s/update at n_steps=720 x 20 envs
(=14 400 env-steps/update, ~260 env-steps/s); 373 updates in 5.74 h.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "morel"))
os.environ.setdefault("WANDB_MODE", "disabled")

import _pin_dataset  # noqa: F401

from morel_amorpheus import PPOConfig
from morel_b2b_amorpheus import train

YEAR_STEPS = 105_120


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--building-types", nargs="+",
                   default=["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"])
    p.add_argument("--n-envs", type=int, default=5, help="buildings per type")
    p.add_argument("--task", default="task_occ_e0")
    p.add_argument("--n-steps", type=int, default=720)
    p.add_argument("--updates", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/_bench_allactive")
    args = p.parse_args()

    n_total_envs = args.n_envs * len(args.building_types)
    total_steps = args.updates * args.n_steps * n_total_envs
    cfg = PPOConfig(learning_rate=args.lr, ent_coef=args.ent_coef, gamma=args.gamma,
                    n_steps=args.n_steps, minibatch_size=max(1, args.n_steps // 2))
    print(f"[bench] {n_total_envs} envs ({args.building_types} x{args.n_envs}) | "
          f"task={args.task} | n_steps={args.n_steps} | {args.updates} updates | "
          f"{total_steps} env-steps", flush=True)

    times: list[float] = []
    t_wall0 = time.time()
    t_last = [t_wall0]

    def log_fn(metrics, step):
        now = time.time()
        upd = metrics.get("time/update_seconds", float("nan"))
        wall = now - t_last[0]
        t_last[0] = now
        times.append(upd)
        env_steps = args.n_steps * n_total_envs
        print(f"[update {len(times):3d}] update_seconds={upd:7.2f}  wall_since_last={wall:7.2f}  "
              f"env-steps/s={env_steps / upd:7.1f}  step_reward={metrics.get('rollout/step_reward', float('nan')):.4f}"
              f"  | rollout={metrics.get('time/rollout_seconds', float('nan')):6.2f}"
              f" gae={metrics.get('time/gae_dataset_seconds', float('nan')):5.2f}"
              f" train={metrics.get('time/train_seconds', float('nan')):6.2f}"
              f" | act={metrics.get('time/rollout_act_seconds', float('nan')):5.2f}"
              f" wait={metrics.get('time/rollout_env_wait_seconds', float('nan')):5.2f}"
              f" obs={metrics.get('time/rollout_observe_seconds', float('nan')):5.2f}",
              flush=True)

    os.makedirs(args.out, exist_ok=True)
    t_setup0 = time.time()
    train(
        building_types=tuple(args.building_types),
        n_envs_per_type=args.n_envs,
        split="train",
        task=args.task,
        total_steps=total_steps,
        cfg=cfg,
        seed=args.seed,
        log_fn=log_fn,
        checkpoint_path=os.path.join(args.out, f"model_s{args.seed}.eqx"),
    )
    t_end = time.time()
    setup = (t_end - t_setup0) - sum(times)
    first, rest = times[0], times[1:]
    steady = sum(rest) / len(rest) if rest else float("nan")
    env_steps = args.n_steps * n_total_envs
    print("[bench] ---- summary ----", flush=True)
    print(f"[bench] total wall {t_end - t_wall0:8.1f} s | env build + non-update overhead {setup:6.1f} s", flush=True)
    print(f"[bench] first update (incl. jit) {first:7.2f} s", flush=True)
    print(f"[bench] steady-state {steady:7.2f} s/update  ->  {env_steps / steady:7.1f} env-steps/s  "
          f"->  {YEAR_STEPS / args.n_steps * steady / 3600:5.2f} h per simulated year on {n_total_envs} envs", flush=True)
    print(f"[bench] old 4-core box reference: ~55 s/update, ~2.2 h/year", flush=True)


if __name__ == "__main__":
    main()
