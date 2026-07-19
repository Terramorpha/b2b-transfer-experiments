"""All-active warm-started fine-tune: train on the WHOLE building pool jointly.

Uses the persistent trainer (`train`) so all `--n-envs` buildings are active every
update — one joint PPO gradient over the full pool, instead of the resampling
loop's one-building-at-a-time. Tests whether joint multi-building training prevents
the training-building overfitting that 1-at-a-time PPO showed (H14). Warm-started
from the RBC clone + pretrained critic.

    python scripts/train_allactive.py --init-checkpoint runs/officesmall_rbc_bcv/model_s0.eqx
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "morel"))

import _pin_dataset  # noqa: F401
import wandb

from morel_amorpheus import PPOConfig
from morel_b2b_amorpheus import load_model, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--building-types", nargs="+", default=["OfficeSmall"])
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--task", default="task_occ_e0")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--out", default="runs/officesmall_allactive")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"model_s{args.seed}.eqx")

    init_model = None
    if args.init_checkpoint is not None:
        init_model = load_model(args.init_checkpoint, d_model=64, n_heads=4, n_layers=3)
        print(f"[setup] warm-starting from {args.init_checkpoint}", flush=True)

    run = wandb.init(
        project="morel-b2b-transfer-port",
        name=f"allactive-{args.task}-{'+'.join(args.building_types)}"
             f"-n{args.n_envs}-s{args.seed}",
        config={"algorithm": "amorpheus", "loop": "persistent_all_active", **vars(args)},
    )
    print(f"[setup] wandb: {run.url}", flush=True)
    print(f"[train] all-active {args.building_types} x{args.n_envs}, "
          f"{args.total_steps} steps, task={args.task}", flush=True)

    cfg = PPOConfig(learning_rate=args.lr, ent_coef=args.ent_coef, gamma=args.gamma)

    def log_fn(metrics, step):
        sr = metrics.get("rollout/step_reward")
        extra = {"rollout/episode_return": sr * cfg.n_steps} if sr is not None else {}
        wandb.log({**metrics, **extra}, step=step)

    train(
        building_types=tuple(args.building_types),
        n_envs_per_type=args.n_envs,
        task=args.task,
        total_steps=args.total_steps,
        cfg=cfg,
        seed=args.seed,
        log_fn=log_fn,
        checkpoint_path=ckpt,
        init_model=init_model,
    )
    print(f"[done] checkpoint {ckpt}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
