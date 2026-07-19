"""Port of building2building's transfer experiment onto the current stack.

Trains one Amorpheus policy across multiple building types with the FAITHFUL
cold-start + resampling loop (`morel_b2b_amorpheus.train_transfer` →
`morel_amorpheus.train_ppo_resampling`): a per-type building pool, one building
per type resampled every `--resample-interval` updates, each update cold-started
to episode start. This mirrors the old `scripts/train_transfer.py` regime, but
on the current building2building dataset + morphology bridge.

    python scripts/train_transfer_port.py --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "morel"))

import _pin_dataset  # noqa: F401  pins the b2b dataset revision before any env
import wandb

from morel_amorpheus import PPOConfig
from morel_b2b_amorpheus import load_model, train_transfer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--building-types", nargs="+",
                   default=["RetailStandalone", "RestaurantFastFood",
                            "OfficeMedium", "OfficeSmall"])
    p.add_argument("--n-buildings-per-type", type=int, default=10)
    p.add_argument("--task", default="task_occ_e0")
    p.add_argument("--total-iterations", type=int, default=400)
    p.add_argument("--resample-interval", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-active", action="store_true",
                   help="every building active each update (joint gradient over the "
                        "whole pool) instead of resampling one per type")
    p.add_argument("--n-workers", type=int, default=1,
                   help="parallel rollout worker processes (one env per worker)")
    p.add_argument("--init-checkpoint", default=None,
                   help="warm-start PPO from this checkpoint instead of a fresh net")
    p.add_argument("--out", default="runs/transfer_port")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"model_s{args.seed}.eqx")

    init_model = None
    if args.init_checkpoint is not None:
        init_model = load_model(args.init_checkpoint, d_model=64, n_heads=4, n_layers=3)
        print(f"[setup] warm-starting from {args.init_checkpoint}", flush=True)

    run = wandb.init(
        project="morel-b2b-transfer-port",
        name=f"transfer-{args.task}-{len(args.building_types)}types"
             f"-pool{args.n_buildings_per_type}-r{args.resample_interval}-s{args.seed}",
        config={"algorithm": "amorpheus", "loop": "cold_start_resampling",
                **vars(args)},
    )
    print(f"[setup] wandb: {run.url}", flush=True)
    print(f"[train] {args.building_types}, pool {args.n_buildings_per_type}/type, "
          f"{args.total_iterations} iters, resample every {args.resample_interval}, "
          f"task={args.task}", flush=True)

    cfg = PPOConfig(learning_rate=args.lr, ent_coef=args.ent_coef, gamma=args.gamma)

    def log_fn(metrics, step):
        # step_reward is the mean per-step reward; scale to an episode-return line
        # comparable to RBC's returns. Only fill in if the trainer didn't already
        # log a real episode_return (from finished episodes).
        sr = metrics.get("rollout/step_reward")
        extra = ({"rollout/episode_return": sr * cfg.n_steps}
                 if sr is not None and "rollout/episode_return" not in metrics else {})
        wandb.log({**metrics, **extra}, step=step)

    train_transfer(
        building_types=tuple(args.building_types),
        n_buildings_per_type=args.n_buildings_per_type,
        task=args.task,
        total_iterations=args.total_iterations,
        resample_interval=args.resample_interval,
        cfg=cfg,
        seed=args.seed,
        log_fn=log_fn,
        checkpoint_path=ckpt,
        init_model=init_model,
        all_active=args.all_active,
        n_workers=args.n_workers,
    )
    print(f"[done] checkpoint {ckpt}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
