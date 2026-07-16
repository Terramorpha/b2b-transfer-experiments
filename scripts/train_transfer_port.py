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
from morel_b2b_amorpheus import train_transfer


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/transfer_port")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"model_s{args.seed}.eqx")

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

    train_transfer(
        building_types=tuple(args.building_types),
        n_buildings_per_type=args.n_buildings_per_type,
        task=args.task,
        total_iterations=args.total_iterations,
        resample_interval=args.resample_interval,
        cfg=PPOConfig(learning_rate=args.lr),
        seed=args.seed,
        log_fn=lambda metrics, step: wandb.log(metrics, step=step),
        checkpoint_path=ckpt,
    )
    print(f"[done] checkpoint {ckpt}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
