"""ModuMorph counterpart of `train_transfer_port.py` — the fig-1 from-scratch
4-type transfer, run with ModuMorph instead of Amorpheus.

Same faithful cold-start + resampling loop (`morel_b2b_modumorph.train_transfer`
-> the shared `morel_amorpheus.train_ppo_resampling` driver), the same building
pools / task / run period / iteration budget as the Amorpheus port, so the two
are directly comparable on fig-1. The ONLY difference is the model + its lens:
ModuMorph keeps the attribute pathway (commons_to_node only, no inline) and
conditions each node's I/O weights on its geometry via hypernetworks. This tests
whether conditioning on static attributes helps learning where Amorpheus got 0/20.

    python scripts/train_transfer_modumorph.py --seed 0 --n-workers 4
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
from morel_b2b_modumorph import load_model, train_transfer


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
    p.add_argument("--d-context", type=int, default=32,
                   help="width of the per-node morphology-context embedding the "
                        "hypernetworks condition on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-active", action="store_true",
                   help="every building active each update (joint gradient over the "
                        "whole pool) instead of resampling one per type")
    p.add_argument("--n-workers", type=int, default=1,
                   help="parallel rollout worker processes (one env per worker)")
    p.add_argument("--init-checkpoint", default=None,
                   help="warm-start PPO from this checkpoint instead of a fresh net")
    p.add_argument("--baseline-json", default=None,
                   help="JSON {suffix: value}; logged as constant baseline/<suffix> "
                        "series each update, for dashed reference lines on wandb")
    p.add_argument("--out", default="runs/transfer_modumorph")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"model_s{args.seed}.eqx")

    init_model = None
    if args.init_checkpoint is not None:
        init_model = load_model(args.init_checkpoint, d_model=64, d_context=args.d_context,
                                n_heads=4, n_layers=3)
        print(f"[setup] warm-starting from {args.init_checkpoint}", flush=True)

    run = wandb.init(
        project="morel-b2b-transfer-port",
        name=f"modumorph-transfer-{args.task}-{len(args.building_types)}types"
             f"-pool{args.n_buildings_per_type}-r{args.resample_interval}-s{args.seed}",
        config={"algorithm": "modumorph", "loop": "cold_start_resampling",
                **vars(args)},
    )
    print(f"[setup] wandb: {run.url}", flush=True)
    print(f"[train] {args.building_types}, pool {args.n_buildings_per_type}/type, "
          f"{args.total_iterations} iters, resample every {args.resample_interval}, "
          f"task={args.task}", flush=True)

    cfg = PPOConfig(learning_rate=args.lr, ent_coef=args.ent_coef, gamma=args.gamma)

    baselines = {}
    if args.baseline_json is not None:
        import json
        with open(args.baseline_json) as f:
            baselines = {f"baseline/{k}": float(v) for k, v in json.load(f).items()}

    def log_fn(metrics, step):
        sr = metrics.get("rollout/step_reward")
        extra = ({"rollout/episode_return": sr * cfg.n_steps}
                 if sr is not None and "rollout/episode_return" not in metrics else {})
        wandb.log({**metrics, **extra, **baselines}, step=step)

    train_transfer(
        building_types=tuple(args.building_types),
        n_buildings_per_type=args.n_buildings_per_type,
        task=args.task,
        total_iterations=args.total_iterations,
        resample_interval=args.resample_interval,
        cfg=cfg,
        d_context=args.d_context,
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
