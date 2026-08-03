"""ModuMorph on the EXACT Amorpheus Part-II protocol: persistent, all-active,
full-year, from scratch. This is the apples-to-apples counterpart of the
Amorpheus `transfer4_persistent_long` run that scored 0/20 full-year.

Pinned to reproduce that run's config (wandb t3x54yyn) — only the model changes:
  20 persistent envs (4 types x 5 buildings), task_occ_e0, full_year, FROM SCRATCH,
  ORIGINAL hyperparameters lr 5e-5 / ent 0.01 / gamma 0.98, n_steps 672,
  minibatch 336, total_steps 5_376_000 = 400 updates = 2.56 passes/year.

The current `train_allactive.py` defaults have since drifted (lr 2e-5 / ent 0 /
gamma 0.99 / n_steps 720); this script hardcodes the Part-II values so a bare
launch is the faithful comparison. Evaluate with `eval_fullyear_panel_modumorph.py`.

    python scripts/train_allactive_modumorph.py --seed 0
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
from morel_b2b_modumorph import load_model, train

YEAR_STEPS = 105_120  # 8760 h x 12 steps/h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--building-types", nargs="+",
                   default=["RetailStandalone", "RestaurantFastFood",
                            "OfficeMedium", "OfficeSmall"])
    p.add_argument("--n-envs", type=int, default=5, help="buildings PER TYPE")
    p.add_argument("--task", default="task_occ_e0")
    p.add_argument("--split", default="train")
    p.add_argument("--indices", type=int, nargs="+", default=None)
    # Part-II protocol values (pinned):
    p.add_argument("--n-steps", type=int, default=672,
                   help="rollout segment per env per update (672 = original Part-II)")
    p.add_argument("--total-steps", type=int, default=5_376_000,
                   help="total env steps across all envs (400 updates x 672 x 20)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--d-context", type=int, default=32)
    p.add_argument("--variant", default="hn", choices=["hn", "faithful", "blind"],
                   help="attribute pathway: hn (channel only), faithful (obs+channel), "
                        "blind (no attributes = retrained knockout)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-checkpoint", default=None, help="default None = FROM SCRATCH")
    p.add_argument("--conc-cap", type=float, default=None)
    p.add_argument("--out", default=None,
                   help="default runs/transfer4_modumorph_<variant>_persistent")
    args = p.parse_args()
    if args.out is None:
        args.out = f"runs/transfer4_modumorph_{args.variant}_persistent"

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"model_s{args.seed}.eqx")

    init_model = None
    if args.init_checkpoint is not None:
        init_model = load_model(args.init_checkpoint, variant=args.variant, d_model=64,
                                d_context=args.d_context, n_heads=4, n_layers=3,
                                conc_cap=args.conc_cap)
        print(f"[setup] warm-starting from {args.init_checkpoint}", flush=True)

    run = wandb.init(
        project="morel-b2b-transfer-port",
        name=f"modumorph-{args.variant}-persistent-{args.task}"
             f"-{'+'.join(args.building_types)}-n{args.n_envs}-s{args.seed}",
        config={"algorithm": f"modumorph_{args.variant}",
                "loop": "persistent_all_active", **vars(args)},
    )
    print(f"[setup] wandb: {run.url}", flush=True)

    n_per_type = len(args.indices) if args.indices else args.n_envs
    n_total_envs = n_per_type * len(args.building_types)
    cfg = PPOConfig(learning_rate=args.lr, ent_coef=args.ent_coef, gamma=args.gamma,
                    n_steps=args.n_steps, minibatch_size=max(1, args.n_steps // 2))
    years = args.total_steps / (YEAR_STEPS * n_total_envs)
    updates = args.total_steps / (args.n_steps * n_total_envs)
    print(f"[protocol] {n_total_envs} persistent envs | {updates:.0f} updates | "
          f"{years:.2f} passes/year | lr {args.lr} ent {args.ent_coef} gamma {args.gamma}",
          flush=True)

    def log_fn(metrics, step):
        sr = metrics.get("rollout/step_reward")
        extra = ({"rollout/episode_return": sr * cfg.n_steps}
                 if sr is not None and "rollout/episode_return" not in metrics else {})
        wandb.log({**metrics, **extra}, step=step)

    train(
        building_types=tuple(args.building_types),
        n_envs_per_type=args.n_envs,
        split=args.split,
        indices=tuple(args.indices) if args.indices else None,
        task=args.task,
        run_period="full_year",
        total_steps=args.total_steps,
        cfg=cfg,
        d_context=args.d_context,
        seed=args.seed,
        log_fn=log_fn,
        checkpoint_path=ckpt,
        init_model=init_model,
        conc_cap=args.conc_cap,
        variant=args.variant,
    )
    print(f"[done] checkpoint {ckpt}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
