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
    # --- protocol: everything is expressed in EnergyPlus YEARS ------------------
    # A year is 8760 h x 12 steps/h = 105_120 steps. n_steps=720 (2.5 days) divides
    # it exactly -> 146 updates == 1 year, so "how many passes over the year" is
    # always an integer you can read off directly. (The old n_steps=672 gave
    # 156.43 updates/year, which is why runs ended on fractions like 2.56 years.)
    p.add_argument("--years", type=float, default=None,
                   help="train for N passes over the EnergyPlus year (per env); "
                        "overrides --total-steps")
    p.add_argument("--n-steps", type=int, default=720,
                   help="rollout segment per env per update (720 = 2.5 d = 146/yr)")
    p.add_argument("--total-steps", type=int, default=1_000_000,
                   help="raw step budget; prefer --years")
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

    YEAR_STEPS = 105_120  # 8760 h x 12 steps/h -- the standard EnergyPlus year
    cfg = PPOConfig(learning_rate=args.lr, ent_coef=args.ent_coef, gamma=args.gamma,
                    n_steps=args.n_steps, minibatch_size=max(1, args.n_steps // 2))

    upd_per_year = YEAR_STEPS / args.n_steps
    if args.years is not None:
        total_steps = int(round(args.years * YEAR_STEPS * args.n_envs))
    else:
        total_steps = args.total_steps
    years = total_steps / (YEAR_STEPS * args.n_envs)
    updates = total_steps / (args.n_steps * args.n_envs)
    print(f"[protocol] year={YEAR_STEPS} steps | n_steps={args.n_steps} "
          f"-> {upd_per_year:.2f} updates/year"
          f"{' (NOT integer!)' if abs(upd_per_year-round(upd_per_year))>1e-9 else ''}",
          flush=True)
    print(f"[protocol] {years:.2f} YEARS per env  ({updates:.0f} updates, "
          f"{total_steps} total steps across {args.n_envs} envs/type)", flush=True)

    def log_fn(metrics, step):
        sr = metrics.get("rollout/step_reward")
        extra = ({"rollout/episode_return": sr * cfg.n_steps}
                 if sr is not None and "rollout/episode_return" not in metrics else {})
        wandb.log({**metrics, **extra}, step=step)

    train(
        building_types=tuple(args.building_types),
        n_envs_per_type=args.n_envs,
        task=args.task,
        total_steps=total_steps,
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
