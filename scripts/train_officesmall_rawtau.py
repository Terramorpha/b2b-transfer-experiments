"""OfficeSmall specialist with the comfort penalty UN-normalized (tau_T=1).

A/B against the normalized specialist: if OfficeSmall's large per-building tau_T
(~117) was starving the comfort gradient and causing the under-heating collapse,
forcing tau_T=1 gives a strong signal and the policy should learn to ramp heat
and beat RBC. Everything else identical (OfficeSmall, pool 10, resample 10, 400
iters, task_occ_e0 => occupancy setpoint + energy_weight 0).
"""
from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "morel"))

import _pin_dataset  # noqa: F401  pins the b2b dataset revision before any env

# --- force tau_T = tau_E = 1 (no per-building normalization) ---------------
import building2building.data.reward_normalizers as _rn  # noqa: E402
_orig_resolve = _rn.resolve_reward_normalizer


def _forced_resolve(building_type, building_id, **kw):
    n = _orig_resolve(building_type, building_id, **kw)
    return dataclasses.replace(n, tau_T=1.0, tau_E=1.0)


_rn.resolve_reward_normalizer = _forced_resolve
# ---------------------------------------------------------------------------

import wandb  # noqa: E402

from morel_amorpheus import PPOConfig  # noqa: E402
from morel_b2b_amorpheus import train_transfer  # noqa: E402

OUT = "runs/officesmall_rawtau_400"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ckpt = os.path.join(OUT, "model_s0.eqx")
    run = wandb.init(
        project="morel-b2b-transfer-port",
        name="officesmall-RAWTAU1-occ_e0-pool10-r10-s0",
        config={"algorithm": "amorpheus", "loop": "cold_start_resampling",
                "building_types": ["OfficeSmall"], "task": "task_occ_e0",
                "total_iterations": 400, "resample_interval": 10,
                "n_buildings_per_type": 10, "seed": 0,
                "tau_T_override": 1.0, "tau_E_override": 1.0,
                "note": "comfort penalty un-normalized (A/B vs normalized specialist)"},
    )
    print(f"[setup] wandb: {run.url}", flush=True)
    print("[train] OfficeSmall specialist with tau_T=1 (un-normalized comfort)", flush=True)
    train_transfer(
        building_types=("OfficeSmall",),
        n_buildings_per_type=10,
        task="task_occ_e0",
        total_iterations=400,
        resample_interval=10,
        cfg=PPOConfig(learning_rate=5e-5),
        seed=0,
        log_fn=lambda metrics, step: wandb.log(metrics, step=step),
        checkpoint_path=ckpt,
    )
    print(f"[done] checkpoint {ckpt}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
