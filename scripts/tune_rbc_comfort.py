"""Optuna-tune a per-building RBC for FULL-YEAR comfort (task_occ_e0).

Dispatches by type: AirLoopPolicy/AirLoopConfig for OfficeMedium (VAV), else
UnitaryHvacPolicy/UnitaryHvacConfig. Objective = full-year episode return (the
reported metric), so the study's best value IS the tuned RBC's full-year comfort
return. Saves the winning param dict to data/rbc_tuned_configs/<bt>_<split>_<idx>.json
for BC-data collection to load.

    python scripts/tune_rbc_comfort.py --building-type RestaurantFastFood --index 0 --split train --n-trials 25
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import numpy as np
import optuna

import building2building as b2b
from baselines.controllers import (
    UnitaryHvacPolicy, UnitaryHvacConfig, AirLoopPolicy, AirLoopConfig,
)
from building2building.data.registry import get_registry

TASK, RP, MAX_STEPS = "task_occ_e0", "full_year", 106000
CONFIG_DIR = os.path.join(REPO, "data", "rbc_tuned_configs")


def _suggest_unitary(trial):
    return dict(
        kp=trial.suggest_float("kp", 0.01, 3.0, log=True),
        ki=trial.suggest_float("ki", 1e-4, 0.1, log=True),
        integral_max=trial.suggest_float("integral_max", 1.0, 50.0),
        min_fan_fraction=trial.suggest_float("min_fan_fraction", 0.01, 0.5),
        sat_min_c=trial.suggest_float("sat_min_c", 5.0, 15.0),
        sat_max_c=trial.suggest_float("sat_max_c", 20.0, 55.0),
        sat_initial_c=trial.suggest_float("sat_initial_c", 10.0, 25.0),
        sat_trim=trial.suggest_float("sat_trim", 0.05, 1.0),
        sat_respond=trial.suggest_float("sat_respond", 0.1, 5.0),
        demand_deadband=trial.suggest_float("demand_deadband", 0.01, 2.0),
        availability_on=trial.suggest_float("availability_on", 0.5, 3.0),
        fan_error_mode=trial.suggest_categorical(
            "fan_error_mode", ["nearest_setpoint", "center_of_band"]),
    )


def _suggest_air_loop(trial):
    return dict(
        sat_neutral=trial.suggest_float("sat_neutral", 14.0, 24.0),
        sat_kp=trial.suggest_float("sat_kp", 0.1, 5.0),
        sat_min=trial.suggest_float("sat_min", 5.0, 15.0),
        sat_max=trial.suggest_float("sat_max", 35.0, 65.0),
        sat_rate_limit=trial.suggest_float("sat_rate_limit", 0.05, 1.0),
        outdoor_sat_gain=trial.suggest_float("outdoor_sat_gain", 0.0, 0.5),
        sat_cold_bias=trial.suggest_float("sat_cold_bias", 0.0, 1.0),
        sat_warm_bias=trial.suggest_float("sat_warm_bias", 0.0, 1.5),
        flow_base=trial.suggest_float("flow_base", 0.1, 0.8),
        flow_kp=trial.suggest_float("flow_kp", 0.05, 1.0),
        flow_ki=trial.suggest_float("flow_ki", 1e-3, 0.1, log=True),
        flow_min=trial.suggest_float("flow_min", 0.05, 0.5),
        flow_max=trial.suggest_float("flow_max", 0.5, 1.0),
        flow_rate_limit=trial.suggest_float("flow_rate_limit", 0.01, 0.3),
        integral_max=trial.suggest_float("integral_max", 5.0, 50.0),
        integral_decay=trial.suggest_float("integral_decay", 0.8, 1.0),
        reheat_sp_min=trial.suggest_float("reheat_sp_min", 8.0, 18.0),
        reheat_sp_max=trial.suggest_float("reheat_sp_max", 20.0, 30.0),
        reheat_sp_deadband=trial.suggest_float("reheat_sp_deadband", 0.1, 2.0),
        reheat_sp_kp=trial.suggest_float("reheat_sp_kp", 0.5, 5.0),
        reheat_sp_rate_limit=trial.suggest_float("reheat_sp_rate_limit", 0.02, 0.3),
        clg_sp_default=trial.suggest_float("clg_sp_default", 20.0, 28.0),
        error_ema_alpha=trial.suggest_float("error_ema_alpha", 0.05, 0.5),
        sat_aware_flow=trial.suggest_categorical("sat_aware_flow", [True, False]),
    )


def _rollout(params, bt, idx, split):
    is_vav = bt == "OfficeMedium"
    cfg = AirLoopConfig(**params) if is_vav else UnitaryHvacConfig(**params)
    pol = AirLoopPolicy(cfg) if is_vav else UnitaryHvacPolicy(cfg)
    env = b2b.make_env(bt, split=split, index=idx, task=TASK, run_period=RP)
    pol.bind_env(env)
    raw, _ = env.reset()
    pol.reset()
    ret = 0.0
    for _t in range(MAX_STEPS):
        act, _ = pol.predict(raw)
        raw, r, term, trunc, _ = env.step(np.asarray(act, dtype=np.float32))
        ret += float(r)
        if term or trunc:
            break
    env.close()
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-type", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-trials", type=int, default=25)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    is_vav = args.building_type == "OfficeMedium"
    suggest = _suggest_air_loop if is_vav else _suggest_unitary
    bid = get_registry().get_building_by_index(
        args.building_type, args.split, args.index).building_id
    print(f"[tune] {bid} ({'VAV' if is_vav else 'unitary'}) {args.split} "
          f"full-year comfort, {args.n_trials} trials", flush=True)

    def objective(trial):
        r = _rollout(suggest(trial), args.building_type, args.index, args.split)
        print(f"  {bid} trial {trial.number:2d}: {r:10.1f}", flush=True)
        return r

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=8, seed=42))
    study.optimize(objective, n_trials=args.n_trials)

    os.makedirs(CONFIG_DIR, exist_ok=True)
    out = os.path.join(CONFIG_DIR, f"{args.building_type}_{args.split}_{args.index}.json")
    with open(out, "w") as f:
        json.dump({"building_type": args.building_type, "index": args.index,
                   "split": args.split, "is_vav": is_vav,
                   "tuned_return": study.best_value, "params": study.best_params}, f, indent=2)
    print(f"[done] {bid} tuned={study.best_value:.1f} -> {out}\nTUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
