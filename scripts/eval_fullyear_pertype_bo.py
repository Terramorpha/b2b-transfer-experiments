"""Full-year eval of the b2b PAPER's per-(type, climate-zone) Bayesian-optimized
reactive baseline -- the strong "BO-tuned" RBC -- on the same 4 test types x 5
buildings, vs OUR default RBC baseline (data/rbc_fullyear_ourharness.json).

This is the counterpart to eval_fullyear_panel.py, but instead of rolling a
neural policy it rolls the reactive controller configured from the paper's tuned
YAMLs in vendor/Building2Building/baselines/configs/tuned_controllers/, selected
per building by its ASHRAE climate zone (the mapping + loader logic mirrors
baselines/run_reactive_control.py). "policy" in the output == the BO-tuned RBC
return; "RBC" == our default-config RBC baseline. So we get the strong BO baseline
alongside the default one, scored through the identical full-year harness.

    python scripts/eval_fullyear_pertype_bo.py --n-workers 3

Writes data/pertype_bo_fullyear_eval.csv and prints a per-type + total summary.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import numpy as np
import pandas as pd
import yaml

import building2building as b2b
from building2building.data.registry import get_registry
from baselines.controllers.air_loop import AirLoopConfig, AirLoopPolicy
from baselines.controllers.unitary_hvac import UnitaryHvacConfig, UnitaryHvacPolicy
from evaluate import _b2b_factory_impl

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
TASK, RP, N, MAX_STEPS = "task_occ_e0", "full_year", 5, 106000  # full year = 105_120 steps

# Same "our-harness" default-RBC baseline that eval_fullyear_panel.py compares to.
OUR_BASELINES = os.path.join(REPO, "data", "rbc_fullyear_ourharness.json")

# Which types use the VAV air-loop controller (mirrors run_reactive_control).
VAV_BUILDING_TYPES = {"OfficeMedium"}
TUNED_CONFIGS_DIR = Path(
    REPO) / "vendor" / "Building2Building" / "baselines" / "configs" / "tuned_controllers"


# --- tuned-config loaders (replicated from baselines/run_reactive_control.py so
#     we do NOT pull in hydra / matplotlib just to read a YAML) -----------------
def _load_tuned_yaml(path: Path) -> dict[str, Any]:
    """unsafe_load: these files were dumped with plain yaml.dump and contain
    !!python/tuple tags (our own tuning pipeline produced them)."""
    return yaml.unsafe_load(path.read_text())


def _load_tuned_unitary_hvac(bt: str, cz: int | None) -> UnitaryHvacConfig:
    if cz is not None:
        p = TUNED_CONFIGS_DIR / f"unitary_hvac_{bt.lower()}_cz{cz}.yaml"
        if p.exists():
            raw = _load_tuned_yaml(p)
            raw.pop("type", None)
            return UnitaryHvacConfig(**{
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in raw.items() if k != "target_schedule"})
    return UnitaryHvacConfig()


def _load_tuned_air_loop(bt: str, cz: int | None) -> AirLoopConfig:
    if cz is not None:
        p = TUNED_CONFIGS_DIR / f"air_loop_{bt.lower()}_cz{cz}.yaml"
        if p.exists():
            raw = _load_tuned_yaml(p)
            raw.pop("type", None)
            return AirLoopConfig(**raw)
    return AirLoopConfig()


def _get_climate_zone(bt: str, bid: str) -> int | None:
    if bt in b2b.TYPES_WITHOUT_CLIMATE_ZONE:
        return None
    return b2b.get_climate_zone(bt, bid)


def _select_policy(bt: str, bid: str, env):
    cz = _get_climate_zone(bt, bid)
    if bt in VAV_BUILDING_TYPES:
        pol = AirLoopPolicy(_load_tuned_air_loop(bt, cz))
    else:
        pol = UnitaryHvacPolicy(_load_tuned_unitary_hvac(bt, cz))
    pol.bind_env(env)
    return pol, cz


def bo_return(bt, idx, bid):
    env, _source = _b2b_factory_impl(bt, idx, "test", TASK, RP)
    pol, cz = _select_policy(bt, bid, env)
    raw, _ = env.reset(); pol.reset()
    ret, steps = 0.0, 0
    for _t in range(MAX_STEPS):
        a, _ = pol.predict(raw)
        raw, r, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32))
        ret += float(r); steps += 1
        if term or trunc:
            break
    env.close()
    return ret, steps, cz


def _worker(args):
    """Spawn-pool worker: one full-year BO-RBC rollout in its own process."""
    bt, idx, bid = args
    ret, steps, cz = bo_return(bt, idx, bid)
    return bt, idx, ret, steps, cz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pertype_bo")
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--building-types", nargs="+", default=None)
    args = ap.parse_args()
    if args.building_types is not None:
        global TYPES
        TYPES = args.building_types

    with open(OUR_BASELINES) as f:
        base = json.load(f)
    reg = get_registry()

    ids = {(bt, idx): reg.get_building_by_index(bt, "test", idx).building_id
           for bt in TYPES for idx in range(N)}
    # idx-major ordering => first wave is one building of EACH type.
    tasks = [(bt, idx, ids[(bt, idx)]) for idx in range(N) for bt in TYPES]

    rows, brows = [], []
    print(f"[eval-bo] {len(tasks)} full-year BO-RBC rollouts | {args.n_workers} "
          f"parallel processes | one-per-type first", flush=True)
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for k, (bt, idx, p, steps, cz) in enumerate(
                pool.imap_unordered(_worker, tasks), 1):
            bid = ids[(bt, idx)]
            rb = base.get(bid, float("nan"))
            rows.append({"building_type": bt, "building_id": bid, "climate_zone": cz,
                         "episode_return": p, "n_steps": steps, "seed": 0})
            brows.append({"building_type": bt, "building_id": bid,
                          "baseline_return": rb})
            print(f"  [{k:2d}/{len(tasks)}] {bid:26s} cz{cz}  BO-RBC {p:10.1f}  "
                  f"defaultRBC {rb:10.1f}  ({steps} steps)  "
                  f"{'BO' if p > rb else 'default'}", flush=True)

    df = pd.DataFrame(rows); bdf = pd.DataFrame(brows)
    out = os.path.join(REPO, "data", f"{args.tag}_fullyear_eval.csv")
    df.to_csv(out, index=False)

    print("\n=== SUMMARY (full year, BO-tuned RBC vs our default RBC) ===")
    tot = 0
    for bt in TYPES:
        s = df[df.building_type == bt]; sb = bdf[bdf.building_type == bt]
        w = int((s.episode_return.to_numpy() > sb.baseline_return.to_numpy()).sum())
        tot += w
        norm = (s.episode_return.to_numpy() / sb.baseline_return.to_numpy()).mean()
        print(f"  {bt:22s} BO-RBC {s.episode_return.mean():10.1f}  "
              f"defaultRBC {sb.baseline_return.mean():10.1f}  norm {norm:.3f}  "
              f"BO-wins {w}/{len(s)}")
    print(f"  TOTAL BO-wins: {tot}/{len(df)}")
    print(f"[done] {out}\nEVAL_BO_DONE")


if __name__ == "__main__":
    main()
