"""Lookup-apply eval of the ENERGY-tuned per-(type, climate-zone) RBC on the
emed TEST split -- the realistic deployment protocol: each test building gets
the config tuned on a TRAIN instance of its (type, CZ) pair, by table lookup
(data/rbc_emed_tuned_pertypecz/, from scripts/tune_rbc_emed_campaign.py).
No per-building experiments on test buildings.

Also prints the RE-JUDGMENT of the energy headline: the DAgger-y2 ->
critic-year -> PPO policy (data/emed_ppoft_test_fullyear_eval.csv) scored
against (a) the energy-tuned bar alone and (b) best-of-three RBC
(default / comfort-BO / energy-tuned) per building.

    python scripts/eval_fullyear_emedtuned_rbc.py --n-workers 3

Writes data/emedtuned_rbc_test_fullyear_eval.csv.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import numpy as np
import pandas as pd

import building2building as b2b
from baselines.controllers import (
    UnitaryHvacPolicy, UnitaryHvacConfig, AirLoopPolicy, AirLoopConfig,
)
from building2building.data.registry import get_registry

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
TASK, RP, N, MAX_STEPS = "task_occ_emed", "full_year", 5, 106000
TUNED_DIR = os.path.join(REPO, "data", "rbc_emed_tuned_pertypecz")


def _lookup_params(bt: str, cz) -> dict:
    """The deployment table: (type, CZ) -> tuned params from the train instance."""
    hits = glob.glob(os.path.join(TUNED_DIR, f"{bt}_cz{cz}", "*.json"))
    assert len(hits) == 1, f"no unique tuned config for ({bt}, cz{cz}): {hits}"
    return json.load(open(hits[0]))["params"]


def _worker(args):
    bt, idx, bid = args
    cz = (None if bt in b2b.TYPES_WITHOUT_CLIMATE_ZONE
          else b2b.get_climate_zone(bt, bid))
    params = _lookup_params(bt, cz)
    is_vav = bt == "OfficeMedium"
    cfg = AirLoopConfig(**params) if is_vav else UnitaryHvacConfig(**params)
    pol = AirLoopPolicy(cfg) if is_vav else UnitaryHvacPolicy(cfg)
    env = b2b.make_env(bt, split="test", index=idx, task=TASK, run_period=RP)
    pol.bind_env(env)
    raw, _ = env.reset()
    pol.reset()
    ret, steps = 0.0, 0
    for _t in range(MAX_STEPS):
        act, _ = pol.predict(raw)
        raw, r, term, trunc, _ = env.step(np.asarray(act, dtype=np.float32))
        ret += float(r)
        steps += 1
        if term or trunc:
            break
    env.close()
    return bt, idx, cz, ret, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=3)
    args = ap.parse_args()

    reg = get_registry()
    ids = {(bt, idx): reg.get_building_by_index(bt, "test", idx).building_id
           for bt in TYPES for idx in range(N)}
    tasks = [(bt, idx, ids[(bt, idx)]) for idx in range(N) for bt in TYPES]

    rows = []
    print(f"[eval-emedtuned] {len(tasks)} full-year lookup-RBC rollouts | "
          f"{args.n_workers} workers", flush=True)
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for k, (bt, idx, cz, ret, steps) in enumerate(
                pool.imap_unordered(_worker, tasks), 1):
            bid = ids[(bt, idx)]
            rows.append({"building_type": bt, "building_id": bid,
                         "climate_zone": cz, "episode_return": ret,
                         "n_steps": steps, "seed": 0})
            print(f"  [{k:2d}/{len(tasks)}] {bid:26s} cz{cz}  "
                  f"emed-tuned RBC {ret:10.1f}  ({steps} steps)", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(REPO, "data", "emedtuned_rbc_test_fullyear_eval.csv")
    df.to_csv(out, index=False)
    print(f"[done] {out}")

    # ---- re-judgment of the energy headline ---------------------------------
    et = df.set_index("building_id").episode_return
    default = pd.Series(json.load(open(os.path.join(REPO, "data",
                                                    "rbc_emed_fullyear.json"))))
    cbo = pd.read_csv(os.path.join(
        REPO, "data", "pertype_bo_emed_fullyear_eval.csv")).set_index(
        "building_id").episode_return
    pol = pd.read_csv(os.path.join(
        REPO, "data", "emed_ppoft_test_fullyear_eval.csv")).set_index(
        "building_id").episode_return

    print("\n=== RE-JUDGMENT: DAgger-y2 -> critic-yr -> PPO vs RBC bars "
          "(emed, test, full year) ===")
    tot_et, tot_b3 = 0, 0
    for bt in TYPES:
        bids = [ids[(bt, i)] for i in range(N)]
        p = pol[bids].to_numpy()
        e = et[bids].to_numpy()
        b3 = np.maximum.reduce([default[bids].to_numpy(),
                                cbo[bids].to_numpy(), e])
        w_et = int((p > e).sum())
        w_b3 = int((p > b3).sum())
        tot_et += w_et
        tot_b3 += w_b3
        print(f"  {bt:22s} policy {p.mean():10.1f}  emed-tunedRBC {e.mean():10.1f}"
              f"  norm {(p / e).mean():.3f}  wins-vs-tuned {w_et}/{N}"
              f"  wins-vs-best3 {w_b3}/{N}")
    print(f"  TOTAL: vs emed-tuned {tot_et}/{len(df)} | "
          f"vs best-of-three {tot_b3}/{len(df)}")
    print("EVAL_EMEDTUNED_DONE")


if __name__ == "__main__":
    main()
