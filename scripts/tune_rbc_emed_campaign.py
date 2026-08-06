"""ENERGY-tuned RBC campaign — the same protocol as the comfort BO baseline:
per-(building_type, climate_zone) Bayesian optimization, tuned on ONE *train*
instance of each pair, deployable to any test building by (type, CZ) table
lookup (NO per-building tuning on test -- the realistic-baseline requirement).

For every (type, CZ) pair present in the TEST set, pick a TRAIN building with
the same (type, CZ) and run tune_rbc_comfort.py on it with
--task task_occ_emed (Optuna TPE, seed 42, 8 startup, 25 trials -- identical to
the comfort campaign). Configs land in data/rbc_emed_tuned_pertypecz/.
Pairs with no train instance are reported and SKIPPED (fallback discussed in
thesis, not silently substituted).

    python scripts/tune_rbc_emed_campaign.py --n-workers 3 [--n-trials 25]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "morel"))
sys.path.insert(0, os.path.join(REPO, "vendor", "Building2Building"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("WANDB_MODE", "offline")

import _pin_dataset  # noqa: F401
import building2building as b2b
from building2building.data.registry import get_registry

TYPES = ["RetailStandalone", "RestaurantFastFood", "OfficeMedium", "OfficeSmall"]
N_TEST = 5
OUT_DIR = os.path.join(REPO, "data", "rbc_emed_tuned_pertypecz")
PY = os.path.join(REPO, "vendor", "morel", ".venv", "bin", "python")


def _cz(bt, bid):
    if bt in b2b.TYPES_WITHOUT_CLIMATE_ZONE:
        return None
    return b2b.get_climate_zone(bt, bid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=25)
    args = ap.parse_args()
    reg = get_registry()

    # 1. (type, CZ) pairs needed to cover the TEST set
    needed = {}
    for bt in TYPES:
        for i in range(N_TEST):
            bid = reg.get_building_by_index(bt, "test", i).building_id
            needed.setdefault((bt, _cz(bt, bid)), []).append(bid)

    # 2. a TRAIN instance for each pair
    train_pick, missing = {}, []
    for bt in TYPES:
        n_train = 0
        while True:  # enumerate train split until exhaustion
            try:
                reg.get_building_by_index(bt, "train", n_train)
                n_train += 1
            except Exception:
                break
        for (t, cz), test_bids in needed.items():
            if t != bt or (t, cz) in train_pick:
                continue
            for i in range(n_train):
                bid = reg.get_building_by_index(bt, "train", i).building_id
                if _cz(bt, bid) == cz:
                    train_pick[(t, cz)] = (i, bid)
                    break
    for pair, bids in needed.items():
        if pair not in train_pick:
            missing.append((pair, bids))

    print(f"[campaign] {len(needed)} (type,CZ) pairs cover the test set; "
          f"{len(train_pick)} have a train instance; {len(missing)} MISSING", flush=True)
    for (t, cz), (i, bid) in sorted(train_pick.items()):
        print(f"  tune {t} cz{cz}  on train[{i}] = {bid}", flush=True)
    for (t, cz), bids in missing:
        print(f"  !! NO TRAIN INSTANCE for {t} cz{cz} (test bldgs {bids}) — skipped", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = []
    for (t, cz), (i, bid) in sorted(train_pick.items()):
        # unambiguous per-pair output name (tuner writes {bt}_{split}_{idx}.json;
        # we run with out-dir per pair to avoid collisions, then flatten)
        pair_dir = os.path.join(OUT_DIR, f"{t}_cz{cz}")
        cfg = os.path.join(pair_dir, f"{t}_train_{i}.json")
        if os.path.exists(cfg):
            print(f"  skip {t} cz{cz} (already tuned)", flush=True)
            continue
        jobs.append(((t, cz, i, bid), [
            PY, os.path.join(REPO, "scripts", "tune_rbc_comfort.py"),
            "--building-type", t, "--index", str(i), "--split", "train",
            "--task", "task_occ_emed", "--n-trials", str(args.n_trials),
            "--out-dir", pair_dir]))

    def run(job):
        (t, cz, i, bid), cmd = job
        log = os.path.join(OUT_DIR, f"tune_{t}_cz{cz}.log")
        with open(log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"  [{'ok' if rc == 0 else f'rc={rc}'}] {t} cz{cz} ({bid})", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        rcs = list(ex.map(run, jobs))
    print(f"CAMPAIGN_DONE ok={sum(1 for r in rcs if r == 0)}/{len(rcs)}", flush=True)


if __name__ == "__main__":
    main()
