"""Per-building EMED ORACLE tuning for the 20 DAgger train buildings.

Distillation teachers: for each (type, index 0..4) train building, Optuna-tune
the RBC on the full-year task_occ_emed objective, WARM-STARTED from the
building's per-(type,CZ) energy-tuned config (enqueued as trial 0), so a short
budget (default 12 trials) recovers most of the per-building headroom.
Configs land in data/rbc_emed_oracle_configs/<bt>_train_<idx>.json -- the
exact filename scheme dagger_stream_amorpheus.py's --config-dir expects.

    python scripts/tune_rbc_emed_oracles.py --n-workers 3 [--n-trials 12]
"""
import argparse
import glob
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
N = 5
OUT_DIR = os.path.join(REPO, "data", "rbc_emed_oracle_configs")
PERTYPECZ_DIR = os.path.join(REPO, "data", "rbc_emed_tuned_pertypecz")
PY = os.path.join(REPO, "vendor", "morel", ".venv", "bin", "python")


def _cz(bt, bid):
    if bt in b2b.TYPES_WITHOUT_CLIMATE_ZONE:
        return None
    return b2b.get_climate_zone(bt, bid)


def _warm_json(bt, cz):
    hits = glob.glob(os.path.join(PERTYPECZ_DIR, f"{bt}_cz{cz}", "*.json"))
    return hits[0] if len(hits) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    reg = get_registry()

    jobs = []
    for bt in TYPES:
        for idx in range(N):
            bid = reg.get_building_by_index(bt, "train", idx).building_id
            out = os.path.join(OUT_DIR, f"{bt}_train_{idx}.json")
            if os.path.exists(out):
                print(f"  [skip] {bt} train[{idx}] ({bid}) — already tuned")
                continue
            warm = _warm_json(bt, _cz(bt, bid))
            jobs.append((bt, idx, bid, warm))
    print(f"[oracles] {len(jobs)} buildings x {args.n_trials} trials "
          f"(warm-started) | {args.n_workers} workers", flush=True)

    def run(job):
        bt, idx, bid, warm = job
        cmd = [PY, os.path.join(REPO, "scripts", "tune_rbc_comfort.py"),
               "--building-type", bt, "--index", str(idx), "--split", "train",
               "--task", "task_occ_emed", "--n-trials", str(args.n_trials),
               "--out-dir", OUT_DIR]
        if warm:
            cmd += ["--enqueue-json", warm]
        log = os.path.join(OUT_DIR, f"tune_{bt}_{idx}.log")
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        tag = "ok" if rc == 0 else f"FAIL rc={rc}"
        print(f"  [{tag}] {bt} train[{idx}] ({bid})", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        rcs = list(ex.map(run, jobs))
    print(f"ORACLES_DONE ok={sum(1 for r in rcs if r == 0)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
