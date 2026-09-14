"""Aggregate multi-seed full-year eval panels into mean +- std (error bars).

    python scripts/aggregate_seeds.py [--seeds 0 1 2] [--out data/seed_summary.csv]

For every panel tag the thesis uses, reads data/<tag>_fullyear_eval.csv (seed 0)
and data/<tag>_seed<k>_fullyear_eval.csv (k > 0), and reports, per panel:
  * win counts vs each bar per seed, and their mean +- std;
  * per-type normalized return (policy / bar), mean +- std across seeds;
  * per-building mean +- std of the raw return.
Writes one long-format CSV (panel, bar, building_type, building_id, seed, return,
bar_return) for the figure builders.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "data")
TYPES = ["OfficeMedium", "OfficeSmall", "RestaurantFastFood", "RetailStandalone"]

PANELS = [
    ("from-scratch comfort", "transfer4_final", ["default_e0", "bo"]),
    ("BC-warm comfort", "transfer4_bcwarm", ["default_e0", "bo"]),
    ("BC-warm on TRAIN vs oracle", "bcwarm_train", ["tuned_train"]),
    ("DAgger y1", "dagger_stream_y1_test", ["default_e0", "bo"]),
    ("DAgger y2", "dagger_stream_y2_test", ["default_e0", "bo"]),
    ("DAgger y2 + PPO", "dagger_ppoft_test", ["default_e0", "bo"]),
    ("SPE y2", "spe_y2_test", ["default_e0", "bo"]),
    ("emed y2->critic->PPO", "emed_ppoft_test", ["default_emed", "emedtuned"]),
    ("emed from-scratch 2.56y", "emed_fromscratch_test", ["default_emed", "emedtuned"]),
    ("emed from-scratch 5.1y", "emed_fromscratch_5y_test", ["default_emed", "emedtuned"]),
]


def bars() -> dict[str, pd.Series]:
    def js(n):
        p = os.path.join(D, n); return pd.Series(json.load(open(p))) if os.path.exists(p) else None
    def csv(n):
        p = os.path.join(D, n)
        return pd.read_csv(p).set_index("building_id").episode_return if os.path.exists(p) else None
    return {"default_e0": js("rbc_fullyear_ourharness.json"), "default_emed": js("rbc_emed_fullyear.json"),
            "bo": csv("pertype_bo_fullyear_eval.csv"), "emedtuned": csv("emedtuned_rbc_test_fullyear_eval.csv"),
            "tuned_train": js("tuned_train_baseline.json")}


def panel(tag: str, seed: int):
    p = os.path.join(D, f"{tag}{'' if seed == 0 else f'_seed{seed}'}_fullyear_eval.csv")
    return pd.read_csv(p).set_index("building_id").episode_return if os.path.exists(p) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default="data/seed_summary.csv")
    args = ap.parse_args()
    B = bars()
    rows = []
    for label, tag, barkeys in PANELS:
        runs = {s: panel(tag, s) for s in args.seeds}
        have = [s for s, r in runs.items() if r is not None]
        print(f"\n=== {label} [{tag}]  seeds present: {have} ===")
        if not have:
            continue
        for bk in barkeys:
            bar = B[bk]
            if bar is None:
                continue
            wins = []
            for s in have:
                r = runs[s]; idx = r.index.intersection(bar.index)
                wins.append(int((r[idx] > bar[idx]).sum()))
                for bid in idx:
                    rows.append({"panel": tag, "bar": bk, "building_type": bid.rsplit("-", 1)[0],
                                 "building_id": bid, "seed": s, "return": float(r[bid]), "bar_return": float(bar[bid])})
            print(f"  wins vs {bk:12s}: per seed {wins}  ->  {np.mean(wins):.1f} +- {np.std(wins, ddof=1) if len(wins) > 1 else 0:.1f} / {len(idx)}")
            for t in TYPES:
                vals = []
                for s in have:
                    r = runs[s]; jb = [i for i in r.index if i.startswith(t) and i in bar.index]
                    if jb:
                        vals.append(float((r[jb] / bar[jb]).mean()))
                if vals:
                    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    print(f"     {t:20s} norm-vs-{bk}: {np.mean(vals):5.2f} +- {sd:4.2f}   (per seed {[round(v, 2) for v in vals]})")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, args.out), index=False)
    print(f"\n[done] {args.out}: {len(df)} rows")


if __name__ == "__main__":
    main()
