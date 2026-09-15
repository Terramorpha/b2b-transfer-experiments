"""Compare a reproduction checkout's results against the banked thesis results.

    python scripts/compare_reproduction.py --banked <main repo> --repro <worktree>

For every full-year eval CSV the thesis uses, prints win tallies against the
relevant bars (default RBC, BO-tuned RBC, energy-tuned RBC), per-type mean
returns, and the normalized return, side by side. Baselines are taken from
EACH checkout (so the reproduction is judged against its own regenerated
baselines) and the bars are also cross-checked against each other.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

TYPES = ["OfficeMedium", "OfficeSmall", "RestaurantFastFood", "RetailStandalone"]

# (label, csv tag, bar): bar in {"default_e0", "bo", "default_emed", "emedtuned"}
PANELS = [
    ("from-scratch comfort", "transfer4_final", ["default_e0", "bo"]),
    ("BC-warm comfort", "transfer4_bcwarm", ["default_e0", "bo"]),
    ("BC-warm on TRAIN vs per-building oracle", "bcwarm_train", ["tuned_train"]),
    ("DAgger y1", "dagger_stream_y1_test", ["default_e0", "bo"]),
    ("DAgger y2 (flagship)", "dagger_stream_y2_test", ["default_e0", "bo"]),
    ("DAgger y2 + PPO", "dagger_ppoft_test", ["default_e0", "bo"]),
    ("SPE y2", "spe_y2_test", ["default_e0", "bo"]),
    ("emed: y2->critic->PPO", "emed_ppoft_test", ["default_emed", "emedtuned"]),
    ("emed: from-scratch 2.56y", "emed_fromscratch_test", ["default_emed", "emedtuned"]),
    ("emed: from-scratch 5.1y", "emed_fromscratch_5y_test", ["default_emed", "emedtuned"]),
]


def load_bars(root: str) -> dict[str, pd.Series]:
    d = os.path.join(root, "data")
    bars: dict[str, pd.Series] = {}

    def js(name):
        p = os.path.join(d, name)
        return pd.Series(json.load(open(p))) if os.path.exists(p) else None

    def csv(name):
        p = os.path.join(d, name)
        return pd.read_csv(p).set_index("building_id").episode_return if os.path.exists(p) else None

    bars["default_e0"] = js("rbc_fullyear_ourharness.json")
    bars["default_emed"] = js("rbc_emed_fullyear.json")
    bars["bo"] = csv("pertype_bo_fullyear_eval.csv")
    bars["emedtuned"] = csv("emedtuned_rbc_test_fullyear_eval.csv")
    bars["tuned_train"] = js("tuned_train_baseline.json")
    return bars


def panel(root: str, tag: str):
    p = os.path.join(root, "data", f"{tag}_fullyear_eval.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).set_index("building_id").episode_return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banked", required=True)
    ap.add_argument("--repro", required=True)
    args = ap.parse_args()
    B, R = load_bars(args.banked), load_bars(args.repro)

    print("=== RBC bars: reproduced vs banked ===")
    for k in B:
        if B[k] is None or R[k] is None:
            print(f"  {k:14s} missing ({'banked' if B[k] is None else 'repro'})"); continue
        common = B[k].index.intersection(R[k].index)
        diff = (R[k][common] - B[k][common])
        print(f"  {k:14s} n={len(common)} identical={(diff == 0).sum()} max|diff|={diff.abs().max():.4g}")

    for label, tag, barkeys in PANELS:
        pb, pr = panel(args.banked, tag), panel(args.repro, tag)
        print(f"\n=== {label} [{tag}] ===")
        if pb is None or pr is None:
            print(f"  missing: banked={pb is not None} repro={pr is not None}"); continue
        for bk in barkeys:
            bb, br = B[bk], R[bk]
            if bb is None or br is None:
                continue
            ib, ir = pb.index.intersection(bb.index), pr.index.intersection(br.index)
            wb, wr = int((pb[ib] > bb[ib]).sum()), int((pr[ir] > br[ir]).sum())
            print(f"  wins vs {bk:12s}: banked {wb}/{len(ib)}   reproduced {wr}/{len(ir)}")
            rows = []
            for t in TYPES:
                jb = [i for i in ib if i.startswith(t)]; jr = [i for i in ir if i.startswith(t)]
                if not jb or not jr:
                    continue
                rows.append((t, pb[jb].mean(), pr[jr].mean(), (pb[jb] / bb[jb]).mean(), (pr[jr] / br[jr]).mean(),
                             int((pb[jb] > bb[jb]).sum()), int((pr[jr] > br[jr]).sum())))
            for t, mb, mr, nb, nr, kb, kr in rows:
                print(f"     {t:20s} mean banked {mb:10.1f} repro {mr:10.1f} | norm-vs-{bk} banked {nb:5.2f} repro {nr:5.2f} | wins {kb}/5 -> {kr}/5")


if __name__ == "__main__":
    main()
