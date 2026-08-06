"""Comfort-campaign summary: every method normalized by the UNTUNED (default) RBC.

One panel, the whole comfort story: for each building type, bars for
  from-scratch PPO -> BC-warm -> streaming DAgger y1 -> y2 -> y2+PPO fine-tune
  -> the TUNED (per-(type,CZ) BO) RBC itself,
each normalized by the DEFAULT RBC on the same test buildings (dashed line =
1.0 = untuned RBC; LOWER = better). The orange hatched bar is the tuned RBC --
the realistic competitor; a method beats it only where its bar dips below the
orange one (OfficeMedium/VAV).

Colours validated pairwise (figures/validate_palette.py
"#e34948,#2a78d6,#4a3aa7,#008300,#c2439b,#eb6834" --pairs all, 2026-08-06):
all PASS except red~orange dE=11.3 (WARN band) -> the tuned-RBC bar carries a
hatch as the required secondary channel (and REACTIVE_COLOR = the reactive
controller's fixed entity colour, per house rules).
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from pub_style import apply_pub_style, save, TEXTWIDTH_IN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = os.path.join(REPO, "data")

TYPES = [("RetailStandalone", "Retail"), ("RestaurantFastFood", "Restaurant"),
         ("OfficeMedium", "OfficeMed"), ("OfficeSmall", "OfficeSmall")]
METHODS = [
    ("from scratch",   "transfer4_final_fullyear_eval.csv",        "#e34948"),
    ("BC-warm",        "transfer4_bcwarm_fullyear_eval.csv",       "#2a78d6"),
    ("DAgger (1 yr)",  "dagger_stream_y1_test_fullyear_eval.csv",  "#4a3aa7"),
    ("DAgger (2 yr)",  "dagger_stream_y2_test_fullyear_eval.csv",  "#008300"),
    ("DAgger + PPO",   "dagger_ppoft_test_fullyear_eval.csv",      "#c2439b"),
]


def _load(path):
    with open(path) as f:
        return {r["building_id"]: float(r["episode_return"])
                for r in csv.DictReader(f)}


def _typ(bid):
    for t, _ in TYPES:
        if bid.startswith(t):
            return t


def main() -> None:
    import json
    apply_pub_style()
    default = json.load(open(os.path.join(D, "rbc_fullyear_ourharness.json")))
    bo = _load(os.path.join(D, "pertype_bo_fullyear_eval.csv"))

    def norm_by_default(pol):
        agg = defaultdict(lambda: [0.0, 0.0])
        for b, v in pol.items():
            if b in default:
                t = _typ(b)
                agg[t][0] += v
                agg[t][1] += default[b]
        return {t: s / sd for t, (s, sd) in agg.items()}

    norms = {name: norm_by_default(_load(os.path.join(D, fname)))
             for name, fname, _c in METHODS}
    norms["tuned RBC"] = norm_by_default(bo)

    series = METHODS + [("tuned RBC", None, "#eb6834")]  # REACTIVE_COLOR

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)

    n_m = len(series)
    width = 0.8 / n_m
    xs = np.arange(len(TYPES))
    for j, (name, _f, color) in enumerate(series):
        vals = [norms[name][t] for t, _ in TYPES]
        hatch = "//" if name == "tuned RBC" else None   # secondary channel (red~orange WARN)
        ax.bar(xs + (j - (n_m - 1) / 2) * width, vals, width * 0.92,
               color=color, alpha=0.6, linewidth=0, hatch=hatch,
               edgecolor=color, label=name)
    ax.axhline(1.0, color="#0b0b0b", linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([short for _, short in TYPES])
    ax.set_ylabel("Return / untuned-RBC return")
    ax.set_ylim(0, 2.45)   # headroom so the upper-left legend clears the tallest bar
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(os.path.join(REPO, "figures_out", "comfort_summary_preview.png"),
                dpi=170, bbox_inches="tight", facecolor="white")  # quick-look raster
    save(fig, os.path.join(REPO, "figures_out", "comfort_summary"))
    print("wrote figures_out/comfort_summary.pdf (+_preview.png)")
    for name, _f, _c in series:
        row = "  ".join(f"{t[1]}:{norms[name][t[0]]:.2f}" for t in TYPES)
        print(f"  {name:14s} {row}")


if __name__ == "__main__":
    main()
