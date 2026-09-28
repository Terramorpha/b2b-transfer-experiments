"""OfficeMedium energy task: what remains of the learned policy's edge once
ventilation is enforced (the "clamped OA" section of the mémoire).

One panel, the five OfficeMedium test buildings, normalized return
(return / default-RBC return, lower is better, dashed line = default RBC):
  * unconstrained from-scratch PPO (5.1 y)          data/emed_fromscratch_5y_test_fullyear_eval.csv
  * the same checkpoint with OA clamped at eval     data/emed_fromscratch_5y_oaclamp_fullyear_eval.csv
  * PPO retrained from scratch with the OA floor    data/emed_oafloor_fullyear_eval.csv
  * energy-tuned RBC (the bar to beat)              data/_emedtuned_ceiling.json

    python figures/build_oa_clamp_figure.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

from pub_style import (apply_pub_style, barh_axes, grouped_barh, preview, save,
                       W_E_COLORS, COLWIDTH_IN)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = os.path.join(REPO, "data")
FIGDIR = os.path.join(REPO, "figures_out")
STEM = "emed_oaclamp_officemed"
BTYPE = "OfficeMedium"

SERIES = [  # (label, colour, source)
    ("PPO (5.1 y), unconstrained", W_E_COLORS["e0"], "emed_fromscratch_5y_test_fullyear_eval.csv"),
    ("PPO (5.1 y), OA clamped at eval", W_E_COLORS["ehigh"], "emed_fromscratch_5y_oaclamp_fullyear_eval.csv"),
    ("PPO (5.1 y), trained with OA floor", W_E_COLORS["emed"], "emed_oafloor_fullyear_eval.csv"),
    ("Energy-tuned RBC", W_E_COLORS["e05"], "_emedtuned_ceiling.json"),
]


def returns(name: str) -> pd.Series:
    p = os.path.join(D, name)
    if name.endswith(".json"):
        return pd.Series(json.load(open(p)), dtype=float)
    df = pd.read_csv(p)
    return df.groupby("building_id")["episode_return"].mean()


def main() -> None:
    base = pd.read_csv(os.path.join(D, "_default_baseline_emed.csv")).set_index("building_id")["baseline_return"]
    ids = sorted(b for b in returns(SERIES[1][2]).index if b.startswith(BTYPE))
    cats = [b.split("-")[-1] for b in ids]
    series = []
    for label, color, src in SERIES:
        r = returns(src)
        vals = [float(r[b] / base[b]) for b in ids]
        series.append((vals, None, color, label))
        print(f"  {label:36s} norm vs default: {[round(v, 3) for v in vals]}")

    apply_pub_style()
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = barh_axes(width=COLWIDTH_IN, height=2.9)
    grouped_barh(ax, cats, series, value_label="Normalized return", invert=False,
                 reference=1.0, reference_label="Default RBC")
    ax.set_ylabel("OfficeMedium test building")
    lo, hi = ax.get_xlim()
    ax.set_xlim(0, hi)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.55 * (y1 - y0))   # headroom so the legend clears the top group
    ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=7)
    fig.tight_layout()
    preview(fig, os.path.join(FIGDIR, f"{STEM}_preview"), dpi=200)
    save(fig, os.path.join(FIGDIR, STEM))
    print("wrote:", os.path.join(FIGDIR, f"{STEM}.pdf"))


if __name__ == "__main__":
    main()
