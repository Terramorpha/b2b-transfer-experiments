"""Progression figure: from-scratch -> BC-warm-500k -> BC-warm-1.5M, per building
type, as % improvement over RBC (0 = RBC, positive = beats it).

The single clearest telling of the thesis: from-scratch RL fails, BC warm-start
rescues it, and adequate budget carries it past RBC -- shown with a plain MLP
(no Amorpheus). The x-axis is clipped near the RBC line so the warm progression is
legible; the catastrophic from-scratch means are drawn to the clip edge and
annotated with their true value.

Colours (validated CVD-safe, figures/validate_palette.py):
  from-scratch #e34948 (red, "fails") -> warm-500k #4a3aa7 (purple) ->
  warm-1.5M #008300 (green, "wins").
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from pub_style import apply_pub_style, save, INK, W_E_COLORS, TEXTWIDTH_IN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIGDIR = os.path.join(REPO, "figures_out_progression")

TYPES = [("OfficeMedium", "OfficeMed"), ("RetailStandalone", "Retail"),
         ("OfficeSmall", "OfficeSmall"), ("RestaurantFastFood", "Restaurant")]
SERIES = [("from-scratch", "mlp_fromscratch", W_E_COLORS["ehigh"]),
          ("BC-warm 500k", "mlp_warm500k", W_E_COLORS["emed"]),
          ("BC-warm 1.5M", "mlp_warm1p5m", W_E_COLORS["e05"])]
XCLIP = -320.0  # clip catastrophic from-scratch bars here; annotate true value


def _load():
    rbc = {}
    import csv
    for r in csv.DictReader(open(os.path.join(REPO, "data", "baseline_fullyear.csv"))):
        rbc[r["building_id"]] = float(r["baseline_return"])
    def asret(name):
        d = json.load(open(os.path.join(REPO, "data", f"{name}_eval.json")))
        # from-scratch stores {bid:{return,...}}, others {bid:return}
        return {b: (v["return"] if isinstance(v, dict) else v) for b, v in d.items()}
    data = {name: asret(name) for _lab, name, _c in SERIES}
    return rbc, data


def _pct(ret, rbc):
    return (ret - rbc) / abs(rbc) * 100.0


def main():
    rbc, data = _load()
    apply_pub_style()
    os.makedirs(FIGDIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.0))
    y = np.arange(len(TYPES))
    n = len(SERIES)
    h = 0.8 / n
    for k, (lab, name, color) in enumerate(SERIES):
        d = data[name]
        means, errs, ns = [], [], []
        for btype, _short in TYPES:
            vals = [_pct(d[b], rbc[b]) for b in d
                    if b.rsplit("-", 1)[0] == btype and b in rbc]
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if len(vals) > 1 else 0.0)
            ns.append(len(vals))
        means = np.array(means); errs = np.array(errs)
        off = ((n - 1) / 2 - k) * h
        drawn = np.clip(means, XCLIP, None)
        # error bars only where the bar is on-scale (clipped bars have no room)
        xerr = np.where(means > XCLIP, errs, 0.0)
        ax.barh(y + off, drawn, height=h * 0.92, color=color, alpha=0.6,
                linewidth=0, label=lab, xerr=xerr,
                error_kw=dict(ecolor=INK["primary"], elinewidth=0.8, capsize=2))
        for yi, (m, dr) in enumerate(zip(means, drawn)):
            if m < XCLIP:  # off-chart from-scratch: annotate true value at clip edge
                ax.text(XCLIP + 6, y[yi] + off, f"{m:.0f}%", va="center",
                        ha="left", fontsize=6.5, color=INK["secondary"])
    ax.axvline(0, color=INK["primary"], linewidth=0.8, linestyle="--",
               label="RBC (G36)")
    ax.set_yticks(y)
    ax.set_yticklabels([s for _t, s in TYPES])
    ax.set_ylabel("Building type")
    ax.set_xlabel("Improvement over RBC  (%,  0 = RBC,  positive = beats it)")
    ax.set_xlim(XCLIP, 95)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "progression_preview.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    save(fig, os.path.join(FIGDIR, "progression"))
    print("saved:", os.path.join(FIGDIR, "progression.pdf"))
    # print the underlying numbers for the record
    for btype, short in TYPES:
        line = f"  {short:12s}"
        for _lab, name, _c in SERIES:
            d = data[name]
            vals = [_pct(d[b], rbc[b]) for b in d if b.rsplit("-", 1)[0] == btype]
            line += f"  {name.split('_')[-1]}={np.mean(vals):+7.1f}(n{len(vals)})"
        print(line)


if __name__ == "__main__":
    main()
