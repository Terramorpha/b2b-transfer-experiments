"""Does the reactive baseline track the task's dynamic setpoint?

Two-panel diagnostic (one week, held-out buildings):
  top    OfficeMedium -- VAV baseline (AirLoopPolicy)
  bottom OfficeSmall  -- unitary baseline (UnitaryHvacPolicy)

Each panel plots the task setpoint, the controller's commanded supply-air
temperature, and the resulting zone temperature. All three are degrees C, so they
share ONE y axis (no dual-axis).

Evidence for the claim that AirLoopPolicy ignores the setpoint: it regulates to a
hard-coded cfg.target_temp=21.0 (air_loop.py:34,278) and never binds a
target_temperature observation, so its SAT is uncorrelated -- in fact
ANTI-correlated -- with the setback the task asks for.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from pub_style import apply_pub_style, save, INK, REACTIVE_COLOR, W_E_COLORS, TEXTWIDTH_IN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NPZ = "/home/terramorpha/.claude/jobs/5c069ce0/tmp/sat_vs_setpoint.npz"
FIGDIR = os.path.join(REPO, "figures_out")

SAT_COLOR = REACTIVE_COLOR        # #eb6834 -- the controller's command
ZONE_COLOR = W_E_COLORS["e0"]     # #2a78d6 -- the resulting zone temperature
STEPS_PER_DAY = 12 * 24

PANELS = [
    ("vav", "OfficeMedium — VAV baseline (AirLoopPolicy)"),
    ("unitary", "OfficeSmall — unitary baseline (UnitaryHvacPolicy)"),
]


def main() -> None:
    d = np.load(NPZ)
    apply_pub_style()
    os.makedirs(FIGDIR, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 4.4), sharex=True, sharey=True)

    for ax, (tag, title) in zip(axes, PANELS):
        sat, tgt, tz = d[f"{tag}_sat"], d[f"{tag}_tgt"], d[f"{tag}_tz"]
        x = np.arange(len(sat)) / STEPS_PER_DAY

        # shade the setback hours (setpoint below its daytime maximum)
        setback = tgt < (np.nanmax(tgt) - 1e-6)
        ax.fill_between(x, 0, 1, where=setback, transform=ax.get_xaxis_transform(),
                        color=INK["grid"], alpha=0.55, linewidth=0, zorder=0)

        ax.plot(x, tgt, color=INK["secondary"], linestyle="--", linewidth=1.6,
                label="Task setpoint", zorder=3)
        ax.plot(x, sat, color=SAT_COLOR, linewidth=2.0,
                label="Commanded supply-air temp", zorder=4)
        ax.plot(x, tz, color=ZONE_COLOR, linewidth=1.6, alpha=0.9,
                label="Zone air temp", zorder=2)

        m = np.isfinite(sat) & np.isfinite(tgt)
        r = np.corrcoef(sat[m], tgt[m])[0, 1]
        span = float(np.nanmax(sat) - np.nanmin(sat))
        ax.set_title(title, loc="left")
        ax.text(0.995, 0.06,
                f"corr(SAT, setpoint) = {r:+.2f}    SAT span = {span:.1f} °C",
                transform=ax.transAxes, ha="right", va="bottom",
                color=INK["secondary"], fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Temperature (°C)")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_xlabel("Days (shaded = task setback hours)")
    axes[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sat_vs_setpoint_preview.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    save(fig, os.path.join(FIGDIR, "sat_vs_setpoint"))
    print("wrote:", os.path.join(FIGDIR, "sat_vs_setpoint.pdf"))
    print("preview:", os.path.join(FIGDIR, "sat_vs_setpoint_preview.png"))


if __name__ == "__main__":
    main()
