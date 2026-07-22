"""Companion to the SAT figure: the fan the policy barely moves.

Same rollout (env driven by the POLICY; RBC asked counterfactually). Fan mass flow
over one week, plotted against the FULL actuator range so the policy's near-constant
command is visible next to RBC's full sweep -- the actuator the policy leaves unused.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from pub_style import apply_pub_style, save, INK, REACTIVE_COLOR, W_E_COLORS, TEXTWIDTH_IN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NPZ = os.path.join(REPO, "data", "policy_vs_rbc_rollout.npz")
FIGDIR = os.path.join(REPO, "figures_out")

POLICY_COLOR = W_E_COLORS["e0"]   # #2a78d6
RBC_COLOR = REACTIVE_COLOR        # #eb6834
STEPS_PER_DAY = 12 * 24
FAN_MAX = 15.0                    # OfficeSmall PSZ fan actuator upper bound (kg/s)


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    p_fan, r_fan, tgt = d["p_fan"], d["r_fan"], d["tgt"]
    bid = str(d["building_id"]) if "building_id" in d.files else "held-out building"
    x = np.arange(len(p_fan)) / STEPS_PER_DAY
    p_span, r_span = np.ptp(p_fan), np.ptp(r_fan)

    apply_pub_style()
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.7))

    # full actuator range as context
    ax.axhline(0.0, color=INK["axis"], linewidth=0.8, zorder=1)
    ax.axhline(FAN_MAX, color=INK["axis"], linewidth=0.8, linestyle=":", zorder=1)
    ax.text(x[-1], FAN_MAX, " actuator max", va="center", ha="left",
            color=INK["muted"], fontsize=7.5)

    setback = tgt < (np.nanmax(tgt) - 1e-6)
    ax.fill_between(x, 0, 1, where=setback, transform=ax.get_xaxis_transform(),
                    color=INK["grid"], alpha=0.55, linewidth=0, zorder=0)
    ax.plot(x, r_fan, color=RBC_COLOR, linewidth=1.8, label="RBC (counterfactual)", zorder=4)
    ax.plot(x, p_fan, color=POLICY_COLOR, linewidth=2.0, label="Policy", zorder=5)

    ax.set_ylim(-0.6, FAN_MAX + 1.2)
    ax.set_ylabel("Fan mass flow (kg/s)")
    ax.set_xlabel("Days (shaded = task setback hours)")
    ax.text(0.5, 0.06,
            f"fan range used:  policy {p_span:.1f} kg/s   vs   RBC {r_span:.1f} kg/s "
            f"(of {FAN_MAX:.0f})",
            transform=ax.transAxes, ha="center", va="bottom",
            color=INK["secondary"], fontsize=8.5)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", visible=False)

    fig.suptitle(bid, x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fan_speed_preview.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    save(fig, os.path.join(FIGDIR, "fan_speed"))
    print("wrote:", os.path.join(FIGDIR, "fan_speed.pdf"))


if __name__ == "__main__":
    main()
