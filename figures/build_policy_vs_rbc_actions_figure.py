"""Policy vs RBC commands in the same state, one week, a held-out building.

Env driven by the POLICY; at each step RBC is asked (counterfactually) what it would
command from the same observation. Top: supply-air-temperature setpoint (policy vs RBC)
plus the task setpoint -- all degrees C, one shared axis. Bottom: resulting zone
temperature vs the task setpoint.
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


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    p_sat, r_sat, tgt, tz = d["p_sat"], d["r_sat"], d["tgt"], d["tz"]
    bid = str(d["building_id"]) if "building_id" in d.files else "held-out building"
    x = np.arange(len(p_sat)) / STEPS_PER_DAY

    apply_pub_style()
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 4.2), sharex=True)

    ax = axes[0]
    setback = tgt < (np.nanmax(tgt) - 1e-6)
    ax.fill_between(x, 0, 1, where=setback, transform=ax.get_xaxis_transform(),
                    color=INK["grid"], alpha=0.55, linewidth=0, zorder=0)
    ax.plot(x, tgt, color=INK["secondary"], linestyle="--", linewidth=1.4,
            label="Task setpoint", zorder=3)
    ax.plot(x, r_sat, color=RBC_COLOR, linewidth=1.8, label="RBC (counterfactual)", zorder=4)
    ax.plot(x, p_sat, color=POLICY_COLOR, linewidth=1.8, label="Policy", zorder=5)
    ax.set_ylabel("Supply-air setpoint (°C)")
    ax.text(0.995, 0.05, f"mean(policy − RBC) = {np.mean(p_sat - r_sat):+.2f} °C",
            transform=ax.transAxes, ha="right", va="bottom",
            color=INK["secondary"], fontsize=8)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)

    ax = axes[1]
    ax.fill_between(x, 0, 1, where=setback, transform=ax.get_xaxis_transform(),
                    color=INK["grid"], alpha=0.55, linewidth=0, zorder=0)
    ax.plot(x, tgt, color=INK["secondary"], linestyle="--", linewidth=1.4,
            label="Task setpoint", zorder=3)
    ax.plot(x, tz, color=POLICY_COLOR, linewidth=1.8, label="Zone temp (under policy)", zorder=4)
    ax.set_ylabel("Zone temp (°C)")
    ax.set_xlabel("Days (shaded = task setback hours)")
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)

    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(axis="x", visible=False)

    fig.suptitle(bid, x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "policy_vs_rbc_actions_preview.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    save(fig, os.path.join(FIGDIR, "policy_vs_rbc_actions"))
    print("wrote:", os.path.join(FIGDIR, "policy_vs_rbc_actions.pdf"))


if __name__ == "__main__":
    main()
