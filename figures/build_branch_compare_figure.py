"""Branched-rollout mechanism figures (OfficeMedium energy dominance).

Input: data/branch_compare_<bt>_<split><idx>.npz from
scripts/rollout_branch_compare.py -- independent same-start trajectories
(policy / energy-tuned RBC / optional default RBC), each controller
on-policy in its own branch, with the emed reward split per step.

Outputs (house style, one PDF per panel):
  branch_cum_energy    cumulative energy TERM (reward units) per branch
  branch_cum_comfort   cumulative comfort TERM per branch
  branch_cum_gas       cumulative raw gas (the dominant lever)
  branch_traces        mean zone temp + target + outdoor, first N days
  branch_sat           mean commanded SAT per branch, first N days

    python figures/build_branch_compare_figure.py \
        [data/branch_compare_OfficeMedium_test0.npz] [trace_days] [suffix] [policy label]
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from pub_style import apply_pub_style, preview, save, INK, W_E_COLORS, REACTIVE_COLOR, \
    COLWIDTH_IN, TEXTWIDTH_IN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NPZ = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(REPO, "data", "branch_compare_OfficeMedium_test0.npz"))
TRACE_DAYS = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""          # e.g. "_oaclamp"
POLICY_LABEL = sys.argv[4] if len(sys.argv) > 4 else "From-scratch PPO (5.1 y)"
FIGDIR = os.path.join(REPO, "figures_out")
STEPS_PER_DAY = 12 * 24

# colour follows the entity (validated triple, all PASS):
STYLE = {  # branch key -> (color, label)
    "policy": (W_E_COLORS["e0"], POLICY_LABEL),
    "tuned_rbc": (W_E_COLORS["e05"], "Energy-tuned RBC"),
    "default_rbc": (REACTIVE_COLOR, "Default RBC"),
}


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    branches = [b for b in ("policy", "tuned_rbc", "default_rbc")
                if f"{b}/reward" in d.files]
    an = [str(x) for x in d["action_names"]]
    sat_i = [i for i, n in enumerate(an) if "supply temp setpoint" in n.lower()]
    oa_i = [i for i, n in enumerate(an)
            if "outdoor air controller" in n.lower() and "mass flow" in n.lower()]
    apply_pub_style()
    os.makedirs(FIGDIR, exist_ok=True)

    def days(x):
        return np.arange(len(x)) / STEPS_PER_DAY

    # --- cumulative panels -------------------------------------------------
    for key, ylabel, stem in [
        ("energy", "Cumulative energy penalty", f"branch_cum_energy{SUFFIX}"),
        ("comfort", "Cumulative comfort penalty", f"branch_cum_comfort{SUFFIX}"),
        ("gas", "Cumulative gas consumption", f"branch_cum_gas{SUFFIX}"),
    ]:
        fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.35))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", visible=False)
        for b in branches:
            v = np.cumsum(np.asarray(d[f"{b}/{key}"], dtype=float))
            c, lab = STYLE[b]
            ax.plot(days(v), v, color=c, linewidth=1.8, label=lab)
        ax.set_xlabel("Days")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
        fig.tight_layout()
        preview(fig, os.path.join(FIGDIR, f"{stem}_preview"), dpi=200)
        save(fig, os.path.join(FIGDIR, stem))
        print("wrote:", os.path.join(FIGDIR, f"{stem}.pdf"))

    # --- trace panels (first TRACE_DAYS days) ------------------------------
    n = int(TRACE_DAYS * STEPS_PER_DAY)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.6))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", visible=False)
    tgt = np.asarray(d["policy/tgts"], dtype=float)[:n].mean(axis=1)
    x = np.arange(len(tgt)) / STEPS_PER_DAY
    setback = tgt < (np.nanmax(tgt) - 1e-6)
    ax.fill_between(x, 0, 1, where=setback, transform=ax.get_xaxis_transform(),
                    color=INK["grid"], alpha=0.55, linewidth=0, zorder=0)
    ax.plot(x, tgt, color=INK["secondary"], linestyle="--", linewidth=1.4,
            label="Target (occupied)", zorder=3)
    for b in branches:
        temps = np.asarray(d[f"{b}/temps"], dtype=float)[:n].mean(axis=1)
        c, lab = STYLE[b]
        ax.plot(x, temps, color=c, linewidth=1.6, label=lab, zorder=4)
    out = np.asarray(d["policy/outdoor"], dtype=float)[:n]
    ax.plot(x, out, color=INK["muted"], linewidth=1.0, label="Outdoor", zorder=2)
    ax.set_xlabel("Days (shaded = setback hours)")
    ax.set_ylabel("Mean zone temp (°C)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))  # headroom so the legend clears the data
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout()
    preview(fig, os.path.join(FIGDIR, f"branch_traces{SUFFIX}_preview"), dpi=200)
    save(fig, os.path.join(FIGDIR, f"branch_traces{SUFFIX}"))
    print("wrote:", os.path.join(FIGDIR, f"branch_traces{SUFFIX}.pdf"))

    if sat_i:
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.4))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", visible=False)
        ax.fill_between(x, 0, 1, where=setback, transform=ax.get_xaxis_transform(),
                        color=INK["grid"], alpha=0.55, linewidth=0, zorder=0)
        for b in branches:
            acts = np.asarray(d[f"{b}/actions"], dtype=float)[:n]
            c, lab = STYLE[b]
            ax.plot(x, acts[:, sat_i].mean(axis=1), color=c, linewidth=1.6,
                    label=lab)
        ax.set_xlabel("Days (shaded = setback hours)")
        ax.set_ylabel("Mean commanded SAT (°C)")
        ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=7.5)
        fig.tight_layout()
        preview(fig, os.path.join(FIGDIR, f"branch_sat{SUFFIX}_preview"), dpi=200)
        save(fig, os.path.join(FIGDIR, f"branch_sat{SUFFIX}"))
        print("wrote:", os.path.join(FIGDIR, f"branch_sat{SUFFIX}.pdf"))

    # --- OA intake: the isolation evidence ----------------------------------
    # Commanded outdoor-air mass flow (mean per air loop), full week, y from 0.
    # Both RBCs sit pinned at the 1.37 kg/s design intake; the policy holds
    # ~0.3 kg/s -- it isolates the building from the air it would have to heat.
    if oa_i:
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.4))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", visible=False)
        seen_levels = []
        for b in branches:
            acts = np.asarray(d[f"{b}/actions"], dtype=float)
            oa = acts[:, oa_i].mean(axis=1)
            c, lab = STYLE[b]
            # the two RBCs coincide exactly at the design intake -> the later
            # one is dashed so both stay visible on the shared line
            coincide = any(abs(oa.mean() - v) < 1e-3 for v in seen_levels)
            ax.plot(days(oa), oa, color=c, linewidth=1.8, label=lab, zorder=3,
                    linestyle="--" if coincide else "-",
                    dashes=(4, 3) if coincide else (None, None))
            ax.text(days(oa)[-1], oa[-1] + (0.09 if coincide else 0), f" {oa.mean():.2f}",
                    color=c, va="bottom" if coincide else "center", ha="left",
                    fontsize=8)
            seen_levels.append(oa.mean())
        ax.set_ylim(0, None)
        lo, hi = ax.get_ylim()
        ax.set_ylim(0, hi + 0.30 * hi)
        ax.set_xlabel("Days")
        ax.set_ylabel("OA flow (kg/s / loop)")
        ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=7.5)
        fig.tight_layout()
        preview(fig, os.path.join(FIGDIR, f"branch_oa_flow{SUFFIX}_preview"), dpi=200)
        save(fig, os.path.join(FIGDIR, f"branch_oa_flow{SUFFIX}"))
        print("wrote:", os.path.join(FIGDIR, f"branch_oa_flow{SUFFIX}.pdf"))

    # --- SAT vs outdoor scatter: the "OA modulation" picture ----------------
    # Each point is one step of the week. The RBCs' outdoor-reset law shows as
    # a steep negative slope (cold outside -> hot supply air, i.e. tempering
    # the constant 1.37 kg/s OA intake); the policy's near-neutral SAT barely
    # responds to outdoor temperature.
    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.7))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    outdoor = np.asarray(d["policy/outdoor"], dtype=float)
    for b in branches:
        acts = np.asarray(d[f"{b}/actions"], dtype=float)
        sat = acts[:, sat_i].mean(axis=1)
        c, lab = STYLE[b]
        ax.scatter(outdoor, sat, s=6, linewidths=0, color=c, alpha=0.35,
                   label=lab, zorder=3)
        k, b0 = np.polyfit(outdoor, sat, 1)
        xs = np.array([outdoor.min(), outdoor.max()])
        ax.plot(xs, k * xs + b0, color=c, linewidth=1.4, zorder=4)
    ax.set_xlabel("Outdoor temperature (°C)")
    ax.set_ylabel("Mean commanded SAT (°C)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    leg = ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    preview(fig, os.path.join(FIGDIR, f"branch_sat_vs_outdoor{SUFFIX}_preview"), dpi=200)
    save(fig, os.path.join(FIGDIR, f"branch_sat_vs_outdoor{SUFFIX}"))
    print("wrote:", os.path.join(FIGDIR, f"branch_sat_vs_outdoor{SUFFIX}.pdf"))

    for b in branches:
        print(f"  {b:12s} cum comfort {float(np.sum(d[f'{b}/comfort'])):9.1f}"
              f"  cum energy {float(np.sum(d[f'{b}/energy'])):9.1f}"
              f"  gas {float(np.sum(d[f'{b}/gas'])):9.0f}"
              f"  elec {float(np.sum(d[f'{b}/elec'])):9.0f}")


if __name__ == "__main__":
    main()
