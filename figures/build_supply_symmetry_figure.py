"""The policy cannot tell the vav_supply loops apart: rollout evidence.

Input: data/supply_symmetry_rollout.npz, written by
    python scripts/check_supply_symmetry.py --steps 1000 \
        --checkpoint runs/dagger_stream_e0_amorpheus_y2/model_s0.eqx \
        --dump data/supply_symmetry_rollout.npz
i.e. a real rollout (env driven by the trained policy) recording each
vav_supply node's deterministic action at every step.

Figure: scatter of loop 2's supply-air-temperature command against loop 1's,
same step. Under the plain Amorpheus bridge the supply nodes are the same
constant token, so every point must sit EXACTLY on the identity line --
which is what the data shows (max deviation 0; loop 3, not plotted, is
identical too and the collection script asserts it).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from pub_style import apply_pub_style, save, INK, W_E_COLORS, COLWIDTH_IN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NPZ = os.path.join(REPO, "data", "supply_symmetry_rollout.npz")
FIGDIR = os.path.join(REPO, "figures_out")

def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    node_ids = [str(n) for n in d["node_ids"]]
    # raw env actions are in real units; pick the per-loop SAT columns by name,
    # ordered to match node_ids (both carry the same "(NN)" schedule token)
    names = [str(n) for n in d["action_names"]]
    cols = []
    for nid in node_ids:
        tok = nid[nid.rindex("("):]  # e.g. "(24)"
        (j,) = [i for i, n in enumerate(names)
                if "supply temp setpoint schedule" in n.lower() and n.endswith(tok)]
        cols.append(j)
    sat = d["raw_actions"][:, cols]  # (steps, loops), degrees C
    assert sat.shape[1] >= 2, "need at least two supply nodes to compare"
    max_dev = float(np.max(np.abs(sat[:, 1] - sat[:, 0])))

    apply_pub_style()
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, COLWIDTH_IN))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    lo, hi = float(sat.min()), float(sat.max())
    pad = 0.06 * (hi - lo) if hi > lo else 1.0
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, linestyle="--", linewidth=0.9, color=INK["secondary"],
            label="identity", zorder=1)

    ax.scatter(sat[:, 0], sat[:, 1], s=26, marker="o", linewidths=0,
               color=W_E_COLORS["e0"], alpha=0.55, label="one rollout step",
               zorder=3)

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Loop 1 supply-air temp. command (°C)")
    ax.set_ylabel("Loop 2 supply-air temp. command (°C)")
    ax.text(0.975, 0.03,
            f"{sat.shape[0]} steps, max |Δ| = {max_dev:.1e}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=INK["secondary"], fontsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "supply_symmetry_scatter_preview.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    save(fig, os.path.join(FIGDIR, "supply_symmetry_scatter"))
    print("wrote:", os.path.join(FIGDIR, "supply_symmetry_scatter.pdf"))
    print("nodes:", *node_ids, sep="\n  ")
    print(f"max pairwise |SAT dev| = {max_dev:.3e} over {sat.shape[0]} steps "
          f"(bridge={d['bridge']}, ckpt={d['checkpoint']})")


if __name__ == "__main__":
    main()
