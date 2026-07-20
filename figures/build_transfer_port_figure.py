"""House-style transfer figure for the PORTED experiment (current stack).

Reads runs/transfer_port_eval/{transfer_port_eval.csv, baseline_chunk.csv}
(per-(seed,building) episode_return over the first-672-step chunk + the RBC
baseline on the same chunk) and renders one transparent vector PDF per
building-type panel via pub_style, plus a PNG preview (viewing only).

Colours from pub_style (nothing by eye): Baseline = reactive orange
(REACTIVE_COLOR); Ours = the comfort-dominant e0 model -> W_E_COLORS["e0"].
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pub_style import (apply_pub_style, barh_axes, grouped_barh, save,
                       REACTIVE_COLOR, W_E_COLORS, TEXTWIDTH_IN)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV = os.path.join(REPO, "data", "transfer_port_eval.csv")
BASE_CSV = os.path.join(REPO, "data", "baseline_chunk.csv")
FIGDIR = os.path.join(REPO, "figures_out")
OURS_COLOR = W_E_COLORS["e0"]

TYPES = [("RetailStandalone", "Retail"), ("RestaurantFastFood", "Restaurant"),
         ("OfficeMedium", "OfficeMed"), ("OfficeSmall", "OfficeSmall")]
PANEL_W = TEXTWIDTH_IN / 4     # a 4-up figure at full text width
VALUE_LABEL = "Normalized return"  # return / baseline on the same 672-step chunk


def _series(sub):
    # Normalized return = return / baseline (b2b compute_normalized_score
    # convention). The baseline is a constant 1.0 -> shown only as the dashed
    # reference line, not a redundant flat bar; Ours is its ratio. Both on the
    # SAME chunk (baseline_chunk.csv).
    return [
        (sub["ours_norm"].to_numpy(), sub["ours_norm_err"].fillna(0.0).to_numpy(),
         OURS_COLOR, "Ours"),
    ]


def _cats(sub):
    return [str(b).split("-")[-1] for b in sub["building_id"]]


def main() -> None:
    df = pd.read_csv(CSV)
    bdf = pd.read_csv(BASE_CSV)
    ours = (df.groupby(["building_type", "building_id"])["episode_return"]
              .agg(["mean", "std"]).reset_index())
    m = ours.merge(bdf, on=["building_type", "building_id"])
    m = m[np.isfinite(m["baseline_return"]) & (m["baseline_return"] != 0)]
    m["ours_norm"] = m["mean"] / m["baseline_return"]
    m["ours_norm_err"] = m["std"] / m["baseline_return"].abs()

    apply_pub_style()
    os.makedirs(FIGDIR, exist_ok=True)

    # --- deliverables: one transparent vector PDF per atomic panel ---
    present = []
    for i, (btype, short) in enumerate(TYPES):
        sub = m[m["building_type"] == btype].sort_values("building_id")
        if not len(sub):
            print(f"  (no data for {btype}, skipping)")
            continue
        present.append((btype, short))
        fig, ax = barh_axes(width=PANEL_W, height=2.35)
        # No per-panel x-label: the four panels share one x quantity, so the
        # "Normalized return" label goes ONCE in LaTeX, centred under the row.
        grouped_barh(ax, _cats(sub), _series(sub), value_label=None,
                     invert=False, reference=1.0, reference_label="Baseline (G36)")
        if i == 0:
            ax.set_ylabel("Building")
        save(fig, os.path.join(FIGDIR, f"transfer_{short.lower()}"))

    # --- combined figure: all panels in one transparent vector PDF. A single
    #     image has no LaTeX subcaptions, so panel names live in the figure as
    #     titles; the shared x quantity is labelled once (supxlabel). No banner
    #     suptitle -- that belonged to the old throwaway preview. ---
    fig, axes = plt.subplots(1, len(present), figsize=(TEXTWIDTH_IN, 2.4))
    axes = np.atleast_1d(axes)
    for ax, (btype, short) in zip(axes, present):
        sub = m[m["building_type"] == btype].sort_values("building_id")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", visible=False)
        grouped_barh(ax, _cats(sub), _series(sub), value_label=None,
                     invert=False, reference=1.0, reference_label="Baseline (G36)")
        ax.set_title(short)
    axes[0].set_ylabel("Building")
    fig.supxlabel(VALUE_LABEL)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "transfer_port_preview.png"),  # quick-look raster
                dpi=200, bbox_inches="tight", facecolor="white")
    save(fig, os.path.join(FIGDIR, "transfer_combined"))  # transparent PDF, closes fig

    print("per-panel PDFs:", sorted(f for f in os.listdir(FIGDIR) if f.endswith(".pdf")))
    print("combined PDF:", os.path.join(FIGDIR, "transfer_combined.pdf"))
    print("raster look:", os.path.join(FIGDIR, "transfer_port_preview.png"))


if __name__ == "__main__":
    main()
