"""House-style transfer figure for the PORTED experiment (current stack).

Reads runs/transfer_port_eval/{transfer_port_eval.csv, baseline_chunk.csv}
(per-(seed,building) episode_return over the first-672-step chunk + the RBC
baseline on the same chunk) and renders one transparent vector PDF per
building-type panel via pub_style, plus a PNG preview (viewing only).

Colours from pub_style (nothing by eye): Baseline = reactive orange
(REACTIVE_COLOR); Ours = the comfort-dominant e0 model -> W_E_COLORS["e0"].
"""
from __future__ import annotations

import argparse
import json
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
# Second series for the specialist-ceiling overlay: green reads as "the
# achievable target". Validated CVD-safe vs OURS_COLOR (dE=19.6 > 12 target,
# figures/validate_palette.py "#2a78d6,#008300").
CEIL_COLOR = W_E_COLORS["e05"]

TYPES = [("RetailStandalone", "Retail"), ("RestaurantFastFood", "Restaurant"),
         ("OfficeMedium", "OfficeMed"), ("OfficeSmall", "OfficeSmall")]
PANEL_W = TEXTWIDTH_IN / 4     # a 4-up figure at full text width
VALUE_LABEL = "Normalized return"  # return / baseline on the same 672-step chunk


def _series(sub, ours_label="Ours", ceiling_label=None, *, ours_color=None,
            ceiling_color=None, ceiling2_label=None, ceiling2_color=None):
    # Normalized return = return / baseline (b2b compute_normalized_score
    # convention). The baseline is a constant 1.0 -> shown only as the dashed
    # reference line, not a redundant flat bar; Ours is its ratio. Both on the
    # SAME chunk (baseline_chunk.csv).
    # Error bars only when there is real seed spread: with a single seed the
    # groupby std is NaN -> filled 0 -> capsize still paints misleading tick
    # marks at every bar end. n=1 => no error channel at all.
    errs = sub["ours_norm_err"].fillna(0.0).to_numpy()
    if not (errs > 0).any():
        errs = None
    s = [
        (sub["ours_norm"].to_numpy(), errs, ours_color or OURS_COLOR, ours_label),
    ]
    # Optional overlays: per-building {building_id: return} series (a specialist
    # "oracle" ceiling, another model, a tuned RBC...). n=1 -> no error bars.
    if ceiling_label is not None and "ceil_norm" in sub:
        s.append((sub["ceil_norm"].to_numpy(), None,
                  ceiling_color or CEIL_COLOR, ceiling_label))
    if ceiling2_label is not None and "ceil2_norm" in sub:
        s.append((sub["ceil2_norm"].to_numpy(), None,
                  ceiling2_color or REACTIVE_COLOR, ceiling2_label))
    return s


def _cats(sub):
    return [str(b).split("-")[-1] for b in sub["building_id"]]


def main() -> None:
    # Defaults reproduce the original camera-ready figure exactly; the flags let
    # the SAME renderer build other evals (e.g. full-year) in identical style.
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", default=CSV)
    ap.add_argument("--baseline-csv", default=BASE_CSV)
    ap.add_argument("--outdir", default=FIGDIR)
    ap.add_argument("--prefix", default="transfer")
    ap.add_argument("--ours-label", default="Ours")
    # Optional upper-bound overlay: {building_id: return} JSON (e.g. per-building
    # specialist "oracle"). When given, a second series is drawn + a legend.
    ap.add_argument("--ceiling-json", default=None)
    ap.add_argument("--ceiling-label", default="Specialist")
    ap.add_argument("--ceiling2-json", default=None,
                    help="optional third series, same {building_id: return} format")
    ap.add_argument("--ceiling2-label", default="Tuned RBC")
    ap.add_argument("--ours-color", default=None)
    ap.add_argument("--ceiling-color", default=None)
    ap.add_argument("--ceiling2-color", default=None)
    ap.add_argument("--reference-label", default="Baseline (G36)",
                    help="label for the normalized-return reference line (=1.0 baseline)")
    args = ap.parse_args()
    figdir, prefix = args.outdir, args.prefix
    clabel = args.ceiling_label if args.ceiling_json else None
    c2label = args.ceiling2_label if args.ceiling2_json else None
    skw = dict(ours_color=args.ours_color, ceiling_color=args.ceiling_color,
               ceiling2_label=c2label, ceiling2_color=args.ceiling2_color)

    df = pd.read_csv(args.eval_csv)
    bdf = pd.read_csv(args.baseline_csv)
    ours = (df.groupby(["building_type", "building_id"])["episode_return"]
              .agg(["mean", "std"]).reset_index())
    m = ours.merge(bdf, on=["building_type", "building_id"])
    m = m[np.isfinite(m["baseline_return"]) & (m["baseline_return"] != 0)]
    m["ours_norm"] = m["mean"] / m["baseline_return"]
    m["ours_norm_err"] = m["std"] / m["baseline_return"].abs()
    if args.ceiling_json:
        cj = json.load(open(args.ceiling_json))
        m["ceil_norm"] = m["building_id"].map(cj) / m["baseline_return"]
        missing = m[m["ceil_norm"].isna()]["building_id"].tolist()
        if missing:
            print(f"  (no ceiling value for {len(missing)} bldgs: {missing})")
    if args.ceiling2_json:
        cj2 = json.load(open(args.ceiling2_json))
        m["ceil2_norm"] = m["building_id"].map(cj2) / m["baseline_return"]

    apply_pub_style()
    os.makedirs(figdir, exist_ok=True)

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
        grouped_barh(ax, _cats(sub), _series(sub, args.ours_label, clabel, **skw),
                     value_label=None, invert=False, reference=1.0,
                     reference_label=args.reference_label)
        if i == 0:
            ax.set_ylabel("Building")
        save(fig, os.path.join(figdir, f"{prefix}_{short.lower()}"))

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
        grouped_barh(ax, _cats(sub), _series(sub, args.ours_label, clabel, **skw),
                     value_label=None, invert=False, reference=1.0,
                     reference_label=args.reference_label)
        ax.set_title(short)
    axes[0].set_ylabel("Building")
    fig.supxlabel(VALUE_LABEL)
    # With the ceiling overlay there are >=2 series -> a shared legend is
    # mandatory (identity must not be colour-alone). One figure-level legend
    # above the panels avoids four redundant copies.
    if clabel is not None:
        h, l = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{prefix}_port_preview.png"),  # quick-look raster
                dpi=200, bbox_inches="tight", facecolor="white")
    save(fig, os.path.join(figdir, f"{prefix}_combined"))  # transparent PDF, closes fig

    print("per-panel PDFs:", sorted(f for f in os.listdir(figdir) if f.endswith(".pdf")))
    print("combined PDF:", os.path.join(figdir, f"{prefix}_combined.pdf"))
    print("raster look:", os.path.join(figdir, f"{prefix}_port_preview.png"))


if __name__ == "__main__":
    main()
