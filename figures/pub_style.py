#!/usr/bin/env python3
"""Publication figure style + validated palette.

Vector PDF at true LaTeX size (no scaling in LaTeX), no titles (LaTeX supplies
them), recessive chrome, and a colourblind-safe palette validated with the
checks in validate_palette.py (all pairwise CVD dE >= 13.3 among the four
series below):

* energy weight ``w_e`` -> genuinely distinct categorical hues (NOT shades of
  one colour -- an earlier revision used a blue ordinal ramp, but overlapping
  alpha-filled histograms of one hue blend and wash out the read).
  e0 = blue, e05 = green, emed = violet, ehigh = red.
* reactive baseline -> orange, distinct from e0/emed.

Colour follows the *entity*: the same weight is the same colour in every
figure, and the reactive baseline is always the same orange. All colours
individually clear >=3:1 contrast on the light surface, which matters because
marks here are PLAIN filled (no outline) -- a low-contrast colour would wash
out at fill alpha alone with nothing else to carry it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- palette (light mode; validated -- see validate_palette.py) ---------------
W_E_COLORS: dict[str, str] = {
    "e0": "#2a78d6",
    "e05": "#008300",
    "emed": "#4a3aa7",
    "ehigh": "#e34948",
}
W_E_LABELS: dict[str, str] = {
    "e0": r"$w_E=0$",
    "e05": r"$w_E=0.5$",
    "emed": r"$w_E=1$",
    "ehigh": r"$w_E=5$",
}
REACTIVE_COLOR = "#eb6834"
REACTIVE_LABEL = "Reactive"

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    # Figures have NO opaque background fill (see apply_pub_style/save below) --
    # they render transparent so the page shows through directly, rather than
    # painting an explicit near-white rectangle that can visibly mismatch the
    # actual page white. "surface" here is only the REFERENCE white used for
    # contrast checks (validate_palette.py) -- not a colour painted anywhere.
    "surface": "#ffffff",
}

# LaTeX target widths (inches). Single-column ~3.3", full text width ~6.9".
COLWIDTH_IN = 3.3
TEXTWIDTH_IN = 6.9


def apply_pub_style() -> None:
    """Set rcParams for publication PDFs. Call once before plotting."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,  # embed TrueType (editable text, not paths)
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": INK["axis"],
            "axes.labelcolor": INK["primary"],
            "text.color": INK["primary"],
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            # No opaque fill -- see save() below, which also forces
            # transparent=True at export time regardless of these rcParams.
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "figure.dpi": 150,
        }
    )


def new_axes(width: float = COLWIDTH_IN, height: float = 2.35):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)  # only horizontal reference lines
    return fig, ax


def overlay_hist(
    ax,
    datasets: list[np.ndarray],
    colors: list[str],
    labels: list[str],
    *,
    bins,
    density: bool = True,
    show_median: bool = False,
    rug: bool = False,
    fill_alpha: float = 0.5,
) -> None:
    """Overlay several distributions as plain, moderately transparent filled
    histograms (no outline). Colour alone identifies each series -- the palette
    is chosen so every colour used together in one chart clears the
    CVD-separation and contrast checks, so no secondary channel is needed.

    ``rug=True`` also draws individual sample ticks (use for small-n
    distributions, e.g. one point per building; omit for pooled per-step data).
    """
    for data, color, label in zip(datasets, colors, labels):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        ax.hist(
            data,
            bins=bins,
            density=density,
            histtype="stepfilled",
            color=color,
            alpha=fill_alpha,
            linewidth=0,
            label=label,
        )
        if show_median:
            ax.axvline(np.median(data), color=color, linestyle="--", linewidth=1.1)
        if rug:
            y0 = -0.02 * ax.get_ylim()[1] if ax.get_ylim()[1] else 0
            ax.plot(
                data,
                np.full_like(data, y0),
                "|",
                color=color,
                markersize=6,
                markeredgewidth=1.2,
                clip_on=False,
            )


def barh_axes(width: float = COLWIDTH_IN, height: float = 2.35):
    """Like `new_axes`, but for HORIZONTAL bar charts: values are read on the
    x-axis, so keep the vertical reference lines and drop the y (category) grid
    — the opposite grid orientation from `new_axes`."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", visible=False)  # value axis (x) keeps its reference lines
    return fig, ax


def grouped_barh(ax, categories, series, *, invert: bool = True,
                 value_label: str | None = None,
                 reference: float | None = None,
                 reference_label: str | None = None) -> None:
    """House-style grouped horizontal bar chart: plain filled bars (no outline),
    `alpha=0.55`, capped error bars, category labels on y. `series` is a list of
    `(values, errs_or_None, color, label)`, drawn top-to-bottom in order. Colour
    is the only series channel (§7) — pass validated/entity colours. `invert`
    puts 0 at the left (for negative-return charts where closer-to-0 is better).
    `reference` draws a dashed vertical marker (e.g. 1.0 for a normalized score
    where 1.0 = baseline) — a semantic reference, not decoration; pass
    `reference_label` to give it a legend entry.
    """
    y = np.arange(len(categories))
    n = len(series)
    h = 0.8 / n
    for k, (vals, errs, color, label) in enumerate(series):
        offset = ((n - 1) / 2 - k) * h
        ax.barh(
            y + offset, vals, height=h * 0.92, color=color, alpha=0.55,
            linewidth=0, label=label, xerr=errs,
            error_kw=(dict(ecolor=INK["primary"], elinewidth=0.8, capsize=2,
                           capthick=0.8) if errs is not None else None),
        )
    ax.set_yticks(y)
    ax.set_yticklabels([str(c) for c in categories])
    ax.axvline(0, color=INK["primary"], linewidth=0.6)
    if reference is not None:
        ax.axvline(reference, color=INK["secondary"], linestyle="--",
                   linewidth=0.8, label=reference_label)
    if invert:
        ax.invert_xaxis()
    if value_label:
        ax.set_xlabel(value_label)


def save(fig, path: str | Path) -> None:
    """Save as vector PDF at true size (tight bbox, no title, transparent
    background). ``transparent=True`` is explicit here (not just the rcParam
    defaults) so every figure has NO opaque fill -- the compiled LaTeX page
    shows through directly, guaranteed to match regardless of the exact white
    the page renderer uses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True
    )
    plt.close(fig)
