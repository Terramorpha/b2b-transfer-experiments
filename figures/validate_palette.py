#!/usr/bin/env python3
"""Validate a categorical figure palette against computable colour-vision and
contrast checks. Same thresholds as the dataviz skill's validator, same Machado
et al. (2009) CVD simulation matrices, same OKLab/OKLCH math.

Run this BEFORE finalizing any new colour for a figure. Do not eyeball
colourblind-safety -- compute it.

Checks:
  1. Lightness band   -- OKLCH L within the mode's band (light: 0.43-0.77)
  2. Chroma floor      -- OKLCH C >= 0.10 (below it a hue reads as gray)
  3. CVD separation    -- CIE76 dE between pairs, simulated for protan/deutan/
                          tritan colour vision (Machado 2009). Target >= 12;
                          8-12 is a floor legal ONLY with a secondary channel
                          (direct label, outline, hatch); < 8 is a hard fail.
  4. Contrast vs surface -- WCAG ratio of each colour against the chart
                          surface. Matters most for PLAIN FILLED marks (bars,
                          filled histograms) with no outline to fall back on.

Usage::

    python validate_palette.py "#2a78d6,#4a3aa7,#e34948"
    python validate_palette.py "#eb6834,#008300" --mode light
    python validate_palette.py "#2a78d6,#eb6834" --pairs all

Exit code 0 unless a check hard-fails; 1 on any FAIL (WARN bands still exit 0).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 12.0, 8.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#ffffff", "dark": "#1a1a19"}

# Machado, Oliveira & Fernandes (2009) CVD transforms, severity 1.0, linear RGB.
MACHADO = {
    "protan": [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ],
    "deutan": [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ],
    "tritan": [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ],
}


def _hex2srgb(h: str) -> list[float]:
    h = h.strip().lstrip("#")
    return [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]


def _s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin(h: str) -> list[float]:
    return [_s2lin(c) for c in _hex2srgb(h)]


def _rel_lum(h: str) -> float:
    r, g, b = _lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted([_rel_lum(a), _rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab(h: str) -> list[float]:
    r, g, b = _lin(h)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]


def oklch(h: str) -> tuple[float, float]:
    L, a, b = _oklab(h)
    return L, math.hypot(a, b)


def _lin2lab(r: float, g: float, b: float) -> list[float]:
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(X / 0.95047), f(Y / 1.0), f(Z / 1.08883)
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]


def _simulate(h: str, kind: str) -> list[float]:
    r, g, b = _lin(h)
    M = MACHADO[kind]
    clamp = lambda c: max(0.0, min(1.0, c))
    return [
        clamp(M[0][0] * r + M[0][1] * g + M[0][2] * b),
        clamp(M[1][0] * r + M[1][1] * g + M[1][2] * b),
        clamp(M[2][0] * r + M[2][1] * g + M[2][2] * b),
    ]


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = _lin2lab(*(_simulate(h1, kind) if kind else _lin(h1)))
    b = _lin2lab(*(_simulate(h2, kind) if kind else _lin(h2)))
    return math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def min_cvd_delta_e(h1: str, h2: str) -> float:
    """Worst-case (most similar) CVD delta-E across protan/deutan/tritan."""
    return min(delta_e(h1, h2, k) for k in MACHADO)


def validate(
    palette: list[str],
    mode: str = "light",
    surface: str | None = None,
    pairs: str = "adjacent",
) -> bool:
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    ok = True

    print(f"Palette: {palette}  mode={mode}  surface={surface}")
    print()

    # 1. lightness band
    offband = [
        (c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)
    ]
    print(
        f"[{'PASS' if not offband else 'FAIL'}] Lightness band ({lo}-{hi}): "
        f"{'all in band' if not offband else offband}"
    )
    ok &= not offband

    # 2. chroma floor
    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    print(
        f"[{'PASS' if not lowc else 'FAIL'}] Chroma floor (>= {CHROMA_FLOOR}): "
        f"{'all above floor' if not lowc else lowc}"
    )
    ok &= not lowc

    # 3. CVD separation
    pair_list = (
        list(itertools.combinations(range(len(palette)), 2))
        if pairs == "all"
        else [(i, i + 1) for i in range(len(palette) - 1)]
    )
    print(
        f"[--] CVD separation ({pairs} pairs, target >= {CVD_TARGET}, floor {CVD_FLOOR}):"
    )
    for i, j in pair_list:
        v = min_cvd_delta_e(palette[i], palette[j])
        verdict = "PASS" if v >= CVD_TARGET else ("WARN" if v >= CVD_FLOOR else "FAIL")
        print(f"       {palette[i]} ~ {palette[j]}: dE={v:5.1f}  [{verdict}]")
        if v < CVD_FLOOR:
            ok = False

    # 4. contrast vs surface
    print(f"[--] Contrast vs surface (target >= {CONTRAST_MIN}):")
    for c in palette:
        ct = contrast(c, surface)
        verdict = (
            "PASS"
            if ct >= CONTRAST_MIN
            else "WARN (relief band -- needs a secondary channel)"
        )
        print(f"       {c}: contrast={ct:.2f}  [{verdict}]")

    print()
    print(
        "Result:",
        (
            "PASS (WARNs are legal only with a secondary channel -- see docstring)"
            if ok
            else "FAIL -- fix before using this palette"
        ),
    )
    return ok


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "palette", help="Comma-separated hex colours, e.g. '#2a78d6,#e34948'"
    )
    p.add_argument("--mode", choices=["light", "dark"], default="light")
    p.add_argument("--surface", default=None, help="Override the chart surface hex")
    p.add_argument(
        "--pairs",
        choices=["adjacent", "all"],
        default="adjacent",
        help="'all' for scatter/bubble/map palettes where every pair can be adjacent",
    )
    a = p.parse_args()

    palette = [c.strip() for c in a.palette.split(",") if c.strip()]
    ok = validate(palette, mode=a.mode, surface=a.surface, pairs=a.pairs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
