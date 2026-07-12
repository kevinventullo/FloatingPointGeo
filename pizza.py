"""North Pole 'pizza' — a standalone, deliberately abstract picture of how
finely IEEE-754 floats slice up the area around the pole.

Looking straight down on the North Pole, the cells cut out by consecutive
representable *longitudes* are pizza slices: all share the pole as apex, all
reach the nearest representable parallel at radius

    radius = ulp(90°) · 111320 m/deg ≈ 0.85 m   (float32)

What varies wildly is the *arc length* of a slice at that rim,

    arc(lon) = ulp(lon) · 111320 · cos(lat_rim),   lat_rim = 90 − ulp(90°)

Since ulp(lon) is constant within each power-of-two binade and halves every
binade toward 0, the pie is sliced in angular bands that crowd toward the
**prime meridian**, where the slices plunge from ~0.2 µm at the date line to
~1e-40 m (flush-to-zero) / ~1e-47 m (subnormals) in a measure-zero spoke.

This is a schematic, not a map: there is deliberately no uniform scale.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

_MPD = 111_320.0  # mean metres per degree of latitude
_DTYPES = {32: np.float32, 64: np.float64}
_SUBNORMAL = {32: 2.0 ** -149, 64: 2.0 ** -1074}
_SMALLEST_NORMAL = {32: 2.0 ** -126, 64: 2.0 ** -1022}


def ulp(values: np.ndarray, dtype) -> np.ndarray:
    """Gap from each value to the next representable float (on the magnitude)."""
    a = np.abs(np.asarray(values, float)).astype(dtype)
    nxt = np.nextafter(a, np.array(np.inf, dtype=dtype))
    return (nxt - a).astype(np.float64)


def _polar_cmap():
    """Viridis grafted onto a red low end (deepest, smallest slices = red)."""
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap

    vir = cm.viridis(np.linspace(0.0, 1.0, 256))
    red = np.array([0.55, 0.0, 0.10, 1.0])
    low = np.linspace(red, vir[0], 96)
    return LinearSegmentedColormap.from_list("viridis_red", np.vstack([low, vir]))


def _fmt_len(metres: float) -> str:
    a = abs(metres)
    for name, fac in [("m", 1.0), ("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9)]:
        if a >= fac:
            return f"{metres / fac:.3g} {name}"
    return f"{metres:.0e} m"


def render(precision: int, subnormals: bool, path: str, n_slices: int = 104):
    """Draw the abstract North-Pole pizza as a single figure.

    Stylised, not to scale: the prime meridian sits at the BOTTOM (where the
    slices collapse to a measure-zero point) and the date line at the TOP (where
    a slice is at its fattest, ~226 nm). Each slice toward the bottom is a fresh
    representable longitude, and the crust arc length halves with every
    power-of-two binade toward 0°, so the slices visibly thin and crowd into the
    red spoke at the prime meridian.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle, FancyArrowPatch

    dtype = _DTYPES[precision]
    floor = _SUBNORMAL[precision] if subnormals else _SMALLEST_NORMAL[precision]

    r_pole_deg = float(ulp(90.0, dtype))                 # ulp of lat at the pole
    radius_m = r_pole_deg * _MPD                          # pizza radius (uniform)
    parallel = 90.0 - r_pole_deg                          # nearest representable lat
    cos_rim = math.sin(math.radians(r_pole_deg))         # cos(lat_rim)
    mpd_lon = _MPD * cos_rim                              # metres per deg lon, rim

    arc_dl = float(ulp(180.0, dtype) * mpd_lon)          # date line crust (fattest)
    arc_pm = floor * mpd_lon                              # prime meridian crust (floor)
    cmap = _polar_cmap()
    mode = "subnormals on" if subnormals else "flush-to-zero"

    # Per-side slice widths from the date line (top) down to the prime meridian
    # (bottom): a geometric taper so they thin and bunch toward the bottom. They
    # span the 180° half exactly; the crust label on each step halves (one binade
    # of ulp(lon)) regardless of the drawn width, which is schematic.
    ratio = 0.965
    widths = ratio ** np.arange(n_slices)
    widths *= 180.0 / widths.sum()                       # normalise to half-turn

    fig, ax = plt.subplots(figsize=(6.4, 7.4), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.85, 1.65)

    # Draw both halves. plot-angle = 90° (top) at the date line, sweeping down to
    # -90° (bottom) at the prime meridian on the right (east); mirrored on the
    # left (west). Colour: yellow (fattest, top) -> red (finest, bottom).
    a_right = 90.0
    for i, w in enumerate(widths):
        t = 1.0 - i / (n_slices - 1)                     # 1 at top -> 0 at bottom
        color = cmap(t)
        for lo, hi in (
            (a_right - w, a_right),                       # east half
            (180.0 - a_right, 180.0 - (a_right - w)),     # west half (mirror)
        ):
            ax.add_patch(Wedge((0, 0), 1.0, lo, hi, facecolor=color,
                               edgecolor="white", lw=0.25, zorder=2))
        a_right -= w

    # Rim = the nearest representable parallel (a real, uniform circle).
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#222", lw=1.4, zorder=4))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#222", lw=0.8, ls=(0, (5, 4)),
                        zorder=4))

    # The two named meridian spokes: prime meridian (bottom, where slices
    # collapse to a point) and date line (top).
    ax.plot([0, 0], [0, -1], color="#ff2222", lw=2.4, zorder=5)
    ax.plot([0, 0], [0, 1], color="#333", lw=1.2, zorder=5)
    ax.scatter([0], [0], s=26, color="k", zorder=6)      # the pole itself

    # The meridian names run sideways along their own spokes, inside the circle.
    ax.text(-0.05, -0.52, "Prime Meridian", fontsize=8.5, rotation=90,
            rotation_mode="anchor", ha="center", va="center", zorder=7)
    ax.text(-0.05, 0.55, "Int'l Date Line", fontsize=8.5, rotation=90,
            rotation_mode="anchor", ha="center", va="center", zorder=7)

    # Arc-length indicators: a narrow rotated brace at each rim end. The bottom
    # brace is drawn smaller than the top to evoke the slice shrinkage.
    ax.text(0, 1.06, "{", fontsize=18, rotation=-90, ha="center", va="center",
            zorder=8)
    ax.text(0, 1.18, f"{_fmt_len(arc_dl)} arclength", fontsize=8.5, ha="center",
            va="bottom", zorder=8)
    ax.text(0, -1.05, "{", fontsize=10, rotation=90, ha="center", va="center",
            zorder=8)
    ax.text(0, -1.14, f"{arc_pm:.0e} m arclength", fontsize=8.5, ha="center",
            va="top", zorder=8)

    # Uniform-radius callout: double-headed arrow at ~2 o'clock, plain label
    # placed below the line so it doesn't sit on the radius.
    ang = math.radians(30)
    rx, ry = math.cos(ang), math.sin(ang)
    ax.add_patch(FancyArrowPatch((0, 0), (rx, ry), arrowstyle="<->",
                                 mutation_scale=12, color="#111", lw=1.3, zorder=6))
    ax.text(rx * 0.5 + 0.06, ry * 0.5 - 0.05, f"r ≈ {radius_m:.2f} m\n(uniform)",
            fontsize=6.5, ha="left", va="top", zorder=7)

    # Pole label sits next to the dot (no arrow); rim parallel labelled at left,
    # sitting just outside the circle.
    ax.text(0.06, -0.06, "North Pole\nlat = 90°", fontsize=7, ha="left",
            va="top", zorder=7)
    ax.text(-1.02, -0.30, f"lat =\n{parallel:.8f}°", fontsize=7, ha="right",
            va="center", zorder=7)

    ax.text(0, 1.42, f"Float{precision}-representable lat/lons in a\n~1 meter "
            f"radius around the North Pole",
            fontsize=11, ha="center", va="bottom", fontweight="bold")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"float{precision} ({mode}): radius {radius_m:.3f} m, "
          f"crust {_fmt_len(arc_dl)} (date line) → {arc_pm:.1e} m "
          f"(prime meridian)  ->  {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", choices=["32", "64", "both"], default="32")
    p.add_argument("--no-subnormals", dest="subnormals", action="store_false",
                   help="model a flush-to-zero system (the floor at 0° becomes "
                        "the smallest normal instead of the smallest subnormal)")
    p.add_argument("--outdir", default="docs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    precisions = [32, 64] if args.precision == "both" else [int(args.precision)]
    for prec in precisions:
        render(prec, args.subnormals, _next_version(args.outdir, prec))


def _next_version(outdir: str, precision: int) -> str:
    """Return the next un-used pizza_float{p}_vN.png path (never overwrite)."""
    n = 1
    while True:
        path = os.path.join(outdir, f"pizza_float{precision}_v{n}.png")
        if not os.path.exists(path):
            return path
        n += 1


if __name__ == "__main__":
    main()
