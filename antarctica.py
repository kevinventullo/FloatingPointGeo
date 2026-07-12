"""Top-down (south-polar) view of IEEE-754 float32 spatial granularity.

This is the same heatmap as ``float32.html`` -- log10 of the ground area of the
smallest cell you can resolve when a coordinate is stored as a 32-bit float --
but drawn on an azimuthal projection centred on the **South Pole** instead of
Web Mercator, so you're looking straight down at Antarctica.

Why bother re-projecting? Mercator can't show the poles at all (it clips near
+/-85 deg), yet the poles are exactly where the story gets wild: meridians
converge, so longitude metres collapse as cos(lat), and the float structure --
power-of-two wedges in longitude (jumps at |lon| = 64, 128 deg) crossed by
latitude bands -- fans out around the pole like a dartboard. Draping that over
the Antarctic coastline makes the pattern legible as a map.

Projection: azimuthal-equidistant on the South Pole (radius = colatitude =
90 + lat, so parallels are evenly spaced circles; longitude = azimuth). Colours
are viridis on the *same* log10(m^2) scale as the world map, so a given colour
means the same physical cell size in both pictures.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os

import numpy as np

import granularity as g   # reuse ulp(), constants, and the world-map value scale

_LAT_BOUND = -50.0        # outer edge of the disc (deg); colatitude 40 deg
_COAST = "docs/antarctica_coast.geojson"


def _world_scale(precision: int) -> tuple[float, float]:
    """(vmin, vmax) of the float32 world map, so colours match ``float32.html``.

    float32.html is rendered with the Mercator projection at 2400x1600; we reuse
    the exact same finite log10(area) range here so a colour is comparable across
    the two maps.
    """
    _, _, vmin, vmax = g.compute_grid(precision, "ground", 2400, 1600, "mercator")
    return vmin, vmax


def _antarctic_rings():
    """Yield (lon, lat) arrays for each Antarctic coastline ring."""
    with open(_COAST) as f:
        gj = json.load(f)
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [geom["coordinates"]])
        for poly in polys:
            for ring in poly:
                arr = np.asarray(ring, float)
                yield arr[:, 0], arr[:, 1]


def compute_polar_grid(precision: int, n_r: int, n_theta: int):
    """log10(cell area, m^2) on a (colatitude, longitude) grid, row 0 at the pole.

    Returns (logA, r_edges, theta_edges) where ``r_edges`` are colatitude edges
    (deg, 0 at the South Pole) and ``theta_edges`` are longitude edges (radians).
    """
    dtype = g._DTYPES[precision]
    r_edges = np.linspace(0.0, 90.0 + _LAT_BOUND, n_r + 1)        # colatitude
    theta_edges = np.radians(np.linspace(-180.0, 180.0, n_theta + 1))
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    th_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    lat = r_c - 90.0                                              # (n_r,)
    lon = np.degrees(th_c)                                        # (n_theta,)

    ulp_lat = g.ulp(lat, dtype)[:, None]                         # (n_r, 1) deg
    ulp_lon = g.ulp(lon, dtype)[None, :]                         # (1, n_theta) deg
    cos_lat = np.cos(np.radians(lat))[:, None]

    d_lat_m = ulp_lat * g._METRES_PER_DEG
    d_lon_m = ulp_lon * g._METRES_PER_DEG * cos_lat
    with np.errstate(divide="ignore"):
        logA = np.log10(d_lat_m * d_lon_m)                       # (n_r, n_theta)
    return logA, r_edges, theta_edges


def render(precision: int, path: str, n_r: int = 900, n_theta: int = 1440) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    logA, r_edges, theta_edges = compute_polar_grid(precision, n_r, n_theta)
    vmin, vmax = _world_scale(precision)

    fig = plt.figure(figsize=(7.4, 8.2), dpi=150)
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.80], projection="polar")

    # Orient like a standard south-polar map: 0 deg (Greenwich) at the top,
    # longitude increasing clockwise, so the Antarctic Peninsula points up-left
    # toward South America and the Ross Sea sits at the bottom (180 deg).
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    TH, R = np.meshgrid(theta_edges, r_edges)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    ax.pcolormesh(TH, R, logA, cmap=cm.viridis, norm=norm, shading="flat",
                  zorder=1, rasterized=True)

    # Antarctic coastline on top, so the heatmap reads as a real map.
    for lon, lat in _antarctic_rings():
        ax.plot(np.radians(lon), lat + 90.0, color="#ffffff", lw=0.9,
                alpha=0.9, solid_joinstyle="round", zorder=3)

    ax.set_rlim(0.0, 90.0 + _LAT_BOUND)
    # Parallels every 10 deg, labelled as degrees south.
    parallels = [10.0, 20.0, 30.0, 40.0]           # colat -> lat -80..-50
    ax.set_rticks(parallels)
    ax.set_yticklabels([f"{int(90 - c)}°S" for c in parallels], fontsize=6.5,
                       color="#dddddd")
    ax.set_rlabel_position(157.5)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°", "45°E", "90°E", "135°E", "180°",
                        "135°W", "90°W", "45°W"], fontsize=7.5)
    ax.tick_params(axis="x", pad=2)
    ax.grid(True, lw=0.4, alpha=0.35, color="white")
    ax.spines["polar"].set_color("#888")

    ax.set_title("Float32 spatial granularity over Antarctica\n"
                 "smallest resolvable cell, looking straight down at the "
                 "South Pole", fontsize=12, fontweight="bold", pad=16)

    # Horizontal colour bar sharing the world-map scale.
    cax = fig.add_axes([0.18, 0.07, 0.64, 0.022])
    sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(f"float{precision} cell area — log₁₀(m²), ground metric   "
                 "(dark = fine, bright = coarse)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    finite = logA[np.isfinite(logA)]
    print(f"float{precision}: log10(m²) scale [{vmin:.2f}, {vmax:.2f}] "
          f"(region reaches {finite.min():.2f})  ->  {path}")


def _next_version(outdir: str, precision: int) -> str:
    n = 1
    while True:
        path = os.path.join(outdir, f"antarctica_float{precision}_v{n}.png")
        if not os.path.exists(path):
            return path
        n += 1


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", choices=["32", "64"], default="32")
    p.add_argument("--outdir", default="docs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prec = int(args.precision)
    render(prec, _next_version(args.outdir, prec))


if __name__ == "__main__":
    main()
