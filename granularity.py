"""Floating-point spatial granularity heatmaps over a world map.

For every (lat, lon) on a grid this computes the size of the smallest "cell" you
can resolve when the coordinate is stored as a 32- or 64-bit IEEE-754 float. The
cell is the rectangle spanned by

    (lat, lon), (nextafter(lat), lon), (nextafter(lat), nextafter(lon)), (lat, nextafter(lon))

Its size is governed by the ULP (unit in the last place) of the coordinate. ULP
grows in discrete powers of two with magnitude -- roughly value * 2^-23 for
float32 -- so the result is a tartan of power-of-two bands, finest at Null Island
(0, 0) where the spacing collapses toward the subnormal floor (~1.4e-45 deg, i.e.
~1.6e-40 m on the ground), and coarsest at the poles/antimeridian.

Two metrics (--metric):
  * ground (default): physical area in m^2; longitude metres scale by cos(lat),
    so meridians converge toward the poles.
  * coord:            pure coordinate area in deg^2 = ulp(lat) * ulp(lon).

The overlay is rendered in Web Mercator so it lines up with the Leaflet basemap,
embedded as a data URI, and dropped onto a folium slippy map -- a self-contained
HTML page suitable for GitHub Pages.
"""

from __future__ import annotations

import argparse
import base64
import math
import os

import numpy as np

# Mean metres per degree of latitude (the equator-to-pole variation is <1%,
# invisible on a log colour scale).
_METRES_PER_DEG = 111_320.0

# Web Mercator clamps latitude here (where mercator-y = +/- pi).
_MERC_LAT = math.degrees(2 * math.atan(math.exp(math.pi)) - math.pi / 2)  # 85.0511

_DTYPES = {32: np.float32, 64: np.float64}

# Smallest positive value per precision, in the two flush regimes. With
# subnormals (IEEE default) the spacing at 0 is the smallest subnormal; under
# flush-to-zero (many GPUs, -ffast-math, some DSPs) anything below the smallest
# normal collapses to 0, so the spacing at 0 is the smallest *normal* magnitude.
_SUBNORMAL = {32: 2.0 ** -149, 64: 2.0 ** -1074}
_SMALLEST_NORMAL = {32: 2.0 ** -126, 64: 2.0 ** -1022}


def _floor_ulp(precision: int, subnormals: bool) -> float:
    """Spacing of the representable grid at 0 (degrees), per flush regime."""
    return _SUBNORMAL[precision] if subnormals else _SMALLEST_NORMAL[precision]


def ulp(values: np.ndarray, dtype) -> np.ndarray:
    """Gap from each value to the next representable float of ``dtype``.

    Computed on the magnitude so it is symmetric about zero, matching how ULP
    behaves either side of a coordinate axis.
    """
    a = np.abs(values).astype(dtype)
    nxt = np.nextafter(a, np.array(np.inf, dtype=dtype))
    return (nxt - a).astype(np.float64)


def _lat_axis(height: int, projection: str):
    """Return (lat_centres, lat_bound) for the given vertical projection."""
    if projection == "mercator":
        ymax = math.pi
        # Row centres uniform in mercator-y, from +ymax (north) to -ymax.
        frac = (np.arange(height) + 0.5) / height
        y = ymax - 2 * ymax * frac
        lats = np.degrees(2 * np.arctan(np.exp(y)) - math.pi / 2)
        return lats, _MERC_LAT
    lats = 90.0 - (np.arange(height) + 0.5) * (180.0 / height)
    return lats, 90.0


def compute_grid(precision: int, metric: str, width: int, height: int,
                 projection: str = "mercator"):
    """Return (logA, lat_bound, vmin, vmax).

    ``logA`` is log10 of the per-cell area, shape (height, width), row 0 north.
    ``vmin``/``vmax`` are the true finite min/max (no percentile clipping, so the
    deep wells along the axes are shown honestly). Longitude is linear in mercator
    so its centres are the same either way.
    """
    dtype = _DTYPES[precision]
    lats, lat_bound = _lat_axis(height, projection)
    lons = -180.0 + (np.arange(width) + 0.5) * (360.0 / width)

    ulp_lat = ulp(lats, dtype)[:, None]          # (H, 1) degrees
    ulp_lon = ulp(lons, dtype)[None, :]          # (1, W) degrees

    if metric == "coord":
        area = np.broadcast_to(ulp_lat * ulp_lon, (height, width))   # deg^2
    else:  # ground
        d_lat_m = ulp_lat * _METRES_PER_DEG
        cos_lat = np.cos(np.radians(lats))[:, None]
        d_lon_m = ulp_lon * _METRES_PER_DEG * cos_lat
        area = np.broadcast_to(d_lat_m * d_lon_m, (height, width))   # m^2

    with np.errstate(divide="ignore"):
        logA = np.log10(area)

    finite = logA[np.isfinite(logA)]
    return logA, lat_bound, float(finite.min()), float(finite.max())


def render_png(logA: np.ndarray, vmin: float, vmax: float, path: str) -> None:
    """Colour-map ``logA`` with viridis and write an opaque RGBA PNG.

    Transparency for the overlay is controlled by Leaflet, not baked in here, so
    the user can tune it without re-rendering.
    """
    from matplotlib import cm, colors
    from PIL import Image

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cm.viridis(norm(logA))                          # (H, W, 4) float 0..1
    rgba[..., 3] = 1.0
    rgba = np.where(np.isfinite(logA)[..., None], rgba, 0.0)  # transparent gaps
    Image.fromarray((rgba * 255).astype(np.uint8)).save(path)


def _polar_cmap():
    """Viridis extended downward into red, for the North Pole zoom.

    The pole inset continues the *same* scale as the main map but reaches cells
    far smaller than anything on the world map, so its low end has to go below
    viridis. We graft a red -> dark-purple ramp beneath viridis: the values that
    are deepest-blue on the main map fade through purple into red here, reading as
    a natural extension "off the bottom" of the main colour bar.
    """
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap

    vir = cm.viridis(np.linspace(0.0, 1.0, 256))
    red = np.array([0.55, 0.0, 0.10, 1.0])               # dark crimson
    low = np.linspace(red, vir[0], 96)                   # red -> viridis purple
    return LinearSegmentedColormap.from_list("viridis_red", np.vstack([low, vir]))


def render_inset(precision: int, metric: str, subnormals: bool, path: str,
                 n: int = 420) -> tuple[float, float]:
    """Render a log-log zoom on Null Island and return its (vmin, vmax).

    A linear zoom on (0,0) shows nothing -- the banding is self-similar at every
    scale. So we plot |lat| and |lon| on log axes, from the spacing floor up to
    10 degrees. Cell area falls steadily toward the origin (a diagonal gradient
    with power-of-two steps) and then *flattens into a plateau* once a coordinate
    drops below the floor: that plateau is the genuinely-finest cell, and it sits
    7 orders of magnitude lower with subnormals than under flush-to-zero.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtype = _DTYPES[precision]
    floor = _floor_ulp(precision, subnormals)            # spacing at 0 (deg)

    # Log-spaced magnitudes from ~one decade below the floor up to 10 deg.
    lo = math.log10(floor) - 1.0
    mags = np.logspace(lo, 1.0, n)                       # |coord| in degrees

    u = np.nextafter(mags.astype(dtype),
                     np.array(np.inf, dtype=dtype)).astype(np.float64) - mags
    u = np.maximum(u, floor)                              # clamp to the floor

    # Work in log space: for float64 the area itself underflows to 0 (the floor
    # squared is far below the smallest double), so log10(side_lat)+log10(side_lon)
    # is the only way to keep it finite. cos(0)=1, so longitude needs no factor.
    log_u = np.log10(u)
    bias = 0.0 if metric == "coord" else 2.0 * math.log10(_METRES_PER_DEG)
    logA = log_u[:, None] + log_u[None, :] + bias        # log10(area)
    vmin, vmax = float(logA.min()), float(logA.max())

    unit = "m²" if metric == "ground" else "deg²"
    fig, ax = plt.subplots(figsize=(3.4, 3.1), dpi=100)
    ext = [lo, 1.0, lo, 1.0]
    # Grayscale -- deliberately *unlike* the main map (and the pole inset), since
    # Null Island is a different story (precision floor, not pole geometry). Dark
    # still reads as "fine", matching the main map's convention.
    im = ax.imshow(logA, origin="lower", extent=ext, cmap="gray",
                   vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")

    # Mark the smallest-normal threshold -- where flush-to-zero would clip.
    sn = math.log10(_SMALLEST_NORMAL[precision])
    if sn >= lo:
        for axis_line in ("h", "v"):
            (ax.axhline if axis_line == "h" else ax.axvline)(
                sn, color="#ff4d4d", lw=0.8, ls="--", alpha=0.8)
        ax.text(sn + 0.3, 0.6, "smallest\nnormal\n(FTZ clips)", color="#ff4d4d",
                fontsize=6, va="top", ha="left")

    ax.scatter([lo], [lo], marker="*", s=70, color="#ff4d4d", zorder=5,
               clip_on=False)
    ax.text(lo + 0.5, lo + 0.5, "(0,0)", color="#ff4d4d", fontsize=7)

    floor_m = floor * _METRES_PER_DEG
    mode = "subnormals on" if subnormals else "flush-to-zero"
    ax.set_title(f"Zoom on Null Island ({mode})\nfloor side ≈ {floor_m:.1e} m",
                 fontsize=7.5)
    ax.set_xlabel("|lon|  (log₁₀ degrees)", fontsize=7)
    ax.set_ylabel("|lat|  (log₁₀ degrees)", fontsize=7)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"log₁₀(cell area, {unit})", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=100, transparent=True)
    plt.close(fig)
    return vmin, vmax


def render_polar(precision: int, metric: str, path: str,
                 colat_min: float = 1e-5, colat_max: float = 5.0,
                 n_r: int = 700, n_theta: int = 720) -> tuple[float, float]:
    """Render a log-radius azimuthal zoom on the North Pole; return (vmin, vmax).

    Looking straight down on the pole: angle = longitude, radius = colatitude
    (90 - lat) on a LOG scale. The log radius is essential -- for the ground
    metric the cell area collapses as cos(lat) = sin(colat) ~ colat, so on a
    linear disc all the deep-blue "rings" are crushed into a couple of pixels at
    the centre. On a log radius the collapse becomes evenly spaced concentric
    rings (one viridis step per decade of colatitude), sweeping from ~5 deg
    (where Mercator clips) down to ~1e-5 deg, i.e. ~1 m from the pole. The
    float32 ULP of longitude additionally bands the disc into radial wedges
    (jumps at |lon| = 64, 128 deg). Net effect: a dartboard of rings (geometry)
    crossed by wedges (float structure).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtype = _DTYPES[precision]
    # Log-spaced colatitude edges so each ring spans equal log-radius.
    r_edges = np.logspace(math.log10(colat_min), math.log10(colat_max), n_r + 1)
    th_edges = np.radians(np.linspace(0.0, 360.0, n_theta + 1))
    r_c = np.sqrt(r_edges[:-1] * r_edges[1:])              # geometric centres
    th_c = 0.5 * (th_edges[:-1] + th_edges[1:])

    lat = 90.0 - r_c                                        # (n_r,)
    lon = np.degrees(th_c) - 180.0                          # (n_theta,)

    ulp_lat = ulp(lat, dtype)[:, None]                     # (n_r, 1) deg
    ulp_lon = ulp(lon, dtype)[None, :]                     # (1, n_theta) deg

    if metric == "coord":
        area = np.broadcast_to(ulp_lat * ulp_lon, (n_r, n_theta))      # deg^2
    else:
        d_lat_m = ulp_lat * _METRES_PER_DEG
        cos_lat = np.cos(np.radians(lat))[:, None]
        d_lon_m = ulp_lon * _METRES_PER_DEG * cos_lat
        area = np.broadcast_to(d_lat_m * d_lon_m, (n_r, n_theta))      # m^2
    logA = np.log10(area)
    vmin, vmax = float(logA.min()), float(logA.max())

    unit = "m²" if metric == "ground" else "deg²"
    fig, ax = plt.subplots(figsize=(3.4, 3.2), dpi=100,
                           subplot_kw={"projection": "polar"})
    TH, R = np.meshgrid(th_edges, r_edges)
    pcm = ax.pcolormesh(TH, R, logA, cmap=_polar_cmap(), vmin=vmin, vmax=vmax,
                        shading="flat")
    ax.set_rscale("log")
    ax.set_rlim(colat_min, colat_max)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    decades = [d for d in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
               if colat_min <= d <= colat_max]
    ax.set_rticks(decades)
    ax.set_yticklabels([f"{d:g}°" for d in decades], fontsize=5)
    ax.set_rlabel_position(135)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°E", "180°", "90°W"], fontsize=5.5)
    ax.tick_params(labelsize=5.5)
    ax.grid(True, lw=0.3, alpha=0.4, color="white")
    ax.set_title(f"Zoom on North Pole ({metric}), log radius\n"
                 f"radius = 90°−lat,  {colat_min:g}°→{colat_max:g}°  "
                 f"(centre ≈ {colat_min * _METRES_PER_DEG:.0f} m from pole)",
                 fontsize=6.8, pad=8)
    cb = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.10)
    cb.set_label(f"log₁₀(cell area, {unit})", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=100, transparent=True)
    plt.close(fig)
    return vmin, vmax


def _human_len(metres: float) -> str:
    a = abs(metres)
    for name, fac in [("km", 1e3), ("m", 1.0), ("mm", 1e-3), ("µm", 1e-6),
                      ("nm", 1e-9), ("pm", 1e-12), ("fm", 1e-15)]:
        if a >= fac:
            return f"{metres / fac:.2f} {name}"
    return f"{metres:.1e} m"


def _collapsible_box(title: str, img_path: str, box_id: str, corner_css: str,
                     start_open: bool = True) -> str:
    """A pinned, click-to-collapse panel wrapping a base64-embedded image.

    ``box_id`` must be unique per panel so multiple panels don't share element
    IDs. ``corner_css`` positions the panel (e.g. "bottom:24px; right:12px;").
    """
    with open(img_path, "rb") as f:
        uri = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    disp = "block" if start_open else "none"
    arrow = "▾" if start_open else "▸"
    return f"""
<div style="position:fixed; {corner_css} z-index:9999;
            background:rgba(255,255,255,.95); border-radius:6px;
            box-shadow:0 1px 4px rgba(0,0,0,.3); overflow:hidden;
            font:12px system-ui,sans-serif;">
  <div onclick="var b=document.getElementById('{box_id}-body');
                var t=document.getElementById('{box_id}-toggle');
                var c=b.style.display!=='none';
                b.style.display=c?'none':'block';
                t.textContent=c?'▸':'▾';"
       style="cursor:pointer; padding:4px 8px; background:#f4f4f4;
              border-bottom:1px solid #ddd; user-select:none; font-weight:600;
              display:flex; justify-content:space-between; align-items:center;">
    <span>{title}</span>
    <span id="{box_id}-toggle">{arrow}</span>
  </div>
  <div id="{box_id}-body" style="display:{disp}; padding:6px;">
    <img src="{uri}" style="display:block; width:300px; height:auto;" alt="{title}">
  </div>
</div>
"""


def build_map(png_path: str, lat_bound: float, vmin: float, vmax: float,
              precision: int, metric: str, opacity: float, html_path: str,
              inset_path: str | None = None,
              polar_path: str | None = None) -> None:
    """Wrap the Mercator overlay PNG in a folium/Leaflet page.

    The PNG is base64-embedded so the page is self-contained. A labels/boundaries
    tile layer is placed in a high-z pane so the real world map reads on top of
    the heatmap, and an optional log-log inset of Null Island is pinned in the
    corner.
    """
    import folium
    from branca.colormap import LinearColormap
    from folium.raster_layers import ImageOverlay
    from matplotlib import cm, colors as mcolors

    with open(png_path, "rb") as f:
        data_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

    unit = "m²" if metric == "ground" else "deg²"

    m = folium.Map(location=[20, 0], zoom_start=2, min_zoom=1, max_bounds=True,
                   tiles=None, world_copy_jump=True)

    # A LABEL-FREE light base (CARTO positron "no labels"). Every label then comes
    # from one English overlay on top (below), so there's a single, coherent label
    # layer instead of two clashing sets rendered at different depths.
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="light base", control=False,
    ).add_to(m)

    ImageOverlay(
        image=data_uri,
        bounds=[[-lat_bound, -180], [lat_bound, 180]],
        opacity=opacity,
        origin="upper",
        name=f"float{precision} granularity",
        control=False,
    ).add_to(m)

    # No label layer: the base is CARTO "no labels", so the map is just clean grey
    # land/sea (coastlines give orientation) with the heatmap on top -- the minimum
    # cartography, and nothing localized to clash.

    n = 16
    swatches = [mcolors.to_hex(cm.viridis(i / (n - 1))) for i in range(n)]
    legend = LinearColormap(
        swatches, vmin=vmin, vmax=vmax,
        caption=(f"float{precision} cell area  —  log₁₀({unit}), "
                 f"{metric} metric   (dark = fine, bright = coarse)"))
    m.add_child(legend)

    if inset_path is not None:
        m.get_root().html.add_child(folium.Element(_collapsible_box(
            "Zoom on Null Island", inset_path, "inset",
            "bottom:24px; right:12px;")))

    if polar_path is not None:
        m.get_root().html.add_child(folium.Element(_collapsible_box(
            "Zoom on North Pole", polar_path, "polar",
            "bottom:24px; left:12px;")))

    m.save(html_path)


def write_index(outdir: str, generated: list[tuple[int, str]]) -> None:
    """Tiny landing page linking the per-precision maps."""
    links = "\n".join(
        f'    <li><a href="{fname}">float{prec} granularity map</a></li>'
        for prec, fname in generated
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Floating-Point Spatial Granularity</title>
  <style>
    body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 42rem;
            margin: 4rem auto; padding: 0 1rem; }}
    code {{ background: #f0f0f0; padding: 0 .25rem; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Floating-Point Spatial Granularity</h1>
  <p>How finely can a latitude/longitude be resolved when stored as an IEEE-754
     float? Each cell is the patch of Earth between a coordinate and its nearest
     representable neighbours (<code>numpy.nextafter</code>).</p>
  <ul>
{links}
  </ul>
</body>
</html>
"""
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(html)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", choices=["32", "64", "both"], default="both")
    p.add_argument("--metric", choices=["ground", "coord"], default="ground")
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=1600)
    p.add_argument("--opacity", type=float, default=0.55,
                   help="overlay opacity 0..1 (default: 0.55)")
    p.add_argument("--no-subnormals", dest="subnormals", action="store_false",
                   help="model a flush-to-zero system (many GPUs, -ffast-math): "
                        "the floor at (0,0) becomes the smallest normal "
                        "(~1e-38 deg) instead of the smallest subnormal")
    p.add_argument("--no-inset", action="store_true",
                   help="skip the log-log Null Island zoom inset")
    p.add_argument("--outdir", default="docs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    precisions = [32, 64] if args.precision == "both" else [int(args.precision)]

    generated = []
    for prec in precisions:
        logA, lat_bound, vmin, vmax = compute_grid(
            prec, args.metric, args.width, args.height, projection="mercator")
        png_name = f"granularity_float{prec}_{args.metric}.png"
        html_name = f"float{prec}.html"
        png_path = os.path.join(args.outdir, png_name)
        render_png(logA, vmin, vmax, png_path)

        inset_path = None
        polar_path = None
        if not args.no_inset:
            inset_path = os.path.join(
                args.outdir, f"inset_float{prec}_{args.metric}.png")
            render_inset(prec, args.metric, args.subnormals, inset_path)
            polar_path = os.path.join(
                args.outdir, f"polar_float{prec}_{args.metric}.png")
            render_polar(prec, args.metric, polar_path)

        build_map(png_path, lat_bound, vmin, vmax, prec, args.metric,
                  args.opacity, os.path.join(args.outdir, html_name),
                  inset_path=inset_path, polar_path=polar_path)
        generated.append((prec, html_name))
        print(f"float{prec}: log10({args.metric}) range "
              f"[{vmin:.2f}, {vmax:.2f}] -> {args.outdir}/{html_name}")

    write_index(args.outdir, generated)
    print(f"index -> {args.outdir}/index.html")


if __name__ == "__main__":
    main()
