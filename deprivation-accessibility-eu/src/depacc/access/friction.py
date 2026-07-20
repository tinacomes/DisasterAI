"""Tier-1 fast travel times from the Weiss et al. (2020) friction surfaces.

The many-city bypass: instead of country-scale OSM extracts + R5 routing
(gigabytes + Java per city), travel time is computed by least-cost paths
over the Malaria Atlas Project 30-arc-second friction rasters (minutes per
metre): the motorised surface for `car`, the walking-only surface for
`walk`. Only the city's raster window is fetched, via a WCS GetCoverage
subset request — typically a few MB per city — and cached with provenance.

Cost distance runs a Dijkstra (scipy.sparse.csgraph) on the 8-connected
pixel graph, per facility, bounded by the routing time cutoff, and emits
the same long-format OD table as the R5 engine, so everything downstream
(2SFCA, soft-min, DLF/DCF, typology, equity) is unchanged. Resolution is
~1 km — coarse but harmonised across all of Europe; r5py remains the
reference engine and Tier-2 cross-checks the rankings (methods.md §5, §7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARTH_M_PER_DEG = 111_320.0  # metres per degree at the equator

# Dataset ids on the Malaria Atlas Project geoserver (verified against
# malariaatlas.org on first use; override via config `friction.wcs`).
DEFAULT_WCS = {
    "base": "https://data.malariaatlas.org/geoserver/Accessibility/ows",
    "coverages": {
        "car": "Accessibility__202001_Global_Motorized_Friction_Surface",
        "walk": "Accessibility__202001_Global_Walking_Only_Friction_Surface",
    },
}
FRICTION_LICENCE = "CC-BY 4.0 (Weiss et al. 2020, Malaria Atlas Project)"


# --------------------------------------------------------------------------
# pure math (unit-tested without rasterio/network)
# --------------------------------------------------------------------------

def _edge_lists(shape: tuple[int, int], friction: np.ndarray,
                dx_m: np.ndarray, dy_m: float):
    """Sparse 8-connected graph; edge weight = mean friction of the two
    pixels x metric distance (dx varies with latitude/row)."""
    n_rows, n_cols = shape
    idx = np.arange(n_rows * n_cols).reshape(shape)
    rows_i, cols_i, weights = [], [], []

    def add(a, b, dist):
        w = 0.5 * (friction.ravel()[a] + friction.ravel()[b]) * dist
        ok = np.isfinite(w)
        rows_i.append(a[ok]); cols_i.append(b[ok]); weights.append(w[ok])

    # horizontal
    a = idx[:, :-1].ravel(); b = idx[:, 1:].ravel()
    add(a, b, np.repeat(dx_m, n_cols - 1))
    # vertical
    a = idx[:-1, :].ravel(); b = idx[1:, :].ravel()
    add(a, b, np.full(a.size, dy_m))
    # diagonals
    diag = np.sqrt(np.repeat(dx_m[:-1], n_cols - 1) ** 2 + dy_m**2)
    a = idx[:-1, :-1].ravel(); b = idx[1:, 1:].ravel()
    add(a, b, diag)
    a = idx[:-1, 1:].ravel(); b = idx[1:, :-1].ravel()
    add(a, b, diag)
    return (np.concatenate(rows_i), np.concatenate(cols_i),
            np.concatenate(weights))


def cost_distance_times(friction: np.ndarray, sources_rc: list[tuple[int, int]],
                        dx_m: np.ndarray, dy_m: float,
                        max_time_min: float) -> np.ndarray:
    """Travel time (minutes) from each source pixel to every pixel.

    ``friction``: minutes-per-metre raster window (NaN = impassable).
    ``dx_m``: per-row pixel width in metres; ``dy_m``: pixel height.
    Returns array (n_sources, n_rows, n_cols) with NaN beyond the cutoff.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    shape = friction.shape
    n = friction.size
    r, c, w = _edge_lists(shape, friction, dx_m, dy_m)
    graph = coo_matrix((w, (r, c)), shape=(n, n))
    src_idx = [rr * shape[1] + cc for rr, cc in sources_rc]
    dist = dijkstra(graph, directed=False, indices=src_idx, limit=max_time_min)
    dist[~np.isfinite(dist)] = np.nan
    return dist.reshape(len(src_idx), *shape)


# --------------------------------------------------------------------------
# raster + pipeline wiring
# --------------------------------------------------------------------------

def _wcs_url(cfg: dict, mode: str, bounds_wgs: tuple[float, float, float, float]) -> str:
    wcs = {**DEFAULT_WCS, **(cfg.get("friction", {}).get("wcs") or {})}
    coverages = {**DEFAULT_WCS["coverages"], **(wcs.get("coverages") or {})}
    minx, miny, maxx, maxy = bounds_wgs
    return (f"{wcs['base']}?service=WCS&version=2.0.1&request=GetCoverage"
            f"&coverageId={coverages[mode]}&format=image/geotiff"
            f"&subset=Long({minx},{maxx})&subset=Lat({miny},{maxy})")


def fetch_friction_window(cfg: dict, mode: str, fua, root: Path, city: str) -> Path:
    """Download (once) the city's friction-raster window for a mode."""
    from depacc.provenance import download

    pad = float(cfg.get("friction", {}).get("pad_deg", 1.0))
    minx, miny, maxx, maxy = fua.to_crs("EPSG:4326").total_bounds
    url = _wcs_url(cfg, mode, (minx - pad, miny - pad, maxx + pad, maxy + pad))
    raw = root / cfg["output"]["raw_root"] / "friction"
    return download(url, raw / f"{city}_{mode}.tif", licence=FRICTION_LICENCE,
                    note=f"Weiss et al. friction window, mode={mode}")


def friction_matrix(cfg: dict, cells: pd.DataFrame, facilities: pd.DataFrame,
                    mode: str, fua, root: Path, city: str) -> pd.DataFrame:
    """Long-format OD table (origin cell, facility, minutes) via cost distance."""
    import rasterio

    if mode not in ("walk", "car"):
        raise ValueError(f"friction engine supports walk/car, not '{mode}' "
                         "(transit needs the r5 engine / Tier 2)")
    by_mode = cfg["routing"].get("max_time_min_by_mode") or {}
    max_time = float(by_mode.get(mode) or cfg["routing"]["max_time_min"])
    tif = fetch_friction_window(cfg, mode, fua, root, city)
    with rasterio.open(tif) as src:
        friction = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            friction[friction == nodata] = np.nan
        transform = src.transform
        # per-row pixel width in metres (geographic raster)
        lat_rows = np.array([rasterio.transform.xy(transform, r_, 0)[1]
                             for r_ in range(src.height)])
        dx_m = np.abs(transform.a) * EARTH_M_PER_DEG * np.cos(np.radians(lat_rows))
        dy_m = abs(transform.e) * EARTH_M_PER_DEG

        def to_rc(lon, lat):
            r_, c_ = src.index(lon, lat)
            return int(np.clip(r_, 0, src.height - 1)), int(np.clip(c_, 0, src.width - 1))

    fac_rc = [to_rc(lo, la) for lo, la in zip(facilities.lon, facilities.lat)]
    times = cost_distance_times(friction, fac_rc, dx_m, dy_m, max_time)

    cell_rc = np.array([to_rc(lo, la) for lo, la in zip(cells.lon, cells.lat)])
    frames = []
    for k, dest in enumerate(facilities.dest_id):
        t = times[k, cell_rc[:, 0], cell_rc[:, 1]]
        ok = ~np.isnan(t)
        frames.append(pd.DataFrame({
            "origin": cells.cell_id.to_numpy()[ok],
            "dest": dest,
            "time": t[ok],
        }))
    od = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["origin", "dest", "time"])
    return od
