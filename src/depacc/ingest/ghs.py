"""GHS-POP 100 m population grid (JRC GHSL, R2023A, epoch 2020).

Downloads the tile-schema shapefile once, resolves which Mollweide tiles the
FUA touches (or takes `sources.ghs_pop.tiles` from config), downloads those
tiles, and extracts the populated cells inside the FUA as a table:

    cell_id, x, y (analysis CRS), lon, lat (WGS84 centroids), population

Only cells with population >= routing.min_cell_population are kept as
origins; the full within-FUA population is preserved for demand weighting.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from depacc.provenance import download

GHS_BASE = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0"
)
TILE_SCHEMA_URL = (
    "https://ghsl.jrc.ec.europa.eu/download/GHSL_data_54009_shapefile.zip"
)
GHS_LICENCE = "CC-BY 4.0 (JRC GHSL, GHS-POP R2023A)"
MOLLWEIDE = "ESRI:54009"


def _tile_url(tile: str) -> str:
    return f"{GHS_BASE}/tiles/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_{tile}.zip"


def resolve_tiles(fua, cfg: dict, root: Path) -> list[str]:
    """Tile ids (e.g. 'R3_C19') intersecting the FUA. Config override:
    sources.ghs_pop.tiles."""
    import geopandas as gpd

    override = (cfg.get("sources", {}).get("ghs_pop", {}) or {}).get("tiles")
    if override:
        return list(override)
    raw = root / cfg["output"]["raw_root"] / "ghs"
    schema_zip = download(TILE_SCHEMA_URL, raw / "GHSL_data_54009_shapefile.zip",
                          licence=GHS_LICENCE, note="tiling schema")
    schema = gpd.read_file(f"zip://{schema_zip}")
    tiles = schema.to_crs(MOLLWEIDE)
    bounds = fua.to_crs(MOLLWEIDE)
    hits = tiles[tiles.intersects(bounds.union_all())]
    id_col = next(c for c in ("tile_id", "TILE_ID", "id") if c in hits.columns)
    return sorted(hits[id_col].astype(str))


def fetch_population_cells(fua, cfg: dict, root: Path) -> pd.DataFrame:
    """Populated GHS-POP cells within the FUA, centroids in analysis CRS and
    WGS84."""
    import rasterio
    import rasterio.mask
    from pyproj import Transformer
    from shapely.geometry import mapping

    raw = root / cfg["output"]["raw_root"] / "ghs"
    frames = []
    fua_moll = fua.to_crs(MOLLWEIDE)
    geoms = [mapping(g) for g in fua_moll.geometry]
    for tile in resolve_tiles(fua, cfg, root):
        tile_zip = download(_tile_url(tile), raw / f"ghs_pop_{tile}.zip",
                            licence=GHS_LICENCE, note=f"GHS-POP 2020 tile {tile}")
        with zipfile.ZipFile(tile_zip) as zf:
            tif = next(n for n in zf.namelist() if n.endswith(".tif"))
        with rasterio.open(f"zip://{tile_zip}!{tif}") as src:
            try:
                data, transform = rasterio.mask.mask(src, geoms, crop=True, filled=True,
                                                     nodata=0.0)
            except ValueError:
                continue  # tile does not overlap after cropping
            band = data[0]
            rows, cols = np.nonzero(band > 0)
            if rows.size == 0:
                continue
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            frames.append(pd.DataFrame({
                "x_moll": np.asarray(xs, dtype=float),
                "y_moll": np.asarray(ys, dtype=float),
                "population": band[rows, cols].astype(float),
            }))
    if not frames:
        raise RuntimeError("No populated GHS-POP cells found inside the FUA")
    cells = pd.concat(frames, ignore_index=True)
    # Cells can straddle two tiles' overlap; keep the max (tiles are identical
    # where they overlap).
    cells = (cells.groupby(["x_moll", "y_moll"], as_index=False)
                  .agg(population=("population", "max")))

    to_analysis = Transformer.from_crs(MOLLWEIDE, cfg["crs"]["analysis"], always_xy=True)
    to_wgs = Transformer.from_crs(MOLLWEIDE, "EPSG:4326", always_xy=True)
    cells["x"], cells["y"] = to_analysis.transform(cells.x_moll, cells.y_moll)
    cells["lon"], cells["lat"] = to_wgs.transform(cells.x_moll, cells.y_moll)
    cells = cells.drop(columns=["x_moll", "y_moll"])
    cells.index.name = "cell_id"
    return cells.reset_index()
