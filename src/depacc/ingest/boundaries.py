"""Functional Urban Area boundaries (Eurostat-OECD FUA via GISCO URAU).

One harmonised city definition for every city: the Eurostat/GISCO Urban
Audit (URAU) 2021 FUA polygons, selected by the `city.fua_code` in the city
config. Cross-checks against GHS-UCDB urban centres happen in quality/.
"""

from __future__ import annotations

from pathlib import Path

from depacc.provenance import download

# GISCO distribution API; 1:100k generalisation is plenty for clipping grids.
URAU_FUA_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/"
    "URAU_RG_100K_2021_3035_FUA.geojson"
)
URAU_LICENCE = "Eurostat/GISCO standard reuse policy (CC-BY 4.0); © EuroGeographics"


def fetch_fua_boundary(cfg: dict, root: Path):
    """Return the city's FUA polygon as a single-row GeoDataFrame in the
    analysis CRS, downloading + caching the continental URAU FUA layer."""
    import geopandas as gpd

    raw = root / cfg["output"]["raw_root"] / "boundaries"
    path = download(URAU_FUA_URL, raw / "URAU_RG_100K_2021_3035_FUA.geojson",
                    licence=URAU_LICENCE)
    fuas = gpd.read_file(path)
    code = cfg["city"]["fua_code"]
    code_col = next(
        (c for c in ("URAU_CODE", "urau_code", "FUA_CODE") if c in fuas.columns), None
    )
    if code_col is None:
        raise RuntimeError(
            f"URAU layer has no recognised code column; columns: {list(fuas.columns)}"
        )
    sel = fuas[fuas[code_col] == code]
    if sel.empty:
        near = sorted(
            c for c in fuas[code_col].astype(str) if c[:4] == str(code)[:4]
        )[:10]
        raise RuntimeError(
            f"FUA code {code!r} not found in URAU 2021 layer; "
            f"codes with the same country/city prefix: {near}"
        )
    return sel.to_crs(cfg["crs"]["analysis"]).reset_index(drop=True)
