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
    name_col = next(
        (c for c in ("URAU_NAME", "urau_name", "FUA_NAME") if c in fuas.columns), None
    )
    sel = fuas[fuas[code_col] == code]
    if sel.empty:
        same_country = fuas[fuas[code_col].astype(str).str[:2] == str(code)[:2]]
        near = sorted(
            f"{r[code_col]} ({r[name_col]})" if name_col else str(r[code_col])
            for _, r in same_country.iterrows()
        )[:30]
        raise RuntimeError(
            f"FUA code {code!r} not found in URAU 2021 layer. "
            f"Same-country codes: {near}. "
            f"Full list: `depacc list-fuas` or the 'depacc — list FUAs' workflow."
        )
    if name_col:
        fua_name = str(sel.iloc[0][name_col])
        expected = str(cfg["city"].get("name", ""))
        if expected and expected.lower()[:4] not in fua_name.lower():
            print(f"WARNING: config city.name '{expected}' does not match the "
                  f"selected FUA's name '{fua_name}' ({code}) — check the "
                  f"fua_code against `depacc list-fuas`.")
        else:
            print(f"FUA {code}: {fua_name}")
    return sel.to_crs(cfg["crs"]["analysis"]).reset_index(drop=True)
