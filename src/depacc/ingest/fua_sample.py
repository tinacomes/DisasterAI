"""Tier-1 city sampling from the harmonised FUA universe.

`list_fuas` loads the GISCO URAU FUA layer and joins FUA populations from a
CSV (config `city_definition.fua_population_csv`; columns fua_code,
population — compile from Eurostat Urban Audit population tables
(urb_lpop1) or by summing GHS-POP over each FUA polygon with
depacc.ingest.ghs). `sample_cities` applies the config sampling mode:

  - "all_eu_fua":  every FUA above `fua_size_threshold`;
  - "stratified":  within `stratified_countries`, up to `per_stratum` cities
                   per population stratum (`strata_bounds`), largest first —
                   the fast route to a cross-city figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_fuas(cfg: dict, root: Path) -> pd.DataFrame:
    import geopandas as gpd

    from depacc.ingest.boundaries import URAU_FUA_URL, URAU_LICENCE
    from depacc.provenance import download

    raw = root / cfg["output"]["raw_root"] / "boundaries"
    path = download(URAU_FUA_URL, raw / "URAU_RG_100K_2021_3035_FUA.geojson",
                    licence=URAU_LICENCE)
    fuas = gpd.read_file(path)
    code_col = next(c for c in ("URAU_CODE", "urau_code", "FUA_CODE") if c in fuas.columns)
    name_col = next((c for c in ("URAU_NAME", "urau_name", "FUA_NAME") if c in fuas.columns),
                    code_col)
    out = pd.DataFrame({
        "fua_code": fuas[code_col].astype(str),
        "name": fuas[name_col].astype(str),
        "country": fuas[code_col].astype(str).str[:2],
    })
    pop_csv = cfg["city_definition"].get("fua_population_csv")
    if pop_csv:
        pops = pd.read_csv(root / pop_csv)
        out = out.merge(pops[["fua_code", "population"]], on="fua_code", how="left")
    else:
        out["population"] = pd.NA
        print("NOTE: no fua_population_csv configured; populations missing — "
              "compile from Eurostat urb_lpop1 or GHS-POP sums before sampling.")
    return out


def sample_cities(fuas: pd.DataFrame, cfg: dict, per_stratum: int = 2) -> pd.DataFrame:
    cd = cfg["city_definition"]
    threshold = float(cd["fua_size_threshold"])
    fuas = fuas.dropna(subset=["population"])
    fuas = fuas[fuas.population >= threshold]
    mode = cd["city_sample_mode"]
    if mode == "all_eu_fua":
        return fuas.sort_values("population", ascending=False).reset_index(drop=True)
    if mode != "stratified":
        raise ValueError(f"Unknown city_sample_mode '{mode}'")
    fuas = fuas[fuas.country.isin(cd["stratified_countries"])]
    bounds = list(cd["strata_bounds"]) + [float("inf")]
    picks = []
    for country, group in fuas.groupby("country"):
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            stratum = group[(group.population >= lo) & (group.population < hi)]
            picks.append(
                stratum.sort_values("population", ascending=False).head(per_stratum)
            )
    return (pd.concat(picks, ignore_index=True)
            .sort_values("population", ascending=False)
            .reset_index(drop=True))
