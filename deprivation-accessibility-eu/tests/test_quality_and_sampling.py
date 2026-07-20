"""OSM completeness table, threshold filter, and Tier-1 stratified sampling."""

import numpy as np
import pandas as pd
import pytest

from depacc.ingest.fua_sample import sample_cities
from depacc.quality.completeness import (
    completeness_table,
    country_completeness,
    filter_cities,
)


def _osm_counts():
    return pd.DataFrame({
        "country": ["DE", "DE", "NL", "NL", "RO", "RO"],
        "service": ["hospital", "pharmacy"] * 3,
        "osm_count": [1800, 17000, 350, 1900, 200, 1500],
    })


def _registry():
    return pd.DataFrame({
        "country": ["DE", "DE", "NL", "NL"],   # no registry rows for RO
        "service": ["hospital", "pharmacy"] * 2,
        "registry_count": [1900, 18000, 370, 2000],
        "source": ["Krankenhausverzeichnis", "ABDA", "CBS", "SFK"],
    })


POP = pd.Series({"DE": 84e6, "NL": 18e6, "RO": 19e6})


def test_completeness_ratio_and_intrinsic_fallback():
    table = completeness_table(_osm_counts(), POP, _registry())
    de_hosp = table[(table.country == "DE") & (table.service == "hospital")].iloc[0]
    assert de_hosp.completeness_ratio == pytest.approx(1800 / 1900)
    ro = table[table.country == "RO"]
    assert ro.completeness_ratio.isna().all()
    assert ro.facilities_per_100k.notna().all()
    score = country_completeness(table)
    assert score["DE"] > 0.9
    # RO falls back to the intrinsic density metric (well below sample median).
    assert score["RO"] < score["DE"]


def test_filter_cities_threshold():
    table = completeness_table(_osm_counts(), POP, _registry())
    cities = pd.DataFrame({
        "city": ["hamburg", "amsterdam", "bucuresti"],
        "country": ["DE", "NL", "RO"],
    })
    kept = filter_cities(cities, table, threshold=0.9)
    assert set(kept.city) == {"hamburg", "amsterdam"}
    # Disabled filter keeps everything.
    assert len(filter_cities(cities, table, threshold=None)) == 3


def _fuas():
    rng = np.random.default_rng(0)
    rows = []
    for country, n in (("DE", 40), ("NL", 15), ("FR", 30), ("ES", 25)):
        pops = rng.uniform(4.5, 6.9, n)  # log10 population
        rows += [{"fua_code": f"{country}{i:03d}L1", "name": f"{country}{i}",
                  "country": country, "population": 10 ** p}
                 for i, p in enumerate(pops)]
    return pd.DataFrame(rows)


def test_stratified_sampling(monkeypatch):
    cfg = {"city_definition": {
        "fua_size_threshold": 100000,
        "city_sample_mode": "stratified",
        "stratified_countries": ["DE", "NL", "FR"],
        "strata_bounds": [100000, 250000, 500000, 1000000, 5000000],
    }}
    sample = sample_cities(_fuas(), cfg, per_stratum=2)
    assert set(sample.country) <= {"DE", "NL", "FR"}   # ES excluded
    assert (sample.population >= 100000).all()
    # At most per_stratum cities per (country, stratum).
    bounds = [100000, 250000, 500000, 1000000, 5000000, float("inf")]
    for _, g in sample.groupby("country"):
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            assert ((g.population >= lo) & (g.population < hi)).sum() <= 2


def test_all_eu_mode_and_threshold():
    cfg = {"city_definition": {
        "fua_size_threshold": 500000,
        "city_sample_mode": "all_eu_fua",
        "stratified_countries": [],
        "strata_bounds": [],
    }}
    sample = sample_cities(_fuas(), cfg)
    assert (sample.population >= 500000).all()
    assert set(sample.country) == {"DE", "NL", "FR", "ES"}
    assert sample.population.is_monotonic_decreasing
