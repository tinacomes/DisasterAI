"""OSM facility completeness per country.

Benchmarks OSM counts of registry-comparable services (default: hospital,
pharmacy) against national registry counts, producing a completeness table

    country, service, osm_count, registry_count, completeness_ratio,
    facilities_per_100k (intrinsic density metric)

Registry counts are supplied as a CSV (config `quality.registry_counts_csv`,
columns: country, service, registry_count, source, year) compiled from
national sources (e.g. DE Krankenhausverzeichnis, national pharmacy
chambers) — each row's `source` is kept in the output so the table is
auditable. Where no registry row exists the ratio is NaN and only the
intrinsic density metric (facilities per 100k inhabitants vs the sample
median) is reported.

`filter_cities` restricts a Tier-1 city sample to countries at or above
`quality.completeness_threshold` (population-weighted mean ratio over the
benchmark services).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def completeness_table(
    osm_counts: pd.DataFrame,
    country_population: pd.Series,
    registry_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the per-country completeness table.

    ``osm_counts``: columns country, service, osm_count.
    ``country_population``: inhabitants per country code.
    ``registry_counts``: columns country, service, registry_count[, source, year].
    """
    table = osm_counts.copy()
    pop = country_population.reindex(table.country).to_numpy()
    table["facilities_per_100k"] = table.osm_count / pop * 1e5
    if registry_counts is not None and not registry_counts.empty:
        table = table.merge(
            registry_counts, on=["country", "service"], how="left", validate="1:1"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            table["completeness_ratio"] = table.osm_count / table.registry_count
    else:
        table["registry_count"] = np.nan
        table["completeness_ratio"] = np.nan
    med = table.groupby("service")["facilities_per_100k"].transform("median")
    table["density_vs_sample_median"] = table.facilities_per_100k / med
    return table


def country_completeness(table: pd.DataFrame) -> pd.Series:
    """Mean completeness ratio per country over the benchmark services;
    falls back to the intrinsic density metric where no registry exists."""
    ratio = table.groupby("country")["completeness_ratio"].mean()
    intrinsic = table.groupby("country")["density_vs_sample_median"].mean()
    return ratio.fillna(intrinsic)


def filter_cities(cities: pd.DataFrame, table: pd.DataFrame,
                  threshold: float | None) -> pd.DataFrame:
    """Restrict a city sample (needs a `country` column) to countries whose
    completeness score meets `threshold`. threshold=None disables filtering.
    Always reports what was dropped."""
    if threshold is None:
        return cities
    score = country_completeness(table)
    ok = score[score >= threshold].index
    dropped = cities[~cities.country.isin(ok)]
    if len(dropped):
        print(f"completeness filter (>= {threshold}): dropping "
              f"{len(dropped)} cities in {sorted(dropped.country.unique())}")
    return cities[cities.country.isin(ok)].copy()
