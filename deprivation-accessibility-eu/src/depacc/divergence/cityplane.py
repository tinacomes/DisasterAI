"""City-level divergence: each city as a point in the everyday-vs-emergency
plane, plus co-location statistics.

The plane's default axes are the population-weighted Ginis of the two
deprivation surfaces; population-weighted mean levels, the HH (compounding)
population share, and a weighted rank correlation between the surfaces are
carried along. Rows accumulate in data/derived/cityplane.csv across cities —
the input to cityvector/ clustering and the size-gradient (space-for-time)
read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from depacc.equity.indices import weighted_gini, weighted_mean


def weighted_rank_corr(a, b, w) -> float:
    """Population-weighted Spearman-type correlation between two surfaces."""
    a = np.asarray(a, float); b = np.asarray(b, float); w = np.asarray(w, float)
    mask = ~np.isnan(a) & ~np.isnan(b) & (w > 0)
    a, b, w = a[mask], b[mask], w[mask]
    if a.size < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    wm = w / w.sum()
    ma, mb = np.sum(wm * ra), np.sum(wm * rb)
    cov = np.sum(wm * (ra - ma) * (rb - mb))
    sa = np.sqrt(np.sum(wm * (ra - ma) ** 2))
    sb = np.sqrt(np.sum(wm * (rb - mb) ** 2))
    return float(cov / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def city_row(surfaces: pd.DataFrame, typology_summary: pd.DataFrame,
             cfg: dict, city: str) -> dict:
    pop = surfaces["population"]
    ev = surfaces["deprivation_everyday"]
    em = surfaces["deprivation_emergency"]
    return {
        "city": city,
        "name": cfg["city"].get("name", city),
        "country": cfg["city"].get("country", ""),
        "tier": cfg["city"].get("tier", 1),
        "synthetic": bool(cfg["city"].get("synthetic", False)),
        "population": float(pop.sum()),
        "mean_everyday": weighted_mean(ev, pop),
        "mean_emergency": weighted_mean(em, pop),
        "gini_everyday": weighted_gini(ev, pop),
        "gini_emergency": weighted_gini(em, pop),
        "gini_divergence": weighted_gini(em, pop) - weighted_gini(ev, pop),
        "rank_corr": weighted_rank_corr(ev, em, pop),
        "hh_pop_share": float(typology_summary.loc["HH", "population_share"]),
        "unreachable_pop_share_everyday": float(
            pop[surfaces["unreachable_everyday"]].sum() / pop.sum()),
        "unreachable_pop_share_emergency": float(
            pop[surfaces["unreachable_emergency"]].sum() / pop.sum()),
    }


def upsert_cityplane(row: dict, table_path: Path) -> pd.DataFrame:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.exists():
        table = pd.read_csv(table_path)
        table = table[table.city != row["city"]]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table = table.sort_values("population", ascending=False)
    table.to_csv(table_path, index=False)
    return table
