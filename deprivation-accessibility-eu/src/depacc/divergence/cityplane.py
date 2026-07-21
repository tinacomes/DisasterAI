"""City-level divergence row: the everyday-vs-emergency plane plus scale-free
co-location scalars.

Ginis are computed on the RAW within-regime values (Gini is scale-invariant,
and this is a within-regime statistic, never a cross-regime comparison). The
plane axes are (gini_everyday, gini_emergency); off-diagonal spread =
divergence. Everything else (Spearman, Jaccard, compounding share) comes from
the percentile surfaces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from depacc.divergence.colocation import (
    compounding_pop_share,
    jaccard_high,
    weighted_spearman,
)
from depacc.equity.indices import weighted_gini, weighted_mean
from depacc.standardize import RegimeSurface, to_percentile


def city_row(everyday_raw: RegimeSurface, emergency_raw: RegimeSurface,
             cfg: dict, city: str, name: str, country: str, tier: int,
             synthetic: bool, population_total: float) -> dict:
    """Assemble the per-city summary from RAW regime surfaces (standardised
    internally where cross-regime comparison is required)."""
    thresholds = cfg.get("typology", {}).get("thresholds", [0.5, 0.75])
    e_p = to_percentile(everyday_raw)
    m_p = to_percentile(emergency_raw)

    pop = everyday_raw.population
    row = {
        "city": city, "name": name, "country": country, "tier": tier,
        "synthetic": bool(synthetic), "population": float(population_total),
        # within-regime means (RAW; regime-internal, not compared across).
        "mean_everyday": weighted_mean(everyday_raw.values, pop),
        "mean_emergency": weighted_mean(emergency_raw.values, pop),
        # the plane: within-regime Ginis (scale-invariant) + off-diagonal gap.
        "gini_everyday": weighted_gini(everyday_raw.values, pop),
        "gini_emergency": weighted_gini(emergency_raw.values, pop),
        # tail-robust inequality for the unbounded emergency surface.
        "p90_p50_ratio_emergency": _p90_p50(emergency_raw.values, pop),
        # coupling (scale-free, from percentiles).
        "spearman_rho": weighted_spearman(e_p, m_p),
    }
    row["divergence_gap"] = row["gini_emergency"] - row["gini_everyday"]
    for thr in thresholds:
        key = f"{int(round(thr * 100)):02d}"
        row[f"compounding_pop_share_{key}"] = compounding_pop_share(e_p, m_p, thr)
        row[f"jaccard_high_{key}"] = jaccard_high(e_p, m_p, thr)
    return row


def _p90_p50(values, weights) -> float:
    from depacc.divergence.typology import CLASSES  # noqa: F401  (keep import light)

    v = np.asarray(values, float); w = np.asarray(weights, float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return float("nan")
    v, w = v[mask], w[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    p50 = v[np.searchsorted(cw, 0.50)]
    p90 = v[np.searchsorted(cw, 0.90)]
    return float(p90 / p50) if p50 > 0 else float("nan")


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
