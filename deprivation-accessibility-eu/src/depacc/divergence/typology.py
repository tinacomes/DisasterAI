"""Cell-level bivariate co-location typology (compounding deprivation).

Each populated cell is classified everyday-hi/lo x emergency-hi/lo against
population-weighted quantile thresholds (default: weighted median) of the two
deprivation surfaces within the city:

    HH  high everyday, high emergency  -> compounding deprivation
    HL  high everyday, low  emergency
    LH  low  everyday, high emergency
    LL  low  everyday, low  emergency

Outputs the per-cell class plus population-share summary. Cells flagged
unreachable on either surface are kept and classified (their capped/excluded
deprivation values speak for themselves under the configured policy), but the
summary also reports their population share separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("LL", "LH", "HL", "HH")


def weighted_quantile(values, q: float, weights) -> float:
    """Population-weighted quantile (inclusive cumulative-weight definition)."""
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    mask = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    if not mask.any():
        return float("nan")
    v, w = v[mask], w[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, q * cum[-1], side="left")])


def bivariate_typology(
    df: pd.DataFrame,
    everyday_col: str = "deprivation_everyday",
    emergency_col: str = "deprivation_emergency",
    pop_col: str = "population",
    quantile: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify cells into LL/LH/HL/HH and summarise population shares.

    Returns ``(cells, summary)``: ``cells`` is ``df`` with added columns
    ``typology`` (categorical, NaN where either surface is NaN) and the two
    booleans ``everyday_high`` / ``emergency_high``; ``summary`` has one row
    per class with population, population_share and cell counts. Population
    shares are taken over classified cells and sum to 1 (tested); the share of
    unclassifiable (NaN-surface) population is reported separately in
    ``summary.attrs['unclassified_pop_share']``.
    """
    ev = df[everyday_col].to_numpy(dtype=float)
    em = df[emergency_col].to_numpy(dtype=float)
    pop = df[pop_col].to_numpy(dtype=float)

    thr_ev = weighted_quantile(ev, quantile, pop)
    thr_em = weighted_quantile(em, quantile, pop)

    valid = ~np.isnan(ev) & ~np.isnan(em)
    ev_high = ev > thr_ev
    em_high = em > thr_em
    labels = np.where(
        valid,
        np.where(
            ev_high,
            np.where(em_high, "HH", "HL"),
            np.where(em_high, "LH", "LL"),
        ),
        None,
    )

    cells = df.copy()
    cells["everyday_high"] = np.where(valid, ev_high, np.nan)
    cells["emergency_high"] = np.where(valid, em_high, np.nan)
    cells["typology"] = pd.Categorical(labels, categories=list(CLASSES))

    total_pop = float(np.nansum(np.where(valid, pop, 0.0)))
    rows = []
    for cls in CLASSES:
        in_cls = (labels == cls)
        cls_pop = float(np.nansum(np.where(in_cls, pop, 0.0)))
        rows.append(
            {
                "typology": cls,
                "n_cells": int(in_cls.sum()),
                "population": cls_pop,
                "population_share": cls_pop / total_pop if total_pop > 0 else np.nan,
            }
        )
    summary = pd.DataFrame(rows).set_index("typology")
    all_pop = float(np.nansum(pop))
    summary.attrs["unclassified_pop_share"] = (
        (all_pop - total_pop) / all_pop if all_pop > 0 else np.nan
    )
    summary.attrs["threshold_everyday"] = thr_ev
    summary.attrs["threshold_emergency"] = thr_em
    return cells, summary
