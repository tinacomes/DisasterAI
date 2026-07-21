"""Cell-level bivariate co-location typology on PERCENTILE surfaces.

The everyday and emergency surfaces are population-weighted percentiles (the
weighted empirical CDF, see depacc.standardize) — never raw magnitudes, which
are incomparable across the two regimes. A cell is "high" on a regime when its
percentile is at or above the threshold (0.50 = pop-weighted median split;
0.75 = acute compounding). Four classes:

    LL  low both        HL  high everyday only
    LH  high emergency  HH  compounding (high both)

Population-weighted class shares are reported; cells NaN on either surface are
left unclassified and reported separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from depacc.standardize import RegimeSurface, require_percentile

CLASSES = ("LL", "LH", "HL", "HH")


def classify(everyday_pct: np.ndarray, emergency_pct: np.ndarray,
             threshold: float) -> np.ndarray:
    """LL/LH/HL/HH label per cell from two percentile arrays; None where either
    percentile is NaN."""
    ev = np.asarray(everyday_pct, float)
    em = np.asarray(emergency_pct, float)
    valid = np.isfinite(ev) & np.isfinite(em)
    ev_hi = ev >= threshold
    em_hi = em >= threshold
    labels = np.where(
        valid,
        np.where(ev_hi,
                 np.where(em_hi, "HH", "HL"),
                 np.where(em_hi, "LH", "LL")),
        None,
    )
    return labels


def class_shares(labels: np.ndarray, population: np.ndarray) -> pd.DataFrame:
    """Population-weighted share of each class (over classified cells; shares
    sum to 1). Unclassified population share stored in ``.attrs``."""
    pop = np.asarray(population, float)
    classified = np.array([lab is not None for lab in labels])
    total = float(np.nansum(np.where(classified, pop, 0.0)))
    rows = []
    for cls in CLASSES:
        in_cls = labels == cls
        cls_pop = float(np.nansum(np.where(in_cls, pop, 0.0)))
        rows.append({
            "typology": cls,
            "n_cells": int(in_cls.sum()),
            "population": cls_pop,
            "population_share": cls_pop / total if total > 0 else np.nan,
        })
    summary = pd.DataFrame(rows).set_index("typology")
    all_pop = float(np.nansum(pop))
    summary.attrs["unclassified_pop_share"] = (
        (all_pop - total) / all_pop if all_pop > 0 else np.nan)
    return summary


def bivariate_typology(everyday: RegimeSurface, emergency: RegimeSurface,
                       threshold: float = 0.5):
    """Classify from two PERCENTILE RegimeSurfaces (guard-enforced). Returns
    ``(labels, summary)``."""
    require_percentile(everyday, emergency)
    labels = classify(everyday.values, emergency.values, threshold)
    summary = class_shares(labels, everyday.population)
    return labels, summary
