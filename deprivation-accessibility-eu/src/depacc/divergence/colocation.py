"""Scale-free co-location statistics between the two regimes.

All operate on PERCENTILE surfaces (guard-enforced) so they are invariant to
the deprivation-function scale; all are population-weighted.
"""

from __future__ import annotations

import numpy as np

from depacc.standardize import RegimeSurface, require_same_standardised


def _weighted_pearson(a, b, w) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float); w = np.asarray(w, float)
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0)
    if mask.sum() < 3:
        return float("nan")
    a, b, w = a[mask], b[mask], w[mask]
    wn = w / w.sum()
    ma, mb = np.sum(wn * a), np.sum(wn * b)
    cov = np.sum(wn * (a - ma) * (b - mb))
    sa = np.sqrt(np.sum(wn * (a - ma) ** 2))
    sb = np.sqrt(np.sum(wn * (b - mb) ** 2))
    return float(cov / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def weighted_spearman(everyday: RegimeSurface, emergency: RegimeSurface) -> float:
    """Population-weighted Spearman correlation of the two surfaces. Percentiles
    are the weighted ranks, so this is the weighted Pearson of the percentiles."""
    require_same_standardised(everyday, emergency)
    return _weighted_pearson(everyday.values, emergency.values, everyday.population)


def jaccard_high(everyday: RegimeSurface, emergency: RegimeSurface,
                 threshold: float = 0.5) -> float:
    """Population-weighted Jaccard of the two 'high' cell sets:
    pop(everyday-high AND emergency-high) / pop(everyday-high OR emergency-high)."""
    require_same_standardised(everyday, emergency)
    ev, em, w = everyday.values, emergency.values, everyday.population
    mask = np.isfinite(ev) & np.isfinite(em) & (w > 0)
    ev_hi = mask & (ev >= threshold)
    em_hi = mask & (em >= threshold)
    inter = float(w[ev_hi & em_hi].sum())
    union = float(w[ev_hi | em_hi].sum())
    return inter / union if union > 0 else float("nan")


def compounding_pop_share(everyday: RegimeSurface, emergency: RegimeSurface,
                          threshold: float = 0.5) -> float:
    """Population share in the HH (high-both) class."""
    require_same_standardised(everyday, emergency)
    ev, em, w = everyday.values, emergency.values, everyday.population
    mask = np.isfinite(ev) & np.isfinite(em) & (w > 0)
    hh = mask & (ev >= threshold) & (em >= threshold)
    total = float(w[mask].sum())
    return float(w[hh].sum()) / total if total > 0 else float("nan")
