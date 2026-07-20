"""Population-weighted inequality indices on deprivation surfaces.

* weighted_mean       — population-weighted mean deprivation.
* weighted_gini       — population-weighted Gini of the deprivation surface
                        (0 = deprivation evenly spread over people,
                        -> 1 = concentrated on few people).
* concentration_index — deprivation concentration against an SES rank
                        (negative = deprivation concentrated among the
                        low-SES-ranked population), standard health-economics
                        definition: twice the covariance between the outcome
                        and the fractional SES rank, divided by the mean.

NaN outcomes (excluded-unreachable cells) are dropped pairwise with their
weights; all functions require at least one positive weight.
"""

from __future__ import annotations

import numpy as np


def _clean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    mask = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    return v[mask], w[mask]


def weighted_mean(values, weights) -> float:
    v, w = _clean(values, weights)
    if v.size == 0:
        return float("nan")
    return float(np.average(v, weights=w))


def weighted_gini(values, weights) -> float:
    """Population-weighted Gini coefficient (values must be >= 0)."""
    v, w = _clean(values, weights)
    if v.size == 0:
        return float("nan")
    if np.any(v < 0):
        raise ValueError("Gini requires non-negative values")
    mu = np.average(v, weights=w)
    if mu == 0:
        return 0.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    p = w / w.sum()
    cum_p = np.cumsum(p)
    # Weighted Gini via the covariance form: G = 2 cov(v, F) / mu with F the
    # mid-interval cumulative population rank.
    rank = cum_p - 0.5 * p
    cov = np.sum(p * (v - mu) * (rank - np.sum(p * rank)))
    return float(2.0 * cov / mu)


def concentration_index(values, ses, weights) -> float:
    """Concentration index of `values` against the SES ranking variable `ses`
    (e.g. income; ranked ascending, so CI < 0 means the burden concentrates
    among the poorest)."""
    v = np.asarray(values, dtype=float)
    s = np.asarray(ses, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (v.shape == s.shape == w.shape):
        raise ValueError("values, ses and weights must share a shape")
    mask = ~np.isnan(v) & ~np.isnan(s) & ~np.isnan(w) & (w > 0)
    v, s, w = v[mask], s[mask], w[mask]
    if v.size == 0:
        return float("nan")
    mu = np.average(v, weights=w)
    if mu == 0:
        return 0.0
    order = np.argsort(s, kind="stable")
    v, w = v[order], w[order]
    p = w / w.sum()
    rank = np.cumsum(p) - 0.5 * p
    cov = np.sum(p * (v - mu) * (rank - np.sum(p * rank)))
    return float(2.0 * cov / mu)
