"""RegimeSurface and the standardisation transforms + guards.

All aggregation is POPULATION-WEIGHTED (deprivation is about people). Empty /
NaN cells are excluded from the weighting throughout (their standardised
value is NaN and never contributes to weighted counts, means or SDs).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

SCALE_STATES = ("raw", "percentile", "zscore")


@dataclass(frozen=True)
class RegimeSurface:
    """A per-cell deprivation surface for one regime of one city, tagged with
    its current scale state so cross-regime consumers can enforce that raw
    magnitudes are never combined.

    ``values`` and ``population`` are 1-D arrays aligned per populated cell.
    """

    values: np.ndarray
    population: np.ndarray
    regime: str            # "everyday" | "emergency"
    city_id: str
    scale_state: str = "raw"

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        p = np.asarray(self.population, dtype=float)
        if v.shape != p.shape or v.ndim != 1:
            raise ValueError("values and population must be 1-D arrays of equal length")
        if self.regime not in ("everyday", "emergency"):
            raise ValueError(f"unknown regime {self.regime!r}")
        if self.scale_state not in SCALE_STATES:
            raise ValueError(f"unknown scale_state {self.scale_state!r}")
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "population", p)

    # -- validity mask (finite value AND positive weight) -------------------
    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.values) & np.isfinite(self.population) & (self.population > 0)


def _weighted_mean_sd(v: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mu = np.average(v, weights=w)
    var = np.average((v - mu) ** 2, weights=w)
    return float(mu), float(np.sqrt(var))


def to_percentile(surface: RegimeSurface) -> RegimeSurface:
    """Population-weighted empirical CDF: each cell -> (weighted pop with
    deprivation <= this cell) / total valid pop. Result in [0, 1], monotone in
    the input and invariant to any strictly increasing rescaling — this is
    what tames the unbounded emergency tail. NaN cells stay NaN."""
    v, w = surface.values, surface.population
    mask = surface.valid
    out = np.full(v.shape, np.nan)
    if not mask.any():
        return replace(surface, values=out, scale_state="percentile")
    vv, ww = v[mask], w[mask]
    order = np.argsort(vv, kind="mergesort")
    vs, wsorted = vv[order], ww[order]
    # Inclusive weighted rank: cumulative weight of all cells with value <=
    # the current value (ties share the group's top cumulative weight).
    cum = np.cumsum(wsorted)
    total = cum[-1]
    # Map each sorted position to the cumulative weight at the end of its tie
    # group, so equal values receive an equal percentile.
    pct_sorted = np.empty_like(vs)
    i = 0
    n = len(vs)
    while i < n:
        j = i
        while j + 1 < n and vs[j + 1] == vs[i]:
            j += 1
        pct_sorted[i:j + 1] = cum[j] / total
        i = j + 1
    pct = np.empty_like(vv)
    pct[order] = pct_sorted
    out[mask] = pct
    return replace(surface, values=out, scale_state="percentile")


def to_zscore(surface: RegimeSurface) -> RegimeSurface:
    """Population-weighted z-score (v - weighted_mean) / weighted_sd. Used for
    feature vectors, NOT the typology. NaN cells stay NaN."""
    v, w = surface.values, surface.population
    mask = surface.valid
    out = np.full(v.shape, np.nan)
    if mask.sum() >= 2:
        mu, sd = _weighted_mean_sd(v[mask], w[mask])
        out[mask] = (v[mask] - mu) / sd if sd > 0 else 0.0
    return replace(surface, values=out, scale_state="zscore")


# -- guards: the contract -----------------------------------------------------

def require_standardised(*surfaces: RegimeSurface) -> None:
    """Reject any raw surface entering a cross-regime operation."""
    for s in surfaces:
        if s.scale_state == "raw":
            raise ValueError(
                "cross-regime op on raw surfaces — standardise first "
                "(to_percentile / to_zscore)"
            )


def require_same_standardised(a: RegimeSurface, b: RegimeSurface) -> None:
    """Two surfaces combined together must be standardised, in the SAME scale
    state, and from the SAME city."""
    require_standardised(a, b)
    if a.scale_state != b.scale_state:
        raise ValueError(
            f"scale_state mismatch: {a.scale_state!r} vs {b.scale_state!r}")
    if a.city_id != b.city_id:
        raise ValueError(f"city_id mismatch: {a.city_id!r} vs {b.city_id!r}")


def require_percentile(*surfaces: RegimeSurface) -> None:
    """The typology is defined on percentiles specifically — raw AND zscore
    are both rejected here."""
    for s in surfaces:
        if s.scale_state != "percentile":
            raise ValueError(
                f"typology requires percentile surfaces, got {s.scale_state!r}"
            )
