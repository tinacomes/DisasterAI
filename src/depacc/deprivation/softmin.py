"""Soft-minimum reducer over reachable facilities.

For travel times t_1..t_n and smoothing parameter kappa > 0:

    softmin(t) = -(1/kappa) * log( sum_j exp(-kappa * t_j) )

Properties (tested):
  * min(t) - log(n)/kappa <= softmin(t) <= min(t)
  * softmin -> min as kappa -> infinity
  * softmin decreases when more (finite-time) facilities are added — the
    deliberate "substitutability bonus" of the everyday regime: several
    similar-time options make a cell effectively less deprived than its
    single nearest option suggests.

NaN entries mark unreachable origin-facility pairs and are ignored; an
all-NaN row yields NaN (the cell is unreachable for this service and is
handled by the configured unreachable policy downstream).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def softmin(times, kappa: float, axis: int = -1) -> np.ndarray:
    """NaN-aware soft-minimum along ``axis`` of an array of travel times."""
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    t = np.asarray(times, dtype=float)
    all_nan = np.all(np.isnan(t), axis=axis)
    # Stable log-sum-exp around the (nan-)minimum.
    with np.errstate(invalid="ignore"):
        tmin = np.nanmin(np.where(np.isnan(t), np.inf, t), axis=axis, keepdims=True)
    tmin = np.where(np.isinf(tmin), np.nan, tmin)
    z = np.exp(-kappa * (t - tmin))
    s = np.nansum(np.where(np.isnan(t), 0.0, z), axis=axis)
    out = np.squeeze(tmin, axis=axis) - np.log(np.where(s > 0, s, np.nan)) / kappa
    return np.where(all_nan, np.nan, out)


def grouped_softmin(
    df: pd.DataFrame,
    kappa: float,
    group_col: str = "origin",
    time_col: str = "time",
) -> pd.Series:
    """Soft-minimum of ``time_col`` per ``group_col`` on a long-format
    travel-time table (vectorised; suitable for millions of rows).

    Rows with NaN time are ignored; groups whose times are all NaN get NaN.
    Returns a Series indexed by group.
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    t = df[time_col]
    tmin = df.groupby(group_col)[time_col].transform("min")
    z = np.exp(-kappa * (t - tmin))
    s = z.groupby(df[group_col]).sum(min_count=1)
    gmin = df.groupby(group_col)[time_col].min()
    return gmin - np.log(s) / kappa
