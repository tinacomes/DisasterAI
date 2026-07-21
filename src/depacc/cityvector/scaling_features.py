"""Cross-city feature standardisation as an enforced contract.

Clustering must never see raw cross-city magnitudes. ``scale_features``
returns a ``ScaledFeatures`` token carrying the standardised matrix and the
fitted center/scale; the clustering functions accept ONLY that token, so an
unscaled DataFrame cannot reach them through the type signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScaledFeatures:
    """Standardised cross-city feature matrix + the scaler that produced it.
    Only object clustering will accept."""

    matrix: np.ndarray            # (n_cities, n_features), standardised
    cities: list[str]
    feature_names: list[str]
    method: str                   # "robust" | "zscore"
    center: np.ndarray
    scale: np.ndarray


def scale_features(vectors: pd.DataFrame, feature_cols: list[str],
                   method: str = "robust", city_col: str = "city") -> ScaledFeatures:
    """Standardise selected feature columns ACROSS cities. Robust = median/IQR
    (default; small, outlier-prone sample); zscore = mean/SD. Columns that are
    all-NaN or zero-spread are dropped (logged). Emits center/scale per feature."""
    cols = [c for c in feature_cols if c in vectors.columns]
    dropped = [c for c in feature_cols if c not in vectors.columns]
    sub = vectors[cols].apply(pd.to_numeric, errors="coerce")
    # drop features with no usable spread across cities
    usable = []
    for c in cols:
        col = sub[c].to_numpy(dtype=float)
        if np.isfinite(col).sum() < 2:
            dropped.append(c)
            continue
        spread = (np.nanpercentile(col, 75) - np.nanpercentile(col, 25)
                  if method == "robust" else np.nanstd(col))
        if not np.isfinite(spread) or spread == 0:
            dropped.append(c)
            continue
        usable.append(c)
    if dropped:
        print(f"cross-city scaler: dropping zero-spread/missing features {sorted(set(dropped))}")
    X = sub[usable].to_numpy(dtype=float)
    if method == "robust":
        center = np.nanmedian(X, axis=0)
        scale = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    elif method == "zscore":
        center = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)
    else:
        raise ValueError(f"unknown scaler method {method!r}")
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - center) / scale
    # impute any residual NaN (a city missing a feature) with the scaled centre 0
    Z = np.where(np.isfinite(Z), Z, 0.0)
    print(f"cross-city scaler={method}: {len(usable)} features, "
          f"center={np.round(center, 3).tolist()}, scale={np.round(scale, 3).tolist()}")
    return ScaledFeatures(
        matrix=Z, cities=list(vectors[city_col].astype(str)),
        feature_names=usable, method=method, center=center, scale=scale,
    )
