"""End-to-end scale invariance: multiplying the raw emergency surface by a
constant must leave every standardised output identical — percentiles,
typology shares, and (across cities) cluster labels."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from depacc.cityvector.clustering import choose_k_and_cluster
from depacc.cityvector.features import feature_columns
from depacc.cityvector.scaling_features import scale_features
from depacc.config import load_config
from depacc.divergence.typology import bivariate_typology
from depacc.standardize import RegimeSurface, to_percentile


def test_typology_shares_invariant_to_emergency_scale():
    rng = np.random.default_rng(0)
    n = 500
    pop = rng.uniform(1, 400, n)
    ev = to_percentile(RegimeSurface(rng.uniform(0, 1, n), pop, "everyday", "c", "raw"))
    em_raw = rng.uniform(0, 900, n)
    m1 = to_percentile(RegimeSurface(em_raw, pop, "emergency", "c", "raw"))
    m2 = to_percentile(RegimeSurface(em_raw * 1000.0, pop, "emergency", "c", "raw"))
    _, s1 = bivariate_typology(ev, m1, 0.5)
    _, s2 = bivariate_typology(ev, m2, 0.5)
    assert np.allclose(s1["population_share"].to_numpy(),
                       s2["population_share"].to_numpy())


def _city_vectors(scale_emergency=1.0, seed=3, n=20):
    """Synthetic per-city feature table whose emergency-derived columns scale
    with `scale_emergency` — standardisation must wash the factor out."""
    rng = np.random.default_rng(seed)
    log_pop = rng.uniform(5, 7, n)
    base_em = rng.uniform(0.1, 0.4, n)
    df = pd.DataFrame({
        "city": [f"c{i}" for i in range(n)],
        "name": [f"City {i}" for i in range(n)],
        "country": rng.choice(["DE", "FR"], n),
        "population": 10 ** log_pop,
        "log10_population": log_pop,
        "gini_everyday": rng.uniform(0.1, 0.3, n),
        "gini_emergency": base_em * scale_emergency,     # scales
        "p90_p50_ratio_emergency": rng.uniform(1.2, 3.0, n) * scale_emergency,
        "spearman_rho": rng.uniform(0.2, 0.8, n),
        "divergence_gap": rng.uniform(-0.1, 0.2, n),
        "compounding_pop_share_50": rng.uniform(0.1, 0.4, n),
    })
    for c in feature_columns(load_config()):
        if c not in df.columns:
            df[c] = rng.normal(0, 1, n)
    return df


def test_cluster_labels_invariant_to_emergency_scale():
    cfg = load_config()
    cols = feature_columns(cfg)
    a = choose_k_and_cluster(scale_features(_city_vectors(1.0), cols), k_range=(2, 5),
                             bootstrap=5)
    b = choose_k_and_cluster(scale_features(_city_vectors(1000.0), cols), k_range=(2, 5),
                             bootstrap=5)
    # Robust median/IQR scaling removes the emergency-scale factor entirely.
    assert a["labels_kmeans"] == b["labels_kmeans"]
    assert a["k"] == b["k"]
