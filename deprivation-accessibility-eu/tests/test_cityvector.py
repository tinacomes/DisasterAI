"""Cross-city feature vectors, scaled-matrix clustering contract, gradient."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")

from depacc.cityvector.clustering import (  # noqa: E402
    choose_k_and_cluster,
    size_gradient,
)
from depacc.cityvector.features import (  # noqa: E402
    feature_columns,
    level_feature_names,
    level_features,
)
from depacc.cityvector.scaling_features import ScaledFeatures, scale_features  # noqa: E402
from depacc.config import load_config  # noqa: E402


def test_level_features_deprivation_free():
    cfg = load_config()
    surf = pd.DataFrame({
        "population": [100.0, 100.0, 100.0, 100.0],
        "t_regime_everyday": [5.0, 10.0, 20.0, 40.0],   # thresholds 15, 30
        "t_regime_emergency": [10.0, 40.0, 50.0, 70.0],  # thresholds 30,45,60
    })
    lf = level_features(surf, cfg)
    assert lf["pop_share_beyond_everyday_15"] == pytest.approx(0.5)   # 20,40 > 15
    assert lf["pop_share_beyond_everyday_30"] == pytest.approx(0.25)  # only 40 > 30
    assert lf["pop_share_beyond_emergency_45"] == pytest.approx(0.5)  # 50,70 > 45
    assert set(lf) == set(level_feature_names(cfg))


def _vectors(n=12, seed=1):
    rng = np.random.default_rng(seed)
    log_pop = rng.uniform(5, 7, n)
    df = pd.DataFrame({
        "city": [f"c{i}" for i in range(n)],
        "name": [f"City {i}" for i in range(n)],
        "country": rng.choice(["DE", "FR", "NL"], n),
        "population": 10 ** log_pop,
        "log10_population": log_pop,
        "gini_everyday": rng.uniform(0.1, 0.3, n),
        "gini_emergency": rng.uniform(0.1, 0.4, n),
        "p90_p50_ratio_emergency": rng.uniform(1.2, 3.0, n),
        "spearman_rho": rng.uniform(0.2, 0.8, n),
        "divergence_gap": rng.uniform(-0.1, 0.2, n),
        "compounding_pop_share_50": rng.uniform(0.1, 0.4, n),
    })
    for c in feature_columns(load_config()):
        if c not in df.columns:
            df[c] = rng.normal(0, 1, n)
    return df


def test_scale_features_returns_token_and_standardises():
    cfg = load_config()
    scaled = scale_features(_vectors(), feature_columns(cfg), method="robust")
    assert isinstance(scaled, ScaledFeatures)
    assert scaled.matrix.shape[0] == 12
    assert np.isfinite(scaled.matrix).all()


def test_clustering_rejects_unscaled_matrix():
    """The contract: a raw DataFrame/array must not reach the clusterer."""
    with pytest.raises(TypeError, match="ScaledFeatures"):
        choose_k_and_cluster(_vectors())          # DataFrame
    with pytest.raises(TypeError, match="ScaledFeatures"):
        choose_k_and_cluster(np.random.rand(12, 5))  # ndarray


def test_clustering_on_scaled_matrix():
    cfg = load_config()
    scaled = scale_features(_vectors(20, seed=3), feature_columns(cfg))
    res = choose_k_and_cluster(scaled, k_range=(2, 5), bootstrap=10)
    assert res["k"] in (2, 3, 4, 5)
    assert len(res["labels_kmeans"]) == 20
    assert -1.0 <= res["silhouette"] <= 1.0


def test_size_gradient_runs():
    grad = size_gradient(_vectors(30, seed=2))
    assert set(grad.outcome) <= {"divergence_gap", "spearman_rho",
                                 "compounding_pop_share_50"}
    assert (grad.inference == "cross-sectional space-for-time").all()
