"""Cross-city feature vectors, clustering, size-gradient regression."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")

from depacc.cityvector.clustering import cluster_cities, size_gradient  # noqa: E402
from depacc.cityvector.features import FEATURES  # noqa: E402


def _vectors(n=12, seed=1):
    rng = np.random.default_rng(seed)
    log_pop = rng.uniform(5, 7, n)
    # Two planted regimes: divergence grows with size + noise.
    div = 0.05 * (log_pop - 5) + rng.normal(0, 0.005, n)
    df = pd.DataFrame({
        "city": [f"c{i}" for i in range(n)],
        "name": [f"City {i}" for i in range(n)],
        "log10_population": log_pop,
        "gini_divergence": div,
        "rank_corr": rng.uniform(0.3, 0.9, n),
        "hh_pop_share": rng.uniform(0.1, 0.4, n),
    })
    for f in FEATURES:
        if f not in df.columns:
            df[f] = rng.normal(0, 1, n)
    return df


def test_clustering_assigns_all_complete_rows():
    v = cluster_cities(_vectors(), n_clusters=3)
    assert v.cluster_kmeans.notna().all()
    assert v.cluster_ward.notna().all()
    assert set(v.cluster_kmeans.astype(int)) == {0, 1, 2}


def test_clustering_skipped_when_too_few():
    v = cluster_cities(_vectors(n=3), n_clusters=3)
    assert v.cluster_kmeans.isna().all()


def test_size_gradient_recovers_planted_slope():
    grad = size_gradient(_vectors(n=40, seed=2))
    row = grad[grad.outcome == "gini_divergence"].iloc[0]
    assert row.slope_per_log10_pop == pytest.approx(0.05, abs=0.01)
    assert row.p < 0.01
    assert row.inference == "cross-sectional space-for-time"


def test_run_cross_city_noop_on_empty_study(tmp_path, capsys):
    """cross must exit cleanly (not crash) when no city summaries exist yet —
    the ingest-only persist path that previously raised FileNotFoundError."""
    from depacc.cityvector.clustering import run_cross_city
    from depacc.config import load_config

    (tmp_path / "data" / "derived").mkdir(parents=True)
    run_cross_city(load_config(), tmp_path)  # must not raise
    assert "no city summaries yet" in capsys.readouterr().out
