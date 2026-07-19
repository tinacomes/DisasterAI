"""Population-weighted Gini and concentration index sanity properties."""

import numpy as np
import pytest

from depacc.equity.indices import concentration_index, weighted_gini, weighted_mean


def test_weighted_mean():
    assert weighted_mean([1.0, 3.0], [1.0, 1.0]) == 2.0
    assert weighted_mean([1.0, 3.0], [3.0, 1.0]) == 1.5
    assert np.isnan(weighted_mean([np.nan], [1.0]))


def test_gini_equal_distribution_zero():
    assert weighted_gini(np.full(10, 5.0), np.ones(10)) == pytest.approx(0.0, abs=1e-12)


def test_gini_concentration_extreme():
    # All deprivation on one of many people -> Gini -> 1.
    v = np.zeros(1000); v[0] = 100.0
    g = weighted_gini(v, np.ones(1000))
    assert g == pytest.approx(1.0, abs=2e-3)


def test_gini_weight_invariance():
    """Duplicating a cell == doubling its weight."""
    g_dup = weighted_gini([1.0, 1.0, 4.0], [1.0, 1.0, 1.0])
    g_w = weighted_gini([1.0, 4.0], [2.0, 1.0])
    assert g_dup == pytest.approx(g_w)


def test_gini_rejects_negative():
    with pytest.raises(ValueError):
        weighted_gini([-1.0, 1.0], [1.0, 1.0])


def test_concentration_index_sign():
    # Deprivation decreasing in SES -> concentrated among the poor -> CI < 0.
    ses = np.arange(10, dtype=float)
    dep = 10.0 - ses
    assert concentration_index(dep, ses, np.ones(10)) < 0
    # Deprivation rising with SES -> CI > 0; flat -> 0.
    assert concentration_index(ses, ses, np.ones(10)) > 0
    assert concentration_index(np.full(10, 3.0), ses, np.ones(10)) == pytest.approx(0.0, abs=1e-12)
