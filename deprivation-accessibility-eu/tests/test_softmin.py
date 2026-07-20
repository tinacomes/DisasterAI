"""Soft-minimum reducer: bounds, hard-min limit, substitutability bonus,
NaN/unreachable handling, grouped (long-format) equivalence."""

import numpy as np
import pandas as pd
import pytest

from depacc.deprivation.softmin import grouped_softmin, softmin


def test_bounds():
    t = np.array([10.0, 12.0, 30.0, 45.0])
    for kappa in (0.1, 0.5, 2.0):
        sm = softmin(t, kappa)
        assert sm <= t.min() + 1e-12
        assert sm >= t.min() - np.log(len(t)) / kappa - 1e-12


def test_hard_min_limit():
    t = np.array([10.0, 12.0, 30.0])
    assert softmin(t, 1000.0) == pytest.approx(10.0, abs=1e-6)


def test_substitutability_bonus():
    """More equally-near facilities -> lower effective time (everyday logic)."""
    one = softmin(np.array([10.0]), 0.5)
    three = softmin(np.array([10.0, 10.0, 10.0]), 0.5)
    assert three < one
    assert one == pytest.approx(10.0)


def test_far_facility_barely_matters():
    base = softmin(np.array([10.0]), 0.5)
    with_far = softmin(np.array([10.0, 120.0]), 0.5)
    assert base - with_far < 1e-10


def test_nan_ignored_and_all_nan_propagates():
    assert softmin(np.array([np.nan, 15.0, np.nan]), 0.5) == pytest.approx(15.0)
    assert np.isnan(softmin(np.array([np.nan, np.nan]), 0.5))


def test_axis_handling():
    t = np.array([[10.0, 20.0], [np.nan, 40.0]])
    out = softmin(t, 5.0, axis=1)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(10.0, abs=0.01)
    assert out[1] == pytest.approx(40.0)


def test_invalid_kappa():
    with pytest.raises(ValueError):
        softmin(np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        grouped_softmin(pd.DataFrame({"origin": [1], "time": [1.0]}), -1.0)


def test_grouped_matches_dense():
    rng = np.random.default_rng(42)
    rows = []
    expected = {}
    for origin in range(20):
        times = rng.uniform(5, 60, size=rng.integers(1, 8))
        # Sprinkle unreachable pairs.
        times = np.where(rng.uniform(size=times.size) < 0.2, np.nan, times)
        expected[origin] = softmin(times, 0.5)
        rows += [{"origin": origin, "dest": j, "time": t} for j, t in enumerate(times)]
    df = pd.DataFrame(rows)
    got = grouped_softmin(df, 0.5)
    for origin, exp in expected.items():
        if np.isnan(exp):
            assert np.isnan(got[origin])
        else:
            assert got[origin] == pytest.approx(exp)
