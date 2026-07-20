"""Divergence typology: weighted quantiles, class assignment, population
shares, NaN handling."""

import numpy as np
import pandas as pd
import pytest

from depacc.divergence.typology import bivariate_typology, weighted_quantile


def test_weighted_quantile_basics():
    v = [1.0, 2.0, 3.0, 4.0]
    assert weighted_quantile(v, 0.5, [1, 1, 1, 1]) in (2.0, 3.0)
    # All weight on the last value drags the median there.
    assert weighted_quantile(v, 0.5, [0.0, 0.0, 0.0, 10.0]) == 4.0
    assert np.isnan(weighted_quantile([np.nan], 0.5, [1.0]))
    with pytest.raises(ValueError):
        weighted_quantile(v, 1.5, [1, 1, 1, 1])


def _frame():
    return pd.DataFrame(
        {
            "deprivation_everyday": [1.0, 1.0, 10.0, 10.0, np.nan],
            "deprivation_emergency": [1.0, 10.0, 1.0, 10.0, 5.0],
            "population": [100.0, 100.0, 100.0, 100.0, 50.0],
        }
    )


def test_four_quadrants():
    cells, summary = bivariate_typology(_frame())
    assert list(cells["typology"][:4]) == ["LL", "LH", "HL", "HH"]
    assert cells["typology"].isna().iloc[4]  # NaN surface -> unclassified


def test_population_shares_sum_to_one():
    _, summary = bivariate_typology(_frame())
    assert summary["population_share"].sum() == pytest.approx(1.0)
    assert summary.loc["HH", "population"] == 100.0
    # 50 of 450 people live in unclassifiable cells.
    assert summary.attrs["unclassified_pop_share"] == pytest.approx(50.0 / 450.0)


def test_compounding_identified():
    """The HH class is exactly the cells bad on BOTH surfaces."""
    cells, _ = bivariate_typology(_frame())
    hh = cells[cells["typology"] == "HH"]
    assert (hh["deprivation_everyday"] > 1.0).all()
    assert (hh["deprivation_emergency"] > 1.0).all()
    assert len(hh) == 1


def test_quantile_shifts_thresholds():
    df = pd.DataFrame(
        {
            "deprivation_everyday": np.linspace(0, 10, 11),
            "deprivation_emergency": np.linspace(0, 10, 11),
            "population": np.ones(11),
        }
    )
    _, summary_med = bivariate_typology(df, quantile=0.5)
    _, summary_p80 = bivariate_typology(df, quantile=0.8)
    assert summary_p80.attrs["threshold_everyday"] > summary_med.attrs["threshold_everyday"]
    # A stricter (higher) threshold shrinks the high-high population.
    assert summary_p80.loc["HH", "population"] <= summary_med.loc["HH", "population"]


def test_all_nan_surfaces():
    df = pd.DataFrame(
        {
            "deprivation_everyday": [np.nan, np.nan],
            "deprivation_emergency": [np.nan, np.nan],
            "population": [1.0, 1.0],
        }
    )
    cells, summary = bivariate_typology(df)
    assert cells["typology"].isna().all()
    assert summary["population"].sum() == 0
