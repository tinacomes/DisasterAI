"""PNAS-style scaling regressions: planted elasticities and the
everyday-vs-emergency gradient-difference test."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from depacc.cityvector.scaling import regime_slope_difference, scaling_table  # noqa: E402


def _vectors(n=60, seed=3, b_ev=0.10, b_em=0.25):
    """Cities whose Ginis scale with planted log-log gradients."""
    rng = np.random.default_rng(seed)
    ln_pop = rng.uniform(np.log(1e5), np.log(5e6), n)
    return pd.DataFrame({
        "city": [f"c{i}" for i in range(n)],
        "country": rng.choice(["DE", "FR", "NL"], n),
        "population": np.exp(ln_pop),
        "gini_everyday": np.exp(-2.0 + b_ev * ln_pop + rng.normal(0, 0.02, n)),
        "gini_emergency": np.exp(-2.5 + b_em * ln_pop + rng.normal(0, 0.02, n)),
        "mean_everyday": np.exp(0.5 + 0.05 * ln_pop + rng.normal(0, 0.02, n)),
        "mean_emergency": np.exp(1.0 + 0.05 * ln_pop + rng.normal(0, 0.02, n)),
        "hh_pop_share": rng.uniform(0.1, 0.4, n),
    })


def test_scaling_recovers_planted_elasticities():
    table = scaling_table(_vectors())
    row = table[table.outcome == "gini_everyday"].iloc[0]
    assert "elasticity" in row.spec
    assert row.gradient_per_ln_pop == pytest.approx(0.10, abs=0.02)
    assert (table.inference == "cross-sectional space-for-time").all()


def test_country_fixed_effects_variant():
    table = scaling_table(_vectors(), country_fe=True)
    assert "country FE" in table.spec.iloc[0]
    row = table[table.outcome == "gini_emergency"].iloc[0]
    assert row.gradient_per_ln_pop == pytest.approx(0.25, abs=0.03)


def test_level_spec_for_nonpositive_outcomes():
    v = _vectors()
    v.loc[0, "hh_pop_share"] = 0.0  # a zero share forces the level-log spec
    table = scaling_table(v)
    assert table[table.outcome == "hh_pop_share"].iloc[0].spec.startswith("level-log")


def test_regime_slope_difference_detects_divergence():
    diff = regime_slope_difference(_vectors(), measure="gini")
    row = diff.iloc[0]
    assert row.gradient_difference_emergency == pytest.approx(0.15, abs=0.03)
    assert row.p < 0.01
    assert "steepens" in row.interpretation
    # Equal planted gradients -> no significant difference.
    same = regime_slope_difference(_vectors(b_ev=0.2, b_em=0.2), measure="gini")
    assert abs(same.iloc[0].gradient_difference_emergency) < 0.03


def test_too_few_cities_is_empty():
    assert regime_slope_difference(_vectors(n=3)).empty
    assert scaling_table(_vectors(n=3)).empty
