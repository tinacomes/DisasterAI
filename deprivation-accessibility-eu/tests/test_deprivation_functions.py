"""DLF/DCF mapping: shape properties, config round-trip, null-parameter guard.

Tests use SYNTHETIC parameters passed explicitly; they never rely on (or leak
into) the literature-transferred values, which remain null placeholders in
config/deprivation.yaml until filled from the cited papers.
"""

import numpy as np
import pytest

from depacc.config import ConfigError, MissingParameterError, load_config, deprivation_spec
from depacc.deprivation.functions import DeprivationFunction

EXP = dict(form="exponential", params={"beta": 0.05, "scale": 1.0})
BC = dict(form="box_cox", params={"lam": 1.5, "scale": 2.0, "shift": 1.0})


@pytest.mark.parametrize("spec", [EXP, BC])
def test_zero_time_zero_deprivation(spec):
    g = DeprivationFunction(**spec)
    assert g(0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("spec", [EXP, BC])
def test_increasing_and_convex(spec):
    g = DeprivationFunction(**spec)
    t = np.linspace(0, 120, 481)
    y = g(t)
    dy = np.diff(y)
    assert np.all(dy > 0), "deprivation must strictly increase in travel time"
    d2y = np.diff(dy)
    assert np.all(d2y >= -1e-9), "deprivation must be convex in travel time"


def test_nan_propagates():
    g = DeprivationFunction(**EXP)
    out = g(np.array([10.0, np.nan]))
    assert np.isnan(out[1]) and not np.isnan(out[0])


def test_negative_time_rejected():
    g = DeprivationFunction(**EXP)
    with pytest.raises(ValueError):
        g(-1.0)


@pytest.mark.parametrize(
    "form,params",
    [
        ("exponential", {"beta": -0.1, "scale": 1.0}),  # decreasing
        ("box_cox", {"lam": 0.5, "scale": 1.0, "shift": 1.0}),  # concave
        ("box_cox", {"lam": 2.0, "scale": -1.0, "shift": 1.0}),  # negative scale
        ("nope", {"beta": 0.1, "scale": 1.0}),  # unknown form
    ],
)
def test_invalid_specs_rejected(form, params):
    with pytest.raises(ConfigError):
        DeprivationFunction(form=form, params=params)


def test_from_spec_round_trip():
    spec = {"kind": "DCF", "form": "box_cox",
            "params": {"lam": 1.2, "scale": 3.0, "shift": 1.0},
            "source": "synthetic test values"}
    g = DeprivationFunction.from_spec(spec)
    assert g.kind == "DCF"
    assert g(30.0) > g(10.0) > 0


def test_shipped_config_placeholders_are_guarded():
    """The shipped config must refuse to run and must cite its sources."""
    cfg = load_config()
    for regime in ("everyday", "emergency"):
        spec = deprivation_spec(cfg, regime)
        with pytest.raises(MissingParameterError) as err:
            DeprivationFunction.from_spec(spec, context=f"{regime} deprivation")
        # The error must repeat the citation so the user knows which paper
        # the parameter must be transferred from.
        assert "TODO(cite)" in str(err.value)
