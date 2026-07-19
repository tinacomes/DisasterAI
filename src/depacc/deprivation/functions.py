"""Deprivation functions g(t): increasing, convex functions of travel time.

The deprivation function IS the impedance function of a gravity model run in
the opposite direction: instead of a decreasing impedance f(t) discounting
opportunities, an increasing convex g(t) prices the burden of the (effective)
travel time itself. The measure is built natively this way — g is evaluated
at an effective time (soft-min of congestion-inflated times for the everyday
regime; nearest-facility time for the emergency regime), never bolted onto a
standard decreasing-impedance gravity score.

Forms (t in minutes; parameters transferred from the literature via config,
see config/deprivation.yaml — nulls are rejected by depacc.config.require_params):

    exponential : g(t) = scale * (exp(beta * t) - 1)                beta > 0
    box_cox     : g(t) = scale * ((t + shift)**lam - shift**lam)/lam,  lam > 1

Both satisfy g(0) = 0, g'(t) > 0 and g''(t) >= 0 for t >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from depacc.config import ConfigError, require_params

FORMS = ("exponential", "box_cox")


def _exponential(t: np.ndarray, beta: float, scale: float) -> np.ndarray:
    return scale * np.expm1(beta * t)


def _box_cox(t: np.ndarray, lam: float, scale: float, shift: float) -> np.ndarray:
    return scale * ((t + shift) ** lam - shift**lam) / lam


@dataclass(frozen=True)
class DeprivationFunction:
    """A configured deprivation function, callable on scalars or arrays.

    ``kind`` distinguishes the dimensionless Deprivation Level Function
    ("DLF", primary for the everyday framing) from the monetary Deprivation
    Cost Function ("DCF", alternative for the emergency framing); it does not
    change the mathematics, only labelling/units of the output surface.
    """

    form: str
    params: Mapping[str, float]
    kind: str = "DLF"
    source: str = ""

    def __post_init__(self) -> None:
        if self.form not in FORMS:
            raise ConfigError(f"Unknown deprivation form '{self.form}'; known: {FORMS}")
        p = self.params
        if self.form == "exponential":
            missing = {"beta", "scale"} - set(p)
            if missing:
                raise ConfigError(f"exponential form missing params {sorted(missing)}")
            if p["beta"] <= 0:
                raise ConfigError("exponential form requires beta > 0 (increasing, convex)")
        else:
            missing = {"lam", "scale", "shift"} - set(p)
            if missing:
                raise ConfigError(f"box_cox form missing params {sorted(missing)}")
            if p["lam"] <= 1:
                raise ConfigError("box_cox form requires lam > 1 for convexity in time")
            if p["shift"] < 0:
                raise ConfigError("box_cox form requires shift >= 0")
        if p.get("scale", 1.0) <= 0:
            raise ConfigError("deprivation scale must be > 0")

    @classmethod
    def from_spec(cls, spec: Mapping[str, object], context: str = "deprivation function") -> "DeprivationFunction":
        """Build from a config spec ({kind, form, params, source}); rejects
        null-placeholder parameters with the citation in the error message."""
        params = require_params(spec, context=context)
        return cls(
            form=str(spec.get("form")),
            params=params,
            kind=str(spec.get("kind", "DLF")),
            source=str(spec.get("source", "")),
        )

    def __call__(self, t) -> np.ndarray:
        """Evaluate g(t); NaN in -> NaN out (unreachable propagates until the
        caller applies the configured unreachable policy)."""
        arr = np.asarray(t, dtype=float)
        neg = arr < 0
        if np.any(neg & ~np.isnan(arr)):
            raise ValueError("travel time must be non-negative")
        if self.form == "exponential":
            out = _exponential(arr, self.params["beta"], self.params["scale"])
        else:
            out = _box_cox(arr, self.params["lam"], self.params["scale"], self.params["shift"])
        return out
