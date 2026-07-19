"""Tier-1 fast path: friction-surface cost distance and Overpass parsing."""

import numpy as np
import pandas as pd
import pytest

from depacc.ingest.overpass import _parse_elements, build_query

scipy = pytest.importorskip("scipy")

from depacc.access.friction import cost_distance_times  # noqa: E402


def test_uniform_friction_matches_straight_line():
    """On a uniform surface, time to an axis-aligned pixel = distance x friction."""
    f = np.full((21, 21), 0.01)  # minutes per metre
    dx = np.full(21, 1000.0)     # 1 km pixels
    t = cost_distance_times(f, [(10, 10)], dx, 1000.0, max_time_min=1000.0)[0]
    assert t[10, 10] == 0.0
    assert t[10, 15] == pytest.approx(5 * 1000 * 0.01)   # 5 px east = 50 min
    assert t[5, 10] == pytest.approx(5 * 1000 * 0.01)    # 5 px north
    # Diagonal uses sqrt(2) edges, cheaper than the manhattan route.
    assert t[15, 15] == pytest.approx(5 * np.sqrt(2) * 1000 * 0.01, rel=1e-6)


def test_barrier_forces_detour_and_cutoff():
    f = np.full((11, 11), 0.01)
    f[:10, 5] = np.nan  # impassable wall with a gap at the bottom row
    dx = np.full(11, 1000.0)
    t = cost_distance_times(f, [(5, 2)], dx, 1000.0, max_time_min=1000.0)[0]
    direct = 6 * 1000 * 0.01
    assert t[5, 8] > direct  # detour around the wall
    assert np.isfinite(t[5, 8])
    # Tight cutoff renders the far side unreachable (NaN).
    t_cut = cost_distance_times(f, [(5, 2)], dx, 1000.0, max_time_min=direct)[0]
    assert np.isnan(t_cut[5, 8])


def test_latitude_dependent_pixel_width():
    """Narrower pixels (higher latitude) mean shorter east-west times."""
    f = np.full((5, 11), 0.01)
    dx_wide = np.full(5, 1000.0)
    dx_narrow = np.full(5, 500.0)
    t_wide = cost_distance_times(f, [(2, 0)], dx_wide, 1000.0, 1e6)[0]
    t_narrow = cost_distance_times(f, [(2, 0)], dx_narrow, 1000.0, 1e6)[0]
    assert t_narrow[2, 10] == pytest.approx(0.5 * t_wide[2, 10])


def test_overpass_query_and_parse():
    rules = [{"key": "amenity", "value": "hospital",
              "require": {"key": "emergency", "value": "yes"}}]
    q = build_query(rules, (53.0, 9.0, 54.0, 10.5))
    assert '["amenity"="hospital"]["emergency"="yes"]' in q
    assert "(53.0,9.0,54.0,10.5)" in q
    assert q.count("(53.0,9.0,54.0,10.5)") == 3  # node + way + relation

    elements = [
        {"type": "node", "id": 1, "lat": 53.5, "lon": 10.0,
         "tags": {"amenity": "hospital", "beds": "450"}},
        {"type": "way", "id": 2, "center": {"lat": 53.6, "lon": 10.1},
         "tags": {"amenity": "hospital", "beds": "n/a"}},
        {"type": "relation", "id": 3},  # no coordinates -> dropped
        {"type": "node", "id": 1, "lat": 53.5, "lon": 10.0},  # duplicate id
    ]
    spec = {"capacity": {"source": "beds", "tag": "beds"}}
    fac = _parse_elements(elements, spec)
    assert len(fac) == 2
    assert fac.loc[0, "capacity"] == 450.0 and not fac.loc[0, "capacity_proxy"]
    assert fac.loc[1, "capacity"] == 1.0 and bool(fac.loc[1, "capacity_proxy"])
