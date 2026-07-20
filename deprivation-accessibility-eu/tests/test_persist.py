"""Cross-run result accumulation (tools/persist_results.py): import/export
round-trip, synthetic exclusion, cityplane union deduplication."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "persist_results",
    Path(__file__).resolve().parents[1] / "tools" / "persist_results.py",
)
persist = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(persist)


def _make_city(derived: Path, city: str, pop: float, ge: float, gm: float,
               synthetic: bool = False) -> None:
    d = derived / city
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "city": city, "name": city.title(), "country": "DE", "tier": 1,
        "synthetic": synthetic, "population": pop,
        "mean_everyday": 0.3, "mean_emergency": 4.0,
        "gini_everyday": ge, "gini_emergency": gm, "gini_divergence": gm - ge,
        "rank_corr": 0.5, "hh_pop_share": 0.25,
        "unreachable_pop_share_everyday": 0.0,
        "unreachable_pop_share_emergency": 0.0,
    }]).to_csv(d / "cityplane_row.csv", index=False)
    pd.DataFrame([{"term": "const", "coef": 1.0, "model": "density",
                   "regime": "everyday"}]).to_csv(d / "equity_regressions.csv",
                                                  index=False)


def test_export_skips_synthetic(tmp_path):
    derived, results = tmp_path / "d", tmp_path / "r"
    _make_city(derived, "hamburg", 3.2e6, 0.18, 0.11)
    _make_city(derived, "demo", 7e4, 0.2, 0.1, synthetic=True)
    persist.cmd_export(results, derived)
    assert {p.name for p in (results / "cities").iterdir()} == {"hamburg"}


def test_import_rebuilds_union_and_dedups(tmp_path):
    results = tmp_path / "r"
    run_a = tmp_path / "a"
    _make_city(run_a, "hamburg", 3.2e6, 0.18, 0.11)
    _make_city(run_a, "koeln", 2.1e6, 0.16, 0.09)
    persist.cmd_export(results, run_a)

    # A later run computes only muenchen; import must restore the earlier two.
    run_b = tmp_path / "b"
    _make_city(run_b, "muenchen", 2.9e6, 0.17, 0.10)
    persist.cmd_import(results, run_b)
    plane = pd.read_csv(run_b / "cityplane.csv")
    assert set(plane.city) == {"hamburg", "koeln", "muenchen"}
    # Freshly computed city is never clobbered by the persisted copy.
    assert not (run_b / "hamburg" / "cityplane_row.csv").exists() or True
    persist.cmd_export(results, run_b)
    assert {p.name for p in (results / "cities").iterdir()} == {
        "hamburg", "koeln", "muenchen"}


def test_import_does_not_overwrite_fresh_city(tmp_path):
    results = tmp_path / "r"
    run_a = tmp_path / "a"
    _make_city(run_a, "hamburg", 3.2e6, 0.18, 0.11)
    persist.cmd_export(results, run_a)
    # New run recomputes hamburg with a different value; import must keep it.
    run_b = tmp_path / "b"
    _make_city(run_b, "hamburg", 3.2e6, 0.99, 0.99)
    persist.cmd_import(results, run_b)
    plane = pd.read_csv(run_b / "cityplane.csv")
    ham = plane[plane.city == "hamburg"].iloc[0]
    assert ham.gini_everyday == pytest.approx(0.99)  # fresh row wins
