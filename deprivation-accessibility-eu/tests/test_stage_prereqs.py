"""Stages give a clear error (not a raw FileNotFoundError traceback) when a
prior stage's outputs are missing — the case that bites when GitHub runs
stages in separate fresh-runner dispatches."""

import pytest

from depacc.access.matrices import run_access
from depacc.config import load_config
from depacc.deprivation.pipeline import run_deprivation

pytest.importorskip("pyarrow")


def test_access_without_ingest_is_clear(tmp_path):
    cfg = load_config("demo")
    with pytest.raises(RuntimeError, match="ingest.*before.*access"):
        run_access(cfg, "demo", tmp_path)


def test_deprivation_without_access_is_clear(tmp_path):
    cfg = load_config("demo")
    with pytest.raises(RuntimeError, match="ingest.*access"):
        run_deprivation(cfg, "demo", tmp_path)


def test_stage_autochain_logic(tmp_path):
    """A dispatched stage pulls in only the missing prerequisites, in order."""
    from depacc.cli import _stages_to_run

    out = tmp_path
    # Nothing done yet: 'deprivation' pulls ingest+access+itself.
    assert _stages_to_run("deprivation", out) == ["ingest", "access", "deprivation"]
    # ingest done -> only access+deprivation.
    (out / "cells.parquet").write_text("x")
    assert _stages_to_run("deprivation", out) == ["access", "deprivation"]
    # access done too -> only the target.
    (out / "od_gp_walk.parquet").write_text("x")
    assert _stages_to_run("deprivation", out) == ["deprivation"]
    # ingest dispatched alone is always just itself.
    assert _stages_to_run("ingest", out) == ["ingest"]
