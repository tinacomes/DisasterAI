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
