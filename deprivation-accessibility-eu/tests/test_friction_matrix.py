"""friction_matrix: batched cost-distance must be memory-bounded AND identical
regardless of source_batch (the fix for the Hamburg OOM)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("scipy")

import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from depacc.access import friction as F  # noqa: E402


class _FUA:
    def to_crs(self, crs):
        return self

    @property
    def total_bounds(self):
        return np.array([9.1, 53.7, 9.3, 53.9])


def _cfg(batch):
    return {"routing": {"max_time_min": 120, "max_time_min_by_mode": {"walk": 30}},
            "output": {"raw_root": "data/raw"},
            "friction": {"pad_deg": 0.25, "source_batch": batch, "wcs": {}}}


@pytest.fixture()
def tiny_raster(tmp_path, monkeypatch):
    tif = tmp_path / "data" / "raw" / "friction" / "tcity_walk.tif"
    tif.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(9.0, 54.0, 0.01, 0.01)
    arr = np.full((40, 40), 0.01, dtype="float32")   # 0.01 min/m, uniform
    with rasterio.open(tif, "w", driver="GTiff", height=40, width=40, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=-1) as dst:
        dst.write(arr, 1)
    monkeypatch.setattr(F, "fetch_friction_window",
                        lambda cfg, mode, fua, root, city: tif)
    return tmp_path


def test_friction_matrix_batch_invariant(tiny_raster):
    cells = pd.DataFrame({"cell_id": [0, 1, 2],
                          "lon": [9.15, 9.2, 9.25], "lat": [53.85, 53.8, 53.75]})
    facs = pd.DataFrame({"dest_id": [f"f{i}" for i in range(5)],
                         "lon": [9.12, 9.18, 9.22, 9.28, 9.15],
                         "lat": [53.88, 53.82, 53.78, 53.72, 53.8]})
    fua = _FUA()
    od1 = F.friction_matrix(_cfg(1), cells, facs, "walk", fua, tiny_raster, "tcity")
    od9 = F.friction_matrix(_cfg(9), cells, facs, "walk", fua, tiny_raster, "tcity")
    key = ["origin", "dest"]
    od1 = od1.sort_values(key).reset_index(drop=True)
    od9 = od9.sort_values(key).reset_index(drop=True)
    assert od1.equals(od9)                       # batch size must not change output
    assert od1.time.notna().all() and (od1.time >= 0).all()
    assert (od1.time <= 30.0 + 1e-9).all()       # respects the walk cutoff
