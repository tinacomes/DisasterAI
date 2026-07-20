"""Travel-time matrices: every populated cell -> every facility, per service
and mode.

Real cities use a single routing engine for all modes — R5 via r5py (JDK 21
required) — so Tier-1 (walk, car) and Tier-2 (+transit) matrices are
methodologically identical. Synthetic demo cities use a deterministic
straight-line router so the pipeline runs without network/Java.

Output: one long-format parquet per (service, mode) under
data/derived/<city>/od_<service>_<mode>.parquet with columns
[origin, dest, time]; pairs beyond routing.max_time_min are absent, and a
cell with no row for a service is unreachable (flagged downstream — see
depacc.deprivation.surfaces).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from depacc.ingest.pipeline import derived_dir

# Straight-line demo speeds (km/h) with a detour factor of 1.3; used ONLY for
# synthetic fixtures.
_SYNTH_SPEED = {"walk": 4.8, "car": 30.0, "transit": 15.0}
_SYNTH_DETOUR = 1.3
_SYNTH_ACCESS_OVERHEAD_MIN = {"walk": 0.0, "car": 3.0, "transit": 5.0}


def run_access(cfg: dict, city: str, root: Path) -> None:
    out = derived_dir(cfg, city, root)
    cells = pd.read_parquet(out / "cells.parquet")
    modes = cfg["routing"].get("modes") or cfg["tiers"]["tier1"]["modes"]
    services = list(cfg.get("everyday_services", {})) + list(cfg.get("emergency_services", {}))
    max_time = float(cfg["routing"]["max_time_min"])

    synthetic = bool(cfg["city"].get("synthetic"))
    engine = cfg["routing"].get("engine", "r5")
    network = None
    fua = None
    for service in services:
        fac_path = out / f"facilities_{service}.parquet"
        if not fac_path.exists():
            print(f"WARNING: no facilities for '{service}'; skipping")
            continue
        facilities = pd.read_parquet(fac_path)
        if facilities.empty:
            print(f"WARNING: zero facilities for '{service}'; skipping")
            continue
        for mode in modes:
            od_path = out / f"od_{service}_{mode}.parquet"
            if od_path.exists():
                continue
            if synthetic:
                od = _synthetic_matrix(cells, facilities, mode, max_time)
            elif engine == "friction":
                import geopandas as gpd

                from depacc.access.friction import friction_matrix

                if fua is None:
                    fua = gpd.read_parquet(out / "fua_boundary.parquet")
                od = friction_matrix(cfg, cells, facilities, mode, fua, root, city)
            elif engine == "r5":
                if network is None:
                    network = _build_r5_network(cfg, city, out)
                od = _r5_matrix(network, cells, facilities, mode, cfg)
            else:
                raise ValueError(f"Unknown routing engine '{engine}'")
            od.to_parquet(od_path)
            reach = od.origin.nunique()
            print(f"od[{service},{mode}]: {len(od)} pairs, "
                  f"{reach}/{len(cells)} cells reach >=1 facility")


def _synthetic_matrix(cells: pd.DataFrame, facilities: pd.DataFrame,
                      mode: str, max_time: float) -> pd.DataFrame:
    """Deterministic straight-line travel times for the demo fixture."""
    speed_m_min = _SYNTH_SPEED[mode] * 1000.0 / 60.0
    dx = cells.x.to_numpy()[:, None] - facilities.x.to_numpy()[None, :]
    dy = cells.y.to_numpy()[:, None] - facilities.y.to_numpy()[None, :]
    t = np.hypot(dx, dy) * _SYNTH_DETOUR / speed_m_min + _SYNTH_ACCESS_OVERHEAD_MIN[mode]
    o, d = np.nonzero(t <= max_time)
    return pd.DataFrame({
        "origin": cells.cell_id.to_numpy()[o],
        "dest": facilities.dest_id.to_numpy()[d],
        "time": t[o, d],
    })


def _build_r5_network(cfg: dict, city: str, out: Path):
    import r5py

    pbf = Path((out / "network_pbf_path.txt").read_text().strip())
    gtfs: list[str] = []
    gtfs_list = out / "gtfs_paths.txt"
    if gtfs_list.exists():
        gtfs = [p for p in gtfs_list.read_text().splitlines() if p]
    print(f"Building R5 network from {pbf.name} + {len(gtfs)} GTFS feed(s)")
    return r5py.TransportNetwork(str(pbf), gtfs)


def _r5_matrix(network, cells: pd.DataFrame, facilities: pd.DataFrame,
               mode: str, cfg: dict) -> pd.DataFrame:
    import datetime

    import geopandas as gpd
    import r5py

    r5_modes = {
        "walk": [r5py.TransportMode.WALK],
        "car": [r5py.TransportMode.CAR],
        "transit": [r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
    }[mode]
    dep = cfg["routing"]["departure"]
    # Next occurrence of the configured weekday must fall inside the GTFS
    # validity window; r5py warns if not. Date is resolved at run time.
    departure = _next_weekday(dep["weekday"], dep["time_window_start"])

    origins = gpd.GeoDataFrame(
        {"id": cells.cell_id},
        geometry=gpd.points_from_xy(cells.lon, cells.lat), crs="EPSG:4326",
    )
    destinations = gpd.GeoDataFrame(
        {"id": facilities.dest_id},
        geometry=gpd.points_from_xy(facilities.lon, facilities.lat), crs="EPSG:4326",
    )
    computer = r5py.TravelTimeMatrixComputer(
        network,
        origins=origins,
        destinations=destinations,
        transport_modes=r5_modes,
        departure=departure,
        departure_time_window=datetime.timedelta(
            minutes=int(cfg["routing"]["departure"]["time_window_minutes"])),
        max_time=datetime.timedelta(minutes=int(cfg["routing"]["max_time_min"])),
        speed_walking=float(cfg["routing"]["walk_speed_kmh"]),
    )
    tt = computer.compute_travel_times()
    tt = tt.rename(columns={"from_id": "origin", "to_id": "dest", "travel_time": "time"})
    return tt.dropna(subset=["time"])[["origin", "dest", "time"]]


def _next_weekday(weekday: str, hhmm: str):
    import datetime

    names = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]
    target = names.index(weekday.lower())
    today = datetime.date.today()
    delta = (target - today.weekday()) % 7 or 7
    day = today + datetime.timedelta(days=delta)
    hour, minute = map(int, hhmm.split(":"))
    return datetime.datetime.combine(day, datetime.time(hour, minute))
