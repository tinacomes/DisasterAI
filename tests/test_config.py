"""Config loading, deep merge, city overlay, missing-parameter guard."""

import pytest

from depacc.config import (
    ConfigError,
    MissingParameterError,
    deep_merge,
    deprivation_spec,
    load_config,
    require_params,
)


def test_deep_merge_nested_override():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 3}, "c": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}  # inputs untouched


def test_load_defaults():
    cfg = load_config()
    assert cfg["crs"]["analysis"] == "EPSG:3035"
    assert cfg["city_definition"]["fua_size_threshold"] == 100000
    assert cfg["city_definition"]["city_sample_mode"] == "stratified"
    assert cfg["tiers"]["tier1"]["modes"] == ["walk", "car"]
    assert "transit" in cfg["tiers"]["tier2"]["modes"]
    assert set(cfg["everyday_services"]) == {
        "gp", "pharmacy", "supermarket", "school", "green_space",
    }
    assert set(cfg["emergency_services"]) == {
        "emergency_dept_hospital", "ambulance_station",
    }


def test_city_overlay_hamburg():
    cfg = load_config("hamburg")
    assert cfg["city"]["fua_code"] == "DE002L1"
    assert cfg["crs"]["local"] == "EPSG:32632"
    assert cfg["crs"]["analysis"] == "EPSG:3035"  # global key survives merge
    assert cfg["routing"]["modes"] == ["walk", "car", "transit"]
    assert cfg["routing"]["max_time_min"] == 120  # inherited default


def test_unknown_city_raises():
    with pytest.raises(ConfigError):
        load_config("atlantis")


def test_require_params_guard():
    with pytest.raises(MissingParameterError, match="Nice Paper"):
        require_params({"params": {"beta": None}, "source": "Nice Paper (2020)"})
    with pytest.raises(MissingParameterError):
        require_params({"params": {}, "source": "s"})
    assert require_params({"params": {"beta": 0.1}, "source": "s"}) == {"beta": 0.1}


def test_deprivation_spec_lookup():
    cfg = load_config()
    assert deprivation_spec(cfg, "everyday")["kind"] == "DLF"
    assert deprivation_spec(cfg, "emergency")["kind"] == "DCF"
    alt = deprivation_spec(cfg, "everyday", alternative="everyday_box_cox")
    assert alt["form"] == "box_cox"
    with pytest.raises(ConfigError):
        deprivation_spec(cfg, "sometimes")
    with pytest.raises(ConfigError):
        deprivation_spec(cfg, "everyday", alternative="nope")
