"""Authored UE exposure bias has one source: the room package field."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEAR = ROOT / "tools/rooms/run_spear_residential_episode.py"
PACKAGES = ROOT / "examples/rooms/packages"


def test_authored_packages_are_the_exposure_bias_source():
    expected = {
        "room_a.json": -3.0,
        "room_b.json": -4.0,
        "room_c.json": -3.0,
    }
    for name, bias in expected.items():
        package = json.loads((PACKAGES / name).read_text(encoding="utf-8"))
        assert package["exposure_bias_ev"] == bias
        assert package["planning_inputs"]["exposure_bias_ev"] == bias


def test_hardcoded_authored_usd_exposure_table_is_deleted():
    source = SPEAR.read_text(encoding="utf-8")
    assert "AUTHORED_USD_EXPOSURE_BIAS_EV" not in source
    assert "authored_usd_map_default" not in source
    assert "return None, \"not_requested\"" in source
    assert "mapping.get(\"exposure_bias_ev\")" in source or "mapping.get('exposure_bias_ev')" in source
