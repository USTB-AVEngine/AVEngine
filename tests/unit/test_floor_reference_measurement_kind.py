"""Floor artifacts must not claim a line trace when none hit."""
from __future__ import annotations

import json
from pathlib import Path

from avengine.rooms.room_package import validate_room_package


ROOT = Path(__file__).resolve().parents[2]


def _classify():
    import importlib.util
    path = ROOT / "tools/rooms/measure_ue_room_floor_reference.py"
    spec = importlib.util.spec_from_file_location("measure_ue_room_floor_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify_ue_floor_measurement


def test_depth_fallback_is_not_labeled_line_trace():
    classify = _classify()
    labeled = classify(
        line_hit_count=0,
        selected_method="depth_capture",
        floor_height_m=0.0001953125,
    )
    assert labeled["measurement_kind"] == "depth_readback_fallback"
    assert labeled["precision_m"] == 0.00025
    assert labeled["status"] == "measured"
    assert "line_trace" not in labeled["measurement_kind"]


def test_line_trace_hits_keep_line_trace_kind():
    classify = _classify()
    labeled = classify(
        line_hit_count=48,
        selected_method="line_trace",
        floor_height_m=0.2711074501,
    )
    assert labeled["measurement_kind"] == "ue_line_trace_down_blockall_complex_v1"
    assert labeled["status"] == "measured"


def test_implausible_height_is_invalid():
    classify = _classify()
    labeled = classify(
        line_hit_count=48,
        selected_method="depth_capture",
        floor_height_m=-74.7625,
    )
    assert labeled["status"] == "invalid"
    assert labeled["measurement_kind"] == "depth_readback_fallback"


def test_authored_packages_declare_depth_fallback_kind():
    for name in ("room_a.json", "room_b.json", "room_c.json", "kujiale_0020_full_home_v1.json"):
        package = json.loads((ROOT / "examples/rooms/packages" / name).read_text(encoding="utf-8"))
        reference = package["floor_reference"]
        assert reference["measurement_kind"] == "depth_readback_fallback"
        assert reference["precision_m"] == 0.00025
        assert validate_room_package(package)["room_id"] == package["room_id"]
