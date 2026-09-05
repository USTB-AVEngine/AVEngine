from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools.acoustics.prepare_authored_room_acoustics import (
    _canonical_anchor,
    _classify_material,
    _connectivity_pairs,
    _material_documents,
    _metadata_path,
    _metadata_seat_points,
    _room_id,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = json.loads(
    (
        ROOT / "examples/acoustics/authored_room_material_assumptions_v1.json"
    ).read_text(encoding="utf-8")
)


def test_authored_anchor_is_converted_to_canonical_y_up():
    assert _canonical_anchor([1.0, -2.0, 0.4], "Blender +Z up metres") == [
        1.0,
        0.4,
        2.0,
    ]


def test_material_classification_is_explicit_and_pbr_independent():
    assert _classify_material("RoomB_WindowGlass", PROFILE) == "glass"
    assert _classify_material("RoomB_SofaTeal", PROFILE) == "fabric"
    assert _classify_material("unlabelled_author_slot", PROFILE) == "generic_hard"


def test_material_documents_cover_each_source_slot_without_fallback():
    scene = SimpleNamespace(
        triangle_source_material_names=("WindowGlass", "OakFloor", "WindowGlass")
    )
    mapping, database, classes = _material_documents(
        "room_b_test",
        scene,
        PROFILE,
    )
    assert [entry["material_id"] for entry in mapping["entries"]] == [0, 1]
    assert all(entry["fallback"] is False for entry in mapping["entries"])
    assert len(database["materials"]) == 2
    assert classes == {"OakFloor": "hard_surface", "WindowGlass": "glass"}


def test_connectivity_pairs_are_derived_from_declared_anchors():
    pairs = _connectivity_pairs(
        {
            "entry": [-1.0, 0.0, 0.0],
            "dining": [1.0, 0.0, 2.0],
            "viewpoint": [0.0, 1.5, 0.0],
        }
    )
    assert pairs
    assert pairs[0]["start_m"] == [-1.0, 0.0, 0.0]
    assert pairs[0]["end_m"] == [1.0, 0.0, 2.0]


def test_polished_root_uses_polish_report_as_build_metadata(tmp_path):
    report = tmp_path / "polish_report.json"
    report.write_text("{}", encoding="utf-8")
    assert _metadata_path(tmp_path) == report


def test_polished_report_derives_room_id_from_source_blend(tmp_path):
    metadata = {
        "source_blend": "/external/authored_compact_home_room_b_v1.blend"
    }
    assert _room_id(metadata, tmp_path) == "authored_compact_home_room_b_v1"


def test_polished_report_seat_points_are_preserved_in_canonical_frame():
    points = _metadata_seat_points(
        {
            "furniture_semantics": [
                {
                    "seat_points": [
                        {
                            "anchor_id": "dining_seat_0",
                            "position_m": [1.0, -2.0, 0.0],
                            "facing_yaw_deg": 0.0,
                            "support_height_m": 0.46,
                        }
                    ]
                }
            ]
        },
        "Blender +Z up metres; exported GLB +Y up",
    )
    assert points[0]["position_canonical_m"] == [1.0, 0.0, 2.0]
