"""P3 regression checks for the seven production-family RoomPackages."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.rooms.room_package import SCHEMA, validate_room_package


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "examples/rooms/packages"
CATALOG = PACKAGE_ROOT / "catalog.json"
EXPECTED = {
    "native_apartment.json": ("apartment", "ue_spear"),
    "room_a.json": ("authored", "ue_spear"),
    "room_b.json": ("authored", "ue_spear"),
    "room_c.json": ("authored", "ue_spear"),
    "kujiale_0020_full_home_v1.json": ("kujiale", "ue_spear"),
    "mp3d_17DRP5sb8fy.json": ("mp3d", "habitat"),
    "hm3d_00800_TEEsavR23oF.json": ("hm3d", "habitat"),
}


@pytest.mark.parametrize("filename,route", EXPECTED.items())
def test_p3_package_is_strict_and_uses_declared_route(filename, route):
    package = json.loads((PACKAGE_ROOT / filename).read_text(encoding="utf-8"))
    assert package["schema"] == SCHEMA
    assert (package["family"], package["renderer"]) == route
    assert package["floor_reference"]
    assert package["static_geometry"]["vertices"]
    assert package["static_geometry"]["triangles"]
    assert package["static_geometry"]["coordinate_frame"] == {
        "linear_unit": "meter", "up_axis": "+Y", "handedness": "right", "forward_axis": "-Z"
    }
    if package["renderer"] == "habitat":
        assert not package["visual_scene"]["scene_glb"].endswith(".basis.glb")
        assert package["coordinate_frame"] == {
            "linear_unit": "meter", "up_axis": "+Y", "handedness": "right", "world_transform": "identity_habitat_y_up_right_v1"
        }
    else:
        assert package["visual_scene"]["map_path"].startswith("/Game/")
    assert validate_room_package(package)["room_id"] == package["room_id"]


def test_p3_catalog_covers_exactly_seven_packages():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert catalog["schema"] == "avengine_qa_room_package_catalog_v1"
    assert catalog["path_bindings"]["AVENGINE_MULTI_HOME_ROOT"] == "/data/datasets/avengine_workspaces/multi_home_activity_20260905"
    assert catalog["path_bindings"]["AVENGINE_MULTI_HOME_UE_ROOT"].endswith("/qa_full_asset_ue_stage_20260907_v1")
    assert catalog["path_bindings"]["AVENGINE_MP3D_ROOT"].endswith("mp3d_example_scene_1.1")
    assert catalog["path_bindings"]["AVENGINE_HM3D_ROOT"] == "/data/datasets/habitat_data"
    entries = catalog["rooms"]
    assert len(entries) == len(EXPECTED)
    assert {Path(entry["room_package"]).name for entry in entries} == set(EXPECTED)
    for entry in entries:
        package = json.loads((ROOT / entry["room_package"]).read_text(encoding="utf-8"))
        assert package["room_id"] == entry["room_id"]
        assert (package["family"], package["renderer"]) == (entry["family"], entry["renderer"])


def test_kujiale_pose_bindings_do_not_invent_furniture_seats():
    pose = json.loads((PACKAGE_ROOT / "kujiale_0020_pose_bindings.json").read_text(encoding="utf-8"))
    assert pose["schema"] == "avengine_qa_room_pose_bindings_v1"
    assert pose["placement_policy"]["seat_affordances"] == "none_declared_for_this_scene"
    assert all("seat_affordance_id" not in asset for asset in pose["assets"])
    assert pose["generation"]["method"].startswith("copy exact imported UE asset")


def test_current_ue_floor_references_are_bound_to_native_measurement_outputs():
    expected = {
        "native_apartment.json": "tmp/p3_room_packages_20260907_v3/floor_reference/",
        "room_a.json": "tmp/p3_room_packages_20260907_v6/ue_floor_measurements/room_a/",
        "room_b.json": "tmp/p3_room_packages_20260907_v6/ue_floor_measurements/room_b/",
        "room_c.json": "tmp/p3_room_packages_20260907_v6/ue_floor_measurements/room_c/",
        "kujiale_0020_full_home_v1.json": "tmp/p3_room_packages_20260907_v6/ue_floor_measurements/kujiale/",
    }
    for filename, prefix in expected.items():
        package = json.loads((PACKAGE_ROOT / filename).read_text(encoding="utf-8"))
        reference = package["floor_reference"]
        assert isinstance(reference, dict)
        assert reference["status"] == "measured"
        assert reference["path"].startswith(prefix)
