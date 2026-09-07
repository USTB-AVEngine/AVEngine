"""RoomPackage existence checks report missing absolute filesystem paths."""
from __future__ import annotations

from pathlib import Path

from avengine.rooms.room_package import (
    missing_filesystem_paths,
    room_package_errors,
    validate_room_package,
)


def _package(**overrides):
    package = {
        "schema": "avengine_qa_room_package_v1",
        "room_id": "room",
        "family": "apartment",
        "renderer": "ue_spear",
        "visual_scene": {"map_path": "/Game/Apartment", "uproject": "stage.uproject"},
        "acoustic_package": "acoustic/manifest.json",
        "walkable_space": {"kind": "route_bank", "path": "routes.json"},
        "floor_reference": "measured_floor.json",
        "static_geometry": {"vertices": "vertices.npy", "triangles": "triangles.npy",
                            "coordinate_frame": {"linear_unit": "meter", "up_axis": "+Y",
                                                 "handedness": "right", "forward_axis": "-Z"}},
        "semantics": {"path": "room.json"},
        "coordinate_frame": {"linear_unit": "centimeter", "up_axis": "+Z",
                             "handedness": "left", "world_transform": "ue_xyz_cm_to_xzy_m_v1"},
        "subrooms": [],
    }
    package.update(overrides)
    return package


def test_game_and_root_paths_are_not_filesystem_checked():
    package = _package()
    package["visual_scene"]["map_path"] = "/Game/AVEngine/MultiHome/room_a_living_props_v7"
    package["semantics"]["usd"] = "/Root/Meshes/livingroom_491/table_lamp_0000"
    assert validate_room_package(package)["room_id"] == "room"
    assert missing_filesystem_paths(package) == []


def test_missing_absolute_path_is_reported(tmp_path):
    missing = tmp_path / "does_not_exist" / "lighting.json"
    present = tmp_path / "present.json"
    present.write_text("{}\n", encoding="utf-8")
    package = _package()
    package["source_assets"] = {
        "lighting": str(missing),
        "present": str(present),
        "ue_map": "/Game/Something",
    }
    errors = room_package_errors(package)
    joined = "; ".join(errors)
    assert str(missing) in joined
    assert str(present) not in joined
    assert missing_filesystem_paths(package) == [("source_assets.lighting", str(missing))]
