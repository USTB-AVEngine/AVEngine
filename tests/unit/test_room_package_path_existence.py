"""RoomPackage existence checks run after path expansion."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.rooms.room_package import (
    REPOSITORY_ROOT,
    missing_filesystem_paths,
    package_from_catalog_entry,
    resolve_room_package_paths,
    room_package_errors,
    validate_room_package,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "examples/rooms/packages"
CATALOG = PACKAGE_ROOT / "catalog.json"


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


def test_unexpanded_template_is_not_a_filesystem_hit():
    package = _package()
    package["acoustic_package"] = "${AVENGINE_MISSING_H4_ROOT}/missing/manifest.json"
    assert missing_filesystem_paths(package) == []
    with pytest.raises(ValueError, match="AVENGINE_MISSING_H4_ROOT"):
        resolve_room_package_paths(package, runtime={"path_bindings": {}})


def test_existence_check_after_expansion_reports_missing_absolute(tmp_path):
    missing = tmp_path / "gone" / "manifest.json"
    package = _package()
    package["acoustic_package"] = "${AVENGINE_TEST_ROOT}/gone/manifest.json"
    resolved = resolve_room_package_paths(
        package, runtime={"path_bindings": {"AVENGINE_TEST_ROOT": str(tmp_path)}}
    )
    assert resolved["acoustic_package"] == str(missing)
    hits = missing_filesystem_paths(resolved)
    assert ("acoustic_package", str(missing)) in hits
    with pytest.raises(ValueError, match="missing path acoustic_package"):
        validate_room_package(resolved)


def test_relative_repo_path_is_checked_against_repository_root(tmp_path):
    present = "examples/rooms/packages/catalog.json"
    missing = "examples/rooms/packages/does_not_exist_h4.json"
    package = _package()
    package["semantics"] = {"path": present}
    assert missing_filesystem_paths(package) == []
    package["semantics"] = {"path": missing}
    assert missing_filesystem_paths(package) == [("semantics.path", missing)]
    extra = tmp_path / "catalog_dir"
    extra.mkdir()
    nested = extra / "nested"
    nested.mkdir()
    (nested / "local_only.json").write_text("{}\n", encoding="utf-8")
    package["semantics"] = {"path": "nested/local_only.json"}
    assert missing_filesystem_paths(package) == []
    catalog_hits = missing_filesystem_paths(package, relative_roots=[extra])
    assert ('semantics.path', 'nested/local_only.json') not in catalog_hits
    package['semantics'] = {'path': 'nested/gone.json'}
    catalog_missing = missing_filesystem_paths(package, relative_roots=[extra])
    assert ('semantics.path', 'nested/gone.json') in catalog_missing


def test_production_catalog_load_reports_zero_missing_from_any_cwd(tmp_path, monkeypatch):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    runtime = {"path_bindings": catalog["path_bindings"]}
    monkeypatch.chdir(tmp_path)
    rows = []
    for entry in catalog["rooms"]:
        package_path = ROOT / entry["room_package"]
        raw = json.loads(package_path.read_text(encoding="utf-8"))
        resolved = resolve_room_package_paths(raw, runtime=runtime)
        missing = missing_filesystem_paths(
            resolved, relative_roots=[ROOT, PACKAGE_ROOT]
        )
        validate_room_package(resolved, relative_roots=[ROOT, PACKAGE_ROOT])
        rows.append((entry["room_id"], Path(entry["room_package"]).name, missing))
    assert len(rows) == 7
    assert all(missing == [] for _, _, missing in rows)


def test_package_from_catalog_entry_validates_after_resolve():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    runtime = {"path_bindings": catalog["path_bindings"]}
    for entry in catalog["rooms"]:
        loaded = package_from_catalog_entry(entry, runtime=runtime)
        assert loaded["room_id"] == entry["room_id"]
        assert missing_filesystem_paths(
            loaded, relative_roots=[ROOT, PACKAGE_ROOT]
        ) == []
    assert REPOSITORY_ROOT == ROOT
