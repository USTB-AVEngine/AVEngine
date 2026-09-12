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
        "native_apartment.json": (
            "tmp/p3_room_packages_20260907_v3/floor_reference/", "measured"
        ),
        "room_a.json": (
            "examples/rooms/packages/floor_reference/room_a/", "depth_readback_fallback"
        ),
        "room_b.json": (
            "examples/rooms/packages/floor_reference/room_b/", "depth_readback_fallback"
        ),
        "room_c.json": (
            "examples/rooms/packages/floor_reference/room_c/", "depth_readback_fallback"
        ),
        "kujiale_0020_full_home_v1.json": (
            "examples/rooms/packages/floor_reference/kujiale_0020/", "depth_readback_fallback"
        ),
    }
    for filename, (prefix, status) in expected.items():
        package = json.loads((PACKAGE_ROOT / filename).read_text(encoding="utf-8"))
        reference = package["floor_reference"]
        assert isinstance(reference, dict)
        assert reference["status"] == status
        assert reference["path"].startswith(prefix)


def test_v1_room_runtime_profiles_agree_with_the_packages_they_describe():
    """A registered transport must not disagree with the room it names.

    The profile carries the render transport and the package carries the
    resources; a UE map named in both has to be the same map, or a run would
    plan against one scene and render another.
    """
    from avengine.rooms.room_providers import (
        BACKEND_RENDERERS, V1_PROFILE_ADAPTERS,
    )

    registry = json.loads(
        (ROOT / "examples/runtime/room_runtime_profiles.json").read_text(
            encoding="utf-8"))
    packages = {}
    for filename in EXPECTED:
        package = json.loads((PACKAGE_ROOT / filename).read_text(encoding="utf-8"))
        packages[package["room_id"]] = package

    v1 = [profile for profile in registry["profiles"]
          if str(profile["adapter_id"]) in V1_PROFILE_ADAPTERS]
    assert v1, "no V1 transport profile is registered"
    for profile in v1:
        room_id = profile["room_ref"]["room_id"]
        package = packages[room_id]
        assert BACKEND_RENDERERS[profile["backend_id"]] == package["renderer"]
        # V1 is 10 seconds of a fixed camera at the established rate.
        render = profile["render"]
        assert render["frame_count"] == 150
        assert render["frame_rate_hz"] == 15
        assert render["frame_count"] / render["frame_rate_hz"] == 10.0
        assert (render["width"], render["height"]) == (1280, 720)
        if package["renderer"] == "ue_spear":
            assert profile["scene"]["map_path"] == package["visual_scene"]["map_path"]
        else:
            # Habitat profiles name a room manifest, never a UE map.
            assert profile["scene"]["map_path"].endswith(".json")
            assert not profile["scene"]["map_path"].startswith("/Game/")


def test_kujiale_reaches_its_transport_without_a_faked_m6_record():
    """R1 reported this gap; R2 closes it through the real registry.

    Kujiale still has no M6 room record and none was invented for it. Its
    profile references the RoomPackage catalog, which is where the room is
    actually registered, and validate_room_runtime_links checks that
    reference against the catalog's identity, revision and package.
    """
    from avengine.rooms.room_providers import load_profile_registry, resolve_room_profile
    from avengine.runtime_profiles import validate_room_runtime_links

    m6_path = ROOT / "examples/registry/rooms/room_registry.json"
    m6 = json.loads(m6_path.read_text(encoding="utf-8"))
    # No kujiale record was added to the legacy registry.
    assert "kujiale_0020_full_home_v1" not in {
        record["room_id"] for record in m6["records"]}
    assert {record["provider_id"] for record in m6["records"]} <= {
        "blender_custom", "replica_cad", "legacy_ue_apartment", "matterport3d"}

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    profiles = load_profile_registry()
    profile = resolve_room_profile(profiles, "kujiale_0020_full_home_v1")
    assert profile is not None
    reference = profile["room_ref"]
    assert reference["registry_id"] == catalog["registry_id"]
    assert reference["revision"] == catalog["revision"]
    assert reference["room_id"] == "kujiale_0020_full_home_v1"
    assert validate_room_runtime_links(
        profiles, m6, room_package_catalog=catalog,
        room_package_catalog_path=CATALOG) == []


def test_all_four_production_rooms_have_a_registered_v1_transport():
    from avengine.rooms.room_providers import (
        V1_PROFILE_ADAPTERS, load_profile_registry,
    )

    profiles = load_profile_registry()
    v1 = {
        profile["room_ref"]["room_id"]
        for profile in profiles["profiles"]
        if str(profile["adapter_id"]) in V1_PROFILE_ADAPTERS
    }
    assert v1 == {
        "legacy_ue_apartment_0000_v1",
        "kujiale_0020_full_home_v1",
        "habitat_mp3d_example_17DRP5sb8fy",
        "hm3d_val_00800_TEEsavR23oF",
    }
