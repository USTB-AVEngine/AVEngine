"""CPU contract tests for the Apartment free-navigation handoff."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from avengine.capture.qa_plan_adapters import (
    load_planning_resources,
    planning_adapter_for_room,
)
from avengine.rooms.room_package import package_from_catalog_entry
from avengine.rooms.room_providers import planning_room_mapping
from avengine.rooms.walkable_space import HabitatWalkableSpace

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "examples/rooms/packages/catalog.json"
PACKAGE_PATH = ROOT / "examples/rooms/packages/native_apartment.json"


def _package() -> dict:
    return json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))


def test_apartment_defaults_to_free_habitat_navmesh_without_changing_ue_route():
    package = _package()
    walkable = package["walkable_space"]

    assert package["renderer"] == "ue_spear"
    assert package["planning_inputs"]["expected_stage_actor_count"] == 0
    assert walkable["kind"] == "habitat_navmesh"
    assert walkable["path"].startswith("${AVENGINE_QA_V3_ROOT}/")
    assert walkable["path"].endswith(
        "apartment_navigation_current_ue_v1/apartment_current_ue.navmesh"
    )
    assert walkable["source_manifest"].endswith(
        "apartment_navigation_current_ue_v1/build_result.json"
    )
    assert walkable["route_bank"].endswith("route_bank.json")
    room = {"room_id": package["room_id"], "room_package": package}
    assert planning_adapter_for_room(room, package) == "habitat_native_navmesh"


def test_route_bank_remains_an_explicit_compatibility_mode():
    package = deepcopy(_package())
    package["walkable_space"]["kind"] = "route_bank"
    package["walkable_space"]["path"] = package["walkable_space"]["route_bank"]
    room = {"room_id": package["room_id"], "room_package": package}

    assert planning_adapter_for_room(room, package) == "native_spear_route_bank"


def _native_runtime_from_environment() -> dict[str, str] | None:
    names = {
        "runtime_prefix": "AVENGINE_WP_B_RUNTIME_PREFIX",
        "magnum_python_site": "AVENGINE_WP_B_MAGNUM_SITE",
        "rlr_sdk_root": "AVENGINE_WP_B_RLR_SDK_ROOT",
    }
    values = {key: os.environ.get(env_name) for key, env_name in names.items()}
    return values if all(values.values()) else None


@pytest.mark.skipif(
    _native_runtime_from_environment() is None,
    reason="set AVENGINE_WP_B_RUNTIME_PREFIX, AVENGINE_WP_B_MAGNUM_SITE, and AVENGINE_WP_B_RLR_SDK_ROOT for the native CPU adapter probe",
)
def test_default_apartment_loads_habitat_pathfinder_and_keeps_spear_layout():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = next(
        row for row in catalog["rooms"]
        if row["room_id"] == "legacy_ue_apartment_0000_v1"
    )
    runtime = {
        "path_bindings": catalog["path_bindings"],
        **(_native_runtime_from_environment() or {}),
    }
    package = package_from_catalog_entry(
        entry, runtime=runtime, catalog_path=CATALOG
    )
    room = planning_room_mapping(package)
    request = {"room_catalog": str(CATALOG), "runtime": runtime}

    space, mesh, layout = load_planning_resources(room, request)

    assert isinstance(space, HabitatWalkableSpace)
    assert space.route_bank() is None
    assert space.metadata["authority"] == "native_apartment_recast_navmesh"
    assert Path(space.metadata["source_manifest"]).is_file()
    assert layout["backend_route"] == "spear_unreal"
    assert layout["native_navigation_mode"] == "habitat_navmesh"
    assert layout["native_navigation_authority"] == "native_apartment_recast_navmesh"
    assert mesh is not None
