"""V1 room dispatch: one entry point resolves every production route.

The point of these checks is that adding a room is a registration, not a code
change. So they assert two things that are easy to lose: that all four
production routes come out of the same entry, and that a room_id the code has
never seen resolves to the same adapter as its backend siblings.

Where a test registers a temporary room_id, it is exercising the interface on
already-registered resources. It is not a new scene, and nothing here counts
toward production scene coverage or dataset admission.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.capture.qa_plan_adapters import (
    _expanded_room,
    capture_adapter_binding,
    planning_adapter_for_room,
    selected_scene_reference,
)
from avengine.rooms.room_package import (
    RUNTIME_KEY_ALIASES,
    canonical_runtime,
    package_from_catalog_entry,
    renderer_runtime_keys,
    resolve_room_runtime,
    room_capability_report,
)
from avengine.rooms.room_providers import (
    CAPTURE_ENTRYPOINTS,
    PLANNING_ADAPTERS,
    PRODUCTION_FAMILIES,
    RoomRouteError,
    catalog_room_ids,
    catalog_runtime,
    enumerate_catalog_rooms,
    load_room_catalog,
    planning_room_mapping,
    require_catalog_room,
    resolve_catalog_room,
    room_route,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "examples/rooms/packages/catalog.json"

# The four production routes and the adapter each must resolve to. A route
# that silently moves to another adapter is the failure this pins down.
PRODUCTION_ROUTES = {
    "legacy_ue_apartment_0000_v1": ("apartment", "ue_spear", "habitat_native_navmesh"),
    "kujiale_0020_full_home_v1": ("kujiale", "ue_spear", "habitat_native_navmesh"),
    "habitat_mp3d_example_17DRP5sb8fy": ("mp3d", "habitat", "habitat_native_navmesh"),
    "hm3d_val_00800_TEEsavR23oF": ("hm3d", "habitat", "habitat_native_navmesh"),
}
AUTHORED_COMPARISON = {
    "aea_loc3_social_rebuild_v1",
    "authored_compact_home_room_b_v1",
    "authored_open_family_home_room_c_v1",
}


@pytest.fixture(scope="module")
def catalog() -> dict:
    return load_room_catalog(CATALOG)


def _resolution(catalog: dict, room_id: str, runtime=None):
    return resolve_catalog_room(
        catalog, room_id, catalog_path=CATALOG,
        runtime=catalog_runtime(catalog, runtime),
    )


# --------------------------------------------------------------------------
# The four routes, from one entry point
# --------------------------------------------------------------------------


@pytest.mark.parametrize("room_id,expected", sorted(PRODUCTION_ROUTES.items()))
def test_production_route_resolves_from_the_unified_entry(catalog, room_id, expected):
    resolution = _resolution(catalog, room_id)
    assert resolution.route is not None, resolution.reason
    route = resolution.route
    assert (route.family, route.renderer, route.planning_adapter) == expected
    assert route.capture_entrypoint == CAPTURE_ENTRYPOINTS[route.renderer]
    assert (ROOT / route.capture_entrypoint).is_file()
    assert route.production_family is True
    # Declared resources resolve; only the executor parameters are absent,
    # and those are supplied per run rather than by the registration.
    assert resolution.resource_status == "pass", resolution.reason
    assert resolution.runtime["missing"] == renderer_runtime_keys(
        route.renderer)["required"]


def test_every_production_route_is_reachable_and_plan_ready(catalog):
    """Planning must not need the renderer's executor parameters."""
    for room_id in PRODUCTION_ROUTES:
        resolution = require_catalog_room(
            catalog, room_id, catalog_path=CATALOG, require_runtime=False)
        assert resolution.planning_room["room_id"] == room_id
        assert isinstance(resolution.planning_room["room_package"], dict)


def test_the_four_families_cover_the_declared_production_set(catalog):
    resolved = enumerate_catalog_rooms(
        catalog, catalog_path=CATALOG, production_only=True)
    assert {item.room_id for item in resolved} == set(PRODUCTION_ROUTES)
    assert {item.route.family for item in resolved} == set(PRODUCTION_FAMILIES)


def test_authored_rooms_resolve_but_are_not_a_production_route(catalog):
    """Retained comparison rooms stay readable without being counted."""
    for room_id in AUTHORED_COMPARISON:
        resolution = _resolution(catalog, room_id)
        assert resolution.route is not None
        assert resolution.route.family == "authored"
        assert resolution.route.production_family is False
    listed = {
        item.room_id for item in enumerate_catalog_rooms(
            catalog, catalog_path=CATALOG, production_only=True)
    }
    assert not (listed & AUTHORED_COMPARISON)


def test_route_family_map_admits_no_undeclared_backend(catalog):
    for room_id in catalog_room_ids(catalog):
        resolution = _resolution(catalog, room_id)
        assert resolution.route.renderer in CAPTURE_ENTRYPOINTS
        assert resolution.route.walkable_kind in PLANNING_ADAPTERS


# --------------------------------------------------------------------------
# A new room_id on a registered backend is a registration, not a code change
# --------------------------------------------------------------------------


def _temporary_room(tmp_path: Path, source_room_id: str, new_room_id: str,
                    *, mutate=None) -> tuple[Path, dict]:
    """Register `new_room_id` against an already-registered room's resources.

    Interface fixture only: the resources are the ones the source room already
    declares, so this adds no scene and claims no new coverage.
    """
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = next(
        item for item in catalog["rooms"] if item["room_id"] == source_room_id)
    package = json.loads(
        (ROOT / entry["room_package"]).read_text(encoding="utf-8"))
    package["room_id"] = new_room_id
    if mutate is not None:
        mutate(package)
    package_path = tmp_path / f"{new_room_id}.json"
    package_path.write_text(
        json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    fresh = {
        "schema": catalog["schema"],
        "revision": "p06_interface_fixture_not_a_scene",
        "claim_boundary": (
            "Temporary catalog for interface tests. It registers an extra "
            "room_id against already-registered resources and is not a new "
            "production scene."
        ),
        "path_bindings": catalog["path_bindings"],
        "rooms": [{
            "room_id": new_room_id,
            "family": package["family"],
            "renderer": package["renderer"],
            "room_package": package_path.name,
        }],
    }
    fresh_path = tmp_path / "catalog.json"
    fresh_path.write_text(
        json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    return fresh_path, fresh


@pytest.mark.parametrize("source_room_id", sorted(PRODUCTION_ROUTES))
def test_new_room_id_on_the_same_backend_needs_no_code_change(
        tmp_path, source_room_id):
    """The route must follow the declared backend, never the name."""
    new_room_id = "p06_interface_probe_room_v1"
    fresh_path, _fresh = _temporary_room(tmp_path, source_room_id, new_room_id)
    fresh_catalog = load_room_catalog(fresh_path)
    resolution = resolve_catalog_room(
        fresh_catalog, new_room_id, catalog_path=fresh_path,
        runtime=catalog_runtime(fresh_catalog, None))
    assert resolution.route is not None, resolution.reason
    family, renderer, adapter = PRODUCTION_ROUTES[source_room_id]
    assert resolution.route.room_id == new_room_id
    assert (resolution.route.family, resolution.route.renderer) == (family, renderer)
    assert resolution.route.planning_adapter == adapter
    assert resolution.route.capture_entrypoint == CAPTURE_ENTRYPOINTS[renderer]
    # Every declared resource still resolves under the new name.
    assert resolution.resource_status == "pass", resolution.reason
    # And the planning mapping the adapters read is shaped the same way.
    assert resolution.planning_room["room_id"] == new_room_id
    assert planning_adapter_for_room(
        resolution.planning_room, resolution.package) == adapter


@pytest.mark.parametrize("source_room_id", sorted(PRODUCTION_ROUTES))
def test_renaming_a_room_does_not_move_its_planning_adapter(source_room_id):
    """No production dispatch may key off room_id."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = next(
        item for item in catalog["rooms"] if item["room_id"] == source_room_id)
    runtime = {"path_bindings": catalog["path_bindings"]}
    package = package_from_catalog_entry(
        entry, runtime=runtime, catalog_path=CATALOG)
    original = planning_adapter_for_room(planning_room_mapping(package), package)
    renamed = dict(package, room_id="some_other_name_entirely_v9")
    assert planning_adapter_for_room(
        planning_room_mapping(renamed), renamed) == original
    assert room_route(renamed).planning_adapter == original
    assert original == PRODUCTION_ROUTES[source_room_id][2]


# --------------------------------------------------------------------------
# Missing resource, unsupported backend and wrong route each say why
# --------------------------------------------------------------------------


def test_missing_resource_names_the_absent_path(tmp_path):
    def break_navmesh(package):
        package["walkable_space"]["path"] = (
            "/data/definitely_absent_p06/no_such.navmesh")
        package["walkable_space"].pop("path_template", None)

    fresh_path, _ = _temporary_room(
        tmp_path, "hm3d_val_00800_TEEsavR23oF", "p06_missing_resource_v1",
        mutate=break_navmesh)
    fresh_catalog = load_room_catalog(fresh_path)
    resolution = resolve_catalog_room(
        fresh_catalog, "p06_missing_resource_v1", catalog_path=fresh_path,
        runtime=catalog_runtime(fresh_catalog, None))
    assert resolution.resource_status == "blocked"
    assert "no_such.navmesh" in (resolution.reason or "")
    with pytest.raises(RoomRouteError) as error:
        require_catalog_room(
            fresh_catalog, "p06_missing_resource_v1", catalog_path=fresh_path,
            require_runtime=False)
    assert "no_such.navmesh" in str(error.value)


def test_unsupported_walkable_kind_names_the_supported_kinds(tmp_path):
    def break_kind(package):
        package["walkable_space"]["kind"] = "hand_drawn_sketch"

    fresh_path, _ = _temporary_room(
        tmp_path, "hm3d_val_00800_TEEsavR23oF", "p06_bad_kind_v1",
        mutate=break_kind)
    # The package validator rejects the kind before routing does, and the
    # reason still has to name the field rather than a stray KeyError.
    fresh_catalog = load_room_catalog(fresh_path)
    resolution = resolve_catalog_room(
        fresh_catalog, "p06_bad_kind_v1", catalog_path=fresh_path,
        runtime=catalog_runtime(fresh_catalog, None))
    assert resolution.status in {"blocked", "fail"}
    assert "walkable_space" in (resolution.reason or "")


def test_unsupported_renderer_names_the_adapter_boundary():
    """A genuinely new backend needs an adapter, not a room_id branch."""
    package = {"family": "apartment", "renderer": "ue_spear",
               "room_id": "x", "walkable_space": {"kind": "route_bank"}}
    assert room_route(package).renderer == "ue_spear"
    with pytest.raises(ValueError) as error:
        room_route(dict(package, family="unknown_engine", renderer="blender_eevee"))
    assert "family" in str(error.value) or "renderer" in str(error.value)
    with pytest.raises(ValueError) as error:
        renderer_runtime_keys("blender_eevee")
    assert "supported renderers" in str(error.value)


def test_catalog_and_package_route_disagreement_is_reported(tmp_path):
    fresh_path, fresh = _temporary_room(
        tmp_path, "hm3d_val_00800_TEEsavR23oF", "p06_route_conflict_v1")
    fresh["rooms"][0]["renderer"] = "ue_spear"
    fresh_path.write_text(
        json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    fresh_catalog = load_room_catalog(fresh_path)
    resolution = resolve_catalog_room(
        fresh_catalog, "p06_route_conflict_v1", catalog_path=fresh_path,
        runtime=catalog_runtime(fresh_catalog, None))
    assert resolution.status == "fail"
    assert "disagree" in (resolution.reason or "")


def test_unknown_room_id_lists_the_registered_rooms(catalog):
    with pytest.raises(RoomRouteError) as error:
        require_catalog_room(catalog, "not_registered_v1", catalog_path=CATALOG)
    message = str(error.value)
    assert "not_registered_v1" in message
    for room_id in PRODUCTION_ROUTES:
        assert room_id in message


def test_catalog_without_the_v1_schema_is_refused(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"schema": "something_else", "rooms": []}))
    with pytest.raises(RoomRouteError) as error:
        load_room_catalog(path)
    assert "avengine_qa_room_package_catalog_v1" in str(error.value)


def test_duplicate_room_id_in_a_catalog_is_refused(tmp_path):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog["rooms"] = [catalog["rooms"][0], dict(catalog["rooms"][0])]
    path = tmp_path / "dupe.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(RoomRouteError) as error:
        load_room_catalog(path)
    assert "duplicate room_id" in str(error.value)


# --------------------------------------------------------------------------
# Runtime parameter pass-through
# --------------------------------------------------------------------------


@pytest.mark.parametrize("room_id", sorted(PRODUCTION_ROUTES))
def test_missing_runtime_parameters_are_named_not_guessed(catalog, room_id):
    resolution = _resolution(catalog, room_id)
    report = resolution.runtime
    required = renderer_runtime_keys(resolution.route.renderer)["required"]
    assert report["missing"] == required
    assert report["status"] == "blocked"
    for key in required:
        assert key in report["reason"]
    with pytest.raises(RoomRouteError) as error:
        require_catalog_room(
            catalog, room_id, catalog_path=CATALOG, require_runtime=True)
    assert required[0] in str(error.value)


def test_supplied_runtime_parameters_clear_the_blocker(catalog):
    runtime = {"runtime_prefix": "/data/example_prefix",
               "rlr_sdk_root": "/data/example_rlr"}
    resolution = _resolution(catalog, "hm3d_val_00800_TEEsavR23oF", runtime)
    assert resolution.runtime["missing"] == ()
    assert resolution.runtime["status"] == "pass"
    assert resolution.runtime["effective"]["runtime_prefix"] == "/data/example_prefix"
    assert resolution.status == "pass", resolution.reason


def test_historical_runtime_key_spellings_still_bind(catalog):
    """The stored hm3d template says magnum_site; the loader reads
    magnum_python_site. Accept the stored spelling instead of reporting the
    value as unset."""
    assert RUNTIME_KEY_ALIASES["magnum_site"] == "magnum_python_site"
    runtime = {"runtime_prefix": "/data/example_prefix",
               "magnum_site": "/data/example_magnum"}
    canonical = canonical_runtime(runtime)
    assert canonical["magnum_python_site"] == "/data/example_magnum"
    assert "magnum_site" not in canonical
    resolution = _resolution(catalog, "hm3d_val_00800_TEEsavR23oF", runtime)
    assert resolution.runtime["effective"]["magnum_python_site"] == "/data/example_magnum"
    assert resolution.runtime["aliased_keys"]["magnum_site"] == "magnum_python_site"


def test_request_runtime_overrides_a_package_declared_default():
    package = {"family": "hm3d", "renderer": "habitat", "room_id": "x",
               "walkable_space": {"kind": "habitat_navmesh"},
               "runtime": {"runtime_prefix": "/from/package",
                           "rlr_sdk_root": "/from/package/rlr"}}
    report = resolve_room_runtime(package, {"runtime_prefix": "/from/request"})
    assert report["effective"]["runtime_prefix"] == "/from/request"
    assert report["effective"]["rlr_sdk_root"] == "/from/package/rlr"
    assert report["missing"] == ()


def test_runtime_parameters_of_the_other_renderer_are_not_accepted():
    package = {"family": "hm3d", "renderer": "habitat", "room_id": "x",
               "walkable_space": {"kind": "habitat_navmesh"}}
    report = resolve_room_runtime(package, {"uproject": "/x.uproject"})
    assert "uproject" not in report["effective"]
    assert report["missing"] == ("runtime_prefix",)


# --------------------------------------------------------------------------
# Capability description and the selected-scene identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("room_id", sorted(PRODUCTION_ROUTES))
def test_capability_report_covers_every_dimension(catalog, room_id):
    resolution = _resolution(catalog, room_id)
    report = resolution.capabilities
    assert report["schema"] == "avengine_qa_room_capability_v1"
    assert report["room_id"] == room_id
    dimensions = report["dimensions"]
    for dimension in ("visual_scene", "navigation", "acoustics", "semantics",
                      "static_geometry", "floor_reference", "coordinate_frame",
                      "subrooms"):
        assert dimensions[dimension]["status"] == "pass", (
            dimension, dimensions[dimension]["reason"])
    assert dimensions["navigation"]["kind"] in PLANNING_ADAPTERS


def test_capability_report_marks_an_undeclared_dimension_not_run():
    package = {"family": "hm3d", "renderer": "habitat", "room_id": "x",
               "walkable_space": {"kind": "habitat_navmesh"},
               "visual_scene": {"scene_glb": "a", "dataset_config": "b",
                                "navmesh": "c"},
               "subrooms": []}
    report = room_capability_report(package)
    assert report["dimensions"]["acoustics"]["status"] == "not_run"
    assert "declares no acoustics" in report["dimensions"]["acoustics"]["reason"]
    assert report["dimensions"]["subrooms"]["status"] == "pass"
    assert report["dimensions"]["subrooms"]["subroom_count"] == 0


@pytest.mark.parametrize("room_id", sorted(PRODUCTION_ROUTES))
def test_capture_binding_names_the_scene_a_readback_must_match(catalog, room_id):
    resolution = _resolution(catalog, room_id)
    binding = capture_adapter_binding(resolution.package, None, repository=ROOT)
    assert binding["renderer"] == resolution.route.renderer
    assert Path(binding["entrypoint"]).is_file()
    scene = binding["selected_scene"]
    assert scene["room_id"] == room_id
    if binding["renderer"] == "ue_spear":
        assert scene["map_path"].startswith("/Game/")
    else:
        # The Habitat scene, dataset config and navmesh this run selected.
        assert scene["scene_glb"] and not scene["scene_glb"].endswith(".basis.glb")
        assert scene["dataset_config"] and scene["navmesh"]
        assert Path(scene["scene_glb"]).is_file()
        assert Path(scene["navmesh"]).is_file()
    assert selected_scene_reference(resolution.package) == scene


# --------------------------------------------------------------------------
# A raw catalog row reaches the planning adapters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("room_id", sorted(PRODUCTION_ROUTES))
def test_raw_catalog_row_is_expanded_before_dispatch(room_id):
    """A row whose room_package is a path used to reach the adapters as a
    string and fail with AttributeError four frames down."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = next(
        item for item in catalog["rooms"] if item["room_id"] == room_id)
    assert isinstance(entry["room_package"], str)
    request = {"runtime": {"path_bindings": catalog["path_bindings"]},
               "room_catalog": str(CATALOG)}
    expanded = _expanded_room(dict(entry), request)
    package = expanded["room_package"]
    assert isinstance(package, dict)
    assert package["room_id"] == room_id
    assert expanded["room_id"] == room_id
    assert planning_adapter_for_room(expanded, package) == (
        PRODUCTION_ROUTES[room_id][2])


def test_an_already_expanded_room_is_passed_through_unchanged():
    room = {"room_id": "x", "room_package": {"family": "hm3d"}}
    assert _expanded_room(room, {}) is room
    bare = {"room_id": "x", "manifest": "m.json"}
    assert _expanded_room(bare, {}) is bare


def test_a_room_package_of_the_wrong_type_says_so():
    with pytest.raises(ValueError) as error:
        _expanded_room({"room_id": "x", "room_package": 17}, {})
    assert "RoomPackage mapping or a path" in str(error.value)


# --------------------------------------------------------------------------
# The CLI reaches the same entry point
# --------------------------------------------------------------------------
#
# A library function nothing calls is not a connected route, so these drive
# tools/studio/run_qa_episode.py itself.


def _controller():
    import importlib.util
    import sys

    path = ROOT / "tools/studio/run_qa_episode.py"
    spec = importlib.util.spec_from_file_location(path.stem + "_p06_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def controller():
    return _controller()


def test_cli_lists_every_registered_room_with_its_route(controller):
    listing = controller.list_rooms(CATALOG)
    assert listing["schema"] == "avengine_qa_room_registry_listing_v1"
    assert listing["native_execution"] == "not_run"
    by_id = {room["room_id"]: room for room in listing["rooms"]}
    assert set(by_id) == set(PRODUCTION_ROUTES) | AUTHORED_COMPARISON
    for room_id, (family, renderer, adapter) in PRODUCTION_ROUTES.items():
        route = by_id[room_id]["route"]
        assert (route["family"], route["renderer"]) == (family, renderer)
        assert route["planning_adapter"] == adapter
        assert by_id[room_id]["resource_status"] == "pass"


def test_cli_production_only_lists_exactly_the_four_routes(controller):
    listing = controller.list_rooms(CATALOG, production_only=True)
    assert {room["room_id"] for room in listing["rooms"]} == set(PRODUCTION_ROUTES)


def test_cli_resolves_one_room_with_its_capture_adapter(controller):
    report = controller.resolve_room(CATALOG, "kujiale_0020_full_home_v1")
    assert report["route"]["planning_adapter"] == "habitat_native_navmesh"
    adapter = report["capture_adapter"]
    assert adapter["entrypoint_repository_relative"] == (
        "tools/rooms/run_spear_residential_episode.py")
    assert adapter["selected_scene"]["map_path"].startswith("/Game/")
    # Plan-only resolution still says which executor parameters are absent.
    assert set(adapter["runtime"]["missing"]) == {
        "uproject", "unreal_editor", "spear_ext_dir"}
    assert report["planning_room"]["room_id"] == "kujiale_0020_full_home_v1"


def test_cli_resolve_for_execution_refuses_without_runtime(controller):
    with pytest.raises(RoomRouteError) as error:
        controller.resolve_room(
            CATALOG, "kujiale_0020_full_home_v1", for_execution=True)
    assert "uproject" in str(error.value)


def test_cli_capture_entrypoint_uses_the_shared_route_table(controller):
    for renderer, relative in CAPTURE_ENTRYPOINTS.items():
        assert controller.renderer_capture_entrypoint(renderer) == ROOT / relative
    with pytest.raises(Exception) as error:
        controller.renderer_capture_entrypoint("blender_eevee")
    assert "supported renderers" in str(error.value)


def test_cli_capture_command_carries_the_selected_ue_room(controller, tmp_path):
    """The UE command line must be built from this room's own parameters.

    Software boundary only: it checks the launch arguments, and no Unreal
    process runs here.
    """
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    runtime = {
        "path_bindings": catalog["path_bindings"],
        "uproject": "/data/example_stage/SpearSim/SpearSim.uproject",
        "unreal_editor": "/data/example_engine/UnrealEditor",
        "spear_ext_dir": "/data/example_spear_ext",
        "graphics_adapter": 2,
    }
    resolution = require_catalog_room(
        catalog, "kujiale_0020_full_home_v1", catalog_path=CATALOG,
        runtime=catalog_runtime(catalog, runtime), require_runtime=True)
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "room_package.json").write_text(
        json.dumps(resolution.package, default=str), encoding="utf-8")
    (plan_dir / "episode_plan.json").write_text(json.dumps({
        "plan_coordinates": "renderer_neutral",
        "visual_plan": {"camera": {"resolution_hw": [720, 1280]}},
        "resources": {"expected_stage_actor_count": 2},
    }), encoding="utf-8")
    request = {"runtime": runtime, "room_catalog": str(CATALOG)}
    command = controller.capture_command(request, tmp_path)
    assert str(ROOT / CAPTURE_ENTRYPOINTS["ue_spear"]) in command
    for key in ("uproject", "unreal_editor", "spear_ext_dir"):
        assert runtime[key] in command
    assert "--graphics-adapter" in command
    assert command[command.index("--graphics-adapter") + 1] == "2"
    assert command[command.index("--width") + 1] == "1280"
    assert command[command.index("--height") + 1] == "720"
    assert command[command.index("--expected-stage-actor-count") + 1] == "2"


def test_cli_capture_command_carries_the_selected_habitat_room(controller, tmp_path):
    """The Habitat command line must name this room's materialized inputs."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    runtime = {
        "path_bindings": catalog["path_bindings"],
        "runtime_prefix": "/data/example_prefix",
        "rlr_sdk_root": "/data/example_rlr",
        "graphics_adapter": 2,
    }
    resolution = require_catalog_room(
        catalog, "hm3d_val_00800_TEEsavR23oF", catalog_path=CATALOG,
        runtime=catalog_runtime(catalog, runtime), require_runtime=True)
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "room_package.json").write_text(
        json.dumps(resolution.package, default=str), encoding="utf-8")
    clock = {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000,
             "sample_count": 160000, "time_base_hz": 48000, "ticks_per_frame": 3200}
    (plan_dir / "episode_plan.json").write_text(json.dumps({
        "plan_coordinates": "renderer_neutral", "clock": clock, "resources": {},
    }), encoding="utf-8")
    (plan_dir / "habitat_room_manifest.json").write_text("{}", encoding="utf-8")
    execution = plan_dir / "habitat_execution"
    execution.mkdir()
    (execution / "case_manifest.json").write_text(
        json.dumps({"clock": clock}), encoding="utf-8")
    (execution / "m1_capture_request.json").write_text("{}", encoding="utf-8")
    request = {"runtime": runtime, "room_catalog": str(CATALOG)}
    command = controller.capture_command(request, tmp_path)
    assert str(ROOT / CAPTURE_ENTRYPOINTS["habitat"]) in command
    assert "--runtime-prefix" in command
    assert command[command.index("--runtime-prefix") + 1] == "/data/example_prefix"
    assert command[command.index("--rlr-sdk-root") + 1] == "/data/example_rlr"
    assert command[command.index("--gpu-device-id") + 1] == "2"
    assert str(plan_dir / "habitat_room_manifest.json") in command


# --------------------------------------------------------------------------
# R1: portable host runtime, registered render transport, process isolation
# --------------------------------------------------------------------------
#
# The gap R1 closes is that a room's resources were registered but the
# machine's executor parameters were not, so every request had to restate
# them. These checks pin the three axes apart and pin the precedence.

from avengine.rooms.room_package import (
    HOST_RUNTIME_CONFIG_SCHEMA,
    HOST_RUNTIME_ENVIRONMENT,
    RUNTIME_ISOLATION_KEYS,
    host_runtime_layers,
    load_host_runtime_config,
    runtime_isolation_key,
)
from avengine.rooms.room_providers import (
    BACKEND_RENDERERS,
    V1_PROFILE_ADAPTERS,
    load_profile_registry,
    profile_route_conflicts,
    resolve_room_profile,
    room_render_parameters,
    runtime_isolation_groups,
)

# V1's fixed transport: a static camera, 10 seconds at 15 Hz.
V1_FRAME_COUNT = 150
V1_FRAME_RATE_HZ = 15
V1_REQUEST = {
    "frame_count": V1_FRAME_COUNT,
    "frame_rate_hz": V1_FRAME_RATE_HZ,
    "camera": {"motion": "static", "fov_deg": 85.0},
}
# Placeholder host values: these tests check resolution and precedence, not
# whether a particular machine has these directories.
FAKE_HOST = {
    "schema": HOST_RUNTIME_CONFIG_SCHEMA,
    "renderers": {
        "ue_spear": {
            "uproject": "/host/stage/SpearSim/SpearSim.uproject",
            "unreal_editor": "/host/engine/UnrealEditor",
            "spear_ext_dir": "/host/spear_ext",
            "graphics_adapter": 2,
        },
        "habitat": {"magnum_python_site": "/host/magnum/site-packages"},
    },
    "rooms": {
        "habitat_mp3d_example_17DRP5sb8fy": {
            "runtime_prefix": "/host/prefix/mp3d",
            "rlr_sdk_root": "/host/rlr/runtime_b",
            "mp3d_root": "/host/datasets/mp3d",
        },
        "hm3d_val_00800_TEEsavR23oF": {
            "runtime_prefix": "/host/prefix/hm3d",
            "rlr_sdk_root": "/host/rlr/base",
        },
    },
}


@pytest.fixture
def host_config(tmp_path):
    path = tmp_path / "host_runtime.json"
    path.write_text(json.dumps(FAKE_HOST), encoding="utf-8")
    return load_host_runtime_config(path)


@pytest.fixture(scope="module")
def profiles():
    return load_profile_registry()


def _resolved_with_host(catalog, room_id, host_config, **kwargs):
    return resolve_catalog_room(
        catalog, room_id, catalog_path=CATALOG,
        runtime=catalog_runtime(catalog, kwargs.pop("runtime", None)),
        host_config=host_config, request=V1_REQUEST, **kwargs)


# --- the distributable examples stay portable -----------------------------


_PRIVATE_PATH_PREFIXES = ('"/data/', '"/home/', '"/mnt/', '"/opt/', '"/srv/')


def _private_path_lines(path):
    return [
        f"{path.name}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if any(prefix in line for prefix in _PRIVATE_PATH_PREFIXES)
    ]


def test_profile_registry_and_room_packages_carry_no_private_server_path():
    """AGENTS.md forbids private absolute paths in current configuration.

    These are shipped files, so a real host path in one would travel to every
    checkout. Scenes use /Game, a repository-relative path or an
    ${AVENGINE_*} override; host paths belong in a run-local host runtime
    config instead.
    """
    shipped = [ROOT / "examples/runtime/room_runtime_profiles.json"]
    shipped += sorted((ROOT / "examples/rooms/packages").glob("*.json"))
    offenders = [
        line for path in shipped if path.name != "catalog.json"
        for line in _private_path_lines(path)
    ]
    assert offenders == [], offenders


def test_catalog_confines_its_declared_roots_to_path_bindings():
    """The catalog ships default roots for this server in one block.

    That block is the remaining deviation from the no-private-path rule and it
    is A0's to decide on; a caller already overrides it per key through
    catalog_runtime. What this pins down is that the deviation stays confined
    to path_bindings instead of spreading through the catalog.
    """
    catalog_path = ROOT / "examples/rooms/packages/catalog.json"
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    without_bindings = json.dumps(
        {key: value for key, value in raw.items() if key != "path_bindings"})
    for prefix in _PRIVATE_PATH_PREFIXES:
        assert prefix not in without_bindings, prefix
    # Overriding one root must not require restating the rest.
    overridden = catalog_runtime(raw, {"path_bindings": {"AVENGINE_HM3D_ROOT": "/x"}})
    assert overridden["path_bindings"]["AVENGINE_HM3D_ROOT"] == "/x"
    assert overridden["path_bindings"]["AVENGINE_MP3D_ROOT"] == (
        raw["path_bindings"]["AVENGINE_MP3D_ROOT"])


def test_v1_profiles_reference_only_relative_or_override_scenes(profiles):
    for profile in profiles["profiles"]:
        if str(profile.get("adapter_id")) not in V1_PROFILE_ADAPTERS:
            continue
        map_path = profile["scene"]["map_path"]
        assert (
            map_path.startswith("/Game/")
            or map_path.startswith("${")
            or not map_path.startswith("/")
        ), (profile["profile_id"], map_path)


# --- the registered render transport --------------------------------------


def test_v1_profiles_keep_the_older_consumers_intact(profiles):
    """spear_apartment_v1 pins a 75-frame transport and validates it exactly.

    A V1 profile therefore registers its own adapter instead of widening that
    one, and the registry default stays the profile the old importer resolves.
    """
    assert profiles["default_profile_id"] == "spear_apartment_0000"
    by_id = {str(p["profile_id"]): p for p in profiles["profiles"]}
    legacy = by_id["spear_apartment_0000"]
    assert legacy["adapter_id"] == "spear_apartment_v1"
    assert legacy["render"]["frame_count"] == 75
    for profile in profiles["profiles"]:
        if str(profile["adapter_id"]) in V1_PROFILE_ADAPTERS:
            assert profile["render"]["frame_count"] == V1_FRAME_COUNT
            assert profile["render"]["frame_rate_hz"] == V1_FRAME_RATE_HZ
            assert profile["adapter_id"] != "spear_apartment_v1"


def test_every_profile_resolves_into_one_of_the_two_room_registries(profiles):
    """Both room registries are checked, and each strictly.

    Rooms with an M6 record keep referencing it. Kujiale has no M6 record, so
    it references the RoomPackage catalog it is actually registered in, and
    that reference is checked against the catalog's own identity, revision and
    a package that really loads.
    """
    from avengine.runtime_profiles import validate_room_runtime_links

    room_registry = json.loads(
        (ROOT / "examples/registry/rooms/room_registry.json").read_text(
            encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert validate_room_runtime_links(
        profiles, room_registry,
        room_package_catalog=catalog, room_package_catalog_path=CATALOG) == []
    by_registry = {}
    for profile in profiles["profiles"]:
        by_registry.setdefault(
            profile["room_ref"]["registry_id"], []).append(
                str(profile["profile_id"]))
    assert set(by_registry) == {
        room_registry["registry_id"], catalog["registry_id"]}
    assert by_registry[catalog["registry_id"]] == [
        "qa_v1_spear_kujiale_0020_full_home"]


@pytest.mark.parametrize("room_id,profile_id", [
    ("legacy_ue_apartment_0000_v1", "qa_v1_spear_apartment_0000"),
    ("kujiale_0020_full_home_v1", "qa_v1_spear_kujiale_0020_full_home"),
    ("habitat_mp3d_example_17DRP5sb8fy", "qa_v1_habitat_mp3d_17DRP5sb8fy"),
    ("hm3d_val_00800_TEEsavR23oF", "qa_v1_habitat_hm3d_val_00800_TEEsavR23oF"),
])
def test_profiled_room_resolves_its_registered_transport(
        catalog, profiles, host_config, room_id, profile_id):
    resolution = _resolved_with_host(
        catalog, room_id, host_config, profile_registry=profiles)
    assert resolution.render["profile_id"] == profile_id
    effective = resolution.render["effective"]
    assert effective["width"] == 1280 and effective["height"] == 720
    assert effective["frame_count"] == V1_FRAME_COUNT
    assert effective["horizontal_fov_deg"] == 85.0
    # Warmups have no request source, so they come from the registration.
    assert resolution.render["provenance"]["streaming_warmup_frames"] == (
        "room_runtime_profile")
    assert resolution.render["conflicts"] == ()
    assert profile_route_conflicts(resolution.profile, resolution.package) == ()


def test_a_room_without_a_registered_profile_still_resolves(
        tmp_path, profiles, host_config):
    """A registered transport must not be a precondition for routing.

    All four production rooms now have one, so this uses a freshly registered
    room_id instead: an interface fixture on already-registered resources, not
    a new scene.
    """
    new_room_id = "p06r2_unprofiled_probe_room_v1"
    fresh_path, _ = _temporary_room(
        tmp_path, "kujiale_0020_full_home_v1", new_room_id)
    fresh_catalog = load_room_catalog(fresh_path)
    resolution = resolve_catalog_room(
        fresh_catalog, new_room_id, catalog_path=fresh_path,
        runtime=catalog_runtime(fresh_catalog, None),
        profile_registry=profiles, host_config=host_config, request=V1_REQUEST)
    assert resolution.profile is None
    assert resolution.render["profile_id"] is None
    assert resolution.route.planning_adapter == "habitat_native_navmesh"
    assert resolution.status == "pass", resolution.reason
    # The request still supplies the transport it states.
    assert resolution.render["effective"]["frame_count"] == V1_FRAME_COUNT
    assert resolution.render["provenance"]["frame_count"] == "request"


def test_kujiale_now_has_its_registered_transport(catalog, profiles, host_config):
    """R2 closes the registration gap R1 reported.

    Kujiale reaches its transport through the RoomPackage catalog rather than
    an M6 record, and the transport itself is the same V1 one the other three
    rooms use.
    """
    resolution = _resolved_with_host(
        catalog, "kujiale_0020_full_home_v1", host_config,
        profile_registry=profiles)
    assert resolution.render["profile_id"] == "qa_v1_spear_kujiale_0020_full_home"
    assert resolution.profile["room_ref"]["registry_id"] == (
        "avengine_qa_room_packages_v1")
    effective = resolution.render["effective"]
    assert effective["frame_count"] == V1_FRAME_COUNT
    assert effective["frame_rate_hz"] == V1_FRAME_RATE_HZ
    assert (effective["width"], effective["height"]) == (1280, 720)
    assert resolution.render["conflicts"] == ()
    assert profile_route_conflicts(resolution.profile, resolution.package) == ()
    assert resolution.status == "pass", resolution.reason


def test_request_overrides_the_registered_transport_and_says_so(profiles):
    profile = resolve_room_profile(profiles, "legacy_ue_apartment_0000_v1")
    render = room_render_parameters(profile, {"frame_count": 90})
    assert render["effective"]["frame_count"] == 90
    assert render["provenance"]["frame_count"] == "request"
    assert any("frame_count" in item for item in render["conflicts"])
    # Untouched keys stay registered rather than being dropped.
    assert render["provenance"]["width"] == "room_runtime_profile"


def test_profile_disagreeing_with_the_package_route_is_reported(catalog, profiles):
    entry = next(item for item in catalog["rooms"]
                 if item["room_id"] == "legacy_ue_apartment_0000_v1")
    package = package_from_catalog_entry(
        entry, runtime=catalog_runtime(catalog, None), catalog_path=CATALOG)
    profile = dict(resolve_room_profile(profiles, "legacy_ue_apartment_0000_v1"))
    profile["backend_id"] = "habitat_native"
    problems = profile_route_conflicts(profile, package)
    assert problems and "renderer" in problems[0]
    profile = dict(resolve_room_profile(profiles, "legacy_ue_apartment_0000_v1"))
    profile["scene"] = dict(profile["scene"], map_path="/Game/Somewhere/Else")
    assert any("map_path" in item for item in
               profile_route_conflicts(profile, package))


def test_unknown_profile_id_and_unmapped_backend_are_named(profiles):
    with pytest.raises(RoomRouteError) as error:
        resolve_room_profile(profiles, "legacy_ue_apartment_0000_v1",
                             profile_id="no_such_profile")
    assert "no_such_profile" in str(error.value)
    package = {"family": "hm3d", "renderer": "habitat", "room_id": "x",
               "walkable_space": {"kind": "habitat_navmesh"}}
    problems = profile_route_conflicts(
        {"backend_id": "unity_hdrp", "scene": {}}, package)
    assert problems and "backend_id" in problems[0]
    assert set(BACKEND_RENDERERS.values()) == {"ue_spear", "habitat"}


# --- host runtime precedence and provenance -------------------------------


@pytest.mark.parametrize("room_id", sorted(PRODUCTION_ROUTES))
def test_host_config_clears_every_executor_blocker(
        catalog, profiles, host_config, room_id):
    """All four rooms become executable from the host config alone."""
    resolution = _resolved_with_host(
        catalog, room_id, host_config, profile_registry=profiles)
    assert resolution.runtime["missing"] == ()
    assert resolution.status == "pass", resolution.reason
    for key in resolution.runtime["required"]:
        assert resolution.runtime["provenance"][key].startswith("host_config:")
    require_catalog_room(
        catalog, room_id, catalog_path=CATALOG,
        runtime=catalog_runtime(catalog, None), profile_registry=profiles,
        host_config=host_config, request=V1_REQUEST, require_runtime=True)


def test_per_room_scope_beats_per_renderer_scope(catalog, profiles, host_config):
    """MP3D and HM3D pin different Habitat prefixes; the room scope decides."""
    mp3d = _resolved_with_host(
        catalog, "habitat_mp3d_example_17DRP5sb8fy", host_config,
        profile_registry=profiles)
    hm3d = _resolved_with_host(
        catalog, "hm3d_val_00800_TEEsavR23oF", host_config,
        profile_registry=profiles)
    assert mp3d.runtime["effective"]["runtime_prefix"] == "/host/prefix/mp3d"
    assert hm3d.runtime["effective"]["runtime_prefix"] == "/host/prefix/hm3d"
    assert mp3d.runtime["provenance"]["runtime_prefix"].endswith(
        "#rooms.habitat_mp3d_example_17DRP5sb8fy")
    # The shared value still comes from the renderer scope for both.
    for resolution in (mp3d, hm3d):
        assert resolution.runtime["effective"]["magnum_python_site"] == (
            "/host/magnum/site-packages")
        assert resolution.runtime["provenance"]["magnum_python_site"].endswith(
            "#renderers.habitat")


def test_request_beats_host_config(catalog, profiles, host_config):
    resolution = _resolved_with_host(
        catalog, "hm3d_val_00800_TEEsavR23oF", host_config,
        profile_registry=profiles,
        runtime={"runtime_prefix": "/from/the/request"})
    assert resolution.runtime["effective"]["runtime_prefix"] == "/from/the/request"
    assert resolution.runtime["provenance"]["runtime_prefix"] == "request"


def test_host_config_layers_are_ordered_least_specific_first(host_config):
    layers = host_runtime_layers(
        host_config, "habitat", "hm3d_val_00800_TEEsavR23oF")
    sources = [source for source, _mapping in layers]
    assert len(sources) == 3
    assert sources[1].endswith("#renderers.habitat")
    assert sources[2].endswith("#rooms.hm3d_val_00800_TEEsavR23oF")


def test_environment_is_opt_in_and_only_for_established_variables(
        catalog, profiles):
    environment = {
        "AVENGINE_HABITAT_RUNTIME_PREFIX": "/env/prefix",
        "AVENGINE_RLR_SDK_ROOT": "/env/rlr",
    }
    without = resolve_catalog_room(
        catalog, "hm3d_val_00800_TEEsavR23oF", catalog_path=CATALOG,
        runtime=catalog_runtime(catalog, None), profile_registry=profiles,
        environment=environment, request=V1_REQUEST)
    assert without.runtime["missing"] == ("runtime_prefix",)
    assert without.runtime["environment_allowed"] is False
    with_env = resolve_catalog_room(
        catalog, "hm3d_val_00800_TEEsavR23oF", catalog_path=CATALOG,
        runtime=catalog_runtime(catalog, None), profile_registry=profiles,
        environment=environment, allow_environment=True, request=V1_REQUEST)
    assert with_env.runtime["effective"]["runtime_prefix"] == "/env/prefix"
    assert with_env.runtime["provenance"]["runtime_prefix"] == (
        "environment:AVENGINE_HABITAT_RUNTIME_PREFIX")
    # No UE key is mapped: nothing in this repository resolves a uproject or
    # an editor binary from the environment.
    assert set(HOST_RUNTIME_ENVIRONMENT) == {
        "runtime_prefix", "magnum_python_site", "mp3d_root", "rlr_sdk_root"}


def test_environment_loses_to_host_config_and_request(catalog, profiles, host_config):
    resolution = resolve_catalog_room(
        catalog, "hm3d_val_00800_TEEsavR23oF", catalog_path=CATALOG,
        runtime=catalog_runtime(catalog, None), profile_registry=profiles,
        host_config=host_config, allow_environment=True, request=V1_REQUEST,
        environment={"AVENGINE_HABITAT_RUNTIME_PREFIX": "/env/prefix"})
    assert resolution.runtime["effective"]["runtime_prefix"] == "/host/prefix/hm3d"


def test_malformed_host_runtime_config_is_refused(tmp_path):
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"schema": "something_else"}))
    with pytest.raises(ValueError) as error:
        load_host_runtime_config(bad_schema)
    assert HOST_RUNTIME_CONFIG_SCHEMA in str(error.value)
    bad_scope = tmp_path / "bad_scope.json"
    bad_scope.write_text(json.dumps({"rooms": ["not", "a", "mapping"]}))
    with pytest.raises(ValueError) as error:
        load_host_runtime_config(bad_scope)
    assert "rooms" in str(error.value)


# --- P18 process isolation ------------------------------------------------


def test_isolation_groups_split_the_two_habitat_prefixes(
        catalog, profiles, host_config):
    resolutions = enumerate_catalog_rooms(
        catalog, catalog_path=CATALOG, production_only=True,
        runtime=catalog_runtime(catalog, None), profile_registry=profiles,
        host_config=host_config, request=V1_REQUEST)
    groups = runtime_isolation_groups(resolutions)
    by_rooms = {tuple(group["room_ids"]): group for group in groups}
    assert len(groups) == 3, [group["room_ids"] for group in groups]
    # The two UE rooms share one runtime and may share one process.
    ue = by_rooms[("kujiale_0020_full_home_v1", "legacy_ue_apartment_0000_v1")]
    assert ue["renderer"] == "ue_spear"
    assert ue["requires_fresh_interpreter"] is False
    # MP3D and HM3D pin different prefixes, so they cannot.
    mp3d = by_rooms[("habitat_mp3d_example_17DRP5sb8fy",)]
    hm3d = by_rooms[("hm3d_val_00800_TEEsavR23oF",)]
    assert mp3d["isolation_values"]["runtime_prefix"] != (
        hm3d["isolation_values"]["runtime_prefix"])
    for group in (mp3d, hm3d):
        assert group["requires_fresh_interpreter"] is True
        assert "fork" in group["reason"]


def test_rooms_sharing_one_prefix_share_one_group(catalog, profiles, tmp_path):
    """Grouping follows the resolved runtime, not the room family."""
    shared = dict(FAKE_HOST)
    shared["rooms"] = {
        room_id: {"runtime_prefix": "/host/prefix/shared",
                  "rlr_sdk_root": "/host/rlr/shared"}
        for room_id in ("habitat_mp3d_example_17DRP5sb8fy",
                        "hm3d_val_00800_TEEsavR23oF")
    }
    path = tmp_path / "shared_host.json"
    path.write_text(json.dumps(shared), encoding="utf-8")
    resolutions = enumerate_catalog_rooms(
        catalog, catalog_path=CATALOG, production_only=True,
        runtime=catalog_runtime(catalog, None), profile_registry=profiles,
        host_config=load_host_runtime_config(path), request=V1_REQUEST)
    groups = runtime_isolation_groups(resolutions)
    habitat = [g for g in groups if g["renderer"] == "habitat"]
    assert len(habitat) == 1
    assert habitat[0]["room_ids"] == [
        "habitat_mp3d_example_17DRP5sb8fy", "hm3d_val_00800_TEEsavR23oF"]


def test_isolation_key_is_the_renderer_plus_its_runtime_identity(
        catalog, profiles, host_config):
    mp3d = _resolved_with_host(
        catalog, "habitat_mp3d_example_17DRP5sb8fy", host_config,
        profile_registry=profiles)
    key = runtime_isolation_key(mp3d.runtime)
    assert key[0] == "habitat"
    assert "/host/prefix/mp3d" in key
    assert RUNTIME_ISOLATION_KEYS["habitat"] == (
        "runtime_prefix", "magnum_python_site", "rlr_sdk_root")
    assert RUNTIME_ISOLATION_KEYS["ue_spear"] == (
        "uproject", "unreal_editor", "spear_ext_dir")


# --------------------------------------------------------------------------
# R2: catalog-backed room references, external roots, camera pass-through
# --------------------------------------------------------------------------

from avengine.rooms.room_package import host_runtime_path_bindings
from avengine.rooms.room_providers import (
    catalog_binding_requirements,
    effective_camera_request,
    request_with_effective_camera,
)

# conditioned_sampler.select_camera_and_schedule falls back to this when the
# request's camera block states no field of view.
_SAMPLER_FOV_DEFAULT = 85.0


def _sampler_field_of_view(request):
    """Exactly what conditioned_sampler computes from a request."""
    camera = request.get("camera", {})
    return float(camera.get(
        "fov_deg", request.get("camera_fov_deg", _SAMPLER_FOV_DEFAULT)))


def test_an_explicit_field_of_view_reaches_the_sampler_unchanged(profiles):
    profile = resolve_room_profile(profiles, "kujiale_0020_full_home_v1")
    request = {"camera": {"motion": "static", "fov_deg": 62.0}}
    render = room_render_parameters(profile, request)
    updated = request_with_effective_camera(request, render)
    assert _sampler_field_of_view(updated) == 62.0
    assert updated["camera_fov_deg"] == 62.0
    # An override of the registration is reported, never silent.
    assert any("horizontal_fov_deg" in item for item in render["conflicts"])


def test_a_registered_field_of_view_is_not_replaced_by_the_sampler_default(
        profiles):
    """The room's registration must win over the sampler's own fallback.

    A room registered at anything other than 85 used to render at 85 whenever
    the request said nothing, because the fallback lived downstream.
    """
    profile = deepcopy(dict(resolve_room_profile(
        profiles, "kujiale_0020_full_home_v1")))
    profile["render"] = dict(profile["render"], horizontal_fov_deg=97.0)
    request = {"camera": {"motion": "static"}}
    assert _sampler_field_of_view(request) == _SAMPLER_FOV_DEFAULT
    render = room_render_parameters(profile, request)
    updated = request_with_effective_camera(request, render)
    assert _sampler_field_of_view(updated) == 97.0


@pytest.mark.parametrize("room_id", sorted(PRODUCTION_ROUTES))
def test_v1_keeps_its_own_85_degrees_for_every_production_room(profiles, room_id):
    profile = resolve_room_profile(profiles, room_id)
    render = room_render_parameters(profile, dict(V1_REQUEST))
    updated = request_with_effective_camera(V1_REQUEST, render)
    assert _sampler_field_of_view(updated) == 85.0
    assert updated["camera"]["resolution_hw"] == [720, 1280]
    assert updated["camera"]["motion"] == "static"
    assert render["conflicts"] == ()


def test_a_stated_resolution_is_not_overwritten_by_the_registration(profiles):
    profile = resolve_room_profile(profiles, "legacy_ue_apartment_0000_v1")
    render = room_render_parameters(profile, {})
    camera = effective_camera_request(
        render, {"camera": {"resolution_hw": [480, 640]}})
    assert camera["resolution_hw"] == [480, 640]


# --- external roots may come from the host instead of the catalog ---------


def test_host_config_supplies_external_roots_under_the_request(host_config, catalog):
    host = dict(host_config)
    host["path_bindings"] = {"AVENGINE_HM3D_ROOT": "/from/the/host"}
    assert host_runtime_path_bindings(host, "habitat") == {
        "AVENGINE_HM3D_ROOT": "/from/the/host"}
    # Catalog default loses to the host, and the host loses to the request.
    from_host = catalog_runtime(catalog, None, host_config=host, renderer="habitat")
    assert from_host["path_bindings"]["AVENGINE_HM3D_ROOT"] == "/from/the/host"
    assert from_host["path_bindings"]["AVENGINE_MP3D_ROOT"] == (
        catalog["path_bindings"]["AVENGINE_MP3D_ROOT"])
    from_request = catalog_runtime(
        catalog, {"path_bindings": {"AVENGINE_HM3D_ROOT": "/from/the/request"}},
        host_config=host, renderer="habitat")
    assert from_request["path_bindings"]["AVENGINE_HM3D_ROOT"] == "/from/the/request"


def test_per_room_scope_can_retarget_one_root(catalog):
    host = {
        "_source": "/run/host.json",
        "path_bindings": {"AVENGINE_HM3D_ROOT": "/host/wide"},
        "rooms": {"hm3d_val_00800_TEEsavR23oF": {
            "path_bindings": {"AVENGINE_HM3D_ROOT": "/host/for/this/room"}}},
    }
    assert host_runtime_path_bindings(
        host, "habitat", "hm3d_val_00800_TEEsavR23oF") == {
            "AVENGINE_HM3D_ROOT": "/host/for/this/room"}
    assert host_runtime_path_bindings(host, "habitat", "other_room") == {
        "AVENGINE_HM3D_ROOT": "/host/wide"}


def test_a_malformed_host_path_bindings_block_is_refused():
    with pytest.raises(ValueError) as error:
        host_runtime_path_bindings(
            {"_source": "/run/host.json", "path_bindings": ["not", "a", "mapping"]},
            "habitat")
    assert "path_bindings must be a mapping" in str(error.value)


def test_catalog_binding_requirements_name_every_root_a_run_must_supply(catalog):
    required = catalog_binding_requirements(catalog)
    assert set(required) == set(catalog["path_bindings"])
    # A migrated catalog declares the names without the values.
    migrated = {k: v for k, v in catalog.items() if k != "path_bindings"}
    migrated["path_binding_requirements"] = sorted(catalog["path_bindings"])
    assert catalog_binding_requirements(migrated) == required


def test_moving_the_roots_to_the_host_changes_no_resolved_path(
        tmp_path, profiles, catalog):
    """The catalog's declared roots may migrate without changing a run.

    Same rooms, same packages, same values - only the file that declares them
    differs. Every expanded filesystem fact has to match, or the migration
    would quietly retarget a room.
    """
    roots = dict(catalog["path_bindings"])
    # Migrated catalog: requirement names only. It lives outside the
    # repository, because the catalog's room_package paths are already
    # repository-relative and a probe file inside examples/ would be visible
    # to whatever test runs next.
    migrated = {k: v for k, v in catalog.items() if k != "path_bindings"}
    migrated["path_binding_requirements"] = sorted(roots)
    migrated_path = tmp_path / "catalog.json"
    migrated_path.write_text(json.dumps(migrated), encoding="utf-8")
    host_path = tmp_path / "host_with_roots.json"
    host_path.write_text(json.dumps({
        "schema": HOST_RUNTIME_CONFIG_SCHEMA, "path_bindings": roots,
    }), encoding="utf-8")
    host = load_host_runtime_config(host_path)

    def facts(catalog_path, host_config):
        loaded = load_room_catalog(catalog_path)
        rows = {}
        for entry in loaded["rooms"]:
            room_id = entry["room_id"]
            resolution = resolve_catalog_room(
                loaded, room_id, catalog_path=catalog_path,
                runtime=catalog_runtime(
                    loaded, None, host_config=host_config,
                    renderer=entry.get("renderer"), room_id=room_id),
                profile_registry=profiles, host_config=host_config,
                request=V1_REQUEST)
            package = resolution.package or {}
            rows[room_id] = {
                "status": resolution.status,
                "resource_status": resolution.resource_status,
                "visual_scene": package.get("visual_scene"),
                "walkable_space": package.get("walkable_space"),
                "acoustic_package": package.get("acoustic_package"),
                "static_geometry": package.get("static_geometry"),
                "semantics": package.get("semantics"),
                "floor_reference": package.get("floor_reference"),
                "planning_inputs": package.get("planning_inputs"),
                "blockers": list(resolution.blockers),
            }
        return rows

    shipped = facts(CATALOG, None)
    moved = facts(migrated_path, host)
    assert shipped == moved
    assert all(row["status"] != "blocked" or row["blockers"]
               for row in moved.values())

    # And the failure the migration must not cause: no roots anywhere has
    # to name the missing roots rather than resolve to something wrong.
    orphaned = facts(migrated_path, None)
    assert all(row["resource_status"] != "pass" for row in orphaned.values())
    reasons = " ".join(
        str(item) for row in orphaned.values() for item in row["blockers"])
    assert "missing configured path roots" in reasons
    for name in ("AVENGINE_MP3D_ROOT", "AVENGINE_HM3D_ROOT"):
        assert name in reasons
