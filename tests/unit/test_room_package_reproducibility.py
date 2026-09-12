"""Room package expansion is cwd-independent and requires explicit path bindings."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.rooms.room_package import (
    SCHEMA,
    package_from_catalog_entry,
    resolve_catalog_room_package_path,
    resolve_room_package_paths,
    write_room_package_plan_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CATALOG = ROOT / "examples/rooms/packages/catalog.json"


def _package(**overrides):
    package = {
        "schema": SCHEMA,
        "room_id": "room",
        "family": "apartment",
        "renderer": "ue_spear",
        "visual_scene": {"map_path": "/Game/Apartment", "uproject": "${AVENGINE_TEST_ROOT}/stage.uproject"},
        "acoustic_package": "${AVENGINE_TEST_ROOT}/acoustic/manifest.json",
        "walkable_space": {"kind": "route_bank", "path": "${AVENGINE_TEST_ROOT}/routes.json"},
        "floor_reference": "measured_floor.json",
        "static_geometry": {"vertices": "vertices.npy", "triangles": "triangles.npy",
                            "coordinate_frame": {"linear_unit": "meter", "up_axis": "+Y",
                                                 "handedness": "right", "forward_axis": "-Z"}},
        "semantics": {"path": "room.json"},
        "coordinate_frame": {"linear_unit": "centimeter", "up_axis": "+Z",
                             "handedness": "left", "world_transform": "ue_xyz_cm_to_xzy_m_v1"},
        "subrooms": [],
        "acoustic_package_template": "${AVENGINE_TEST_ROOT}/acoustic/manifest.json",
    }
    package.update(overrides)
    return package


def test_missing_path_bindings_error_lists_variable_names(monkeypatch):
    monkeypatch.setenv("AVENGINE_MISSING_A", "/from-env")
    monkeypatch.setenv("AVENGINE_MISSING_B", "/from-env-b")
    with pytest.raises(ValueError, match="AVENGINE_MISSING_A") as excinfo:
        resolve_room_package_paths(
            {"scene": "${AVENGINE_MISSING_A}/x/${AVENGINE_MISSING_B}/y.glb"},
            runtime={"path_bindings": {}},
        )
    message = str(excinfo.value)
    assert "AVENGINE_MISSING_A" in message
    assert "AVENGINE_MISSING_B" in message
    assert "missing configured path roots" in message


def test_process_environ_is_not_a_binding_source(monkeypatch):
    monkeypatch.setenv("AVENGINE_TEST_ROOT", "/from-env")
    with pytest.raises(ValueError, match="AVENGINE_TEST_ROOT"):
        resolve_room_package_paths({"scene": "${AVENGINE_TEST_ROOT}/scene.glb"})


def test_relative_room_package_path_is_cwd_independent(tmp_path, monkeypatch):
    catalog_dir = tmp_path / "packages"
    decoy_dir = tmp_path / "decoy"
    catalog_dir.mkdir()
    decoy_dir.mkdir()
    bound = tmp_path / "bound"
    (bound / "acoustic").mkdir(parents=True)
    (bound / "stage.uproject").write_text("", encoding="utf-8")
    (bound / "acoustic" / "manifest.json").write_text("{}\n", encoding="utf-8")
    (bound / "routes.json").write_text("{}\n", encoding="utf-8")
    package = _package(room_id="cwd_room")
    package_path = catalog_dir / "room.json"
    package_path.write_text(json.dumps(package) + "\n", encoding="utf-8")
    for name in ("measured_floor.json", "vertices.npy", "triangles.npy", "room.json"):
        if name == "room.json":
            continue
        (catalog_dir / name).write_bytes(b"")
    catalog_path = catalog_dir / "catalog.json"
    catalog_path.write_text(json.dumps({
        "path_bindings": {"AVENGINE_TEST_ROOT": str(bound)},
        "rooms": [{"room_id": "cwd_room", "family": "apartment", "renderer": "ue_spear",
                   "room_package": "room.json"}],
    }) + "\n", encoding="utf-8")
    decoy = decoy_dir / "room.json"
    decoy.write_text(json.dumps(_package(room_id="decoy_room")) + "\n", encoding="utf-8")
    entry = {"room_id": "cwd_room", "family": "apartment", "renderer": "ue_spear",
             "room_package": "room.json"}
    runtime = {"path_bindings": {"AVENGINE_TEST_ROOT": str(bound)}}
    monkeypatch.chdir(decoy_dir)
    first = package_from_catalog_entry(entry, runtime=runtime, catalog_path=catalog_path)
    monkeypatch.chdir(tmp_path)
    second = package_from_catalog_entry(entry, runtime=runtime, catalog_path=catalog_path)
    assert first == second
    assert first["room_id"] == "cwd_room"
    assert first["acoustic_package"] == str(bound / "acoustic" / "manifest.json")
    assert first["acoustic_package_template"] == "${AVENGINE_TEST_ROOT}/acoustic/manifest.json"
    assert first["visual_scene"]["uproject"] == str(bound / "stage.uproject")


def test_repo_style_relative_path_resolves_next_to_catalog(tmp_path, monkeypatch):
    catalog_dir = tmp_path / "examples" / "rooms" / "packages"
    catalog_dir.mkdir(parents=True)
    package_path = catalog_dir / "room_a.json"
    package_path.write_text(json.dumps(_package(room_id="repo_style")) + "\n", encoding="utf-8")
    catalog_path = catalog_dir / "catalog.json"
    catalog_path.write_text("{}\n", encoding="utf-8")
    declared = "examples/rooms/packages/room_a.json"
    monkeypatch.chdir(tmp_path / "examples")
    resolved = resolve_catalog_room_package_path(declared, catalog_path=catalog_path)
    assert resolved == package_path.resolve()
    monkeypatch.chdir(tmp_path)
    assert resolve_catalog_room_package_path(declared, catalog_path=catalog_path) == resolved


def test_relative_room_package_without_catalog_path_errors():
    with pytest.raises(ValueError, match="requires catalog_path"):
        resolve_catalog_room_package_path("room.json")


def test_plan_snapshot_writes_expanded_package_and_bindings(tmp_path):
    package = resolve_room_package_paths(
        _package(), runtime={"path_bindings": {"AVENGINE_TEST_ROOT": "/bound"}})
    bindings = {"AVENGINE_TEST_ROOT": "/bound", "AVENGINE_OTHER": "/other"}
    catalog_path = tmp_path / "catalog.json"
    paths = write_room_package_plan_snapshot(
        tmp_path / "plan", package, path_bindings=bindings, catalog_path=catalog_path)
    saved = json.loads(paths["room_package"].read_text(encoding="utf-8"))
    record = json.loads(paths["path_bindings"].read_text(encoding="utf-8"))
    assert saved["acoustic_package"] == "/bound/acoustic/manifest.json"
    assert record["path_bindings"] == bindings
    assert record["catalog_path"] == str(catalog_path)


def test_production_catalog_relative_package_is_cwd_independent(tmp_path, monkeypatch):
    catalog = json.loads(PRODUCTION_CATALOG.read_text(encoding="utf-8"))
    entry = next(row for row in catalog["rooms"] if row["room_package"].endswith("room_a.json"))
    runtime = {"path_bindings": catalog["path_bindings"]}
    monkeypatch.chdir(tmp_path)
    first = package_from_catalog_entry(entry, runtime=runtime, catalog_path=PRODUCTION_CATALOG)
    monkeypatch.chdir(ROOT)
    second = package_from_catalog_entry(entry, runtime=runtime, catalog_path=PRODUCTION_CATALOG)
    assert first == second
    assert first["room_id"] == entry["room_id"]
    assert not str(first["acoustic_package"]).startswith("${")
    assert Path(first["floor_reference"]["path"]).is_absolute()
    assert Path(first["static_geometry"]["vertices"]).is_absolute()


def test_host_runtime_resolution_does_not_depend_on_the_process_directory(
        tmp_path, monkeypatch):
    """The same room and the same config resolve identically from any cwd."""
    from avengine.rooms.room_package import (
        HOST_RUNTIME_CONFIG_SCHEMA, load_host_runtime_config, resolve_room_runtime,
    )

    config = tmp_path / "host_runtime.json"
    config.write_text(json.dumps({
        "schema": HOST_RUNTIME_CONFIG_SCHEMA,
        "renderers": {"habitat": {"magnum_python_site": "/host/magnum"}},
        "rooms": {"room": {"runtime_prefix": "/host/prefix",
                           "rlr_sdk_root": "/host/rlr"}},
    }), encoding="utf-8")
    package = {
        "schema": SCHEMA, "room_id": "room", "family": "hm3d",
        "renderer": "habitat",
        "walkable_space": {"kind": "habitat_navmesh", "path": "nav.navmesh"},
    }

    def resolve():
        return resolve_room_runtime(
            package, {}, host_config=load_host_runtime_config(config),
            room_id="room")

    first = resolve()
    monkeypatch.chdir(tmp_path)
    second = resolve()
    assert first["effective"] == second["effective"]
    assert first["provenance"] == second["provenance"]
    assert first["effective"]["runtime_prefix"] == "/host/prefix"
    assert first["effective"]["magnum_python_site"] == "/host/magnum"


def test_shell_variables_never_substitute_for_a_declared_host_runtime(
        monkeypatch):
    """Environment stays opt-in, so a stale shell cannot change a run."""
    from avengine.rooms.room_package import resolve_room_runtime

    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", "/from/the/shell")
    package = {
        "schema": SCHEMA, "room_id": "room", "family": "hm3d",
        "renderer": "habitat",
        "walkable_space": {"kind": "habitat_navmesh", "path": "nav.navmesh"},
    }
    default = resolve_room_runtime(package, {})
    assert default["missing"] == ("runtime_prefix",)
    assert "runtime_prefix" not in default["effective"]
    opted_in = resolve_room_runtime(package, {}, allow_environment=True)
    assert opted_in["effective"]["runtime_prefix"] == "/from/the/shell"
    assert opted_in["provenance"]["runtime_prefix"] == (
        "environment:AVENGINE_HABITAT_RUNTIME_PREFIX")


def test_room_package_declared_runtime_is_the_lowest_precedence():
    """A package may carry working defaults without overriding a run."""
    from avengine.rooms.room_package import resolve_room_runtime

    package = {
        "schema": SCHEMA, "room_id": "room", "family": "hm3d",
        "renderer": "habitat",
        "walkable_space": {"kind": "habitat_navmesh", "path": "nav.navmesh"},
        "runtime": {"runtime_prefix": "/from/package",
                    "rlr_sdk_root": "/from/package/rlr"},
    }
    host = {"_source": "/run/host.json",
            "rooms": {"room": {"runtime_prefix": "/from/host"}}}
    report = resolve_room_runtime(
        package, {}, host_config=host, room_id="room")
    assert report["effective"]["runtime_prefix"] == "/from/host"
    assert report["provenance"]["runtime_prefix"].startswith("host_config:")
    # The package still supplies what nothing else did.
    assert report["effective"]["rlr_sdk_root"] == "/from/package/rlr"
    assert report["provenance"]["rlr_sdk_root"] == "room_package"


def test_external_roots_from_the_host_config_expand_the_same_from_any_cwd(
        tmp_path, monkeypatch):
    """Where a root is declared must not change what it expands to."""
    from avengine.rooms.room_package import (
        HOST_RUNTIME_CONFIG_SCHEMA, host_runtime_path_bindings,
        load_host_runtime_config,
    )
    from avengine.rooms.room_providers import catalog_runtime

    config = tmp_path / "host_runtime.json"
    config.write_text(json.dumps({
        "schema": HOST_RUNTIME_CONFIG_SCHEMA,
        "path_bindings": {"AVENGINE_TEST_ROOT": "/external/roots/test"},
    }), encoding="utf-8")
    package = _package()
    catalog = {"schema": "avengine_qa_room_package_catalog_v1",
               "registry_id": "probe_v1", "revision": "r1", "rooms": []}

    def expand():
        host = load_host_runtime_config(config)
        runtime = catalog_runtime(catalog, None, host_config=host,
                                  renderer="ue_spear")
        return resolve_room_package_paths(package, runtime=runtime)

    first = expand()
    monkeypatch.chdir(tmp_path)
    second = expand()
    assert first == second
    assert first["acoustic_package"] == "/external/roots/test/acoustic/manifest.json"
    assert host_runtime_path_bindings(
        load_host_runtime_config(config), "ue_spear") == {
            "AVENGINE_TEST_ROOT": "/external/roots/test"}


def test_a_catalog_root_and_a_host_root_expand_identically(tmp_path):
    """Same value, two declaring files, one expansion."""
    from avengine.rooms.room_package import (
        HOST_RUNTIME_CONFIG_SCHEMA, load_host_runtime_config,
    )
    from avengine.rooms.room_providers import catalog_runtime

    roots = {"AVENGINE_TEST_ROOT": "/external/roots/shared"}
    package = _package()
    from_catalog = resolve_room_package_paths(
        package,
        runtime=catalog_runtime(
            {"schema": "avengine_qa_room_package_catalog_v1",
             "registry_id": "probe_v1", "revision": "r1", "rooms": [],
             "path_bindings": roots}, None))
    config = tmp_path / "host_runtime.json"
    config.write_text(json.dumps({
        "schema": HOST_RUNTIME_CONFIG_SCHEMA, "path_bindings": roots,
    }), encoding="utf-8")
    from_host = resolve_room_package_paths(
        package,
        runtime=catalog_runtime(
            {"schema": "avengine_qa_room_package_catalog_v1",
             "registry_id": "probe_v1", "revision": "r1", "rooms": []},
            None, host_config=load_host_runtime_config(config),
            renderer="ue_spear"))
    assert from_catalog == from_host
