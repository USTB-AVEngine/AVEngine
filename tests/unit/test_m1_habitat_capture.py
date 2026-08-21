from __future__ import annotations

import copy
from importlib.machinery import EXTENSION_SUFFIXES
import re
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from avengine.backends.rlr import sdk as rlr_sdk_module
from avengine.backends.rlr.sdk import ExternalRlrSdk, ExternalRlrSdkError
from avengine.contracts.json_io import sha256_file
from avengine.m1.habitat_capture import (
    _activate_runtime_prefix,
    _import_installed_habitat,
    _import_prepared_installed_habitat,
    _import_prepared_installed_habitat_dependencies,
    _import_prepared_installed_habitat_with_rlr,
    _prepare_installed_habitat_import,
    _installed_runtime_paths,
    _logical_listener_pose,
    _make_configuration,
    _validate_loaded_habitat_sim_origins,
    _validate_magnum_python_origins,
    _ue_project_asset_package_closure,
    discover_magnum_python_site,
    discover_pbr_asset_root,
    discover_mp3d_root,
    discover_runtime_prefix,
    prepare_installed_habitat_runtime,
    resolve_installed_runtime_prefix,
)


MESH_OBJECT_PATH = "/Game/Test/SM_Test.SM_Test"
MATERIAL_OBJECT_PATH = "/Game/Test/M_Test.M_Test"
ENGINE_OBJECT_PATH = "/Engine/EngineMaterials/DefaultMaterial.DefaultMaterial"


def test_discover_mp3d_root_can_ignore_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "ambient-mp3d"
    (ambient / "scene_datasets").mkdir(parents=True)
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(ambient))

    assert discover_mp3d_root() == ambient.resolve()
    assert discover_mp3d_root(allow_environment=False) is None


def test_discover_mp3d_root_explicit_wins_when_environment_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "ambient-mp3d"
    explicit = tmp_path / "explicit-mp3d"
    (ambient / "scene_datasets").mkdir(parents=True)
    (explicit / "scene_datasets").mkdir(parents=True)
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(ambient))

    assert (
        discover_mp3d_root(explicit, allow_environment=False)
        == explicit.resolve()
    )


def _run_git(source_root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _package_path(source_root: Path, package_name: str) -> Path:
    return (
        source_root
        / "cpp/unreal_projects/SpearSim/Content"
        / f"{package_name.removeprefix('/Game/')}.uasset"
    )


def _package_record(
    source_root: Path, package_name: str, object_path: str
) -> dict[str, object]:
    path = _package_path(source_root, package_name)
    relative = path.relative_to(source_root).as_posix()
    return {
        "package_name": package_name,
        "repository_relative_path": relative,
        "resolved_path": str(path.resolve()),
        "git_tracked": True,
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "asset_object_paths": [object_path],
    }


def _tracked_package_report(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source_root = tmp_path / "SPEAR"
    source_root.mkdir()
    _run_git(source_root, "init", "--quiet")

    package_payloads = {
        "/Game/Test/SM_Test": b"tracked static mesh package\n",
        "/Game/Test/M_Test": b"tracked material package\n",
    }
    for package_name, payload in package_payloads.items():
        path = _package_path(source_root, package_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _run_git(source_root, "add", "--", "cpp/unreal_projects/SpearSim/Content")

    records = [
        _package_record(source_root, "/Game/Test/SM_Test", MESH_OBJECT_PATH),
        _package_record(source_root, "/Game/Test/M_Test", MATERIAL_OBJECT_PATH),
    ]
    report: dict[str, object] = {
        "actors": [
            {
                "static_mesh_components": [
                    {
                        "static_mesh_asset": MESH_OBJECT_PATH,
                        "material_assets": [
                            MATERIAL_OBJECT_PATH,
                            ENGINE_OBJECT_PATH,
                        ],
                    }
                ]
            }
        ],
        "selected_project_asset_package_count": len(records),
        "selected_project_asset_packages": records,
        "selected_engine_asset_references": [ENGINE_OBJECT_PATH],
    }
    return source_root, report


def test_ue_project_asset_package_closure_accepts_exact_tracked_packages(
    tmp_path: Path,
) -> None:
    source_root, report = _tracked_package_report(tmp_path)

    passed, measured = _ue_project_asset_package_closure(report, source_root)

    assert passed is True
    assert measured == {
        "record_count": 2,
        "declared_count": 2,
        "errors": [],
        "selected_project_object_count": 2,
        "recorded_project_object_count": 2,
        "selected_engine_reference_count": 1,
    }


def test_ue_project_asset_package_closure_rejects_missing_exact_record(
    tmp_path: Path,
) -> None:
    source_root, baseline_report = _tracked_package_report(tmp_path)
    report = copy.deepcopy(baseline_report)
    records = report["selected_project_asset_packages"]
    assert isinstance(records, list)
    records.pop()
    report["selected_project_asset_package_count"] = len(records)

    passed, measured = _ue_project_asset_package_closure(report, source_root)

    assert passed is False
    assert measured["errors"] == [
        "selected /Game actor assets differ from the package closure"
    ]


def test_ue_project_asset_package_closure_rejects_changed_package_bytes(
    tmp_path: Path,
) -> None:
    source_root, report = _tracked_package_report(tmp_path)
    path = _package_path(source_root, "/Game/Test/SM_Test")
    original_size = path.stat().st_size
    path.write_bytes(b"x" * original_size)

    passed, measured = _ue_project_asset_package_closure(report, source_root)

    assert passed is False
    assert measured["errors"] == [
        "package bytes or tracking changed: /Game/Test/SM_Test"
    ]


def test_ue_project_asset_package_closure_rejects_existing_untracked_package(
    tmp_path: Path,
) -> None:
    source_root, report = _tracked_package_report(tmp_path)
    package_name = "/Game/Test/SM_Test"
    path = _package_path(source_root, package_name)
    relative = path.relative_to(source_root).as_posix()
    _run_git(source_root, "rm", "--cached", "--quiet", "--", relative)
    assert path.is_file()

    passed, measured = _ue_project_asset_package_closure(report, source_root)

    assert passed is False
    assert measured["errors"] == [
        f"package path is not the tracked expected file: {package_name}",
        "selected /Game actor assets differ from the package closure",
    ]


def test_m1_audio_off_configuration_never_constructs_audio_sensor(
    tmp_path: Path, monkeypatch
) -> None:
    class CameraSensorSpec:
        pass

    class AudioSensorSpec:
        def __init__(self) -> None:
            raise AssertionError("AudioSensorSpec must not be constructed for M1")

    class SimulatorConfiguration:
        pass

    class AgentConfiguration:
        pass

    class NavMeshSettings:
        def set_defaults(self) -> None:
            return None

    class Configuration:
        def __init__(self, sim_cfg, agent_configs) -> None:
            self.sim_cfg = sim_cfg
            self.agent_configs = agent_configs

    class SensorType:
        COLOR = "COLOR"
        DEPTH = "DEPTH"
        SEMANTIC = "SEMANTIC"

    class SensorSubType:
        PINHOLE = "PINHOLE"

    FakeHabitat = type(
        "FakeHabitat",
        (),
        {
            "CameraSensorSpec": CameraSensorSpec,
            "AudioSensorSpec": AudioSensorSpec,
            "SimulatorConfiguration": SimulatorConfiguration,
            "AgentConfiguration": AgentConfiguration,
            "NavMeshSettings": NavMeshSettings,
            "Configuration": Configuration,
            "SensorType": SensorType,
            "SensorSubType": SensorSubType,
        },
    )

    class FakeMagnum:
        @staticmethod
        def Vector2i(value):
            return tuple(value)

        @staticmethod
        def Vector3(*value):
            return tuple(value[0]) if len(value) == 1 else tuple(value)

    identity = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    inputs = type(
        "Inputs",
        (),
        {
            "room": {
                "scene": {
                    "scene_id_kind": "handle",
                    "scene_id": "fixture_scene",
                    "dataset_config_path": "default",
                    "navmesh_policy": "load_declared",
                    "load_semantic_mesh": False,
                    "enable_physics": False,
                },
                "navigation": {
                    "agent_height_m": 1.5,
                    "agent_radius_m": 0.2,
                    "include_static_objects": False,
                },
            },
            "request": {
                "seed": 7,
                "primary_camera_rig": {
                    "shared_calibration": {
                        "resolution_hw": [2, 3],
                        "rig_from_sensor": identity,
                        "hfov_degrees": 90.0,
                        "near_m": 0.05,
                        "far_m": 10.0,
                    },
                    "modalities": [
                        {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                        {"modality": "depth", "sensor_uuid": "rig_depth"},
                        {"modality": "semantic", "sensor_uuid": "rig_semantic"},
                    ],
                },
                "listener": {
                    "listener_id": "listener0",
                    "rig_from_listener": identity,
                },
            },
            "room_path": tmp_path / "room.json",
        },
    )()
    physics_path = tmp_path / "default.physics_config.json"
    physics_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_habitat",
        lambda: (object(), FakeHabitat, FakeMagnum, object()),
    )

    configuration, modalities, listener_id, _ = _make_configuration(
        inputs,
        None,
        tmp_path,
        mp3d_root=tmp_path,
        include_audio_sensor=False,
        physics_config_path=physics_path,
    )

    specs = configuration.agent_configs[0].sensor_specifications
    assert [spec.uuid for spec in specs] == ["rig_rgb", "rig_depth", "rig_semantic"]
    assert modalities == {
        "rgb": "rig_rgb",
        "depth": "rig_depth",
        "semantic": "rig_semantic",
    }
    assert listener_id == "listener0"
    assert configuration.sim_cfg.physics_config_file == str(physics_path.resolve())


def test_logical_listener_pose_is_composed_from_actual_agent_state() -> None:
    snapshot = {
        "agent": {
            "translation_m": [1.0, 2.0, 3.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    listener = {
        "rig_from_listener": {
            "translation_m": [0.0, 0.5, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }

    assert _logical_listener_pose(snapshot, listener) == {
        "translation_m": [1.0, 2.5, 3.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _magnum_python_site(tmp_path: Path) -> Path:
    site = tmp_path / "magnum-site"
    for package in ("corrade", "magnum"):
        package_path = site / package
        package_path.mkdir(parents=True)
        (package_path / "__init__.py").write_text(
            f"# {package}\n", encoding="utf-8"
        )
    suffix = EXTENSION_SUFFIXES[0]
    for module_name in ("_corrade", "_magnum"):
        (site / f"{module_name}{suffix}").write_bytes(b"extension")
    return site


def _pbr_asset_root(tmp_path: Path) -> Path:
    root = tmp_path / "pbr-assets"
    lut = root / "bluts/brdflut_ldr_512x512.png"
    environment = root / "env_maps/brown_photostudio_02_1k.hdr"
    lut.parent.mkdir(parents=True)
    environment.parent.mkdir(parents=True)
    lut.write_bytes(b"fixture lut")
    environment.write_bytes(b"fixture environment")
    (root / "license.txt").write_text(
        "fixture MIT LUT and CC0 Brown Photostudio notice\n",
        encoding="utf-8",
    )
    return root


def _external_rlr_sdk(tmp_path: Path) -> ExternalRlrSdk:
    root = tmp_path / "external-rlr-sdk"
    header = root / "headers" / "RLRAudioPropagation.h"
    library = root / "libs" / "linux" / "x64" / "libRLRAudioPropagation.so"
    header.parent.mkdir(parents=True)
    library.parent.mkdir(parents=True)
    header.write_text("// fixture\n", encoding="utf-8")
    library.write_bytes(b"fixture rlr")
    return ExternalRlrSdk(
        root=root.resolve(),
        header=header.resolve(),
        library=library.resolve(),
    )


def _module_at(module_name: str, path: Path) -> ModuleType:
    module = ModuleType(module_name)
    module.__file__ = str(path)
    return module


def test_installed_prefix_rejects_a_git_checkout_and_accepts_root_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "installed-prefix"
    prefix.mkdir()

    assert resolve_installed_runtime_prefix(prefix) == prefix.resolve()
    assert resolve_installed_runtime_prefix(runtime_root=prefix) == prefix.resolve()
    with pytest.raises(ValueError, match="Specify only one"):
        resolve_installed_runtime_prefix(prefix, runtime_root=prefix)

    checkout = tmp_path / "legacy-checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    nested_prefix = checkout / "build" / "installed-prefix"
    nested_prefix.mkdir(parents=True)
    with pytest.raises(ValueError, match="must not be inside a Git checkout"):
        discover_runtime_prefix(nested_prefix)

    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(nested_prefix))
    with pytest.raises(ValueError, match="must not be inside a Git checkout"):
        resolve_installed_runtime_prefix()


def test_discover_pbr_asset_root_is_explicit_complete_and_non_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _pbr_asset_root(tmp_path)
    monkeypatch.setenv("AVENGINE_HABITAT_PBR_ASSET_ROOT", str(root))

    assert discover_pbr_asset_root() is None
    assert discover_pbr_asset_root(root) == root.resolve()

    (root / "license.txt").unlink()
    with pytest.raises(FileNotFoundError, match="license.txt"):
        discover_pbr_asset_root(root)

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    checkout_root = _pbr_asset_root(checkout)
    with pytest.raises(ValueError, match="must not be inside a Git checkout"):
        discover_pbr_asset_root(checkout_root)


def test_discover_pbr_asset_root_rejects_required_file_escape(
    tmp_path: Path,
) -> None:
    root = _pbr_asset_root(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    lut = root / "bluts/brdflut_ldr_512x512.png"
    lut.unlink()
    lut.symlink_to(outside)

    with pytest.raises(RuntimeError, match="below its root"):
        discover_pbr_asset_root(root)


def test_discover_magnum_python_site_checks_layout_and_current_abi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", raising=False)
    with pytest.raises(FileNotFoundError, match="AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"):
        discover_magnum_python_site()

    site = _magnum_python_site(tmp_path)
    monkeypatch.setenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(site))
    assert discover_magnum_python_site() == site.resolve()

    extension = site / f"_magnum{EXTENSION_SUFFIXES[0]}"
    extension.unlink()
    with pytest.raises(FileNotFoundError, match="_magnum"):
        discover_magnum_python_site()

    outside = tmp_path / "outside.so"
    outside.write_bytes(b"extension")
    extension.symlink_to(outside)
    with pytest.raises(RuntimeError, match="site root"):
        discover_magnum_python_site()


def test_activate_runtime_prefix_places_magnum_site_after_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    site = tmp_path / "site"
    prefix.mkdir()
    site.mkdir()
    prefix_alias = tmp_path / "prefix-alias"
    prefix_alias.symlink_to(prefix)
    monkeypatch.setattr(sys, "path", [str(site), str(prefix_alias), "retained"])

    _activate_runtime_prefix(prefix, magnum_python_site=site)

    assert sys.path == [str(prefix.resolve()), str(site.resolve()), "retained"]


def test_validate_magnum_python_origins_rejects_preloaded_outside_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _magnum_python_site(tmp_path)
    modules = {
        "corrade": _module_at("corrade", site / "corrade" / "__init__.py"),
        "magnum": _module_at("magnum", site / "magnum" / "__init__.py"),
        "_corrade": _module_at(
            "_corrade", site / f"_corrade{EXTENSION_SUFFIXES[0]}"
        ),
        "_magnum": _module_at("_magnum", site / f"_magnum{EXTENSION_SUFFIXES[0]}"),
    }
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    _validate_magnum_python_origins(site, modules["magnum"])

    outside = tmp_path / "old-checkout" / "magnum.py"
    outside.parent.mkdir()
    outside.write_text("# wrong module\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "_magnum", _module_at("_magnum", outside))
    with pytest.raises(RuntimeError, match="outside the required external site"):
        _validate_magnum_python_origins(site, modules["magnum"])


def test_prepare_installed_habitat_removes_editable_without_importing_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    site = _magnum_python_site(tmp_path)
    binding_name = "habitat_sim._ext.habitat_sim_bindings"
    imported = False

    class EditableHabitatFinder:
        pass

    EditableHabitatFinder.__module__ = "_editable_skbc_habitat_sim"
    removable = EditableHabitatFinder()
    retained = object()
    monkeypatch.setattr(sys, "meta_path", [retained, removable])
    for module_name in tuple(sys.modules):
        if module_name == binding_name or module_name.startswith(binding_name + "."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.discover_magnum_python_site",
        lambda: site,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._activate_runtime_prefix",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_loaded_habitat_sim_origins",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_preloaded_magnum_python_origins",
        lambda *_args, **_kwargs: None,
    )

    def should_not_import() -> tuple[object, object, object, object]:
        nonlocal imported
        imported = True
        raise AssertionError("preparation must not import Habitat")

    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_habitat", should_not_import
    )

    prepared = _prepare_installed_habitat_import(prefix)

    assert prepared.prefix == prefix
    assert prepared.magnum_python_site == site
    assert sys.meta_path == [retained]
    assert binding_name not in sys.modules
    assert imported is False


def test_import_prepared_installed_habitat_runs_origin_postconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    site = tmp_path / "magnum-site"
    imported = (object(), object(), object(), object())
    prepared = type("Prepared", (), {"prefix": prefix, "magnum_python_site": site})()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_habitat", lambda: imported
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_loaded_habitat_sim_origins",
        lambda active_prefix: calls.append(("habitat", active_prefix)),
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_magnum_python_origins",
        lambda active_site, magnum: calls.append(("magnum", (active_site, magnum))),
    )

    assert _import_prepared_installed_habitat(prepared) == imported
    assert calls == [
        ("habitat", prefix),
        ("magnum", (site, imported[2])),
    ]


def test_explicit_rlr_sdk_preloads_before_prepared_habitat_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    site = tmp_path / "magnum-site"
    prepared = type("Prepared", (), {"prefix": prefix, "magnum_python_site": site})()
    sdk = _external_rlr_sdk(tmp_path)
    imported = (object(), object(), object(), object())
    binding_name = "habitat_sim._ext.habitat_sim_bindings"
    for module_name in tuple(sys.modules):
        if module_name == binding_name or module_name.startswith(binding_name + "."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        rlr_sdk_module,
        "discover_external_rlr_sdk",
        lambda root: events.append(("discover", root)) or sdk,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_prepared_installed_habitat_dependencies",
        lambda observed: events.append(("dependencies", observed))
        or (object(), object()),
    )

    def preload(observed_sdk: ExternalRlrSdk) -> None:
        assert binding_name not in sys.modules
        events.append(("preload", observed_sdk))

    monkeypatch.setattr(rlr_sdk_module, "preload_external_rlr_sdk", preload)
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_prepared_installed_habitat",
        lambda observed: events.append(("import", observed)) or imported,
    )
    monkeypatch.setattr(
        rlr_sdk_module,
        "validate_loaded_external_rlr_sdk",
        lambda observed_sdk: events.append(("validate", observed_sdk)),
    )

    observed_import, observed_sdk = _import_prepared_installed_habitat_with_rlr(
        prepared,
        rlr_sdk_root=sdk.root,
    )

    assert observed_import == imported
    assert observed_sdk == sdk
    assert events == [
        ("dependencies", prepared),
        ("discover", sdk.root),
        ("preload", sdk),
        ("import", prepared),
        ("validate", sdk),
    ]


def test_prepared_dependencies_load_before_habitat_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "magnum-site"
    prepared = type(
        "Prepared",
        (),
        {"prefix": tmp_path / "prefix", "magnum_python_site": site},
    )()
    quaternion = ModuleType("quaternion")
    magnum = ModuleType("magnum")
    monkeypatch.setitem(sys.modules, "quaternion", quaternion)
    monkeypatch.setitem(sys.modules, "magnum", magnum)
    binding_name = "habitat_sim._ext.habitat_sim_bindings"
    monkeypatch.delitem(sys.modules, binding_name, raising=False)
    validations: list[tuple[Path, ModuleType]] = []
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_magnum_python_origins",
        lambda observed_site, observed_magnum: validations.append(
            (observed_site, observed_magnum)
        ),
    )

    assert _import_prepared_installed_habitat_dependencies(prepared) == (
        quaternion,
        magnum,
    )
    assert validations == [(site, magnum)]
    assert binding_name not in sys.modules


def test_explicit_rlr_sdk_rejects_loaded_binding_mismatch_before_cdll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = type(
        "Prepared",
        (),
        {"prefix": tmp_path / "prefix", "magnum_python_site": tmp_path / "site"},
    )()
    sdk = _external_rlr_sdk(tmp_path)
    binding_name = "habitat_sim._ext.habitat_sim_bindings"
    monkeypatch.setitem(sys.modules, binding_name, ModuleType(binding_name))
    events: list[str] = []
    monkeypatch.setattr(
        rlr_sdk_module,
        "discover_external_rlr_sdk",
        lambda _root: events.append("discover") or sdk,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_prepared_installed_habitat_dependencies",
        lambda _prepared: events.append("dependencies") or (object(), object()),
    )

    def reject_mapping(_sdk: ExternalRlrSdk) -> None:
        events.append("validate")
        raise ExternalRlrSdkError("wrong preloaded RLR mapping")

    monkeypatch.setattr(
        rlr_sdk_module,
        "validate_loaded_external_rlr_sdk",
        reject_mapping,
    )
    monkeypatch.setattr(
        rlr_sdk_module,
        "preload_external_rlr_sdk",
        lambda _sdk: events.append("preload"),
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_prepared_installed_habitat",
        lambda _prepared: events.append("import"),
    )

    with pytest.raises(ExternalRlrSdkError, match="wrong preloaded RLR mapping"):
        _import_prepared_installed_habitat_with_rlr(
            prepared,
            rlr_sdk_root=sdk.root,
        )

    assert events == ["dependencies", "discover", "validate"]


def test_adapter_linked_import_without_sdk_has_explicit_remediation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = type(
        "Prepared",
        (),
        {"prefix": tmp_path / "prefix", "magnum_python_site": tmp_path / "site"},
    )()
    def missing_rlr() -> tuple[object, object, object, object]:
        raise ImportError(
            "libRLRAudioPropagation.so: cannot open shared object file"
        )

    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_habitat",
        missing_rlr,
    )

    with pytest.raises(RuntimeError, match="explicit rlr_sdk_root") as raised:
        _import_prepared_installed_habitat(prepared)

    assert "AVENGINE_HABITAT_BUILD_RLR_ADAPTER=OFF" in str(raised.value)


def test_public_installed_runtime_dispatches_optional_explicit_rlr_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    site = tmp_path / "magnum-site"
    module_path = prefix / "habitat_sim" / "__init__.py"
    binding_path = prefix / "habitat_sim" / "_ext" / "habitat_sim_bindings.unit.so"
    physics_path = prefix / "config" / "default.physics_config.json"
    module_path.parent.mkdir(parents=True)
    binding_path.parent.mkdir(parents=True)
    physics_path.parent.mkdir(parents=True)
    module_path.write_text("# fixture\n", encoding="utf-8")
    binding_path.write_bytes(b"binding")
    physics_path.write_text("{}\n", encoding="utf-8")
    habitat = _module_at("habitat_sim", module_path)
    habitat.__path__ = []
    binding = ModuleType("habitat_sim._ext.habitat_sim_bindings")
    binding.__file__ = str(binding_path)
    extension_package = ModuleType("habitat_sim._ext")
    extension_package.__path__ = []
    extension_package.habitat_sim_bindings = binding
    habitat._ext = extension_package
    monkeypatch.setitem(sys.modules, "habitat_sim", habitat)
    monkeypatch.setitem(sys.modules, "habitat_sim._ext", extension_package)
    monkeypatch.setitem(
        sys.modules,
        "habitat_sim._ext.habitat_sim_bindings",
        binding,
    )
    sdk = _external_rlr_sdk(tmp_path)
    pbr_root = _pbr_asset_root(tmp_path)
    imported = (object(), habitat, object(), object())
    prepared = type(
        "Prepared", (), {"prefix": prefix, "magnum_python_site": site}
    )()
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.resolve_installed_runtime_prefix",
        lambda *_args, **_kwargs: prefix,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.discover_magnum_python_site",
        lambda _explicit=None: site,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.discover_mp3d_root",
        lambda _explicit=None: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._prepare_installed_habitat_import",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_prepared_installed_habitat_with_rlr",
        lambda observed, *, rlr_sdk_root: events.append(
            ("with_rlr", (observed, rlr_sdk_root))
        )
        or (imported, sdk),
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_prepared_installed_habitat",
        lambda observed: events.append(("plain", observed)) or imported,
    )

    with_sdk = prepare_installed_habitat_runtime(
        runtime_prefix=prefix,
        pbr_asset_root=pbr_root,
        magnum_python_site=site,
        rlr_sdk_root=sdk.root,
    )
    without_sdk = prepare_installed_habitat_runtime(
        runtime_prefix=prefix,
        magnum_python_site=site,
    )

    assert with_sdk.habitat_sim is habitat
    assert without_sdk.habitat_sim is habitat
    assert with_sdk.pbr_asset_root == pbr_root.resolve()
    assert without_sdk.pbr_asset_root is None
    assert events == [
        ("with_rlr", (prepared, sdk.root)),
        ("plain", prepared),
    ]


def test_import_installed_habitat_removes_only_editable_habitat_meta_finder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    site = _magnum_python_site(tmp_path)
    imported = (object(), object(), object(), object())

    class EditableHabitatFinder:
        calls = 0

        def find_spec(self, *_args: object, **_kwargs: object) -> None:
            type(self).calls += 1
            raise AssertionError("editable Habitat finder must not be invoked")

    EditableHabitatFinder.__module__ = "_editable_skbc_habitat_sim"

    class RetainedFinder:
        def find_spec(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("test does not import through retained finders")

    RetainedFinder.__module__ = "_editable_unrelated_package"

    class HabitatOnlyFinder:
        def find_spec(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("test does not import through retained finders")

    HabitatOnlyFinder.__module__ = "habitat_sim_runtime"

    removable = EditableHabitatFinder()
    retained = RetainedFinder()
    habitat_only = HabitatOnlyFinder()
    monkeypatch.setattr(sys, "meta_path", [retained, removable, habitat_only])
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.discover_magnum_python_site",
        lambda: site,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._activate_runtime_prefix",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_loaded_habitat_sim_origins",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_preloaded_magnum_python_origins",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_magnum_python_origins",
        lambda *_args, **_kwargs: None,
    )

    def fake_import_habitat() -> tuple[object, object, object, object]:
        assert sys.meta_path == [retained, habitat_only]
        return imported

    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_habitat", fake_import_habitat
    )

    assert _import_installed_habitat(prefix) == imported
    assert sys.meta_path == [retained, habitat_only]
    assert EditableHabitatFinder.calls == 0


@pytest.mark.parametrize("module_name", ["habitat_sim", "habitat_sim.utils.common"])
def test_import_installed_habitat_rejects_preloaded_module_outside_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    site = _magnum_python_site(tmp_path)
    outside = tmp_path / "old-checkout" / f"{module_name.rsplit('.', 1)[-1]}.py"
    outside.parent.mkdir()
    outside.write_text("# wrong habitat module\n", encoding="utf-8")
    for existing_name in tuple(sys.modules):
        if existing_name == "habitat_sim" or existing_name.startswith("habitat_sim."):
            monkeypatch.delitem(sys.modules, existing_name, raising=False)
    monkeypatch.setitem(sys.modules, module_name, _module_at(module_name, outside))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.discover_magnum_python_site",
        lambda: site,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._activate_runtime_prefix",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_preloaded_magnum_python_origins",
        lambda *_args, **_kwargs: None,
    )
    imported = False

    def should_not_import() -> tuple[object, object, object, object]:
        nonlocal imported
        imported = True
        raise AssertionError("wrong preloaded Habitat module must fail before import")

    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_habitat", should_not_import
    )

    with pytest.raises(RuntimeError, match="outside the required --runtime-prefix"):
        _import_installed_habitat(prefix)
    assert imported is False


def test_validate_loaded_habitat_origins_accepts_originless_native_binding_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    binding_name = "habitat_sim._ext.habitat_sim_bindings"
    child_name = f"{binding_name}.core"
    binding_path = prefix / "habitat_sim" / "_ext" / f"habitat_sim_bindings{EXTENSION_SUFFIXES[0]}"
    binding_path.parent.mkdir(parents=True)
    binding_path.write_bytes(b"binding")
    binding = _module_at(binding_name, binding_path)
    child = ModuleType(child_name)
    setattr(binding, "core", child)
    monkeypatch.setitem(sys.modules, binding_name, binding)
    monkeypatch.setitem(sys.modules, child_name, child)

    _validate_loaded_habitat_sim_origins(prefix)


def test_validate_loaded_habitat_origins_rejects_originless_child_without_binding_relation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    binding_name = "habitat_sim._ext.habitat_sim_bindings"
    child_name = f"{binding_name}.core"
    binding_path = prefix / "habitat_sim" / "_ext" / f"habitat_sim_bindings{EXTENSION_SUFFIXES[0]}"
    binding_path.parent.mkdir(parents=True)
    binding_path.write_bytes(b"binding")
    binding = _module_at(binding_name, binding_path)
    child = ModuleType(child_name)
    setattr(binding, "core", ModuleType(child_name))
    monkeypatch.setitem(sys.modules, binding_name, binding)
    monkeypatch.setitem(sys.modules, child_name, child)

    with pytest.raises(RuntimeError, match="no filesystem origin"):
        _validate_loaded_habitat_sim_origins(prefix)


def test_validate_loaded_habitat_origins_rejects_originless_child_of_external_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    binding_name = "habitat_sim._ext.habitat_sim_bindings"
    child_name = f"{binding_name}.geo"
    binding_path = tmp_path / "old-checkout" / f"habitat_sim_bindings{EXTENSION_SUFFIXES[0]}"
    binding_path.parent.mkdir()
    binding_path.write_bytes(b"binding")
    binding = _module_at(binding_name, binding_path)
    child = ModuleType(child_name)
    setattr(binding, "geo", child)
    monkeypatch.setitem(sys.modules, binding_name, binding)
    monkeypatch.setitem(sys.modules, child_name, child)

    with pytest.raises(RuntimeError, match="outside the required --runtime-prefix"):
        _validate_loaded_habitat_sim_origins(prefix)


def test_import_installed_habitat_rejects_wrong_module_loaded_during_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    site = _magnum_python_site(tmp_path)
    outside = tmp_path / "old-checkout" / "bindings.py"
    outside.parent.mkdir()
    outside.write_text("# wrong habitat module\n", encoding="utf-8")
    for existing_name in tuple(sys.modules):
        if existing_name == "habitat_sim" or existing_name.startswith("habitat_sim."):
            monkeypatch.delitem(sys.modules, existing_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.discover_magnum_python_site",
        lambda: site,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._activate_runtime_prefix",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_preloaded_magnum_python_origins",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_magnum_python_origins",
        lambda *_args, **_kwargs: None,
    )

    def import_wrong_habitat() -> tuple[object, object, object, object]:
        monkeypatch.setitem(
            sys.modules,
            "habitat_sim._ext.habitat_sim_bindings",
            _module_at("habitat_sim._ext.habitat_sim_bindings", outside),
        )
        return object(), object(), object(), object()

    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_habitat", import_wrong_habitat
    )

    with pytest.raises(RuntimeError, match="outside the required --runtime-prefix"):
        _import_installed_habitat(prefix)


def test_import_installed_habitat_activates_site_before_habitat_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    site = _magnum_python_site(tmp_path)
    calls: list[tuple[str, object]] = []
    imported = (object(), object(), object(), object())
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.discover_magnum_python_site",
        lambda: site,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._remove_editable_habitat_sim_meta_finders",
        lambda: calls.append(("remove_editable_habitat_finder", None)),
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_loaded_habitat_sim_origins",
        lambda active_prefix: calls.append(("validate_habitat", active_prefix)),
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._activate_runtime_prefix",
        lambda active_prefix, *, magnum_python_site: calls.append(
            ("activate", (active_prefix, magnum_python_site))
        ),
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_preloaded_magnum_python_origins",
        lambda active_site: calls.append(("preloaded", active_site)),
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._import_habitat",
        lambda: calls.append(("import", None)) or imported,
    )
    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_magnum_python_origins",
        lambda active_site, active_magnum: calls.append(
            ("validate", (active_site, active_magnum))
        ),
    )

    assert _import_installed_habitat(prefix) == imported
    assert calls == [
        ("remove_editable_habitat_finder", None),
        ("validate_habitat", prefix),
        ("activate", (prefix, site)),
        ("preloaded", site),
        ("import", None),
        ("validate_habitat", prefix),
        ("validate", (site, imported[2])),
    ]


def test_installed_runtime_rejects_physics_config_symlink_outside_prefix(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "prefix"
    module_path = prefix / "habitat_sim" / "__init__.py"
    binding_path = prefix / "habitat_sim" / "_ext" / "bindings.so"
    physics_path = prefix / "config" / "default.physics_config.json"
    module_path.parent.mkdir(parents=True)
    binding_path.parent.mkdir(parents=True)
    physics_path.parent.mkdir(parents=True)
    module_path.write_text("# module\n", encoding="utf-8")
    binding_path.write_bytes(b"binding")
    outside = tmp_path / "outside.physics_config.json"
    outside.write_text("{}\n", encoding="utf-8")
    physics_path.symlink_to(outside)

    habitat_module = type("HabitatModule", (), {"__file__": str(module_path)})()
    bindings_module = type("BindingsModule", (), {"__file__": str(binding_path)})()

    with pytest.raises(RuntimeError, match="physics config"):
        _installed_runtime_paths(prefix, habitat_module, bindings_module)


def test_import_installed_habitat_real_import_bypasses_editable_redirector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "installed-prefix"
    habitat_root = prefix / "habitat_sim"
    utils_root = habitat_root / "utils"
    utils_root.mkdir(parents=True)
    (habitat_root / "__init__.py").write_text(
        "RLR_ADAPTER_ENABLED = True\n", encoding="utf-8"
    )
    (utils_root / "__init__.py").write_text("", encoding="utf-8")
    (utils_root / "common.py").write_text(
        "def quat_to_coeffs(value):\n    return value\n", encoding="utf-8"
    )
    site = tmp_path / "external-magnum"
    magnum_root = site / "magnum"
    magnum_root.mkdir(parents=True)
    (magnum_root / "__init__.py").write_text("", encoding="utf-8")
    old_checkout = tmp_path / "habitat-sim-AVEngine"
    old_habitat = old_checkout / "habitat_sim"
    old_habitat.mkdir(parents=True)
    (old_habitat / "__init__.py").write_text(
        "RLR_ADAPTER_ENABLED = False\n", encoding="utf-8"
    )

    class EditableHabitatFinder:
        calls = 0

        def find_spec(self, *_args: object, **_kwargs: object) -> None:
            type(self).calls += 1
            raise AssertionError("editable Habitat finder must be removed before import")

    EditableHabitatFinder.__module__ = "_editable_skbc_habitat_sim"
    finder = EditableHabitatFinder()
    monkeypatch.setattr(sys, "path", list(sys.path))
    for module_name in tuple(sys.modules):
        if module_name == "habitat_sim" or module_name.startswith("habitat_sim."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        if module_name == "magnum":
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    quaternion = ModuleType("quaternion")
    quaternion.__file__ = str(tmp_path / "quaternion.py")
    monkeypatch.setitem(sys.modules, "quaternion", quaternion)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    monkeypatch.setattr(
        "avengine.m1.habitat_capture.discover_magnum_python_site",
        lambda _explicit=None: site,
    )

    def validate_magnum(active_site: Path, magnum: ModuleType) -> None:
        assert Path(magnum.__file__).resolve().is_relative_to(active_site.resolve())

    monkeypatch.setattr(
        "avengine.m1.habitat_capture._validate_magnum_python_origins",
        validate_magnum,
    )

    _qt, habitat, _magnum, _quat_to_coeffs = _import_installed_habitat(
        prefix, magnum_python_site=site
    )

    assert EditableHabitatFinder.calls == 0
    assert finder not in sys.meta_path
    assert habitat.RLR_ADAPTER_ENABLED is True
    origins = [
        Path(module.__file__).resolve()
        for name, module in sys.modules.items()
        if name == "habitat_sim" or name.startswith("habitat_sim.")
        if getattr(module, "__file__", None)
    ]
    assert origins
    assert all(path.is_relative_to(prefix.resolve()) for path in origins)
    assert all(not path.is_relative_to(old_checkout.resolve()) for path in origins)
    assert sys.path[:2] == [str(prefix.resolve()), str(site.resolve())]
    for module_name in tuple(sys.modules):
        if module_name == "habitat_sim" or module_name.startswith("habitat_sim."):
            sys.modules.pop(module_name, None)
        if module_name == "magnum":
            sys.modules.pop(module_name, None)


def test_habitat_python_binding_explicitly_disables_build_and_install_rpaths() -> None:
    repository = Path(__file__).resolve().parents[2]
    cmake = (repository / "native/habitat/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    target = re.search(
        r"set_target_properties\(\s*avengine_habitat_python_bindings\s+"
        r"PROPERTIES(?P<body>.*?)\n\s*\)",
        cmake,
        flags=re.DOTALL,
    )
    assert target is not None
    properties = target.group("body")
    expected = {
        "BUILD_RPATH": '""',
        "SKIP_BUILD_RPATH": "TRUE",
        "BUILD_WITH_INSTALL_RPATH": "FALSE",
        "INSTALL_RPATH": '""',
        "INSTALL_RPATH_USE_LINK_PATH": "FALSE",
    }
    for name, value in expected.items():
        assert re.search(
            rf"(?m)^\s*{name}\s+{re.escape(value)}\s*$",
            properties,
        ), name
