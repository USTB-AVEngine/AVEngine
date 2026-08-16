from __future__ import annotations

import copy
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from avengine.contracts.json_io import sha256_file
from avengine.m1.habitat_capture import (
    _activate_runtime_prefix,
    _import_installed_habitat,
    _installed_runtime_paths,
    _logical_listener_pose,
    _make_configuration,
    _validate_magnum_python_origins,
    _ue_project_asset_package_closure,
    discover_magnum_python_site,
)


MESH_OBJECT_PATH = "/Game/Test/SM_Test.SM_Test"
MATERIAL_OBJECT_PATH = "/Game/Test/M_Test.M_Test"
ENGINE_OBJECT_PATH = "/Engine/EngineMaterials/DefaultMaterial.DefaultMaterial"


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


def _module_at(module_name: str, path: Path) -> ModuleType:
    module = ModuleType(module_name)
    module.__file__ = str(path)
    return module


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
        ("activate", (prefix, site)),
        ("preloaded", site),
        ("import", None),
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
