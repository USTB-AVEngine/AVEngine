from __future__ import annotations

import copy
from pathlib import Path
import subprocess

from avengine.contracts.json_io import sha256_file
from avengine.m1.habitat_capture import _ue_project_asset_package_closure


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
