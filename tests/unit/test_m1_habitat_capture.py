from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import pytest

from avengine.contracts.json_io import sha256_file, write_json
from avengine.m1.contracts import ValidatedM1Inputs
from avengine.m1.habitat_capture import (
    _provenance_source_locator_report,
    _surface_provenance_check,
    _ue_project_asset_package_closure,
)


MESH_OBJECT_PATH = "/Game/Test/SM_Test.SM_Test"
MATERIAL_OBJECT_PATH = "/Game/Test/M_Test.M_Test"
ENGINE_OBJECT_PATH = "/Engine/EngineMaterials/DefaultMaterial.DefaultMaterial"
SPEAR_MAP_RELATIVE = Path(
    "cpp/unreal_projects/SpearSim/Content/SPEAR/Scenes/"
    "apartment_0000/Maps/apartment_0000.umap"
)


def _run_git(source_root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _commit(source_root: Path, message: str) -> str:
    _run_git(
        source_root,
        "-c",
        "user.name=AVEngine Tests",
        "-c",
        "user.email=tests@example.com",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        "source_snapshot": {
            "repository_root": str(source_root.resolve()),
            "actual_project_dir": str(
                (source_root / "cpp/unreal_projects/SpearSim").resolve()
            ),
            "map_package_path": str(
                (
                    source_root / "cpp/unreal_projects/SpearSim/Content/SPEAR/Scenes/"
                    "apartment_0000/Maps/apartment_0000.umap"
                ).resolve()
            ),
        },
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


def _file_asset_record(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "resolved_path": str(path.resolve()),
        "exists": path.is_file(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _relocated_surface_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    Path,
    ValidatedM1Inputs,
    list[dict[str, object]],
    Path,
    dict[str, object],
]:
    producer_root, report = _tracked_package_report(tmp_path)
    producer_root = producer_root.resolve()
    map_path = producer_root / SPEAR_MAP_RELATIVE
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"tracked apartment map\n")
    _run_git(producer_root, "add", "--", SPEAR_MAP_RELATIVE.as_posix())
    commit = _commit(producer_root, "tracked Apartment fixture")
    map_sha256 = sha256_file(map_path)
    snapshot = {
        "schema": "avengine_spear_source_snapshot_v1",
        "capture_phase": "before_ue_gltf_export",
        "repository_root": str(producer_root),
        "actual_project_dir": str(producer_root / "cpp/unreal_projects/SpearSim"),
        "commit": commit,
        "tracked_worktree_dirty": False,
        "map_asset": "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000",
        "map_package_path": str(producer_root / SPEAR_MAP_RELATIVE),
        "map_package_sha256": map_sha256,
    }
    report.update(
        {
            "schema": "avengine_legacy_ue_apartment_export_v1",
            "status": "pass",
            "source_map_asset": snapshot["map_asset"],
            "source_snapshot": snapshot,
            "source_snapshot_after_export": {
                **snapshot,
                "capture_phase": "after_ue_gltf_export",
            },
            "actual_project_dir": snapshot["actual_project_dir"],
            "loaded_editor_world": (
                "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000.apartment_0000"
            ),
            "engine_version": "5.5.4",
            "gltf_exporter_plugin": {"version_name": "1.3.1"},
            "geometry_representation": "real_surface_mesh",
            "geometry_source": "UE StaticMesh render data LOD0",
            "uses_actor_bounds_as_geometry": False,
            "option_warnings": [],
            "export_messages": {"errors": []},
            "selected_actor_count": 1,
            "static_mesh_component_count": 1,
            "unique_static_mesh_asset_count": 1,
            "dirty_packages": {
                "before_reload": {"content": [], "maps": []},
                "after_reload": {"content": [], "maps": []},
                "after_export": {"content": [], "maps": []},
            },
        }
    )
    shared = tmp_path / "shared"
    shared.mkdir()
    source_root = producer_root.rename(shared / "SPEAR").resolve()
    current_map = source_root / SPEAR_MAP_RELATIVE
    render_path = tmp_path / "scene.glb"
    render_path.write_bytes(b"audited real surface GLB\n")
    render_sha256 = sha256_file(render_path)
    report["output"] = {
        "sha256": render_sha256,
        "byte_size": render_path.stat().st_size,
    }
    ue_report_path = tmp_path / "ue_export_manifest.json"
    write_json(ue_report_path, report)
    mesh_report = {
        "schema": "avengine_real_surface_mesh_audit_v1",
        "real_surface_gate": {"status": "pass"},
        "triangles": 300,
        "meshes": 1,
        "materials": 1,
        "aabb_proxy_indicators": {
            "known_legacy_triangle_signature": False,
            "all_mesh_nodes_are_simple_boxes": False,
        },
        "sha256": render_sha256,
        "bytes": render_path.stat().st_size,
    }
    mesh_report_path = tmp_path / "mesh_audit.json"
    write_json(mesh_report_path, mesh_report)
    inputs = ValidatedM1Inputs(
        room_path=tmp_path / "room_manifest.json",
        request_path=tmp_path / "capture_request.json",
        room={
            "room_kind": "legacy_ue_real_surface_export",
            "provenance": {
                "source_repository_root": str(producer_root),
                "source_revision": commit,
                "source_repository_tracked_dirty": False,
                "source_map_package_path": str(producer_root / SPEAR_MAP_RELATIVE),
                "source_map_package_sha256": map_sha256,
                "exported_scene_sha256": render_sha256,
            },
            "surface_audit": {
                "aabb_proxy": False,
                "triangle_count": 300,
                "mesh_sha256": render_sha256,
                "real_surface_gate_status": "pass",
            },
        },
        request={},
    )
    records = [
        _file_asset_record("render_surface_mesh", render_path),
        _file_asset_record("ue_export_manifest", ue_report_path),
        _file_asset_record("real_surface_mesh_audit", mesh_report_path),
        _file_asset_record("legacy_source_map_package", current_map),
    ]
    monkeypatch.setenv("AVENGINE_SPEAR_ROOT", str(source_root))
    return source_root, inputs, records, ue_report_path, report


def test_surface_provenance_accepts_relocated_clean_tracked_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, inputs, records, _, _ = _relocated_surface_fixture(
        tmp_path, monkeypatch
    )

    check = _surface_provenance_check(inputs, records)

    assert check is not None
    assert check["status"] == "pass"
    assert check["measured"]["resolved_source_repository_root"] == str(source_root)
    assert check["measured"]["current_source_map_tracked"] == (
        SPEAR_MAP_RELATIVE.as_posix()
    )


def test_surface_provenance_rejects_legacy_producer_locator_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, inputs, records, _, _ = _relocated_surface_fixture(tmp_path, monkeypatch)
    rebound_root = tmp_path / "different_producer_root"
    inputs.room["provenance"]["source_repository_root"] = str(rebound_root)
    inputs.room["provenance"]["source_map_package_path"] = str(
        rebound_root / SPEAR_MAP_RELATIVE
    )

    check = _surface_provenance_check(inputs, records)

    assert check is not None
    assert check["status"] == "fail"
    assert check["measured"]["producer_locator_matches_provenance"] is False


def test_surface_provenance_rejects_clean_untracked_map_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, inputs, records, ue_report_path, report = _relocated_surface_fixture(
        tmp_path, monkeypatch
    )
    _run_git(
        source_root,
        "rm",
        "--cached",
        "--quiet",
        "--",
        SPEAR_MAP_RELATIVE.as_posix(),
    )
    commit = _commit(source_root, "remove tracked Apartment map")
    snapshot = report["source_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["commit"] = commit
    report["source_snapshot_after_export"] = {
        **snapshot,
        "capture_phase": "after_ue_gltf_export",
    }
    inputs.room["provenance"]["source_revision"] = commit
    write_json(ue_report_path, report)

    check = _surface_provenance_check(inputs, records)

    assert check is not None
    assert check["status"] == "fail"
    assert check["measured"]["current_source_tracked_status"] == ""
    assert check["measured"]["current_source_map_tracked"] is None


def test_surface_provenance_rejects_dirty_relocated_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, inputs, records, _, _ = _relocated_surface_fixture(
        tmp_path, monkeypatch
    )
    (source_root / SPEAR_MAP_RELATIVE).write_bytes(b"dirty apartment map\n")

    check = _surface_provenance_check(inputs, records)

    assert check is not None
    assert check["status"] == "fail"
    assert check["measured"]["current_source_tracked_status"] != ""


def test_ue_project_asset_package_closure_accepts_exact_tracked_packages(
    tmp_path: Path,
) -> None:
    source_root, report = _tracked_package_report(tmp_path)

    passed, measured = _ue_project_asset_package_closure(report, source_root)

    assert passed is True
    assert measured == {
        "record_count": 2,
        "declared_count": 2,
        "producer_source_locator": {
            "repository_root": str(source_root.resolve()),
            "project_repository_relative_path": "cpp/unreal_projects/SpearSim",
            "declared_project_path": str(
                (source_root / "cpp/unreal_projects/SpearSim").resolve()
            ),
            "project_locator_matches": True,
            "map_package_repository_relative_path": (
                "cpp/unreal_projects/SpearSim/Content/SPEAR/Scenes/"
                "apartment_0000/Maps/apartment_0000.umap"
            ),
            "declared_map_package_path": str(
                (
                    source_root / "cpp/unreal_projects/SpearSim/Content/SPEAR/Scenes/"
                    "apartment_0000/Maps/apartment_0000.umap"
                ).resolve()
            ),
            "map_locator_matches": True,
        },
        "errors": [],
        "selected_project_object_count": 2,
        "recorded_project_object_count": 2,
        "selected_engine_reference_count": 1,
    }


def test_ue_project_asset_package_closure_accepts_relocated_checkout(
    tmp_path: Path,
) -> None:
    producer_root, report = _tracked_package_report(tmp_path)
    relocated_parent = tmp_path / "shared"
    relocated_parent.mkdir()
    consumer_root = producer_root.rename(relocated_parent / "SPEAR")

    passed, measured = _ue_project_asset_package_closure(report, consumer_root)

    assert passed is True
    assert measured["errors"] == []
    assert measured["producer_source_locator"]["repository_root"] == str(
        producer_root.resolve()
    )


def test_ue_project_asset_package_closure_rejects_inconsistent_producer_locator(
    tmp_path: Path,
) -> None:
    source_root, baseline_report = _tracked_package_report(tmp_path)
    report = copy.deepcopy(baseline_report)
    records = report["selected_project_asset_packages"]
    assert isinstance(records, list)
    records[0]["resolved_path"] = "/unrelated/SM_Test.uasset"

    passed, measured = _ue_project_asset_package_closure(report, source_root)

    assert passed is False
    assert measured["errors"] == [
        "producer package locator mismatch: /Game/Test/SM_Test"
    ]


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


@pytest.mark.parametrize(
    "relative_fields",
    [
        {
            "source_project_repository_relative_path": None,
            "source_map_package_repository_relative_path": SPEAR_MAP_RELATIVE.as_posix(),
        },
        {"source_project_repository_relative_path": ""},
        {"source_map_package_repository_relative_path": SPEAR_MAP_RELATIVE.as_posix()},
    ],
)
def test_provenance_source_locator_rejects_incomplete_portable_fields(
    tmp_path: Path, relative_fields: dict[str, object]
) -> None:
    source_root = tmp_path / "SPEAR"
    (source_root / "cpp/unreal_projects/SpearSim").mkdir(parents=True)
    provenance = {
        "source_repository_root": "${AVENGINE_SPEAR_ROOT}",
        "source_map_package_path": (
            "${AVENGINE_SPEAR_ROOT}/" + SPEAR_MAP_RELATIVE.as_posix()
        ),
        **relative_fields,
    }
    inputs = ValidatedM1Inputs(
        room_path=tmp_path / "room_manifest.json",
        request_path=tmp_path / "capture_request.json",
        room={"provenance": provenance},
        request={},
    )

    passed, measured = _provenance_source_locator_report(
        inputs,
        source_root.resolve(),
        {"AVENGINE_SPEAR_ROOT": str(source_root)},
    )

    assert passed is False
    assert measured["relative_paths_match"] is False
