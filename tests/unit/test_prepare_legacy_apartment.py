from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tools.m1.prepare_legacy_apartment import (
    FULL_GIT_COMMIT,
    SPEAR_MAP_PACKAGE,
    make_room_manifest,
    sha256_file,
    spear_source_snapshot,
    validate_selected_project_packages,
)


MESH_OBJECT_PATH = "/Game/Test/SM_Test.SM_Test"


def _git(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(source_root: Path, message: str) -> str:
    _git(
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
    return _git(source_root, "rev-parse", "HEAD")


def _tracked_spear_checkout(tmp_path: Path) -> tuple[Path, str]:
    source_root = tmp_path / "SPEAR"
    source_root.mkdir()
    _git(source_root, "init", "--quiet")
    map_path = source_root / SPEAR_MAP_PACKAGE
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"tracked apartment map\n")
    package_path = (
        source_root / "cpp/unreal_projects/SpearSim/Content/Test/SM_Test.uasset"
    )
    package_path.parent.mkdir(parents=True)
    package_path.write_bytes(b"tracked static mesh package\n")
    _git(
        source_root,
        "add",
        "--",
        SPEAR_MAP_PACKAGE.as_posix(),
        str(package_path.relative_to(source_root)),
    )
    return source_root, _commit(source_root, "fixture")


def _producer_package_report(source_root: Path, commit: str) -> dict[str, object]:
    package_path = (
        source_root / "cpp/unreal_projects/SpearSim/Content/Test/SM_Test.uasset"
    )
    relative = package_path.relative_to(source_root).as_posix()
    snapshot = {
        "repository_root": str(source_root.resolve()),
        "actual_project_dir": str(
            (source_root / "cpp/unreal_projects/SpearSim").resolve()
        ),
        "map_package_path": str((source_root / SPEAR_MAP_PACKAGE).resolve()),
        "commit": commit,
    }
    return {
        "source_snapshot": snapshot,
        "selected_project_asset_package_count": 1,
        "selected_project_asset_packages": [
            {
                "package_name": "/Game/Test/SM_Test",
                "repository_relative_path": relative,
                "resolved_path": str(package_path.resolve()),
                "git_tracked": True,
                "byte_size": package_path.stat().st_size,
                "sha256": sha256_file(package_path),
                "asset_object_paths": [MESH_OBJECT_PATH],
            }
        ],
        "actors": [
            {
                "static_mesh_components": [
                    {"static_mesh_asset": MESH_OBJECT_PATH, "material_assets": []}
                ]
            }
        ],
    }


def test_make_room_manifest_uses_portable_spear_locators() -> None:
    manifest = make_room_manifest(
        scene_glb=Path("scene.glb"),
        ue_manifest_path=Path("ue_export_manifest.json"),
        mesh_audit_path=Path("mesh_audit.json"),
        ue_manifest={"source_map_asset": "/Game/Apartment"},
        mesh_audit={
            "real_surface_gate": {"status": "pass"},
            "aabb_proxy_indicators": {},
            "triangles": 300,
        },
        scene_sha256="a" * 64,
        spear_snapshot={
            "commit": "b" * 40,
            "tracked_worktree_dirty": False,
            "map_package_sha256": "c" * 64,
            "repository_root": "/data/jzy/code/SPEAR",
            "map_package_path": f"/data/jzy/code/SPEAR/{SPEAR_MAP_PACKAGE}",
        },
    )

    source_asset = next(
        asset
        for asset in manifest["assets"]
        if asset["role"] == "legacy_source_map_package"
    )
    provenance = manifest["provenance"]
    assert source_asset["path"] == f"${{AVENGINE_SPEAR_ROOT}}/{SPEAR_MAP_PACKAGE}"
    assert provenance["source_repository_root"] == "${AVENGINE_SPEAR_ROOT}"
    assert provenance["source_project_repository_relative_path"] == (
        "cpp/unreal_projects/SpearSim"
    )
    assert provenance["source_map_package_repository_relative_path"] == (
        SPEAR_MAP_PACKAGE.as_posix()
    )
    assert "/data/jzy" not in json.dumps(
        {"source_asset": source_asset, "provenance": provenance}
    )


def test_selected_package_validation_accepts_relocated_checkout(
    tmp_path: Path,
) -> None:
    producer_root, commit = _tracked_spear_checkout(tmp_path)
    report = _producer_package_report(producer_root, commit)
    shared = tmp_path / "shared"
    shared.mkdir()
    consumer_root = producer_root.rename(shared / "SPEAR")

    validate_selected_project_packages(report, consumer_root)


def test_selected_package_validation_rejects_bad_producer_locator(
    tmp_path: Path,
) -> None:
    source_root, commit = _tracked_spear_checkout(tmp_path)
    report = _producer_package_report(source_root, commit)
    report["selected_project_asset_packages"][0]["resolved_path"] = (
        "/unrelated/SM_Test.uasset"
    )

    with pytest.raises(ValueError, match="producer locator"):
        validate_selected_project_packages(report, source_root)


def test_spear_snapshot_requires_git_tracked_map(tmp_path: Path) -> None:
    source_root, _ = _tracked_spear_checkout(tmp_path)

    snapshot = spear_source_snapshot(source_root.resolve())

    assert snapshot["map_package_sha256"] == sha256_file(
        source_root / SPEAR_MAP_PACKAGE
    )
    assert FULL_GIT_COMMIT.fullmatch(snapshot["commit"]) is not None

    _git(source_root, "rm", "--cached", "--quiet", "--", SPEAR_MAP_PACKAGE.as_posix())
    _commit(source_root, "remove tracked map")
    assert (source_root / SPEAR_MAP_PACKAGE).is_file()

    with pytest.raises(RuntimeError, match="ls-files"):
        spear_source_snapshot(source_root.resolve())


def test_spear_snapshot_rejects_symlinked_map_package(tmp_path: Path) -> None:
    source_root, _ = _tracked_spear_checkout(tmp_path)
    map_path = source_root / SPEAR_MAP_PACKAGE
    outside_map = tmp_path / "outside_apartment.umap"
    outside_map.write_bytes(b"outside apartment map\n")
    map_path.unlink()
    map_path.symlink_to(outside_map)
    _git(source_root, "add", "--", SPEAR_MAP_PACKAGE.as_posix())
    _commit(source_root, "symlink map")

    with pytest.raises(ValueError, match="must not be a symlink"):
        spear_source_snapshot(source_root)


def test_spear_snapshot_rejects_nested_directory_as_repository_root(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "parent_repository"
    repository_root.mkdir()
    _git(repository_root, "init", "--quiet")
    nested_root = repository_root / "nested_spear"
    map_path = nested_root / SPEAR_MAP_PACKAGE
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"nested apartment map\n")
    _git(
        repository_root,
        "add",
        "--",
        map_path.relative_to(repository_root).as_posix(),
    )
    _commit(repository_root, "nested fixture")

    with pytest.raises(ValueError, match="Git repository top level"):
        spear_source_snapshot(nested_root)


@pytest.mark.parametrize(
    "value",
    ["a" * 39, "a" * 41, "A" * 40, "not-a-commit"],
)
def test_full_git_commit_pattern_rejects_noncanonical_values(value: str) -> None:
    assert FULL_GIT_COMMIT.fullmatch(value) is None
