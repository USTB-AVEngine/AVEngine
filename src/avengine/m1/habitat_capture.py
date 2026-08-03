from __future__ import annotations

from datetime import datetime, timezone
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PIL import __version__ as pillow_version
from PIL import Image

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    resolve_declared_path,
    sha256_file,
    write_json,
)
from avengine.contracts.transforms import (
    compose_transforms,
    invert_transform,
    round_trip_via_parent,
    transform_error,
)
from avengine.m1.contracts import (
    EVIDENCE_SCHEMA,
    ValidatedM1Inputs,
    validate_loaded_scene_asset_graph,
    validate_scene_asset_graph,
)
from avengine.m1.evidence import (
    array_sha256,
    finalize_evidence,
    make_check,
    save_observations,
    verify_evidence_artifacts,
)
from avengine.runtime_lock import RuntimeLockError, resolve_runtime_profile


VISUAL_SENSOR_TYPES = {
    "rgb": "COLOR",
    "depth": "DEPTH",
    "semantic": "SEMANTIC",
}

SPEAR_PROJECT_REPOSITORY_RELATIVE = Path("cpp/unreal_projects/SpearSim")
SPEAR_PROJECT_CONTENT_REPOSITORY_RELATIVE = (
    SPEAR_PROJECT_REPOSITORY_RELATIVE / "Content"
)
SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE = Path(
    "cpp/unreal_projects/SpearSim/Content/SPEAR/Scenes/"
    "apartment_0000/Maps/apartment_0000.umap"
)

PROCESS_INSTANCE_ID = str(uuid.uuid4())
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat()
PROCESS_INITIAL_PID = os.getpid()


def _producer_process_identity() -> dict[str, Any]:
    return {
        "process_instance_id": PROCESS_INSTANCE_ID,
        "pid": os.getpid(),
        "initial_pid": PROCESS_INITIAL_PID,
        "started_at_utc": PROCESS_STARTED_AT_UTC,
    }


def discover_runtime_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    configured = os.environ.get("AVENGINE_HABITAT_RUNTIME_ROOT")
    if configured:
        return Path(configured).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    sibling = repository_root.parent / "habitat-sim-AVEngine"
    if sibling.is_dir():
        return sibling.resolve()
    raise FileNotFoundError("Set AVENGINE_HABITAT_RUNTIME_ROOT or pass --runtime-root")


def _git_value(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _runtime_lock_commit(repository_root: Path) -> str | None:
    try:
        lock_path = resolve_runtime_profile(repository_root, "m1")
    except RuntimeLockError:
        return None
    text = lock_path.read_text(encoding="utf-8")
    match = re.search(
        r"^habitat_runtime:\s*$.*?^\s+fork_governance_commit:\s+([0-9a-f]{40})\s*$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else None


def _import_habitat() -> tuple[Any, Any, Any, Any]:
    # The pinned audio-enabled build aborts if habitat_sim is imported before
    # numpy-quaternion. Keep this order local and explicit.
    import quaternion as qt

    import habitat_sim
    import magnum as mn
    from habitat_sim.utils.common import quat_to_coeffs

    return qt, habitat_sim, mn, quat_to_coeffs


def _environment_for_paths(runtime_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["AVENGINE_HABITAT_RUNTIME_ROOT"] = str(runtime_root)
    return environment


def _source_map_repository_relative(inputs: ValidatedM1Inputs) -> Path:
    provenance = inputs.room.get("provenance", {})
    key = "source_map_package_repository_relative_path"
    if key not in provenance:
        return SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE
    raw = provenance.get(key)
    if raw != SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE.as_posix():
        raise ValueError(
            "Legacy source map repository-relative path does not match Apartment"
        )
    return SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE


def _resolved_spear_source_root(
    inputs: ValidatedM1Inputs, environment: dict[str, str]
) -> Path | None:
    provenance = inputs.room.get("provenance", {})
    raw = environment.get("AVENGINE_SPEAR_ROOT") or provenance.get(
        "source_repository_root"
    )
    if not isinstance(raw, str) or not raw:
        return None
    return resolve_declared_path(
        raw,
        manifest_dir=inputs.room_path.parent,
        environment=environment,
    )


def _resolved_room_asset_path(
    inputs: ValidatedM1Inputs,
    asset: dict[str, Any],
    environment: dict[str, str],
) -> Path:
    if asset.get("role") == "legacy_source_map_package" and environment.get(
        "AVENGINE_SPEAR_ROOT"
    ):
        source_root = _resolved_spear_source_root(inputs, environment)
        if source_root is None:
            raise ValueError("AVENGINE_SPEAR_ROOT did not resolve to a source root")
        path = (source_root / _source_map_repository_relative(inputs)).resolve()
        provenance = inputs.room.get("provenance", {})
        relative_keys = [
            "source_project_repository_relative_path" in provenance,
            "source_map_package_repository_relative_path" in provenance,
        ]
        if all(relative_keys):
            declared_path = resolve_declared_path(
                asset.get("path"),
                manifest_dir=inputs.room_path.parent,
                environment=environment,
            )
            if declared_path != path:
                raise ValueError(
                    "Portable legacy source-map asset does not match its fixed locator"
                )
        elif not any(relative_keys):
            declared_raw = asset.get("path")
            producer_map_raw = provenance.get("source_map_package_path")
            if not (
                isinstance(declared_raw, str)
                and Path(declared_raw).is_absolute()
                and isinstance(producer_map_raw, str)
                and Path(producer_map_raw).is_absolute()
                and Path(declared_raw) == Path(producer_map_raw)
            ):
                raise ValueError(
                    "Legacy source-map asset does not match its producer locator"
                )
        else:
            raise ValueError("Portable SPEAR provenance locator is incomplete")
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise ValueError("Legacy source map escapes the SPEAR checkout") from error
        return path
    return resolve_declared_path(
        asset["path"],
        manifest_dir=inputs.room_path.parent,
        environment=environment,
    )


def _resolved_scene(inputs: ValidatedM1Inputs, runtime_root: Path) -> dict[str, Any]:
    scene = inputs.room["scene"]
    environment = _environment_for_paths(runtime_root)
    manifest_dir = inputs.room_path.parent

    dataset_raw = scene["dataset_config_path"]
    dataset_config: str | Path
    if dataset_raw == "default":
        dataset_config = "default"
    else:
        dataset_config = resolve_declared_path(
            dataset_raw, manifest_dir=manifest_dir, environment=environment
        )

    if scene["scene_id_kind"] == "path":
        scene_id: str | Path = resolve_declared_path(
            scene["scene_id"], manifest_dir=manifest_dir, environment=environment
        )
    else:
        scene_id = scene["scene_id"]

    navmesh = None
    if scene.get("navmesh_path"):
        navmesh = resolve_declared_path(
            scene["navmesh_path"],
            manifest_dir=manifest_dir,
            environment=environment,
        )
    return {
        "dataset_config": dataset_config,
        "scene_id": scene_id,
        "navmesh": navmesh,
        "navmesh_policy": scene["navmesh_policy"],
        "load_semantic_mesh": bool(scene.get("load_semantic_mesh", False)),
        "enable_physics": bool(scene.get("enable_physics", False)),
    }


def _resolved_assets(
    inputs: ValidatedM1Inputs, runtime_root: Path
) -> list[dict[str, Any]]:
    environment = _environment_for_paths(runtime_root)
    records: list[dict[str, Any]] = []
    for asset in inputs.room["assets"]:
        path = _resolved_room_asset_path(inputs, asset, environment)
        record = {
            "role": asset["role"],
            "declared_path": asset["path"],
            "resolved_path": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            record.update(
                {
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        records.append(record)
    return records


def _asset_by_role(records: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("role") == role), None)


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _producer_source_locator_report(
    snapshot: Any,
) -> tuple[bool, dict[str, Any]]:
    value = snapshot if isinstance(snapshot, dict) else {}
    root_raw = value.get("repository_root")
    root = (
        Path(root_raw)
        if isinstance(root_raw, str) and root_raw and Path(root_raw).is_absolute()
        else None
    )
    expected_project = (
        root / SPEAR_PROJECT_REPOSITORY_RELATIVE if root is not None else None
    )
    expected_map = (
        root / SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE if root is not None else None
    )
    project_matches = bool(
        expected_project is not None
        and value.get("actual_project_dir") == str(expected_project)
    )
    map_matches = bool(
        expected_map is not None and value.get("map_package_path") == str(expected_map)
    )
    measured = {
        "repository_root": root_raw,
        "project_repository_relative_path": (
            SPEAR_PROJECT_REPOSITORY_RELATIVE.as_posix()
        ),
        "declared_project_path": value.get("actual_project_dir"),
        "project_locator_matches": project_matches,
        "map_package_repository_relative_path": (
            SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE.as_posix()
        ),
        "declared_map_package_path": value.get("map_package_path"),
        "map_locator_matches": map_matches,
    }
    return bool(root is not None and project_matches and map_matches), measured


def _ue_project_asset_package_closure(
    report: dict[str, Any], source_root: Path | None
) -> tuple[bool, dict[str, Any]]:
    records = report.get("selected_project_asset_packages")
    producer_locator_passed, producer_locator = _producer_source_locator_report(
        report.get("source_snapshot")
    )
    measured: dict[str, Any] = {
        "record_count": len(records) if isinstance(records, list) else None,
        "declared_count": report.get("selected_project_asset_package_count"),
        "producer_source_locator": producer_locator,
        "errors": [],
    }
    errors: list[str] = measured["errors"]
    if not producer_locator_passed:
        errors.append("producer SPEAR source locator is inconsistent")
        return False, measured
    producer_root = Path(producer_locator["repository_root"])
    if source_root is None or not source_root.is_dir():
        errors.append("SPEAR source root is unavailable")
        return False, measured
    if not isinstance(records, list) or not records:
        errors.append("selected project package closure is missing")
        return False, measured
    if report.get("selected_project_asset_package_count") != len(records):
        errors.append("selected project package count differs from records")
    tracked_result = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if tracked_result.returncode != 0:
        errors.append("unable to enumerate SPEAR tracked files")
        return False, measured
    tracked = {
        value.decode("utf-8") for value in tracked_result.stdout.split(b"\0") if value
    }
    actor_project_paths: set[str] = set()
    actor_engine_paths: set[str] = set()
    for actor in report.get("actors", []):
        for component in actor.get("static_mesh_components", []):
            for value in [
                component.get("static_mesh_asset"),
                *component.get("material_assets", []),
            ]:
                if not isinstance(value, str):
                    continue
                if value.startswith("/Game/"):
                    actor_project_paths.add(value)
                elif value.startswith("/Engine/"):
                    actor_engine_paths.add(value)
    recorded_paths: set[str] = set()
    package_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"package record {index} is not an object")
            continue
        package_name = record.get("package_name")
        if not isinstance(package_name, str) or not package_name.startswith("/Game/"):
            errors.append(f"package record {index} has an invalid package name")
            continue
        if package_name in package_names:
            errors.append(f"duplicate package record: {package_name}")
        package_names.add(package_name)
        expected_relative = (
            SPEAR_PROJECT_CONTENT_REPOSITORY_RELATIVE
            / f"{package_name.removeprefix('/Game/')}.uasset"
        ).as_posix()
        relative = record.get("repository_relative_path")
        if relative != expected_relative or relative not in tracked:
            errors.append(
                f"package path is not the tracked expected file: {package_name}"
            )
            continue
        producer_path_raw = record.get("resolved_path")
        if not (
            isinstance(producer_path_raw, str)
            and Path(producer_path_raw).is_absolute()
            and Path(producer_path_raw) == producer_root / relative
        ):
            errors.append(f"producer package locator mismatch: {package_name}")
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError:
            errors.append(f"package path escapes SPEAR: {package_name}")
            continue
        if (
            not path.is_file()
            or record.get("git_tracked") is not True
            or record.get("byte_size") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            errors.append(f"package bytes or tracking changed: {package_name}")
        object_paths = record.get("asset_object_paths")
        if not isinstance(object_paths, list) or not object_paths:
            errors.append(f"package has no object paths: {package_name}")
            continue
        if not all(
            isinstance(value, str) and value.split(".", 1)[0] == package_name
            for value in object_paths
        ):
            errors.append(f"package object path mismatch: {package_name}")
        recorded_paths.update(value for value in object_paths if isinstance(value, str))
    if recorded_paths != actor_project_paths:
        errors.append("selected /Game actor assets differ from the package closure")
    if sorted(actor_engine_paths) != report.get("selected_engine_asset_references"):
        errors.append("selected /Engine asset reference set changed")
    measured.update(
        {
            "selected_project_object_count": len(actor_project_paths),
            "recorded_project_object_count": len(recorded_paths),
            "selected_engine_reference_count": len(actor_engine_paths),
        }
    )
    return not errors, measured


def _load_record_json(
    records: list[dict[str, Any]], role: str
) -> tuple[dict[str, Any] | None, str | None]:
    record = _asset_by_role(records, role)
    if record is None:
        return None, f"Missing asset role: {role}"
    try:
        value = load_json(record["resolved_path"])
    except (OSError, ValueError) as error:
        return None, f"Unable to load {role}: {type(error).__name__}: {error}"
    return value, None


def _provenance_source_locator_report(
    inputs: ValidatedM1Inputs,
    source_root: Path | None,
    environment: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    provenance = inputs.room.get("provenance", {})
    project_key = "source_project_repository_relative_path"
    map_key = "source_map_package_repository_relative_path"
    project_raw = provenance.get(project_key)
    map_relative_raw = provenance.get(map_key)
    relative_keys_present = [project_key in provenance, map_key in provenance]
    portable_locator = all(relative_keys_present)
    legacy_locator = not any(relative_keys_present)
    relative_paths_match = bool(
        legacy_locator
        or (
            portable_locator
            and project_raw == SPEAR_PROJECT_REPOSITORY_RELATIVE.as_posix()
            and map_relative_raw == SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE.as_posix()
        )
    )

    root_raw = provenance.get("source_repository_root")
    map_path_raw = provenance.get("source_map_package_path")
    root_locator_matches = False
    map_locator_matches = False
    locator_error: str | None = None
    try:
        if portable_locator:
            declared_root = resolve_declared_path(
                root_raw,
                manifest_dir=inputs.room_path.parent,
                environment=environment,
            )
            declared_map = resolve_declared_path(
                map_path_raw,
                manifest_dir=inputs.room_path.parent,
                environment=environment,
            )
            root_locator_matches = (
                source_root is not None and declared_root == source_root
            )
            map_locator_matches = bool(
                source_root is not None
                and declared_map
                == (source_root / SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE).resolve()
            )
        elif legacy_locator:
            producer_root = (
                Path(root_raw)
                if isinstance(root_raw, str)
                and root_raw
                and Path(root_raw).is_absolute()
                else None
            )
            root_locator_matches = producer_root is not None
            map_locator_matches = bool(
                producer_root is not None
                and isinstance(map_path_raw, str)
                and Path(map_path_raw).is_absolute()
                and Path(map_path_raw)
                == producer_root / SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE
            )
    except (OSError, TypeError, ValueError) as error:
        locator_error = f"{type(error).__name__}: {error}"

    current_project_exists = bool(
        source_root is not None
        and (source_root / SPEAR_PROJECT_REPOSITORY_RELATIVE).is_dir()
    )
    measured = {
        "source_repository_root": root_raw,
        "source_project_repository_relative_path": (
            project_raw or SPEAR_PROJECT_REPOSITORY_RELATIVE.as_posix()
        ),
        "source_map_package_path": map_path_raw,
        "source_map_package_repository_relative_path": (
            map_relative_raw or SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE.as_posix()
        ),
        "portable_locator": portable_locator,
        "legacy_locator": legacy_locator,
        "relative_paths_match": relative_paths_match,
        "root_locator_matches": root_locator_matches,
        "map_locator_matches": map_locator_matches,
        "current_project_exists": current_project_exists,
        "locator_error": locator_error,
    }
    return bool(
        relative_paths_match
        and root_locator_matches
        and map_locator_matches
        and current_project_exists
        and locator_error is None
    ), measured


def _surface_provenance_check(
    inputs: ValidatedM1Inputs, asset_records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    room_kind = inputs.room["room_kind"]
    render_record = _asset_by_role(asset_records, "render_surface_mesh")
    surface_audit = inputs.room.get("surface_audit", {})
    if room_kind == "blender_custom":
        report, error = _load_record_json(asset_records, "blender_build_report")
        opening_ids = sorted(
            opening["opening_id"] for opening in inputs.room.get("openings", [])
        )
        measured: dict[str, Any] = {
            "load_error": error,
            "aabb_proxy": surface_audit.get("aabb_proxy"),
            "declared_openings": opening_ids,
        }
        passed = error is None and report is not None and render_record is not None
        if report is not None and render_record is not None:
            stage_output = report.get("outputs", {}).get(
                "stages/m1_custom_room.glb", {}
            )
            measured.update(
                {
                    "schema": report.get("schema"),
                    "geometry_representation": report.get("geometry_representation"),
                    "stage_object_count": report.get("stage_object_count"),
                    "stage_triangle_count": report.get("stage_triangle_count"),
                    "report_openings": sorted(report.get("openings", [])),
                    "report_mesh_sha256": stage_output.get("sha256"),
                    "actual_mesh_sha256": render_record.get("sha256"),
                }
            )
            passed = bool(
                passed
                and report.get("schema") == "avengine_blender_room_build_report_v1"
                and report.get("geometry_representation") == "real_surface_mesh"
                and isinstance(report.get("stage_object_count"), int)
                and report["stage_object_count"] > 0
                and isinstance(report.get("stage_triangle_count"), int)
                and report["stage_triangle_count"] > 0
                and sorted(report.get("openings", [])) == opening_ids
                and stage_output.get("sha256") == render_record.get("sha256")
                and surface_audit.get("aabb_proxy") is False
            )
        return make_check(
            "blender_authored_surface_provenance",
            "pass" if passed else "fail",
            measured=measured,
            threshold={
                "tracked_build_report_matches_mesh": True,
                "modeled_openings_match": True,
                "aabb_proxy": False,
            },
            failure_reason=None
            if passed
            else "Blender authored-surface provenance did not validate",
        )

    if room_kind != "legacy_ue_real_surface_export":
        return None

    ue_report, ue_error = _load_record_json(asset_records, "ue_export_manifest")
    mesh_report, mesh_error = _load_record_json(
        asset_records, "real_surface_mesh_audit"
    )
    source_map_record = _asset_by_role(asset_records, "legacy_source_map_package")
    provenance = inputs.room.get("provenance", {})
    source_root_raw = provenance.get("source_repository_root")
    environment = dict(os.environ)
    source_root_error: str | None = None
    try:
        source_root = _resolved_spear_source_root(inputs, environment)
    except (OSError, TypeError, ValueError) as error:
        source_root = None
        source_root_error = f"{type(error).__name__}: {error}"
    provenance_locator_passed, provenance_locator = _provenance_source_locator_report(
        inputs, source_root, environment
    )
    expected_source_map = (
        (source_root / SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE).resolve()
        if source_root is not None
        else None
    )
    current_source_commit = (
        _git_value(source_root, "rev-parse", "HEAD")
        if source_root is not None and source_root.is_dir()
        else None
    )
    current_source_tracked_status = (
        _git_value(
            source_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        if source_root is not None and source_root.is_dir()
        else None
    )
    current_source_toplevel = (
        _git_value(source_root, "rev-parse", "--show-toplevel")
        if source_root is not None and source_root.is_dir()
        else None
    )
    current_source_map_tracked = (
        _git_value(
            source_root,
            "ls-files",
            "--error-unmatch",
            "--",
            SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE.as_posix(),
        )
        if source_root is not None and source_root.is_dir()
        else None
    )
    measured = {
        "ue_manifest_error": ue_error,
        "mesh_audit_error": mesh_error,
        "manifest_surface_audit": surface_audit,
        "source_repository_locator": source_root_raw,
        "resolved_source_repository_root": (
            str(source_root) if source_root is not None else None
        ),
        "source_repository_resolution_error": source_root_error,
        "provenance_source_locator": provenance_locator,
        "current_source_commit": current_source_commit,
        "current_source_tracked_status": current_source_tracked_status,
        "current_source_toplevel": current_source_toplevel,
        "current_source_map_tracked": current_source_map_tracked,
        "source_map_record": source_map_record,
    }
    passed = (
        ue_error is None
        and mesh_error is None
        and ue_report is not None
        and mesh_report is not None
        and render_record is not None
        and source_map_record is not None
    )
    if ue_report is not None and mesh_report is not None and render_record is not None:
        export_output = ue_report.get("output", {})
        export_messages = ue_report.get("export_messages", {})
        export_source_snapshot = ue_report.get("source_snapshot", {})
        export_source_snapshot_after = ue_report.get("source_snapshot_after_export", {})
        producer_locator_passed, producer_locator = _producer_source_locator_report(
            export_source_snapshot
        )
        producer_after_locator_passed, producer_after_locator = (
            _producer_source_locator_report(export_source_snapshot_after)
        )
        producer_locator_matches_provenance = bool(
            provenance_locator.get("portable_locator") is True
            or (
                provenance_locator.get("legacy_locator") is True
                and export_source_snapshot.get("repository_root")
                == provenance.get("source_repository_root")
                and export_source_snapshot.get("map_package_path")
                == provenance.get("source_map_package_path")
            )
        )

        dirty_packages = ue_report.get("dirty_packages", {})
        gate = mesh_report.get("real_surface_gate", {})
        indicators = mesh_report.get("aabb_proxy_indicators", {})
        package_closure_passed, package_closure = _ue_project_asset_package_closure(
            ue_report, source_root
        )
        measured.update(
            {
                "ue_schema": ue_report.get("schema"),
                "loaded_editor_world": ue_report.get("loaded_editor_world"),
                "engine_version": ue_report.get("engine_version"),
                "gltf_exporter_plugin": ue_report.get("gltf_exporter_plugin"),
                "geometry_source": ue_report.get("geometry_source"),
                "uses_actor_bounds_as_geometry": ue_report.get(
                    "uses_actor_bounds_as_geometry"
                ),
                "selected_actor_count": ue_report.get("selected_actor_count"),
                "static_mesh_component_count": ue_report.get(
                    "static_mesh_component_count"
                ),
                "unique_static_mesh_asset_count": ue_report.get(
                    "unique_static_mesh_asset_count"
                ),
                "option_warnings": ue_report.get("option_warnings"),
                "export_errors": export_messages.get("errors"),
                "export_source_snapshot": export_source_snapshot,
                "export_source_snapshot_after": export_source_snapshot_after,
                "producer_source_locator": producer_locator,
                "producer_source_locator_after_export": producer_after_locator,
                "producer_locator_matches_provenance": (
                    producer_locator_matches_provenance
                ),
                "actual_project_dir": ue_report.get("actual_project_dir"),
                "dirty_packages": dirty_packages,
                "triangles": mesh_report.get("triangles"),
                "meshes": mesh_report.get("meshes"),
                "materials": mesh_report.get("materials"),
                "mesh_gate": gate,
                "aabb_indicators": indicators,
                "actual_mesh_sha256": render_record.get("sha256"),
                "ue_mesh_sha256": export_output.get("sha256"),
                "audit_mesh_sha256": mesh_report.get("sha256"),
                "selected_project_asset_package_closure": package_closure,
            }
        )
        triangles = mesh_report.get("triangles")
        passed = bool(
            passed
            and ue_report.get("schema") == "avengine_legacy_ue_apartment_export_v1"
            and ue_report.get("status") == "pass"
            and ue_report.get("source_map_asset")
            == "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
            and export_source_snapshot.get("schema")
            == "avengine_spear_source_snapshot_v1"
            and export_source_snapshot.get("capture_phase") == "before_ue_gltf_export"
            and export_source_snapshot_after
            == {
                **export_source_snapshot,
                "capture_phase": "after_ue_gltf_export",
            }
            and producer_locator_passed
            and producer_after_locator_passed
            and producer_locator_matches_provenance
            and export_source_snapshot.get("map_asset")
            == ue_report.get("source_map_asset")
            and str(ue_report.get("loaded_editor_world", "")).startswith(
                "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000."
            )
            and isinstance(ue_report.get("engine_version"), str)
            and ue_report["engine_version"].startswith("5.5.")
            and ue_report.get("gltf_exporter_plugin", {}).get("version_name") == "1.3.1"
            and package_closure_passed
            and ue_report.get("geometry_source") == "UE StaticMesh render data LOD0"
            and ue_report.get("uses_actor_bounds_as_geometry") is False
            and ue_report.get("option_warnings") == []
            and export_messages.get("errors") == []
            and _positive_integer(ue_report.get("selected_actor_count"))
            and _positive_integer(ue_report.get("static_mesh_component_count"))
            and _positive_integer(ue_report.get("unique_static_mesh_asset_count"))
            and mesh_report.get("schema") == "avengine_real_surface_mesh_audit_v1"
            and gate.get("status") == "pass"
            and isinstance(triangles, int)
            and triangles > 252
            and _positive_integer(mesh_report.get("meshes"))
            and _positive_integer(mesh_report.get("materials"))
            and indicators.get("known_legacy_triangle_signature") is False
            and indicators.get("all_mesh_nodes_are_simple_boxes") is False
            and export_output.get("sha256") == render_record.get("sha256")
            and mesh_report.get("sha256") == render_record.get("sha256")
            and export_output.get("byte_size") == render_record.get("byte_size")
            and mesh_report.get("bytes") == render_record.get("byte_size")
            and surface_audit.get("aabb_proxy") is False
            and surface_audit.get("triangle_count") == triangles
            and surface_audit.get("mesh_sha256") == render_record.get("sha256")
            and surface_audit.get("real_surface_gate_status") == "pass"
            and ue_report.get("actual_project_dir")
            == export_source_snapshot.get("actual_project_dir")
            and provenance_locator_passed
            and dirty_packages.get("before_reload") == {"content": [], "maps": []}
            and dirty_packages.get("after_reload") == {"content": [], "maps": []}
            and dirty_packages.get("after_export") == {"content": [], "maps": []}
            and export_source_snapshot.get("commit")
            == provenance.get("source_revision")
            == current_source_commit
            and isinstance(current_source_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", current_source_commit) is not None
            and export_source_snapshot.get("tracked_worktree_dirty") is False
            and provenance.get("source_repository_tracked_dirty") is False
            and current_source_tracked_status == ""
            and source_root is not None
            and current_source_toplevel == str(source_root)
            and expected_source_map is not None
            and current_source_map_tracked
            == SPEAR_MAP_PACKAGE_REPOSITORY_RELATIVE.as_posix()
            and source_map_record.get("resolved_path") == str(expected_source_map)
            and export_source_snapshot.get("map_package_sha256")
            == provenance.get("source_map_package_sha256")
            == source_map_record.get("sha256")
            and provenance.get("exported_scene_sha256") == render_record.get("sha256")
        )
    return make_check(
        "legacy_real_surface_provenance",
        "pass" if passed else "fail",
        measured=measured,
        threshold={
            "source": "UE StaticMesh render data LOD0",
            "minimum_triangles": 253,
            "known_252_triangle_proxy": False,
            "all_simple_boxes": False,
            "hash_chain_matches": True,
            "pre_export_clean_spear_commit_and_map_package_hash_match": True,
        },
        failure_reason=None
        if passed
        else "Legacy real-surface export/audit hash chain did not validate",
    )


def _make_configuration(
    inputs: ValidatedM1Inputs,
    runtime_root: Path,
    output_dir: Path,
) -> tuple[Any, dict[str, str], str, dict[str, Any]]:
    qt, habitat_sim, mn, _ = _import_habitat()
    del qt
    resolved = _resolved_scene(inputs, runtime_root)
    rig = inputs.request["primary_camera_rig"]
    calibration = rig["shared_calibration"]
    height, width = calibration["resolution_hw"]
    local = calibration["rig_from_sensor"]
    local_position = mn.Vector3(local["translation_m"])
    local_orientation = mn.Vector3(0.0, 0.0, 0.0)

    modality_to_uuid = {
        item["modality"]: item["sensor_uuid"] for item in rig["modalities"]
    }
    sensor_specs: list[Any] = []
    for modality in ("rgb", "depth", "semantic"):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = modality_to_uuid[modality]
        spec.sensor_type = getattr(
            habitat_sim.SensorType, VISUAL_SENSOR_TYPES[modality]
        )
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = mn.Vector2i([height, width])
        spec.position = local_position
        spec.orientation = local_orientation
        spec.hfov = float(calibration["hfov_degrees"])
        spec.near = float(calibration["near_m"])
        spec.far = float(calibration["far_m"])
        spec.gpu2gpu_transfer = False
        spec.noise_model = "None"
        if modality != "rgb":
            spec.channels = 1
        if modality == "semantic" and hasattr(spec, "semantic_target"):
            from habitat_sim._ext import habitat_sim_bindings

            spec.semantic_target = habitat_sim_bindings.SemanticSensorTarget.SEMANTIC_ID
        sensor_specs.append(spec)

    listener = inputs.request["listener"]
    audio_spec = habitat_sim.AudioSensorSpec()
    audio_spec.uuid = listener["listener_id"]
    audio_spec.position = local_position
    audio_spec.orientation = local_orientation
    # This sensor is a pose anchor in M1. It is deliberately excluded from
    # render_sensors(), so the deprecated one-source RLR wrapper never runs.
    audio_spec.outputDirectory = str(output_dir / "audio_not_run")
    sensor_specs.append(audio_spec)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(resolved["scene_id"])
    sim_cfg.scene_dataset_config_file = str(resolved["dataset_config"])
    sim_cfg.load_semantic_mesh = resolved["load_semantic_mesh"]
    sim_cfg.enable_physics = resolved["enable_physics"]
    sim_cfg.random_seed = int(inputs.request["seed"])
    sim_cfg.gpu_device_id = 0

    agent_cfg = habitat_sim.AgentConfiguration()
    navigation = inputs.room.get("navigation", {})
    agent_cfg.height = float(navigation.get("agent_height_m", 1.5))
    agent_cfg.radius = float(navigation.get("agent_radius_m", 0.2))
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {}

    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.agent_height = agent_cfg.height
    nav_settings.agent_radius = agent_cfg.radius
    nav_settings.include_static_objects = bool(
        navigation.get("include_static_objects", False)
    )
    sim_cfg.navmesh_settings = nav_settings
    return (
        habitat_sim.Configuration(sim_cfg, [agent_cfg]),
        modality_to_uuid,
        listener["listener_id"],
        resolved,
    )


def _numpy_quaternion(xyzw: list[float], qt: Any) -> Any:
    x, y, z, w = xyzw
    return qt.quaternion(w, x, y, z)


def _pose_dict(pose: Any, quat_to_coeffs: Any) -> dict[str, list[float]]:
    return {
        "translation_m": np.asarray(pose.position, dtype=np.float64).tolist(),
        "rotation_xyzw": np.asarray(
            quat_to_coeffs(pose.rotation), dtype=np.float64
        ).tolist(),
    }


def _state_snapshot(
    sim: Any,
    agent: Any,
    sensor_uuids: list[str],
    quat_to_coeffs: Any,
) -> dict[str, Any]:
    state = agent.get_state()
    return {
        "world_time_seconds": float(sim.get_world_time()),
        "agent": _pose_dict(state, quat_to_coeffs),
        "sensors": {
            uuid: _pose_dict(state.sensor_states[uuid], quat_to_coeffs)
            for uuid in sorted(sensor_uuids)
        },
    }


def _repeat_hashes(
    captures: list[dict[str, np.ndarray]], modality_to_uuid: dict[str, str]
) -> list[dict[str, str]]:
    return [
        {
            modality: array_sha256(uuid, capture[uuid])
            for modality, uuid in sorted(modality_to_uuid.items())
        }
        for capture in captures
    ]


def _connectivity_checks(
    sim: Any, inputs: ValidatedM1Inputs
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, habitat_sim, _, _ = _import_habitat()
    reports: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    pairs = inputs.room.get("connectivity_pairs", [])
    if not pairs:
        return reports, checks
    if not sim.pathfinder.is_loaded:
        for pair in pairs:
            checks.append(
                make_check(
                    f"connectivity_{pair['pair_id']}",
                    "blocked",
                    measured={"pathfinder_loaded": False},
                    threshold={"path_found": True},
                    failure_reason="No navmesh is loaded",
                )
            )
        return reports, checks

    for pair in pairs:
        start = np.asarray(pair["start_m"], dtype=np.float32)
        end = np.asarray(pair["end_m"], dtype=np.float32)
        snapped_start = np.asarray(sim.pathfinder.snap_point(start), dtype=np.float64)
        snapped_end = np.asarray(sim.pathfinder.snap_point(end), dtype=np.float64)
        start_snap_distance = float(
            np.linalg.norm(snapped_start - start.astype(np.float64))
        )
        end_snap_distance = float(np.linalg.norm(snapped_end - end.astype(np.float64)))
        maximum_snap_distance = 0.30
        query = habitat_sim.ShortestPath()
        query.requested_start = snapped_start
        query.requested_end = snapped_end
        found = bool(sim.pathfinder.find_path(query))
        report = {
            "pair_id": pair["pair_id"],
            "requested_start_m": start.astype(np.float64).tolist(),
            "requested_end_m": end.astype(np.float64).tolist(),
            "snapped_start_m": snapped_start.tolist(),
            "snapped_end_m": snapped_end.tolist(),
            "start_snap_distance_m": start_snap_distance,
            "end_snap_distance_m": end_snap_distance,
            "found": found,
            "geodesic_distance_m": float(query.geodesic_distance) if found else None,
            "path_point_count": len(query.points) if found else 0,
        }
        reports.append(report)
        passed = bool(
            found
            and np.isfinite(query.geodesic_distance)
            and np.isfinite(start_snap_distance)
            and np.isfinite(end_snap_distance)
            and start_snap_distance <= maximum_snap_distance
            and end_snap_distance <= maximum_snap_distance
        )
        checks.append(
            make_check(
                f"connectivity_{pair['pair_id']}",
                "pass" if passed else "fail",
                measured=report,
                threshold={
                    "path_found": True,
                    "finite_distance": True,
                    "maximum_snap_distance_m": maximum_snap_distance,
                },
                failure_reason=None
                if passed
                else "ShortestPath did not connect the pair",
            )
        )
    return reports, checks


def _ray_checks(
    sim: Any, inputs: ValidatedM1Inputs
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, habitat_sim, mn, _ = _import_habitat()
    reports: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for declared in inputs.room.get("ray_checks", []):
        ray = habitat_sim.geo.Ray(
            mn.Vector3(declared["origin_m"]), mn.Vector3(declared["direction"])
        )
        results = sim.cast_ray(ray, buffer_distance=0.0)
        nearest = float(results.hits[0].ray_distance) if results.has_hits() else None
        distance = float(declared["distance_m"])
        if declared["expectation"] == "clear_until_m":
            passed = nearest is None or nearest > distance
        else:
            passed = nearest is not None and nearest <= distance
        report = {
            "check_id": declared["check_id"],
            "expectation": declared["expectation"],
            "distance_m": distance,
            "nearest_hit_m": nearest,
            "passed": passed,
        }
        reports.append(report)
        checks.append(
            make_check(
                f"ray_{declared['check_id']}",
                "pass" if passed else "fail",
                measured=report,
                threshold={
                    "expectation": declared["expectation"],
                    "distance_m": distance,
                },
                failure_reason=None if passed else "Opening/control ray did not match",
            )
        )
    return reports, checks


def _save_topdown(
    sim: Any, inputs: ValidatedM1Inputs, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for view in inputs.request.get("qa_views", []):
        if view.get("kind") != "topdown":
            continue
        if not sim.pathfinder.is_loaded:
            checks.append(
                make_check(
                    f"qa_{view['qa_id']}",
                    "blocked",
                    measured={"pathfinder_loaded": False},
                    threshold={"artifact_written": True},
                    failure_reason="No navmesh is loaded",
                )
            )
            continue
        meters_per_pixel = float(view.get("meters_per_pixel", 0.05))
        height = float(view.get("height_m", 0.1))
        topdown = np.asarray(
            sim.pathfinder.get_topdown_view(meters_per_pixel, height), dtype=np.uint8
        )
        qa_dir = output_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        path = qa_dir / f"{view['qa_id']}.png"
        Image.fromarray(topdown * 255, mode="L").save(path)
        report = {
            "qa_id": view["qa_id"],
            "kind": "topdown",
            "formal_view": False,
            "meters_per_pixel": meters_per_pixel,
            "height_m": height,
            "shape": list(topdown.shape),
            "navigable_pixel_count": int(np.count_nonzero(topdown)),
            "artifact": file_record(path, relative_to=output_dir),
        }
        reports.append(report)
        passed = topdown.size > 0 and report["navigable_pixel_count"] > 0
        checks.append(
            make_check(
                f"qa_{view['qa_id']}",
                "pass" if passed else "fail",
                measured={
                    "shape": list(topdown.shape),
                    "formal_view": False,
                    "navigable_pixel_count": report["navigable_pixel_count"],
                },
                threshold={
                    "nonempty": True,
                    "minimum_navigable_pixel_count": 1,
                    "formal_view": False,
                },
                artifact=report["artifact"]["path"],
                failure_reason=None
                if passed
                else "Topdown QA artifact has no navigable pixels",
            )
        )
    return reports, checks


def _source_roundtrip(inputs: ValidatedM1Inputs) -> tuple[list[dict[str, Any]], float]:
    world_from_rig = inputs.request["primary_camera_rig"]["world_from_rig"]
    rig_from_world = invert_transform(world_from_rig)
    reports: list[dict[str, Any]] = []
    maximum = 0.0
    for source in inputs.request["sources"]:
        world_from_source = source["world_from_source"]
        rig_from_source = compose_transforms(rig_from_world, world_from_source)
        recovered, error = round_trip_via_parent(world_from_rig, world_from_source)
        maximum = max(maximum, error)
        reports.append(
            {
                "source_id": source["source_id"],
                "world_from_source": world_from_source,
                "rig_from_source": rig_from_source,
                "recovered_world_from_source": recovered,
                "roundtrip_max_error": error,
            }
        )
    return reports, maximum


def _independent_process_repeatability_check(
    *,
    reference_path: str | Path | None,
    inputs: ValidatedM1Inputs,
    observation_records: dict[str, dict[str, Any]],
    state_hash: str,
    runtime_commit: str | None,
    native_binding_sha256: str,
    asset_records: list[dict[str, Any]],
    avengine_commit: str | None,
    repository_clean: bool,
    runtime_clean: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if reference_path is None:
        return (
            make_check(
                "independent_process_repeatability",
                "not_run",
                measured={"reference_evidence": None},
                threshold={"independent_reference_matches": True},
                failure_reason=(
                    "Run capture once, then rerun in a fresh process with "
                    "--reference-evidence pointing to the first evidence.json"
                ),
            ),
            None,
        )

    resolved = Path(reference_path).resolve()
    measured: dict[str, Any] = {"reference_evidence": str(resolved)}
    try:
        reference_verification_status, reference_verification_checks = (
            verify_evidence_artifacts(resolved, _allow_reference=True)
        )
        reference = load_json(resolved)
        expected_hash = reference.get("evidence_content_sha256")
        actual_hash = canonical_json_sha256(
            {
                key: value
                for key, value in reference.items()
                if key != "evidence_content_sha256"
            }
        )
        reference_observation_hashes = {
            modality: reference.get("observations", {})
            .get(modality, {})
            .get("raw_array_sha256")
            for modality in ("rgb", "depth", "semantic")
        }
        current_observation_hashes = {
            modality: observation_records[modality]["raw_array_sha256"]
            for modality in ("rgb", "depth", "semantic")
        }
        reference_assets = {
            record.get("role"): (record.get("sha256"), record.get("byte_size"))
            for record in reference.get("scene_assets", [])
            if isinstance(record, dict)
        }
        current_assets = {
            record.get("role"): (record.get("sha256"), record.get("byte_size"))
            for record in asset_records
        }
        comparisons = {
            "reference_fully_verified": reference_verification_status == "pass",
            "content_hash_valid": expected_hash == actual_hash,
            "schema_matches": reference.get("schema") == EVIDENCE_SCHEMA,
            "evidence_kind_matches": reference.get("evidence_kind")
            == "completed_capture",
            "reference_is_first_run": reference.get("overall_status") == "not_run"
            and reference.get("independent_reference") is None,
            "room_id_matches": reference.get("room_id") == inputs.room["room_id"],
            "request_id_matches": reference.get("request_id")
            == inputs.request["request_id"],
            "room_manifest_hash_matches": reference.get("room_manifest", {}).get(
                "sha256"
            )
            == sha256_file(inputs.room_path),
            "capture_request_hash_matches": reference.get("capture_request", {}).get(
                "sha256"
            )
            == sha256_file(inputs.request_path),
            "runtime_commit_matches": reference.get("runtime", {}).get(
                "habitat_runtime_commit"
            )
            == runtime_commit,
            "runtime_was_and_is_clean": reference.get("runtime", {}).get(
                "habitat_runtime_worktree_dirty"
            )
            is False
            and runtime_clean,
            "avengine_commit_matches": reference.get("runtime", {}).get(
                "avengine_commit"
            )
            == avengine_commit,
            "avengine_was_and_is_clean": reference.get("runtime", {}).get(
                "avengine_worktree_dirty"
            )
            is False
            and repository_clean,
            "native_binding_matches": reference.get("runtime", {}).get(
                "native_binding_sha256"
            )
            == native_binding_sha256,
            "scene_assets_match": reference_assets == current_assets,
            "initial_state_matches": reference.get("capture_state", {}).get(
                "before_sha256"
            )
            == state_hash,
            "raw_observation_hashes_match": reference_observation_hashes
            == current_observation_hashes,
            "fresh_process_instance": reference.get("producer_process", {}).get(
                "process_instance_id"
            )
            != PROCESS_INSTANCE_ID,
        }
        passed = all(comparisons.values())
        measured.update(
            {
                "reference_content_sha256": expected_hash,
                "reference_overall_status": reference.get("overall_status"),
                "reference_verification_status": reference_verification_status,
                "reference_verification_checks": reference_verification_checks,
                "comparisons": comparisons,
                "reference_observation_hashes": reference_observation_hashes,
                "current_observation_hashes": current_observation_hashes,
                "reference_process": reference.get("producer_process"),
                "current_process": _producer_process_identity(),
            }
        )
        if passed:
            copied_root = output_dir / "independent_reference"
            if copied_root.is_dir():
                shutil.rmtree(copied_root)
            copied_root.mkdir(parents=True, exist_ok=True)
            copied_evidence = copied_root / "evidence.json"
            shutil.copy2(resolved, copied_evidence)
            for directory_name in ("observations", "qa"):
                source_directory = resolved.parent / directory_name
                if source_directory.is_dir():
                    shutil.copytree(source_directory, copied_root / directory_name)
            reference_record: dict[str, Any] | None = {
                "path": copied_evidence.relative_to(output_dir).as_posix(),
                "evidence_content_sha256": reference.get("evidence_content_sha256"),
                "artifact": file_record(copied_evidence, relative_to=output_dir),
            }
        else:
            reference_record = None
    except (OSError, ValueError, TypeError) as error:
        reference = None
        passed = False
        reference_record = None
        measured["exception"] = f"{type(error).__name__}: {error}"

    return (
        make_check(
            "independent_process_repeatability",
            "pass" if passed else "fail",
            measured=measured,
            threshold={"all_identity_state_asset_and_observation_comparisons": True},
            failure_reason=None
            if passed
            else "Fresh-process evidence does not match the reference run",
        ),
        reference_record,
    )


def capture_m1(
    inputs: ValidatedM1Inputs,
    output_dir: str | Path,
    *,
    runtime_root: str | Path | None = None,
    repeat_count: int = 3,
    reference_evidence: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in (
        output / "observations",
        output / "qa",
        output / "independent_reference",
    ):
        if stale.is_dir():
            shutil.rmtree(stale)
    (output / "evidence.json").unlink(missing_ok=True)
    runtime = discover_runtime_root(runtime_root)
    repository_root = Path(__file__).resolve().parents[3]
    qt, habitat_sim, _, quat_to_coeffs = _import_habitat()
    from habitat_sim._ext import habitat_sim_bindings

    configuration, modality_to_uuid, listener_uuid, resolved_scene = (
        _make_configuration(inputs, runtime, output)
    )
    repeat_count = max(2, int(repeat_count))
    asset_records = _resolved_assets(inputs, runtime)
    missing_assets = [record for record in asset_records if not record["exists"]]
    if missing_assets:
        raise FileNotFoundError(
            "Required room assets are missing: "
            + ", ".join(record["declared_path"] for record in missing_assets)
        )
    scene_graph_errors = validate_scene_asset_graph(inputs, runtime)

    formal_view_passed = (
        inputs.request["primary_camera_rig"]["rig_id"] == "camera_rig_0"
        and inputs.request["primary_camera_rig"]["view_id"] == "view0"
        and set(modality_to_uuid) == set(VISUAL_SENSOR_TYPES)
    )
    checks: list[dict[str, Any]] = [
        make_check(
            "single_formal_view",
            "pass" if formal_view_passed else "fail",
            measured={
                "view_ids": [inputs.request["primary_camera_rig"]["view_id"]],
                "modality_count": len(modality_to_uuid),
            },
            threshold={
                "formal_view_count": 1,
                "modalities": sorted(VISUAL_SENSOR_TYPES),
            },
            failure_reason=None
            if formal_view_passed
            else "M1 requires camera_rig_0/view0 and exactly three shared modalities",
        )
    ]
    scene_graph_check = make_check(
        "scene_load_graph_closure",
        "pass" if not scene_graph_errors else "fail",
        measured={
            "errors": scene_graph_errors,
            "static_errors": scene_graph_errors,
            "loaded_errors": ["Simulator has not been constructed"],
            "loaded_graph": None,
        },
        threshold={
            "errors": [],
            "actual_habitat_scene_resolves_to_declared_assets": True,
        },
        failure_reason=None
        if not scene_graph_errors
        else "Habitat scene selection diverges from the declared asset closure",
    )
    checks.append(scene_graph_check)
    runtime_commit = _git_value(runtime, "rev-parse", "HEAD")
    locked_runtime_commit = _runtime_lock_commit(repository_root)
    runtime_matches = (
        runtime_commit is not None and runtime_commit == locked_runtime_commit
    )
    checks.append(
        make_check(
            "runtime_commit_matches_lock",
            "pass" if runtime_matches else "fail",
            measured=runtime_commit,
            threshold=locked_runtime_commit,
            failure_reason=None
            if runtime_matches
            else "Runtime checkout differs from lock",
        )
    )
    runtime_status = _git_value(runtime, "status", "--porcelain")
    runtime_clean = runtime_status == ""
    checks.append(
        make_check(
            "runtime_worktree_clean",
            "pass" if runtime_clean else "fail",
            measured={"git_status": runtime_status},
            threshold={"dirty": False},
            failure_reason=None
            if runtime_clean
            else "Pinned Habitat runtime worktree is dirty",
        )
    )
    repository_status = _git_value(repository_root, "status", "--porcelain")
    repository_clean = repository_status == ""
    avengine_commit = _git_value(repository_root, "rev-parse", "HEAD")
    checks.append(
        make_check(
            "avengine_worktree_clean",
            "pass" if repository_clean else "fail",
            measured={"git_status": repository_status},
            threshold={"dirty": False},
            failure_reason=None
            if repository_clean
            else "AVEngine worktree is dirty; final evidence must bind a clean commit",
        )
    )
    habitat_module_path = Path(habitat_sim.__file__).resolve()
    native_binding_path = Path(habitat_sim_bindings.__file__).resolve()
    try:
        habitat_module_path.relative_to(runtime)
        native_binding_path.relative_to(runtime)
        binary_origin_passed = True
    except ValueError:
        binary_origin_passed = False
    native_binding_sha256 = sha256_file(native_binding_path)
    checks.append(
        make_check(
            "runtime_binary_origin",
            "pass" if binary_origin_passed else "fail",
            measured={
                "habitat_module": str(habitat_module_path),
                "native_binding": str(native_binding_path),
                "native_binding_sha256": native_binding_sha256,
            },
            threshold={"both_paths_within_runtime_root": str(runtime)},
            failure_reason=None
            if binary_origin_passed
            else "Imported Habitat Python/native binding is not from --runtime-root",
        )
    )
    checks.append(
        make_check(
            "scene_asset_closure",
            "pass",
            measured={
                "asset_count": len(asset_records),
                "roles": sorted(record["role"] for record in asset_records),
                "all_exist": True,
            },
            threshold={"all_declared_assets_exist_and_are_hashed": True},
        )
    )
    surface_check = _surface_provenance_check(inputs, asset_records)
    if surface_check is not None:
        checks.append(surface_check)

    rig = inputs.request["primary_camera_rig"]
    world_from_rig = rig["world_from_rig"]
    calibration = rig["shared_calibration"]
    expected_world_from_sensor = compose_transforms(
        world_from_rig, calibration["rig_from_sensor"]
    )
    state = habitat_sim.AgentState()
    state.position = np.asarray(world_from_rig["translation_m"], dtype=np.float64)
    state.rotation = _numpy_quaternion(world_from_rig["rotation_xyzw"], qt)

    visual_uuids = [modality_to_uuid[key] for key in ("rgb", "depth", "semantic")]
    all_sensor_uuids = visual_uuids + [listener_uuid]
    with habitat_sim.Simulator(configuration) as sim:
        navmesh_path = resolved_scene["navmesh"]
        declared_navmesh_loaded = False
        if navmesh_path is not None and Path(navmesh_path).is_file():
            declared_navmesh_loaded = bool(
                sim.pathfinder.load_nav_mesh(str(navmesh_path))
            )
            if not declared_navmesh_loaded:
                raise RuntimeError("Habitat failed to load the declared navmesh")

        loaded_graph_errors, loaded_graph = validate_loaded_scene_asset_graph(
            inputs,
            runtime,
            sim,
            declared_navmesh_loaded=declared_navmesh_loaded,
        )
        combined_scene_graph_errors = scene_graph_errors + loaded_graph_errors
        scene_graph_check.clear()
        scene_graph_check.update(
            make_check(
                "scene_load_graph_closure",
                "pass" if not combined_scene_graph_errors else "fail",
                measured={
                    "errors": combined_scene_graph_errors,
                    "static_errors": scene_graph_errors,
                    "loaded_errors": loaded_graph_errors,
                    "loaded_graph": loaded_graph,
                },
                threshold={
                    "errors": [],
                    "actual_habitat_scene_resolves_to_declared_assets": True,
                    "declared_navmesh_explicitly_loaded_and_fingerprinted": True,
                },
                failure_reason=None
                if not combined_scene_graph_errors
                else "Habitat loaded scene graph diverges from declared assets",
            )
        )

        requires_navigation_evidence = bool(
            inputs.room.get("connectivity_pairs")
            or any(
                view.get("kind") == "topdown"
                for view in inputs.request.get("qa_views", [])
            )
        )
        if requires_navigation_evidence and not sim.pathfinder.is_loaded:
            raise RuntimeError(
                "M1 navigation evidence requires a loaded navmesh; capture is blocked"
            )

        sim.seed(int(inputs.request["seed"]))
        agent = sim.initialize_agent(0, state)
        before = _state_snapshot(sim, agent, all_sensor_uuids, quat_to_coeffs)
        before_hash = canonical_json_sha256(before)

        captures: list[dict[str, np.ndarray]] = []
        wrappers = [sim.sensors[uuid] for uuid in visual_uuids]
        for _ in range(repeat_count):
            observation = sim.render_sensors(wrappers)
            captures.append(
                {
                    uuid: np.ascontiguousarray(observation[uuid]).copy()
                    for uuid in visual_uuids
                }
            )
        after = _state_snapshot(sim, agent, all_sensor_uuids, quat_to_coeffs)
        after_hash = canonical_json_sha256(after)

        observation_records = save_observations(captures[0], modality_to_uuid, output)
        repeated_hashes = _repeat_hashes(captures, modality_to_uuid)
        repeat_passed = all(item == repeated_hashes[0] for item in repeated_hashes[1:])

        state_unchanged = before_hash == after_hash
        checks.append(
            make_check(
                "capture_state_unchanged",
                "pass" if state_unchanged else "fail",
                measured={"before": before_hash, "after": after_hash},
                threshold={"hashes_equal": True, "world_time_advance_seconds": 0.0},
                failure_reason=None
                if state_unchanged
                else "Capture advanced or changed state",
            )
        )
        checks.append(
            make_check(
                "repeatability_same_process",
                "pass" if repeat_passed else "fail",
                measured=repeated_hashes,
                threshold={"all_repeats_identical": True, "repeat_count": repeat_count},
                failure_reason=None if repeat_passed else "Observation bytes changed",
            )
        )

        readback_poses = before["sensors"]
        pose_errors = {
            uuid: transform_error(readback_poses[uuid], expected_world_from_sensor)
            for uuid in all_sensor_uuids
        }
        maximum_pose_error = max(pose_errors.values())
        alignment_passed = maximum_pose_error <= 1e-7
        checks.append(
            make_check(
                "rig_visual_listener_alignment",
                "pass" if alignment_passed else "fail",
                measured={"pose_errors": pose_errors, "maximum": maximum_pose_error},
                threshold={"maximum_transform_error": 1e-7},
                failure_reason=None
                if alignment_passed
                else "Visual modalities or listener diverged from shared rig pose",
            )
        )

        rgb_stats = observation_records["rgb"]["statistics"]
        rgb_passed = (
            rgb_stats["color_standard_deviation"] > 1.0
            and max(rgb_stats["per_channel_standard_deviation"]) > 1.0
        )
        checks.append(
            make_check(
                "rgb_nonconstant",
                "pass" if rgb_passed else "fail",
                measured=rgb_stats,
                threshold={
                    "minimum_color_standard_deviation": 1.0,
                    "minimum_one_channel_standard_deviation": 1.0,
                    "alpha_channel_excluded": True,
                },
                artifact=observation_records["rgb"]["artifact"]["path"],
                failure_reason=None if rgb_passed else "RGB observation is constant",
            )
        )

        depth_stats = observation_records["depth"]["statistics"]
        depth_max = depth_stats["maximum_finite_m"]
        depth_passed = (
            depth_stats["finite_positive_fraction"] > 0.05
            and depth_max is not None
            and depth_max <= float(calibration["far_m"]) + 1e-4
        )
        checks.append(
            make_check(
                "depth_valid",
                "pass" if depth_passed else "fail",
                measured=depth_stats,
                threshold={
                    "minimum_finite_positive_fraction": 0.05,
                    "maximum_depth_m": calibration["far_m"],
                },
                artifact=observation_records["depth"]["artifact"]["path"],
                failure_reason=None if depth_passed else "Depth observation is invalid",
            )
        )

        semantic_stats = observation_records["semantic"]["statistics"]
        declared_semantic_ids = {
            int(value)
            for value in inputs.room.get("semantics", {}).get("id_to_label", {})
            if str(value).lstrip("-").isdigit()
        }
        visible_semantic_ids = set(semantic_stats["unique_ids"])
        expected_nonzero_ids = declared_semantic_ids - {0}
        declared_marker_visible = not expected_nonzero_ids or bool(
            expected_nonzero_ids & visible_semantic_ids
        )
        semantic_passed = (
            semantic_stats["unique_id_count"] > 1
            and declared_marker_visible
            and all(value >= 0 for value in visible_semantic_ids)
        )
        checks.append(
            make_check(
                "semantic_nontrivial_raw_ids",
                "pass" if semantic_passed else "fail",
                measured={
                    **semantic_stats,
                    "declared_nonzero_ids": sorted(expected_nonzero_ids),
                    "declared_marker_visible": declared_marker_visible,
                },
                threshold={
                    "minimum_unique_id_count": 2,
                    "at_least_one_declared_nonzero_id_visible": bool(
                        expected_nonzero_ids
                    ),
                    "nonnegative_raw_ids": True,
                },
                artifact=observation_records["semantic"]["artifact"]["path"],
                failure_reason=None
                if semantic_passed
                else "Semantic IDs were trivial, invalid, or missed declared markers",
            )
        )

        source_reports, source_error = _source_roundtrip(inputs)
        source_passed = source_error <= 1e-9
        checks.append(
            make_check(
                "named_source_transform_roundtrip",
                "pass" if source_passed else "fail",
                measured={
                    "source_count": len(source_reports),
                    "maximum_transform_error": source_error,
                },
                threshold={
                    "minimum_source_count_for_canary": 2,
                    "maximum_transform_error": 1e-9,
                },
                failure_reason=None
                if source_passed and len(source_reports) >= 2
                else "Need two named sources with stable transform round-trip",
            )
        )
        if len(source_reports) < 2:
            checks[-1]["status"] = "fail"

        connectivity, connectivity_checks = _connectivity_checks(sim, inputs)
        checks.extend(connectivity_checks)
        rays, ray_checks = _ray_checks(sim, inputs)
        checks.extend(ray_checks)
        qa_observations, qa_checks = _save_topdown(sim, inputs, output)
        checks.extend(qa_checks)

    independent_check, reference_record = _independent_process_repeatability_check(
        reference_path=reference_evidence,
        inputs=inputs,
        observation_records=observation_records,
        state_hash=before_hash,
        runtime_commit=runtime_commit,
        native_binding_sha256=native_binding_sha256,
        asset_records=asset_records,
        avengine_commit=avengine_commit,
        repository_clean=repository_clean,
        runtime_clean=runtime_clean,
        output_dir=output,
    )
    checks.append(independent_check)

    lock_path = resolve_runtime_profile(repository_root, "m1")
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "evidence_kind": "completed_capture",
        "room_id": inputs.room["room_id"],
        "room_kind": inputs.room["room_kind"],
        "request_id": inputs.request["request_id"],
        "producer_process": _producer_process_identity(),
        "capture_batch_id": canonical_json_sha256(
            {
                "room_manifest_sha256": sha256_file(inputs.room_path),
                "capture_request_sha256": sha256_file(inputs.request_path),
                "scene_assets": asset_records,
                "avengine_commit": avengine_commit,
                "habitat_runtime_commit": runtime_commit,
                "native_binding_sha256": native_binding_sha256,
                "state": before,
                "repeat_count": repeat_count,
            }
        ),
        "formal_view_ids": [rig["view_id"]],
        "room_manifest": {
            "path": str(inputs.room_path),
            "sha256": sha256_file(inputs.room_path),
        },
        "capture_request": {
            "path": str(inputs.request_path),
            "sha256": sha256_file(inputs.request_path),
        },
        "runtime": {
            "avengine_commit": avengine_commit,
            "avengine_worktree_dirty": not repository_clean,
            "habitat_runtime_root": str(runtime),
            "habitat_runtime_commit": runtime_commit,
            "habitat_runtime_worktree_dirty": not runtime_clean,
            "locked_habitat_runtime_commit": locked_runtime_commit,
            "runtime_lock_sha256": sha256_file(lock_path),
            "habitat_module_path": str(habitat_module_path),
            "native_binding_path": str(native_binding_path),
            "native_binding_sha256": native_binding_sha256,
            "habitat_python_version": getattr(habitat_sim, "__version__", None),
            "habitat_audio_enabled": bool(habitat_sim.audio_enabled),
            "habitat_bullet_enabled": bool(habitat_sim.built_with_bullet),
            "habitat_cuda_enabled": bool(habitat_sim.cuda_enabled),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "scene_assets": asset_records,
        "sensor_contract": {
            "rig_id": rig["rig_id"],
            "view_id": rig["view_id"],
            "world_from_rig": world_from_rig,
            "shared_calibration": calibration,
            "modalities": rig["modalities"],
            "listener": inputs.request["listener"],
            "audio_propagation_status": "not_run",
            "audio_propagation_reason": "M1 listener is a pose anchor; multi-source RLR is M4",
        },
        "capture_state": {
            "before": before,
            "after": after,
            "before_sha256": before_hash,
            "after_sha256": after_hash,
        },
        "repeat_observation_hashes": repeated_hashes,
        "observations": observation_records,
        "sources": source_reports,
        "connectivity": connectivity,
        "ray_checks": rays,
        "qa_observations": qa_observations,
        "independent_reference": reference_record,
        "known_runtime_failures_carried_forward": [
            {
                "check_id": "habitat_direct_import",
                "status": "fail",
                "reason": "fresh direct import aborts unless quaternion is imported first",
            },
            {
                "check_id": "habitat_greedy_follower_binding_cases",
                "status": "fail",
                "reason": "21 PyCapsule iterator cases remain; M1 uses ShortestPath only",
            },
        ],
        "checks": checks,
    }
    finalize_evidence(evidence)
    write_json(output / "evidence.json", evidence)
    return evidence


def build_navmesh(
    inputs: ValidatedM1Inputs,
    *,
    runtime_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    runtime = discover_runtime_root(runtime_root)
    temporary_output = inputs.room_path.parent / "tmp" / "m1_navmesh_build"
    configuration, _, _, resolved_scene = _make_configuration(
        inputs, runtime, temporary_output
    )
    _, habitat_sim, _, _ = _import_habitat()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else resolved_scene["navmesh"]
    )
    if destination is None:
        raise ValueError("Room manifest does not declare scene.navmesh_path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with habitat_sim.Simulator(configuration) as sim:
        settings = configuration.sim_cfg.navmesh_settings
        success = bool(sim.recompute_navmesh(sim.pathfinder, settings))
        if not success or not sim.pathfinder.is_loaded:
            raise RuntimeError("Habitat navmesh recomputation failed")
        sim.pathfinder.save_nav_mesh(str(destination))
        return {
            "status": "pass",
            "navigable_area_m2": float(sim.pathfinder.navigable_area),
            "num_islands": int(sim.pathfinder.num_islands),
            "artifact": {
                "path": str(destination),
                "byte_size": destination.stat().st_size,
                "sha256": sha256_file(destination),
            },
        }
