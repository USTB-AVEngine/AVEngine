#!/usr/bin/env python3
"""Replay declared rays from an actual room region or route path.

The input owns the probe origins and directions.  This tool never generates a
Fibonacci sphere and never treats a surface self-hit as evidence that a room
has no leak.  Each declaration may carry its region/path/frame and an explicit
wall, door, window, or other semantic object expected at the first hit.  CPU
Moller--Trumbore results are reported alongside the matched compiled-package
object; optional native TraceRay replay is available when all runtime paths
are supplied.

Accepted input forms are either a list of ray objects, ``{"rays": [...]}``,
or ``{"regions": [{"region_id": ..., "path_id": ..., "rays": [...]}]}``.
Every ray needs ``origin_m``, ``direction``, ``distance_m`` (or
``maximum_distance_m``), and ``expectation`` (``hit_within_m`` or
``clear_until_m``).  A route producer should copy its measured path/frame
metadata into each ray rather than reconstructing coordinates here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.acoustics.qa import ray_leakage_report  # noqa: E402
from avengine.acoustics.contracts import load_and_validate_acoustic_scene_package  # noqa: E402
from avengine.acoustics.gltf import load_glb  # noqa: E402
from avengine.acoustics.runtime import (  # noqa: E402
    RLRSimulationConfig,
    RuntimeAnchor,
    RUNTIME_MODE_CURRENT_INSTALLED,
    RuntimeContractError,
    RuntimeExecutionError,
    RuntimeUnavailableError,
    load_compiled_acoustic_scene,
    simulate_compiled_acoustic_scene,
)
from avengine.contracts.json_io import load_json  # noqa: E402


SCHEMA = "avengine_region_path_ray_diagnostic_v1"
_EXPECTATIONS = {"hit_within_m", "clear_until_m"}


class RegionRayDiagnosticError(ValueError):
    """The selected region/path ray declarations are malformed."""


def _vector(value: Any, *, owner: str) -> list[float]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RegionRayDiagnosticError(f"{owner} must be a numeric 3-vector") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise RegionRayDiagnosticError(f"{owner} must be a finite 3-vector")
    return [float(item) for item in result]


def _ray_declarations(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        regions = [(None, None, value)]
    elif isinstance(value, Mapping) and isinstance(value.get("rays"), list):
        regions = [(value.get("region_id"), value.get("path_id"), value["rays"])]
    elif isinstance(value, Mapping) and isinstance(value.get("regions"), list):
        regions = []
        for index, region in enumerate(value["regions"]):
            if not isinstance(region, Mapping) or not isinstance(region.get("rays"), list):
                raise RegionRayDiagnosticError(f"regions[{index}] must carry a rays list")
            regions.append((region.get("region_id"), region.get("path_id"), region["rays"]))
    else:
        raise RegionRayDiagnosticError("probe input must be a ray list, rays object, or regions object")

    declarations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for region_id, path_id, rays in regions:
        for index, raw in enumerate(rays):
            if not isinstance(raw, Mapping):
                raise RegionRayDiagnosticError("each region/path ray must be an object")
            check_id = raw.get("check_id", raw.get("ray_id"))
            if not isinstance(check_id, str) or not check_id:
                raise RegionRayDiagnosticError("each ray needs a nonempty check_id")
            if check_id in seen:
                raise RegionRayDiagnosticError(f"duplicate ray check_id: {check_id}")
            seen.add(check_id)
            expectation = raw.get("expectation")
            if expectation not in _EXPECTATIONS:
                raise RegionRayDiagnosticError(
                    f"{check_id} expectation must be hit_within_m or clear_until_m"
                )
            maximum_distance = raw.get("distance_m", raw.get("maximum_distance_m"))
            if isinstance(maximum_distance, bool) or not isinstance(maximum_distance, (int, float)):
                raise RegionRayDiagnosticError(f"{check_id} needs numeric distance_m")
            maximum_distance = float(maximum_distance)
            if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
                raise RegionRayDiagnosticError(f"{check_id} distance_m must be positive and finite")
            declaration = {
                "check_id": check_id,
                "origin_m": _vector(raw.get("origin_m"), owner=f"{check_id}.origin_m"),
                "direction": _vector(raw.get("direction"), owner=f"{check_id}.direction"),
                "distance_m": maximum_distance,
                "expectation": expectation,
            }
            direction_array = np.asarray(declaration["direction"], dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction_array))
            if direction_norm <= 1.0e-12:
                raise RegionRayDiagnosticError(f"{check_id}.direction must be nonzero")
            declaration["direction"] = (direction_array / direction_norm).tolist()
            # Preserve provenance for the report, while the CPU/native ray
            # API receives only its original geometric declaration fields.
            for key in (
                "region_id",
                "path_id",
                "frame_index",
                "source_id",
                "direction_source",
                "expected_object_id",
                "expected_object_kind",
                "opening_id",
                "target_object_id",
                "target_authority",
                "target_bound_authority",
                "target_bound_x_m",
                "target_distance_m",
                "range_margin_m",
                "first_hit_object_id",
                "first_hit_source_node_index",
                "first_hit_authority",
                "expectation_basis",
            ):
                if key in raw:
                    declaration[key] = raw[key]
            if declaration.get("region_id") is None and region_id is not None:
                declaration["region_id"] = region_id
            if declaration.get("path_id") is None and path_id is not None:
                declaration["path_id"] = path_id
            declarations.append(declaration)
    if not declarations:
        raise RegionRayDiagnosticError("probe input contains no rays")
    return declarations


def _semantic_objects(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    objects: dict[str, Mapping[str, Any]] = {}
    raw_objects = value.get("objects")
    if isinstance(raw_objects, list):
        for item in raw_objects:
            if isinstance(item, Mapping) and isinstance(item.get("object_id"), str):
                objects[item["object_id"]] = item
    for key in ("furniture_objects", "scene_objects"):
        raw_group = value.get(key)
        if isinstance(raw_group, list):
            for item in raw_group:
                if isinstance(item, Mapping) and isinstance(item.get("object_id"), str):
                    objects[item["object_id"]] = item
    return objects


def _compiled_object_for_triangle(scene: Any, triangle_index: Any) -> Mapping[str, Any] | None:
    if isinstance(triangle_index, bool) or not isinstance(triangle_index, int):
        return None
    offset = 0
    for item in getattr(scene, "objects", ()):
        if not isinstance(item, Mapping):
            continue
        triangles = item.get("triangles")
        count = len(triangles) if isinstance(triangles, np.ndarray) else item.get("triangle_count")
        if isinstance(count, int) and offset <= triangle_index < offset + count:
            return item
        if isinstance(count, int):
            offset += count
    return None



def _semantic_package_mapping(
    room_semantics: Path,
    semantics: Mapping[str, Mapping[str, Any]],
    scene: Any,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    """Resolve authored sidecar object IDs to package objects and ranges.

    The sidecar names the authored Blender object, while the acoustic package
    names expanded GLB node/mesh primitives. The GLB node name and the package
    manifest source_node_index are the authoritative bridge; cumulative
    triangle offsets are read from that same manifest rather than guessed
    from sidecar ordering.
    """

    node_indices_by_name: dict[str, list[int]] = {}
    mesh_indices_by_name: dict[str, list[int]] = {}
    visual_paths: set[Path] = set()
    for item in semantics.values():
        geometry_ref = item.get("geometry_ref")
        if not isinstance(geometry_ref, Mapping):
            continue
        visual_glb = geometry_ref.get("visual_glb")
        if not isinstance(visual_glb, str) or not visual_glb:
            continue
        visual_path = Path(visual_glb).expanduser()
        if not visual_path.is_absolute():
            visual_path = room_semantics.parent / visual_path
        visual_paths.add(visual_path.resolve())

    mapping_errors: list[str] = []
    for visual_path in sorted(visual_paths, key=str):
        try:
            document = load_glb(visual_path).document
        except (OSError, ValueError, KeyError, TypeError) as error:
            mapping_errors.append(f"{visual_path}: {type(error).__name__}: {error}")
            continue
        nodes = document.get("nodes", [])
        if not isinstance(nodes, list):
            mapping_errors.append(f"{visual_path}: GLB nodes is not a list")
            continue
        for node_index, node in enumerate(nodes):
            if not isinstance(node, Mapping) or not isinstance(node.get("name"), str):
                continue
            node_indices_by_name.setdefault(str(node["name"]), []).append(node_index)
        meshes = document.get("meshes", [])
        if isinstance(meshes, list):
            for mesh_index, mesh in enumerate(meshes):
                if not isinstance(mesh, Mapping) or not isinstance(mesh.get("name"), str):
                    continue
                mesh_indices_by_name.setdefault(str(mesh["name"]), []).append(mesh_index)

    manifest = getattr(scene, "manifest", {})
    manifest_objects = manifest.get("objects", []) if isinstance(manifest, Mapping) else []
    package_by_source_node: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]] = {}
    package_by_source_mesh: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]] = {}
    runtime_objects = list(getattr(scene, "objects", ()))
    if isinstance(manifest_objects, list):
        for index, manifest_object in enumerate(manifest_objects):
            if not isinstance(manifest_object, Mapping):
                continue
            source_node_index = manifest_object.get("source_node_index")
            if isinstance(source_node_index, bool) or not isinstance(source_node_index, int):
                continue
            runtime_object = runtime_objects[index] if index < len(runtime_objects) else None
            package_record = (
                manifest_object,
                runtime_object if isinstance(runtime_object, Mapping) else None,
            )
            package_by_source_node.setdefault(source_node_index, []).append(package_record)
            source_mesh_index = manifest_object.get("source_mesh_index")
            if isinstance(source_mesh_index, int) and not isinstance(source_mesh_index, bool):
                package_by_source_mesh.setdefault(source_mesh_index, []).append(package_record)

    resolved: dict[str, Mapping[str, Any]] = {}
    for sidecar_id, sidecar_object in semantics.items():
        geometry_ref = sidecar_object.get("geometry_ref")
        blend_name = (
            geometry_ref.get("blend_object")
            if isinstance(geometry_ref, Mapping)
            else None
        )
        if not isinstance(blend_name, str) or not blend_name:
            blend_name = sidecar_object.get("kind")
        if not isinstance(blend_name, str) or not blend_name:
            mapping_errors.append(f"{sidecar_id}: missing geometry_ref.blend_object")
            continue
        node_indices = node_indices_by_name.get(blend_name, [])
        mapping_mode = "node_name"
        package_candidates = [
            candidate
            for node_index in node_indices
            for candidate in package_by_source_node.get(node_index, [])
        ]
        if not package_candidates:
            geometry_ref = sidecar_object.get("geometry_ref")
            mesh_datablock = (
                geometry_ref.get("mesh_datablock")
                if isinstance(geometry_ref, Mapping)
                else None
            )
            mesh_indices = (
                mesh_indices_by_name.get(mesh_datablock, [])
                if isinstance(mesh_datablock, str)
                else []
            )
            package_candidates = [
                candidate
                for mesh_index in mesh_indices
                for candidate in package_by_source_mesh.get(mesh_index, [])
            ]
            if package_candidates:
                mapping_mode = "mesh_datablock"
            else:
                mapping_errors.append(
                    f"{sidecar_id}: GLB node {blend_name!r} and mesh datablock "
                    f"{mesh_datablock!r} have no package object"
                )
                continue
        package_records: list[dict[str, Any]] = []
        invalid_candidate = False
        for manifest_object, runtime_object in package_candidates:
            package_object_id = manifest_object.get("object_id")
            triangle_offset = manifest_object.get("triangle_offset")
            triangle_count = manifest_object.get("triangle_count")
            if not isinstance(package_object_id, str):
                mapping_errors.append(f"{sidecar_id}: package object has no object_id")
                invalid_candidate = True
                break
            if (
                isinstance(triangle_offset, bool)
                or not isinstance(triangle_offset, int)
                or isinstance(triangle_count, bool)
                or not isinstance(triangle_count, int)
                or triangle_offset < 0
                or triangle_count < 0
            ):
                mapping_errors.append(f"{sidecar_id}: package object has invalid triangle range")
                invalid_candidate = True
                break
            package_records.append(
                {
                    "package_object_id": package_object_id,
                    "triangle_offset": triangle_offset,
                    "triangle_count": triangle_count,
                    "triangle_end_exclusive": triangle_offset + triangle_count,
                    "package_source_material_name": manifest_object.get("source_material_name"),
                    "runtime_object_present": runtime_object is not None,
                }
            )
        if invalid_candidate or not package_records:
            continue
        primary = package_records[0]
        source_node_indices = sorted(
            {
                int(manifest_object.get("source_node_index"))
                for manifest_object, _runtime_object in package_candidates
                if isinstance(manifest_object.get("source_node_index"), int)
                and not isinstance(manifest_object.get("source_node_index"), bool)
            }
        )
        source_mesh_indices = sorted(
            {
                int(manifest_object.get("source_mesh_index"))
                for manifest_object, _runtime_object in package_candidates
                if isinstance(manifest_object.get("source_mesh_index"), int)
                and not isinstance(manifest_object.get("source_mesh_index"), bool)
            }
        )
        if not source_node_indices:
            mapping_errors.append(f"{sidecar_id}: package object has no source_node_index")
            continue
        resolved[sidecar_id] = {
            "sidecar_object_id": sidecar_id,
            "sidecar_category": sidecar_object.get("category"),
            "sidecar_kind": sidecar_object.get("kind"),
            "package_object_id": primary["package_object_id"],
            "package_object_ids": [item["package_object_id"] for item in package_records],
            "source_node_index": source_node_indices[0],
            "source_node_indices": source_node_indices,
            "source_mesh_indices": source_mesh_indices,
            "triangle_offset": primary["triangle_offset"],
            "triangle_count": primary["triangle_count"],
            "triangle_end_exclusive": primary["triangle_end_exclusive"],
            "triangle_ranges": [
                {
                    "package_object_id": item["package_object_id"],
                    "triangle_offset": item["triangle_offset"],
                    "triangle_count": item["triangle_count"],
                    "triangle_end_exclusive": item["triangle_end_exclusive"],
                }
                for item in package_records
            ],
            "package_source_material_name": primary["package_source_material_name"],
            "package_source_material_names": [
                item["package_source_material_name"] for item in package_records
            ],
            "runtime_object_present": all(
                item["runtime_object_present"] for item in package_records
            ),
            "mapping_status": (
                f"resolved_{mapping_mode}_one_to_many"
                if len(package_records) > 1
                else f"resolved_{mapping_mode}"
            ),
        }
    mapping_report = {
        "status": "pass" if resolved and not mapping_errors else (
            "partial" if resolved else "not_run"
        ),
        "resolved_object_count": len(resolved),
        "sidecar_object_count": len(semantics),
        "unresolved_object_count": max(0, len(semantics) - len(resolved)),
        "errors": mapping_errors,
        "authority": (
            "sidecar geometry_ref.visual_glb node or mesh_datablock -> package "
            "manifest source_node_index/source_mesh_index -> triangle ranges"
        ),
    }
    return resolved, mapping_report

def _annotate_cpu_checks(
    checks: list[Mapping[str, Any]],
    *,
    scene: Any,
    semantics: Mapping[str, Mapping[str, Any]],
    semantic_mapping: Mapping[str, Mapping[str, Any]],
    declarations: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    annotated: list[dict[str, Any]] = []
    semantic_resolved = False
    semantic_unresolved = 0
    semantic_fail = False
    compiled_ids = {
        str(item.get("object_id"))
        for item in getattr(scene, "objects", ())
        if isinstance(item, Mapping) and isinstance(item.get("object_id"), str)
    }
    for check in checks:
        item = dict(check)
        declaration = declarations.get(str(item.get("check_id")))
        if declaration is not None:
            for key in (
                "region_id",
                "path_id",
                "frame_index",
                "source_id",
                "direction_source",
                "expected_object_id",
                "expected_object_kind",
                "opening_id",
                "target_object_id",
                "target_authority",
                "target_bound_authority",
                "target_bound_x_m",
                "target_distance_m",
                "range_margin_m",
                "first_hit_object_id",
                "first_hit_source_node_index",
                "first_hit_authority",
                "expectation_basis",
            ):
                if key in declaration:
                    item[key] = declaration[key]
        expected_id = item.get("expected_object_id")
        expected_id_text = str(expected_id) if expected_id is not None else None
        expected_kind = item.get("expected_object_kind")
        hit_object = _compiled_object_for_triangle(scene, item.get("measured_triangle_index"))
        measured_id = hit_object.get("object_id") if hit_object else None
        item["measured_object_id"] = measured_id
        material_category = None
        if hit_object is not None and isinstance(item.get("measured_triangle_index"), int):
            material_ids = hit_object.get("triangle_material_ids")
            relative = item["measured_triangle_index"]
            offset = 0
            for candidate in getattr(scene, "objects", ()):
                if candidate is hit_object:
                    break
                if isinstance(candidate, Mapping) and isinstance(candidate.get("triangles"), np.ndarray):
                    offset += len(candidate["triangles"])
            relative -= offset
            if isinstance(material_ids, np.ndarray) and 0 <= relative < len(material_ids):
                categories = getattr(scene, "material_categories", ())
                material_index = int(material_ids[relative])
                if 0 <= material_index < len(categories):
                    material_category = str(categories[material_index])
        item["measured_object_material_category"] = material_category
        if expected_id is not None or expected_kind is not None:
            expected_record = semantics.get(expected_id_text) if expected_id_text is not None else None
            expected_package = (
                semantic_mapping.get(expected_id_text)
                if expected_id_text is not None
                else None
            )
            if expected_package is None and expected_id_text in compiled_ids:
                expected_package = {
                    "package_object_id": str(expected_id),
                    "mapping_status": "direct_package_object_id",
                }
            if expected_id is not None and expected_package is None:
                item["semantic_match"] = None
                item["semantic_review_status"] = "unresolved_source_to_package_object_id"
                semantic_unresolved += 1
                item["mapping_status"] = "unresolved"
            elif expected_id is None:
                item["semantic_match"] = None
                item["semantic_review_status"] = "not_run_expected_kind_only"
                item["mapping_status"] = "not_applicable"
            else:
                expected_known = expected_record is not None
                item["expected_package_object_id"] = expected_package.get("package_object_id")
                for key in (
                    "source_node_index",
                    "triangle_offset",
                    "triangle_count",
                    "triangle_end_exclusive",
                    "sidecar_category",
                    "sidecar_kind",
                    "package_object_ids",
                    "triangle_ranges",
                    "package_source_material_names",
                    "mapping_status",
                ):
                    if key in expected_package:
                        item[f"expected_{key}"] = expected_package[key]
                item["mapping_status"] = expected_package.get("mapping_status", "resolved")
                expected_values = set()
                if expected_record is not None:
                    for key in ("kind", "category"):
                        value = expected_record.get(key)
                        if isinstance(value, str):
                            expected_values.add(value)
                kind_matches = expected_kind is None or expected_kind in expected_values
                expected_package_ids = expected_package.get("package_object_ids")
                if not isinstance(expected_package_ids, list):
                    expected_package_ids = [expected_package.get("package_object_id")]
                object_matches = measured_id in expected_package_ids
                semantic_resolved = True
                item["semantic_match"] = bool(expected_known and kind_matches and object_matches)
                item["semantic_review_status"] = "pass" if item["semantic_match"] else "mismatch"
                semantic_fail |= not item["semantic_match"]
        else:
            item["semantic_match"] = None
            item["semantic_review_status"] = "not_run"
        annotated.append(item)
    semantic_status = (
        "pass" if semantic_resolved and not semantic_fail else
        "fail" if semantic_resolved else
        "not_run"
    )
    return annotated, semantic_status
def _simulation_mapping(path: Path | None) -> dict[str, Any]:
    source = path or (REPOSITORY / "examples/runtime/rir_cache_simulation_request_v2.json")
    value = load_json(source)
    simulation = dict(value["simulation"])
    simulation["channel_layout"] = {"type": "ambisonics", "channel_count": 4}
    return simulation


def _native_replay(
    scene: Any,
    declarations: list[Mapping[str, Any]],
    *,
    runtime_prefix: str | Path,
    magnum_site: str | Path,
    rlr_sdk_root: str | Path,
    simulation_request: Path | None,
) -> dict[str, Any]:
    try:
        simulation = RLRSimulationConfig.from_mapping(_simulation_mapping(simulation_request))
        native_declarations = [
            {
                key: item[key]
                for key in ("check_id", "origin_m", "direction", "distance_m", "expectation")
            }
            for item in declarations
        ]
        result = simulate_compiled_acoustic_scene(
            scene,
            simulation,
            source=RuntimeAnchor(anchor_id="region_ray_source", position_m=declarations[0]["origin_m"]),
            listener=RuntimeAnchor(anchor_id="region_ray_listener", position_m=declarations[-1]["origin_m"]),
            runtime_mode=RUNTIME_MODE_CURRENT_INSTALLED,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_site,
            ray_checks=native_declarations,
        )
        reports = [dict(item) for item in result.ray_checks]
        return {
            "status": "pass" if reports and all(item.get("passed") is True for item in reports) else "fail",
            "backend": "avengine_modern_rlr_context_trace_ray",
            "ray_checks": reports,
            "count": len(reports),
        }
    except RuntimeUnavailableError as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}", "count": len(declarations)}
    except (RuntimeExecutionError, RuntimeContractError, OSError, ValueError) as error:
        return {"status": "error", "reason": f"{type(error).__name__}: {error}", "count": len(declarations)}


def diagnose(
    *,
    package_manifest: Path,
    probes: Path,
    room_semantics: Path | None = None,
    runtime_prefix: str | Path | None = None,
    magnum_site: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    simulation_request: Path | None = None,
) -> dict[str, Any]:
    scene = load_compiled_acoustic_scene(
        package_manifest,
        allow_nonpassing_research_qa=True,
    )
    raw_probes = load_json(probes)
    declarations = _ray_declarations(raw_probes)
    # ``load_compiled_acoustic_scene`` keeps runtime-owned arrays private. The
    # package validator is the authoritative readback for this CPU diagnostic;
    # it reuses the same declared package arrays and does not infer geometry.
    validated = load_and_validate_acoustic_scene_package(package_manifest)
    cpu_base = ray_leakage_report(validated.vertices, validated.triangles, declarations)
    semantics = _semantic_objects(load_json(room_semantics)) if room_semantics else {}
    semantic_mapping, mapping_report = (
        _semantic_package_mapping(room_semantics, semantics, scene)
        if room_semantics
        else (
            {},
            {
                "status": "not_run",
                "resolved_object_count": 0,
                "sidecar_object_count": 0,
                "unresolved_object_count": 0,
                "errors": [],
            },
        )
    )
    cpu_checks, semantic_status = _annotate_cpu_checks(
        cpu_base.get("checks", []),
        scene=scene,
        semantics=semantics,
        semantic_mapping=semantic_mapping,
        declarations={str(item["check_id"]): item for item in declarations},
    )
    cpu_base["checks"] = cpu_checks
    cpu_base["declared_check_count"] = len(cpu_checks)
    cpu_base["status"] = (
        "pass" if cpu_checks and all(item.get("status") == "pass" for item in cpu_checks) else "fail"
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": cpu_base["status"],
        "claim_boundary": (
            "Declared region/path rays only. CPU hit results and optional native "
            "TraceRay replay do not prove whole-room enclosure or physical acoustic accuracy."
        ),
        "package_manifest": str(package_manifest.resolve()),
        "probe_input": str(probes.resolve()),
        "room_semantics": str(room_semantics.resolve()) if room_semantics else None,
        "ray_source_policy": "caller_supplied_actual_region_path_origins_and_directions_v1",
        "self_hit_interpretation": "a self-hit is recorded as a hit and is not evidence of no leakage",
        "cpu": {
            "backend": cpu_base.get("backend"),
            "status": cpu_base["status"],
            "declared_check_count": len(cpu_checks),
            "checks": cpu_checks,
            "automatic_enclosure_probe": {
                "status": "not_run",
                "reason": "this diagnostic does not synthesize directions or origins",
            },
        },
        "semantic_review": {
            "status": semantic_status,
            "object_count": len(semantics),
            "unresolved_count": sum(
                item.get("semantic_review_status", "").startswith("unresolved")
                for item in cpu_checks
            ),
            "scope": "explicit expected_object_id/expected_object_kind metadata only",
            "package_mapping": mapping_report,
        },
        "native": {
            "status": "not_run",
            "reason": "runtime paths not supplied",
        },
    }
    if all(value is not None for value in (runtime_prefix, magnum_site, rlr_sdk_root)):
        report["native"] = _native_replay(
            scene,
            declarations,
            runtime_prefix=runtime_prefix,
            magnum_site=magnum_site,
            rlr_sdk_root=rlr_sdk_root,
            simulation_request=simulation_request,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--probes", required=True, type=Path)
    parser.add_argument("--room-semantics", type=Path)
    parser.add_argument("--runtime-prefix")
    parser.add_argument("--magnum-site")
    parser.add_argument("--rlr-sdk-root")
    parser.add_argument("--simulation-request", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise RegionRayDiagnosticError(f"refusing to overwrite: {args.output}")
    report = diagnose(
        package_manifest=args.package_manifest.resolve(),
        probes=args.probes.resolve(),
        room_semantics=args.room_semantics.resolve() if args.room_semantics else None,
        runtime_prefix=args.runtime_prefix,
        magnum_site=args.magnum_site,
        rlr_sdk_root=args.rlr_sdk_root,
        simulation_request=args.simulation_request.resolve() if args.simulation_request else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "semantic_review": report["semantic_review"], "native": report["native"], "output": str(args.output.resolve())}, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegionRayDiagnosticError as error:
        print(f"REGION_RAY_DIAGNOSTIC_FAILED {error}", file=sys.stderr)
        raise SystemExit(2)
