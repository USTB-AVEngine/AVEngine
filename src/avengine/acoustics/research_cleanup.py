"""Fail-closed research-only cleanup for RLR-incompatible acoustic surfaces.

Some scanned-room exports contain repeated-index or near-zero-area triangles.
RLR rejects them before simulation. A research review may derive a new package
that removes only triangles rejected by either the M3 geometry-QA scale rule
or the pinned native RLR cross-product rule. The source package is never
edited, production packages are rejected, and every removal remains hash-bound
in the derived package's failing compiler-parity report.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.acoustics.contracts import load_and_validate_acoustic_scene_package
from avengine.acoustics.qa import (
    array_sha256,
    debug_obj_array_parity_bytes,
    geometry_report,
    material_coverage_report,
    triangle_areas,
    write_debug_obj,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


CLEANUP_POLICY = "m3_research_remove_rlr_incompatible_triangles_v2"
DERIVED_PACKAGE_SUFFIX = "rlr_incompatible_filter_v2"
RLR_MIN_CROSS_NORM_SQUARED = 1.0e-20


class ResearchCleanupError(RuntimeError):
    """A package cannot be safely derived under the bounded cleanup policy."""


@dataclass(frozen=True)
class FilteredResearchGeometry:
    """Filtered arrays, object ranges and exact derivation facts."""

    vertices: np.ndarray
    triangles: np.ndarray
    material_ids: np.ndarray
    objects: tuple[dict[str, Any], ...]
    record: Mapping[str, Any]


def _derived_package_id(source_package_id: str) -> str:
    """Keep the derived identity version aligned with the cleanup policy."""

    if not isinstance(source_package_id, str) or not source_package_id:
        raise ResearchCleanupError("source package_id must be non-empty")
    return f"{source_package_id}_{DERIVED_PACKAGE_SUFFIX}"


def _npy_record(path: Path, *, root: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        **file_record(path, relative_to=root),
        "format": "npy",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "memory_order": "C",
    }


def _json_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {**file_record(path, relative_to=root), "format": "json"}


def _obj_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {**file_record(path, relative_to=root), "format": "obj"}


def filter_research_geometry(
    vertices: Any,
    triangles: Any,
    material_ids: Any,
    objects: Sequence[Mapping[str, Any]],
) -> FilteredResearchGeometry:
    """Remove only M3-QA- or native-RLR-incompatible faces and empty objects.

    Vertex ranges for objects retaining at least one triangle are preserved
    byte-for-byte.  A vertex range is removed only when every triangle in that
    object is rejected by at least one pinned rule.  This minimizes changes
    while maintaining the M3 package's contiguous per-object index contract.
    """

    source_vertices = np.ascontiguousarray(vertices, dtype="<f4")
    source_triangles = np.ascontiguousarray(triangles, dtype="<u4")
    source_material_ids = np.ascontiguousarray(material_ids, dtype="<u4")
    if (
        source_vertices.ndim != 2
        or source_vertices.shape[1:] != (3,)
        or source_triangles.ndim != 2
        or source_triangles.shape[1:] != (3,)
        or source_material_ids.shape != (len(source_triangles),)
        or not np.all(np.isfinite(source_vertices))
    ):
        raise ResearchCleanupError("research cleanup inputs have invalid array shapes")
    if not objects:
        raise ResearchCleanupError("research cleanup requires object partitions")

    areas = triangle_areas(source_vertices, source_triangles)
    bounds_min = source_vertices.min(axis=0).astype(np.float64)
    bounds_max = source_vertices.max(axis=0).astype(np.float64)
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    qa_area_threshold = max(1.0e-14, diagonal * diagonal * 1.0e-14)
    # Match RLRAcousticContext.cpp exactly: coordinate differences are first
    # evaluated as float, products are promoted to double, then squared cross
    # magnitude is compared to 1e-20. Do not infer this from rounded area.
    points_f32 = source_vertices[source_triangles]
    ab_f32 = points_f32[:, 1] - points_f32[:, 0]
    ac_f32 = points_f32[:, 2] - points_f32[:, 0]
    ab = ab_f32.astype(np.float64)
    ac = ac_f32.astype(np.float64)
    cross = np.cross(ab, ac)
    cross_norm_squared = np.einsum("ij,ij->i", cross, cross)
    rlr_compatible = cross_norm_squared > RLR_MIN_CROSS_NORM_SQUARED
    keep = np.asarray(
        (areas > qa_area_threshold) & rlr_compatible,
        dtype=np.bool_,
    )
    removed_indices = np.flatnonzero(~keep)
    if removed_indices.size == 0:
        raise ResearchCleanupError("source package has no RLR-incompatible triangles")

    expected_vertex = 0
    expected_triangle = 0
    vertex_chunks: list[np.ndarray] = []
    triangle_chunks: list[np.ndarray] = []
    material_chunks: list[np.ndarray] = []
    retained_objects: list[dict[str, Any]] = []
    removed_objects: list[dict[str, Any]] = []
    removed_by_object: dict[str, int] = {}
    new_vertex_offset = 0
    new_triangle_offset = 0
    for object_index, raw in enumerate(objects):
        item = dict(raw)
        vertex_offset = item.get("vertex_offset")
        vertex_count = item.get("vertex_count")
        triangle_offset = item.get("triangle_offset")
        triangle_count = item.get("triangle_count")
        if (
            not isinstance(vertex_offset, int)
            or not isinstance(vertex_count, int)
            or not isinstance(triangle_offset, int)
            or not isinstance(triangle_count, int)
            or vertex_offset != expected_vertex
            or triangle_offset != expected_triangle
            or vertex_count < 1
            or triangle_count < 1
        ):
            raise ResearchCleanupError(
                f"object partition {object_index} is not positive and contiguous"
            )
        vertex_stop = vertex_offset + vertex_count
        triangle_stop = triangle_offset + triangle_count
        if vertex_stop > len(source_vertices) or triangle_stop > len(source_triangles):
            raise ResearchCleanupError(f"object partition {object_index} exceeds arrays")
        object_triangles = source_triangles[triangle_offset:triangle_stop]
        if object_triangles.size and (
            int(object_triangles.min()) < vertex_offset
            or int(object_triangles.max()) >= vertex_stop
        ):
            raise ResearchCleanupError(
                f"object partition {object_index} triangles escape its vertex range"
            )
        object_keep = keep[triangle_offset:triangle_stop]
        removed_count = int(np.count_nonzero(~object_keep))
        object_id = str(item.get("object_id", f"object_{object_index}"))
        if removed_count:
            removed_by_object[object_id] = removed_count
        if not np.any(object_keep):
            removed_objects.append(
                {
                    "object_id": object_id,
                    "vertex_offset": vertex_offset,
                    "vertex_count": vertex_count,
                    "triangle_offset": triangle_offset,
                    "triangle_count": triangle_count,
                    "reason": (
                        "all_triangles_geometry_qa_or_native_rlr_incompatible"
                    ),
                }
            )
        else:
            vertex_chunks.append(source_vertices[vertex_offset:vertex_stop])
            kept_triangles = object_triangles[object_keep].astype(np.int64)
            kept_triangles += new_vertex_offset - vertex_offset
            triangle_chunks.append(np.ascontiguousarray(kept_triangles, dtype="<u4"))
            material_chunks.append(
                np.ascontiguousarray(
                    source_material_ids[triangle_offset:triangle_stop][object_keep],
                    dtype="<u4",
                )
            )
            item["vertex_offset"] = new_vertex_offset
            item["triangle_offset"] = new_triangle_offset
            item["triangle_count"] = int(np.count_nonzero(object_keep))
            retained_objects.append(item)
            new_vertex_offset += vertex_count
            new_triangle_offset += item["triangle_count"]
        expected_vertex = vertex_stop
        expected_triangle = triangle_stop
    if expected_vertex != len(source_vertices) or expected_triangle != len(source_triangles):
        raise ResearchCleanupError("object partitions do not cover the source arrays")
    if not retained_objects:
        raise ResearchCleanupError("cleanup would remove every acoustic object")

    output_vertices = np.ascontiguousarray(np.concatenate(vertex_chunks), dtype="<f4")
    output_triangles = np.ascontiguousarray(np.concatenate(triangle_chunks), dtype="<u4")
    output_material_ids = np.ascontiguousarray(np.concatenate(material_chunks), dtype="<u4")
    remaining_areas = triangle_areas(output_vertices, output_triangles)
    remaining_points = output_vertices[output_triangles]
    remaining_ab = (remaining_points[:, 1] - remaining_points[:, 0]).astype(
        np.float64
    )
    remaining_ac = (remaining_points[:, 2] - remaining_points[:, 0]).astype(
        np.float64
    )
    remaining_cross = np.cross(remaining_ab, remaining_ac)
    remaining_cross_norm_squared = np.einsum(
        "ij,ij->i", remaining_cross, remaining_cross
    )
    if np.any(remaining_areas <= qa_area_threshold) or np.any(
        remaining_cross_norm_squared <= RLR_MIN_CROSS_NORM_SQUARED
    ):
        raise ResearchCleanupError("derived geometry still contains degenerate triangles")
    if set(np.unique(output_material_ids)) != set(np.unique(source_material_ids)):
        raise ResearchCleanupError(
            "cleanup would remove every triangle of a declared material category"
        )

    indices_payload = np.ascontiguousarray(removed_indices, dtype="<u8")
    record: dict[str, Any] = {
        "policy": CLEANUP_POLICY,
        "research_only": True,
        "qualification_claim": False,
        "qa_area_threshold_m2_inclusive": qa_area_threshold,
        "rlr_cross_norm_squared_threshold_inclusive": RLR_MIN_CROSS_NORM_SQUARED,
        "rlr_equivalent_area_threshold_m2_inclusive": (
            0.5 * math.sqrt(RLR_MIN_CROSS_NORM_SQUARED)
        ),
        "source_vertex_count": int(len(source_vertices)),
        "source_triangle_count": int(len(source_triangles)),
        "derived_vertex_count": int(len(output_vertices)),
        "derived_triangle_count": int(len(output_triangles)),
        "removed_vertex_count": int(len(source_vertices) - len(output_vertices)),
        "removed_triangle_count": int(len(removed_indices)),
        "removed_triangle_indices": removed_indices.astype(int).tolist(),
        "removed_triangle_indices_sha256": array_sha256(indices_payload),
        "removed_triangle_area_min_m2": float(np.min(areas[removed_indices])),
        "removed_triangle_area_max_m2": float(np.max(areas[removed_indices])),
        "minimum_retained_triangle_area_m2": float(np.min(remaining_areas)),
        "minimum_retained_cross_norm_squared": float(
            np.min(remaining_cross_norm_squared)
        ),
        "removed_by_object": removed_by_object,
        "removed_objects": removed_objects,
        "source_arrays": {
            "vertices": array_sha256(source_vertices),
            "triangles": array_sha256(source_triangles),
            "triangle_material_ids": array_sha256(source_material_ids),
        },
        "derived_arrays": {
            "vertices": array_sha256(output_vertices),
            "triangles": array_sha256(output_triangles),
            "triangle_material_ids": array_sha256(output_material_ids),
        },
    }
    record["record_content_sha256"] = canonical_json_sha256(record)
    return FilteredResearchGeometry(
        vertices=output_vertices,
        triangles=output_triangles,
        material_ids=output_material_ids,
        objects=tuple(retained_objects),
        record=record,
    )


def _derived_parity_report(
    *,
    source: Mapping[str, Any],
    geometry: FilteredResearchGeometry,
    debug_obj_payload: bytes,
) -> dict[str, Any]:
    report = deepcopy(dict(source))
    vertices = geometry.vertices
    triangles = geometry.triangles
    bounds_min = vertices.min(axis=0).astype(float).tolist()
    bounds_max = vertices.max(axis=0).astype(float).tolist()
    report.update(
        {
            "schema": "avengine_m3_compiler_source_to_package_parity_v1",
            "status": "fail",
            "comparison_scope": "research_runtime_derivation_from_validated_package",
            "visual_runtime_parity_claim": False,
            "expected_package_vertex_count": len(vertices),
            "package_vertex_count": len(vertices),
            "expected_package_triangle_count": len(triangles),
            "package_triangle_count": len(triangles),
            "expected_vertex_bytes_identical_to_npy": True,
            "expected_triangle_bytes_identical_to_npy": True,
            "bounds_identical_within_m": True,
            "expected_package_bounds_m": {"min": bounds_min, "max": bounds_max},
            "package_bounds_m": {"min": bounds_min, "max": bounds_max},
            "debug_obj_parity": debug_obj_array_parity_bytes(
                debug_obj_payload, vertices, triangles
            ),
            "research_cleanup": dict(geometry.record),
        }
    )
    hashes = dict(report.get("array_hashes", {}))
    hashes.update(
        {
            "canonical_expected_vertices": array_sha256(vertices),
            "canonical_expected_triangles": array_sha256(triangles),
            "emitted_npy_vertices": array_sha256(vertices),
            "emitted_npy_triangles": array_sha256(triangles),
        }
    )
    report["array_hashes"] = hashes
    report["derivation_note"] = (
        "Status intentionally remains fail because this is not byte-identical "
        "source compilation; canonical_expected_* describes the explicit "
        "cleanup-derived expectation bound above."
    )
    return report


def derive_rlr_compatible_research_package(
    source_manifest_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create one atomic, internally valid, research-only derived package."""

    source_manifest = Path(source_manifest_path).resolve()
    unresolved = Path(output_dir).expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    if os.path.lexists(unresolved):
        raise ResearchCleanupError(f"refusing to replace cleanup output: {unresolved}")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    output_parent = unresolved.parent.resolve(strict=True)
    output = output_parent / unresolved.name
    policy = WorkspacePathPolicy.from_roots([output_parent])
    try:
        output = policy.resolve_output(output, owner="research cleanup package")
        staging = policy.resolve_output(
            output.with_name(f".{output.name}.staging-{uuid4().hex}"),
            owner="research cleanup staging directory",
        )
    except (FileExistsError, ValueError) as exc:
        raise ResearchCleanupError(str(exc)) from exc
    validated = load_and_validate_acoustic_scene_package(source_manifest)
    manifest = validated.manifest
    if (
        manifest.get("package_mode") != "research_candidate"
        or manifest.get("materials", {}).get("material_semantics")
        != "research_placeholder"
        or manifest.get("materials", {}).get("qualification_claim")
        != "unqualified_research_placeholder"
    ):
        raise ResearchCleanupError(
            "degenerate cleanup is allowed only for unqualified research-placeholder packages"
        )
    filtered = filter_research_geometry(
        validated.vertices,
        validated.triangles,
        validated.triangle_material_ids,
        manifest["objects"],
    )

    staging.mkdir(parents=True)
    try:
        acoustic = staging / "acoustic"
        provenance = staging / "provenance"
        qa = staging / "qa"
        acoustic.mkdir()
        provenance.mkdir()
        qa.mkdir()
        vertices_path = acoustic / "vertices.npy"
        triangles_path = acoustic / "triangles.npy"
        material_ids_path = acoustic / "triangle_material_ids.npy"
        np.save(vertices_path, filtered.vertices, allow_pickle=False)
        np.save(triangles_path, filtered.triangles, allow_pickle=False)
        np.save(material_ids_path, filtered.material_ids, allow_pickle=False)

        copy_roles = {
            "materials.categories": acoustic / "material_categories.json",
            "materials.rlr_database": acoustic / "material_database.json",
            "materials.source_mapping": provenance / "source_material_mapping.json",
            "materials.source_database": provenance / "source_material_database.json",
        }
        copied: dict[str, Path] = {}
        for role, destination in copy_roles.items():
            parent, child = role.split(".")
            record = manifest[parent][child]
            source = (validated.package_root / record["path"]).resolve()
            if (
                not source.is_file()
                or source.stat().st_size != record["byte_size"]
                or sha256_file(source) != record["sha256"]
            ):
                raise ResearchCleanupError(f"validated package dependency changed: {role}")
            shutil.copyfile(source, destination)
            copied[role] = destination

        categories = load_json(copied["materials.categories"])
        geometry_qa = geometry_report(
            filtered.vertices,
            filtered.triangles,
            source_sha256=manifest["source_room"]["geometry_asset_sha256"],
            representation=manifest["geometry"]["representation"],
            source_to_canonical=manifest["geometry"]["source_to_canonical"],
            objects=filtered.objects,
        )
        geometry_qa["research_cleanup"] = dict(filtered.record)
        coverage_qa = material_coverage_report(
            filtered.vertices,
            filtered.triangles,
            filtered.material_ids,
            categories,
        )
        debug_path = qa / "compiler_acoustic_mesh.obj"
        write_debug_obj(
            debug_path,
            filtered.vertices,
            filtered.triangles,
            filtered.material_ids,
            categories,
            filtered.objects,
        )
        source_parity = validated.qa_reports["compiler_source_to_package_parity"]
        parity_qa = _derived_parity_report(
            source=source_parity,
            geometry=filtered,
            debug_obj_payload=debug_path.read_bytes(),
        )
        leakage_qa = deepcopy(validated.qa_reports["ray_leakage"])
        qa_values = {
            "geometry_report": geometry_qa,
            "material_coverage": coverage_qa,
            "ray_leakage": leakage_qa,
            "compiler_source_to_package_parity": parity_qa,
        }
        qa_paths: dict[str, Path] = {}
        for name, value in qa_values.items():
            path = qa / f"{name}.json"
            write_json(path, value)
            qa_paths[name] = path

        derived = deepcopy(manifest)
        derived["package_id"] = _derived_package_id(manifest["package_id"])
        derived["source_room"]["source_revision"] = (
            f"{manifest['source_room']['source_revision']}; derived research-only "
            f"by {CLEANUP_POLICY} from source package "
            f"{sha256_file(source_manifest)}"
        )
        derived["geometry"]["vertex_count"] = len(filtered.vertices)
        derived["geometry"]["triangle_count"] = len(filtered.triangles)
        derived["arrays"] = {
            "vertices": _npy_record(
                vertices_path, root=staging, array=filtered.vertices
            ),
            "triangles": _npy_record(
                triangles_path, root=staging, array=filtered.triangles
            ),
            "triangle_material_ids": _npy_record(
                material_ids_path, root=staging, array=filtered.material_ids
            ),
        }
        derived["objects"] = list(filtered.objects)
        derived["materials"]["categories"] = _json_record(
            copied["materials.categories"], root=staging
        )
        derived["materials"]["rlr_database"] = _json_record(
            copied["materials.rlr_database"], root=staging
        )
        derived["materials"]["source_mapping"] = _json_record(
            copied["materials.source_mapping"], root=staging
        )
        derived["materials"]["source_database"] = _json_record(
            copied["materials.source_database"], root=staging
        )
        derived["qa"] = {
            name: _json_record(path, root=staging) for name, path in qa_paths.items()
        }
        derived["debug_mesh"] = _obj_record(debug_path, root=staging)
        components = dict(derived["compiler"]["components"])
        components["research_cleanup.py"] = sha256_file(Path(__file__).resolve())
        derived["compiler"]["components"] = components
        derived["compiler"]["implementation_sha256"] = canonical_json_sha256(
            components
        )
        derived.pop("package_content_sha256", None)
        derived["package_content_sha256"] = canonical_json_sha256(derived)
        output_manifest = staging / "manifest.json"
        write_json(output_manifest, derived)
        load_and_validate_acoustic_scene_package(output_manifest)
        published = atomic_publish_directory(policy, staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return published / "manifest.json"


__all__ = [
    "CLEANUP_POLICY",
    "DERIVED_PACKAGE_SUFFIX",
    "FilteredResearchGeometry",
    "ResearchCleanupError",
    "derive_rlr_compatible_research_package",
    "filter_research_geometry",
]
