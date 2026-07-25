from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\x00")
    digest.update(",".join(map(str, contiguous.shape)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points = vertices[triangles].astype(np.float64, copy=False)
    return 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        axis=1,
    )


def _welded_indices(vertices: np.ndarray) -> np.ndarray:
    # Exact float32 equality is intentional: compilation already canonicalizes
    # to stable float32 bytes, and QA must not hide authoring gaps behind an
    # undocumented tolerance.
    contiguous = np.ascontiguousarray(vertices, dtype="<f4")
    byte_rows = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * 3)))
    _unique, inverse = np.unique(byte_rows.reshape(-1), return_inverse=True)
    if len(_unique) > np.iinfo(np.uint32).max:
        raise ValueError("welded vertex index exceeds uint32")
    return np.ascontiguousarray(inverse, dtype=np.uint32)


def geometry_report(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    source_sha256: str,
    representation: str,
    source_to_canonical: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    areas = triangle_areas(vertices, triangles)
    bounds_min = vertices.min(axis=0).astype(float)
    bounds_max = vertices.max(axis=0).astype(float)
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    area_epsilon = max(1e-14, diagonal * diagonal * 1e-14)
    degenerate = int(np.count_nonzero(areas <= area_epsilon))

    welded = _welded_indices(vertices)
    welded_triangles = welded[triangles]
    edges = np.concatenate(
        (
            welded_triangles[:, [0, 1]],
            welded_triangles[:, [1, 2]],
            welded_triangles[:, [2, 0]],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
    sorted_triangles = np.sort(welded_triangles, axis=1)
    _, triangle_counts = np.unique(sorted_triangles, axis=0, return_counts=True)
    duplicate_triangles = int(np.sum(np.maximum(triangle_counts - 1, 0)))
    object_topology: list[dict[str, Any]] = []
    object_boundary_total = 0
    object_nonmanifold_total = 0
    for item in objects:
        triangle_start = int(item["triangle_offset"])
        triangle_stop = triangle_start + int(item["triangle_count"])
        vertex_start = int(item["vertex_offset"])
        object_triangles = triangles[triangle_start:triangle_stop] - vertex_start
        object_vertices = vertices[
            vertex_start : vertex_start + int(item["vertex_count"])
        ]
        object_welded = _welded_indices(object_vertices)
        welded_object_triangles = object_welded[object_triangles]
        object_edges = np.concatenate(
            (
                welded_object_triangles[:, [0, 1]],
                welded_object_triangles[:, [1, 2]],
                welded_object_triangles[:, [2, 0]],
            ),
            axis=0,
        )
        object_edges.sort(axis=1)
        _, object_edge_counts = np.unique(object_edges, axis=0, return_counts=True)
        object_boundary = int(np.count_nonzero(object_edge_counts == 1))
        object_nonmanifold = int(np.count_nonzero(object_edge_counts > 2))
        object_boundary_total += object_boundary
        object_nonmanifold_total += object_nonmanifold
        object_topology.append(
            {
                "object_id": item["object_id"],
                "triangle_count": int(item["triangle_count"]),
                "boundary_edge_count_after_exact_weld": object_boundary,
                "nonmanifold_edge_count_after_exact_weld": object_nonmanifold,
            }
        )
    passed = (
        degenerate == 0
        and duplicate_triangles == 0
        and object_boundary_total == 0
        and object_nonmanifold_total == 0
    )
    return {
        "schema": "avengine_m3_geometry_report_v1",
        "status": "pass" if passed else "fail",
        "geometry_representation": representation,
        "aabb_proxy": representation == "debug_aabb_proxy",
        "source_geometry_sha256": source_sha256,
        "source_to_canonical": dict(source_to_canonical),
        "vertex_count": int(len(vertices)),
        "welded_vertex_count": int(len(np.unique(welded))),
        "triangle_count": int(len(triangles)),
        "object_count": int(len(objects)),
        "surface_area_m2": float(np.sum(areas)),
        "bounds_m": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
            "extent": (bounds_max - bounds_min).tolist(),
            "diagonal": diagonal,
        },
        "topology": {
            "degenerate_triangle_count": degenerate,
            "duplicate_triangle_count": duplicate_triangles,
            "boundary_edge_count_after_exact_weld": boundary_edges,
            "nonmanifold_edge_count_after_exact_weld": nonmanifold_edges,
            "global_nonmanifold_is_inter_object_junction_diagnostic": True,
            "per_object_boundary_edge_count_after_exact_weld": object_boundary_total,
            "per_object_nonmanifold_edge_count_after_exact_weld": object_nonmanifold_total,
            "per_object": object_topology,
            "boundary_edges_are_reported_not_silently_filled": True,
        },
        "thresholds": {
            "degenerate_triangle_count": 0,
            "duplicate_triangle_count": 0,
            "per_object_boundary_edge_count_after_exact_weld": 0,
            "per_object_nonmanifold_edge_count_after_exact_weld": 0,
            "production_aabb_proxy": False,
        },
        "array_hashes": {
            "vertices": array_sha256(vertices),
            "triangles": array_sha256(triangles),
        },
    }


def material_coverage_report(
    vertices: np.ndarray,
    triangles: np.ndarray,
    material_ids: np.ndarray,
    categories_document: Mapping[str, Any],
) -> dict[str, Any]:
    categories = categories_document.get("categories", [])
    areas = triangle_areas(vertices, triangles)
    total_area = float(np.sum(areas))
    records: list[dict[str, Any]] = []
    covered = 0
    fallback_triangles = 0
    for category in categories:
        material_id = int(category["material_id"])
        selected = material_ids == material_id
        count = int(np.count_nonzero(selected))
        covered += count
        if category.get("fallback") is not False:
            fallback_triangles += count
        area = float(np.sum(areas[selected]))
        records.append(
            {
                "material_id": material_id,
                "category_name": category["category_name"],
                "source_material_name": category["source_material_name"],
                "triangle_count": count,
                "surface_area_m2": area,
                "surface_area_fraction": area / total_area if total_area > 0 else 0.0,
                "fallback": category.get("fallback"),
                "mapping_confidence": category.get("mapping_confidence"),
                "human_override": category.get("human_override"),
                "randomized": category.get("randomized"),
                "rlr_match": category.get("rlr_match"),
            }
        )
    coverage = covered / len(triangles) if len(triangles) else 0.0
    all_categories_used = all(record["triangle_count"] > 0 for record in records)
    passed = (
        len(material_ids) == len(triangles)
        and covered == len(triangles)
        and fallback_triangles == 0
        and all_categories_used
    )
    return {
        "schema": "avengine_m3_material_coverage_v1",
        "status": "pass" if passed else "fail",
        "triangle_count": int(len(triangles)),
        "assigned_triangle_count": covered,
        "coverage_fraction": coverage,
        "fallback_triangle_count": fallback_triangles,
        "unmatched_triangle_count": int(len(triangles) - covered),
        "all_declared_categories_used": all_categories_used,
        "categories": records,
        "thresholds": {
            "coverage_fraction": 1.0,
            "fallback_triangle_count": 0,
            "unmatched_triangle_count": 0,
            "all_declared_categories_used": True,
        },
    }


def _reference_source_to_package_arrays(
    source_vertices: np.ndarray,
    source_triangles: np.ndarray,
    source_to_canonical: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Independently replay the declared transform for compiler QA.

    This intentionally does not call the compiler transform helper.  It starts
    from the extracted GLB arrays and uses a separate row-vector expression so
    the report is not a comparison of a package array with itself.
    """

    matrix = np.asarray(
        source_to_canonical["matrix_row_major"], dtype=np.float64
    ).reshape(4, 4)
    expected_vertices = np.ascontiguousarray(
        source_vertices.astype(np.float64, copy=False) @ matrix[:3, :3].T
        + matrix[:3, 3],
        dtype="<f4",
    )
    expected_triangles = np.ascontiguousarray(source_triangles, dtype="<u4")
    if float(np.linalg.det(matrix[:3, :3])) < 0.0:
        expected_triangles = expected_triangles.copy()
        expected_triangles[:, [1, 2]] = expected_triangles[:, [2, 1]]
    return expected_vertices, expected_triangles


def _debug_obj_array_parity(
    path: str | Path,
    expected_vertices: np.ndarray,
    expected_triangles: np.ndarray,
) -> dict[str, Any]:
    try:
        payload = Path(path).read_bytes()
    except OSError:
        payload = b""
    return debug_obj_array_parity_bytes(
        payload, expected_vertices, expected_triangles
    )


def debug_obj_array_parity_bytes(
    payload: bytes,
    expected_vertices: np.ndarray,
    expected_triangles: np.ndarray,
) -> dict[str, Any]:
    """Validate compiler OBJ lines from the exact hash-checked byte snapshot."""

    vertex_index = 0
    triangle_index = 0
    vertex_values_equal = True
    triangle_indices_equal = True
    malformed_line_count = 0
    try:
        text = payload.decode("utf-8")
        with io.StringIO(text) as handle:
            for line in handle:
                if line.startswith("v "):
                    fields = line.split()
                    if len(fields) != 4 or vertex_index >= len(expected_vertices):
                        malformed_line_count += 1
                        vertex_values_equal = False
                    else:
                        try:
                            parsed = np.asarray(fields[1:], dtype="<f4")
                        except ValueError:
                            malformed_line_count += 1
                            vertex_values_equal = False
                        else:
                            if not np.array_equal(
                                parsed, expected_vertices[vertex_index]
                            ):
                                vertex_values_equal = False
                    vertex_index += 1
                elif line.startswith("f "):
                    fields = line.split()
                    if len(fields) != 4 or triangle_index >= len(expected_triangles):
                        malformed_line_count += 1
                        triangle_indices_equal = False
                    else:
                        try:
                            parsed = np.asarray(
                                [int(value) - 1 for value in fields[1:]], dtype="<u4"
                            )
                        except (TypeError, ValueError, OverflowError):
                            malformed_line_count += 1
                            triangle_indices_equal = False
                        else:
                            if not np.array_equal(
                                parsed, expected_triangles[triangle_index]
                            ):
                                triangle_indices_equal = False
                    triangle_index += 1
    except UnicodeDecodeError:
        malformed_line_count += 1
        vertex_values_equal = False
        triangle_indices_equal = False
    return {
        "vertex_count": vertex_index,
        "triangle_count": triangle_index,
        "vertex_values_float32_roundtrip_identical": bool(
            vertex_values_equal and vertex_index == len(expected_vertices)
        ),
        "triangle_indices_identical": bool(
            triangle_indices_equal and triangle_index == len(expected_triangles)
        ),
        "malformed_line_count": malformed_line_count,
    }


def compiler_source_to_package_parity_report(
    *,
    source_vertices: np.ndarray,
    source_triangles: np.ndarray,
    package_vertices: np.ndarray,
    package_triangles: np.ndarray,
    source_geometry_sha256: str,
    openings: Sequence[Mapping[str, Any]],
    source_to_canonical: Mapping[str, Any],
    debug_obj_path: str | Path,
    declared_surface_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_vertices, expected_triangles = _reference_source_to_package_arrays(
        source_vertices, source_triangles, source_to_canonical
    )
    expected_bounds = [
        expected_vertices.min(axis=0).astype(float).tolist(),
        expected_vertices.max(axis=0).astype(float).tolist(),
    ]
    package_bounds = [
        package_vertices.min(axis=0).astype(float).tolist(),
        package_vertices.max(axis=0).astype(float).tolist(),
    ]
    vertices_equal = bool(np.array_equal(expected_vertices, package_vertices))
    triangles_equal = bool(np.array_equal(expected_triangles, package_triangles))
    bounds_equal = bool(
        np.allclose(expected_bounds, package_bounds, rtol=0.0, atol=1e-7)
    )
    debug_obj = _debug_obj_array_parity(
        debug_obj_path, expected_vertices, expected_triangles
    )
    declared_audit = declared_surface_audit or {}
    declared_vertex_count = declared_audit.get("vertex_count")
    declared_triangle_count = declared_audit.get("triangle_count")
    declared_bounds = declared_audit.get("bounds")
    declared_count_match = bool(
        (declared_vertex_count is None or declared_vertex_count == len(package_vertices))
        and (
            declared_triangle_count is None
            or declared_triangle_count == len(package_triangles)
        )
    )
    declared_bounds_match: bool | None = None
    if isinstance(declared_bounds, Mapping):
        declared_min = declared_bounds.get("min")
        declared_max = declared_bounds.get("max")
        if isinstance(declared_min, list) and isinstance(declared_max, list):
            declared_bounds_match = bool(
                np.allclose(declared_min, package_bounds[0], rtol=0.0, atol=1e-7)
                and np.allclose(
                    declared_max, package_bounds[1], rtol=0.0, atol=1e-7
                )
            )
    passed = (
        vertices_equal
        and triangles_equal
        and bounds_equal
        and debug_obj["vertex_values_float32_roundtrip_identical"]
        and debug_obj["triangle_indices_identical"]
        and debug_obj["malformed_line_count"] == 0
        and declared_count_match
        and declared_bounds_match is not False
    )
    return {
        "schema": "avengine_m3_compiler_source_to_package_parity_v1",
        "status": "pass" if passed else "fail",
        "comparison_scope": "compiler_source_to_package_parity",
        "visual_runtime_parity_claim": False,
        "source_geometry_sha256": source_geometry_sha256,
        "source_to_canonical": dict(source_to_canonical),
        "source_vertex_count": int(len(source_vertices)),
        "expected_package_vertex_count": int(len(expected_vertices)),
        "package_vertex_count": int(len(package_vertices)),
        "source_triangle_count": int(len(source_triangles)),
        "expected_package_triangle_count": int(len(expected_triangles)),
        "package_triangle_count": int(len(package_triangles)),
        "expected_vertex_bytes_identical_to_npy": vertices_equal,
        "expected_triangle_bytes_identical_to_npy": triangles_equal,
        "bounds_identical_within_m": bounds_equal,
        "expected_package_bounds_m": {
            "min": expected_bounds[0],
            "max": expected_bounds[1],
        },
        "package_bounds_m": {
            "min": package_bounds[0],
            "max": package_bounds[1],
        },
        "debug_obj_parity": debug_obj,
        "array_hashes": {
            "raw_expanded_vertices": array_sha256(source_vertices),
            "raw_expanded_triangles": array_sha256(source_triangles),
            "canonical_expected_vertices": array_sha256(expected_vertices),
            "canonical_expected_triangles": array_sha256(expected_triangles),
            "emitted_npy_vertices": array_sha256(package_vertices),
            "emitted_npy_triangles": array_sha256(package_triangles),
        },
        "declared_openings": [dict(opening) for opening in openings],
        "declared_surface_audit": {
            "vertex_count": declared_vertex_count,
            "triangle_count": declared_triangle_count,
            "bounds": declared_bounds,
            "count_match": declared_count_match,
            "bounds_match_within_m": declared_bounds_match,
        },
        "thresholds": {
            "expected_vertex_bytes_identical_to_npy": True,
            "expected_triangle_bytes_identical_to_npy": True,
            "debug_obj_float32_vertices_and_indices_identical": True,
            "bounds_absolute_tolerance_m": 1e-7,
            "declared_surface_audit_count_match": True,
            "declared_surface_audit_bounds_match_if_present": True,
        },
    }


def _trace_first_hit(
    vertices: np.ndarray,
    triangles: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    maximum_distance: float,
) -> tuple[bool, float | None, int | None]:
    epsilon = 1e-7
    first_distance = math.inf
    first_triangle: int | None = None
    chunk_size = 200_000
    for start in range(0, len(triangles), chunk_size):
        chunk = triangles[start : start + chunk_size]
        points = vertices[chunk].astype(np.float64, copy=False)
        edge1 = points[:, 1] - points[:, 0]
        edge2 = points[:, 2] - points[:, 0]
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        a = np.einsum("ij,ij->i", edge1, h)
        valid = np.abs(a) > epsilon
        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]
        s = origin - points[:, 0]
        u = f * np.einsum("ij,ij->i", s, h)
        valid &= (u >= -epsilon) & (u <= 1.0 + epsilon)
        q = np.cross(s, edge1)
        v = f * np.einsum("j,ij->i", direction, q)
        valid &= (v >= -epsilon) & (u + v <= 1.0 + epsilon)
        distance = f * np.einsum("ij,ij->i", edge2, q)
        valid &= (distance > epsilon) & (distance <= maximum_distance + epsilon)
        indices = np.flatnonzero(valid)
        if indices.size:
            local = int(indices[np.argmin(distance[indices])])
            candidate = float(distance[local])
            if candidate < first_distance:
                first_distance = candidate
                first_triangle = start + local
    if first_triangle is None:
        return False, None, None
    return True, first_distance, first_triangle


def _fibonacci_sphere_directions(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("direction count must be positive")
    indices = np.arange(count, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * ((indices + 0.5) / count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = indices * golden_angle
    return np.column_stack((np.cos(angle) * radius, y, np.sin(angle) * radius))


def automatic_mesh_leakage_report(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    origins: Sequence[Sequence[float]],
    direction_count: int = 32,
    directions: np.ndarray | None = None,
    maximum_distance_m: float | None = None,
) -> dict[str, Any]:
    """Probe whether rays from declared interior points escape the surface mesh.

    This is a functional enclosure diagnostic.  It deliberately does not label
    every escape as a defect because valid doors, windows and intentionally open
    scene boundaries require human or dataset-specific interpretation.
    """

    raw_origins = np.asarray(origins, dtype=np.float64)
    if raw_origins.ndim != 2 or raw_origins.shape[1] != 3 or not len(raw_origins):
        raise ValueError("automatic leakage origins must have shape (N, 3)")
    if not np.isfinite(raw_origins).all():
        raise ValueError("automatic leakage origins must be finite")
    if directions is None:
        unit_directions = _fibonacci_sphere_directions(direction_count)
        direction_source = "deterministic_fibonacci_sphere"
    else:
        unit_directions = np.asarray(directions, dtype=np.float64)
        if (
            unit_directions.ndim != 2
            or unit_directions.shape[1] != 3
            or not len(unit_directions)
            or not np.isfinite(unit_directions).all()
        ):
            raise ValueError("automatic leakage directions must have shape (N, 3)")
        norms = np.linalg.norm(unit_directions, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("automatic leakage directions must be non-zero")
        unit_directions = unit_directions / norms[:, None]
        direction_source = "caller_supplied"
    bounds_min = vertices.min(axis=0).astype(np.float64)
    bounds_max = vertices.max(axis=0).astype(np.float64)
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    maximum_distance = (
        float(maximum_distance_m)
        if maximum_distance_m is not None
        else max(diagonal * 1.05, 1.0)
    )
    if not math.isfinite(maximum_distance) or maximum_distance <= 0:
        raise ValueError("automatic leakage maximum distance must be positive")

    origin_reports: list[dict[str, Any]] = []
    total_hits = 0
    hit_distances: list[float] = []
    for origin_index, origin in enumerate(raw_origins):
        escaped: list[int] = []
        hits: list[dict[str, Any]] = []
        for direction_index, direction in enumerate(unit_directions):
            hit, distance, triangle_index = _trace_first_hit(
                vertices,
                triangles,
                origin,
                direction,
                maximum_distance,
            )
            if not hit:
                escaped.append(direction_index)
                continue
            assert distance is not None
            total_hits += 1
            hit_distances.append(distance)
            hits.append(
                {
                    "direction_index": direction_index,
                    "first_hit_distance_m": distance,
                    "triangle_index": triangle_index,
                }
            )
        ray_count = len(unit_directions)
        origin_reports.append(
            {
                "origin_index": origin_index,
                "origin_m": origin.astype(float).tolist(),
                "ray_count": ray_count,
                "hit_ray_count": ray_count - len(escaped),
                "escaped_ray_count": len(escaped),
                "escape_fraction": len(escaped) / ray_count,
                "escaped_direction_indices": escaped,
                "hits": hits,
            }
        )
    total_rays = len(raw_origins) * len(unit_directions)
    escaped_rays = total_rays - total_hits
    distance_summary = {
        "minimum_m": min(hit_distances) if hit_distances else None,
        "maximum_m": max(hit_distances) if hit_distances else None,
        "mean_m": (
            float(np.mean(np.asarray(hit_distances, dtype=np.float64)))
            if hit_distances
            else None
        ),
    }
    return {
        "schema": "avengine_m3_automatic_mesh_leakage_diagnostic_v1",
        "status": "diagnostic_complete",
        "admission_claim": False,
        "interpretation": (
            "Escaped rays identify open directions from reviewed interior probes. "
            "They may be scan holes or legitimate openings and require review."
        ),
        "backend": "compiler_cpu_reference_moller_trumbore",
        "direction_source": direction_source,
        "directions": unit_directions.astype(float).tolist(),
        "maximum_distance_m": maximum_distance,
        "origin_count": len(raw_origins),
        "ray_count": total_rays,
        "hit_ray_count": total_hits,
        "escaped_ray_count": escaped_rays,
        "escape_fraction": escaped_rays / total_rays,
        "first_hit_distance_summary": distance_summary,
        "origins": origin_reports,
    }


def ray_leakage_report(
    vertices: np.ndarray,
    triangles: np.ndarray,
    ray_checks: Sequence[Mapping[str, Any]],
    *,
    automatic_origins: Sequence[Sequence[float]] | None = None,
    automatic_direction_count: int = 32,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for declaration in ray_checks:
        origin = np.asarray(declaration["origin_m"], dtype=np.float64)
        direction = np.asarray(declaration["direction"], dtype=np.float64)
        distance = float(declaration["distance_m"])
        hit, hit_distance, triangle_index = _trace_first_hit(
            vertices, triangles, origin, direction, distance
        )
        expectation = declaration["expectation"]
        passed = (expectation == "clear_until_m" and not hit) or (
            expectation == "hit_within_m" and hit
        )
        checks.append(
            {
                "check_id": declaration["check_id"],
                "status": "pass" if passed else "fail",
                "origin_m": origin.tolist(),
                "direction": direction.tolist(),
                "maximum_distance_m": distance,
                "expectation": expectation,
                "measured_hit": hit,
                "measured_first_hit_distance_m": hit_distance,
                "measured_triangle_index": triangle_index,
            }
        )
    if not checks:
        status = "not_run"
    else:
        status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    has_automatic_origins = (
        automatic_origins is not None and len(automatic_origins) > 0
    )
    automatic = (
        automatic_mesh_leakage_report(
            vertices,
            triangles,
            origins=automatic_origins,
            direction_count=automatic_direction_count,
        )
        if has_automatic_origins
        else {
            "schema": "avengine_m3_automatic_mesh_leakage_diagnostic_v1",
            "status": "not_run",
            "admission_claim": False,
            "reason": "no interior probe origins were supplied",
        }
    )
    return {
        "schema": "avengine_m3_ray_leakage_v1",
        "status": status,
        "backend": "compiler_cpu_reference_moller_trumbore",
        "rlr_runtime_ray_check_status": "not_run",
        "rlr_runtime_note": (
            "This compiler preflight does not replace the post-RLRA_Simulate "
            "TraceRay canary required for final M3 admission."
        ),
        "declared_check_count": len(checks),
        "checks": checks,
        "automatic_enclosure_probe": automatic,
    }


def write_debug_obj(
    path: str | Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    material_ids: np.ndarray,
    categories_document: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> Path:
    output = Path(path)
    category_names = {
        int(category["material_id"]): category["category_name"]
        for category in categories_document["categories"]
    }
    ordered_objects = sorted(objects, key=lambda item: int(item["triangle_offset"]))
    object_index = 0
    current_object_record = ordered_objects[0] if ordered_objects else None
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# AVEngine compiler-side acoustic surface preview\n")
        handle.write(
            "# This is not RLRA_WriteSceneMeshOBJ runtime-ingestion evidence.\n"
        )
        for vertex in vertices:
            handle.write(
                "v "
                + " ".join(format(float(value), ".9g") for value in vertex)
                + "\n"
            )
        current_object: str | None = None
        current_material: int | None = None
        for index, (triangle, material_id_value) in enumerate(
            zip(triangles, material_ids)
        ):
            while current_object_record is not None and index >= int(
                current_object_record["triangle_offset"]
            ) + int(current_object_record["triangle_count"]):
                object_index += 1
                current_object_record = (
                    ordered_objects[object_index]
                    if object_index < len(ordered_objects)
                    else None
                )
            object_id = (
                str(current_object_record["object_id"])
                if current_object_record is not None
                and index >= int(current_object_record["triangle_offset"])
                else "unassigned"
            )
            if object_id != current_object:
                handle.write(f"g {object_id}\n")
                current_object = object_id
                current_material = None
            material_id = int(material_id_value)
            if material_id != current_material:
                handle.write(f"usemtl {category_names[material_id]}\n")
                current_material = material_id
            one_based = triangle.astype(np.uint64) + 1
            handle.write(f"f {one_based[0]} {one_based[1]} {one_based[2]}\n")
    return output
