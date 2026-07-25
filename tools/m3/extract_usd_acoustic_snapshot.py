#!/usr/bin/env python3
"""Expand a static USD room into one auditable M3 acoustic snapshot.

Run this tool in an optional environment that provides Pixar USD.  It follows
the composed USD stage, keeps visible Mesh prims, bakes their world transforms,
preserves bound-material and object/category identities, and writes one NPZ
snapshot consumed by the normal AVEngine/Habitat environment.

The source USD and all referenced dataset files remain untouched.  This tool
does not infer calibrated acoustic coefficients and does not repair holes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np


SNAPSHOT_SCHEMA = "avengine_m3_usd_acoustic_snapshot_v1"
TRANSFORM_PROFILES = {
    "kujiale_z_up_y_back_to_habitat": {
        "matrix_row_major": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "source": (
            "Reviewed InteriorAgent/Kujiale runtime convention: source "
            "[x, y, z] Z-up metres -> AVEngine/Habitat [x, z, y] "
            "[right, up, back]"
        ),
        "reviewed": True,
    }
}
IDENTITY_MATRIX = np.eye(4, dtype=float).reshape(-1).tolist()


class UsdSnapshotExtractionError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _object_type(value: str) -> str:
    token = _token(value)
    token = re.sub(r"_\d+$", "", token)
    return token or "unknown_object"


def _semantic_category(prim_path: str) -> tuple[str, str, str]:
    parts = [part for part in prim_path.split("/") if part]
    if len(parts) < 4 or parts[:2] != ["Root", "Meshes"]:
        raise UsdSnapshotExtractionError(
            f"Mesh prim is outside /Root/Meshes: {prim_path}"
        )
    scope = _token(parts[2])
    object_name = parts[3] if len(parts) >= 4 else parts[2]
    if scope in {"wall", "floor", "ceiling"}:
        category = scope
    else:
        category = _object_type(object_name)
    return category, object_name, scope


def _bound_material_path(prim: Any, UsdShade: Any) -> str:
    material, _relationship = UsdShade.MaterialBindingAPI(
        prim
    ).ComputeBoundMaterial()
    if material and material.GetPrim().IsValid():
        return str(material.GetPath())
    return "<unbound>"


def _matrix_rows(matrix: Any) -> list[float]:
    values = [float(matrix[row][column]) for row in range(4) for column in range(4)]
    if not all(math.isfinite(value) for value in values):
        raise UsdSnapshotExtractionError("USD world transform contains non-finite values")
    return values


def _world_points(points: Any, matrix: Any, Gf: Any) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray([[float(v) for v in point] for point in points], dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or len(source) < 3:
        raise UsdSnapshotExtractionError("USD Mesh points must have shape [N, 3]")
    world_matrix = np.asarray(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )
    homogeneous = np.concatenate(
        [source, np.ones((len(source), 1), dtype=np.float64)], axis=1
    )
    transformed = (homogeneous @ world_matrix)[:, :3]
    # USD/Gf uses row-vector affine matrices.  Check the vectorized convention
    # against the authoritative API before using it for the full mesh.
    reference = matrix.Transform(Gf.Vec3d(*source[0]))
    if not np.allclose(
        transformed[0],
        [float(reference[0]), float(reference[1]), float(reference[2])],
        atol=1e-8,
    ):
        raise UsdSnapshotExtractionError(
            "USD matrix convention check failed while baking Mesh points"
        )
    if not np.isfinite(transformed).all():
        raise UsdSnapshotExtractionError(
            "USD Mesh world transform produced non-finite points"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        encoded = transformed.astype("<f4")
    if not np.isfinite(encoded).all():
        raise UsdSnapshotExtractionError(
            "USD Mesh world points overflow float32 acoustic encoding"
        )
    return encoded, world_matrix


def _triangles_by_material(
    mesh: Any,
    prim: Any,
    *,
    Usd: Any,
    UsdGeom: Any,
    UsdShade: Any,
) -> dict[str, np.ndarray]:
    counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if counts is None or indices is None:
        raise UsdSnapshotExtractionError(f"USD Mesh lacks face arrays: {prim.GetPath()}")
    face_counts = [int(value) for value in counts]
    flat_indices = np.asarray(indices, dtype=np.int64)
    if sum(face_counts) != len(flat_indices):
        raise UsdSnapshotExtractionError(
            f"USD Mesh face counts/indices disagree: {prim.GetPath()}"
        )
    hole_faces = set(
        int(value)
        for value in (
            mesh.GetHoleIndicesAttr().Get(Usd.TimeCode.Default()) or []
        )
    )
    if any(index < 0 or index >= len(face_counts) for index in hole_faces):
        raise UsdSnapshotExtractionError(
            f"USD Mesh has an invalid hole face index: {prim.GetPath()}"
        )

    base_material = _bound_material_path(prim, UsdShade)
    face_materials = [base_material] * len(face_counts)
    assigned_by_subset: set[int] = set()
    for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
        subset_prim = subset.GetPrim()
        material_path = _bound_material_path(subset_prim, UsdShade)
        if material_path == "<unbound>":
            continue
        subset_indices = subset.GetIndicesAttr().Get(Usd.TimeCode.Default()) or []
        for raw_index in subset_indices:
            face_index = int(raw_index)
            if face_index < 0 or face_index >= len(face_counts):
                raise UsdSnapshotExtractionError(
                    f"USD material subset has an invalid face index: {subset_prim.GetPath()}"
                )
            if face_index in assigned_by_subset:
                raise UsdSnapshotExtractionError(
                    f"USD material subsets overlap on face {face_index}: {prim.GetPath()}"
                )
            assigned_by_subset.add(face_index)
            face_materials[face_index] = material_path

    groups: dict[str, list[tuple[int, int, int]]] = {}
    offset = 0
    for face_index, count in enumerate(face_counts):
        face = flat_indices[offset : offset + count]
        offset += count
        if face_index in hole_faces:
            continue
        if count < 3:
            raise UsdSnapshotExtractionError(
                f"USD Mesh contains a face with fewer than three vertices: {prim.GetPath()}"
            )
        if np.any(face < 0):
            raise UsdSnapshotExtractionError(
                f"USD Mesh contains a negative vertex index: {prim.GetPath()}"
            )
        target = groups.setdefault(face_materials[face_index], [])
        first = int(face[0])
        for triangle_index in range(1, count - 1):
            target.append(
                (first, int(face[triangle_index]), int(face[triangle_index + 1]))
            )
    return {
        material: np.asarray(triangles, dtype="<u4")
        for material, triangles in groups.items()
        if triangles
    }


def _extract_stage(
    *,
    source: Path,
    room_id: str,
    transform_profile: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdShade
    except ImportError as exc:
        raise UsdSnapshotExtractionError(
            "Pixar USD Python bindings are required for USD extraction"
        ) from exc

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise UsdSnapshotExtractionError(f"could not open USD stage: {source}")
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if up_axis.upper() != "Z":
        raise UsdSnapshotExtractionError(
            f"USD acoustic extractor expected Z-up, got {up_axis!r}"
        )
    if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise UsdSnapshotExtractionError(
            f"USD acoustic extractor expected metresPerUnit=1, got {meters_per_unit}"
        )

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertex_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    objects: list[dict[str, Any]] = []
    surface_records: dict[str, dict[str, str]] = {}
    category_triangle_counts: Counter[str] = Counter()
    scope_mesh_counts: Counter[str] = Counter()
    hidden_mesh_count = 0
    mesh_count = 0
    visible_mesh_count = 0
    vertex_offset = 0
    triangle_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = str(prim.GetPath())
        if not prim_path.startswith("/Root/Meshes/"):
            continue
        mesh_count += 1
        if (
            UsdGeom.Imageable(prim).ComputeVisibility()
            == UsdGeom.Tokens.invisible
        ):
            hidden_mesh_count += 1
            continue
        visible_mesh_count += 1
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetPointsAttr().ValueMightBeTimeVarying():
            raise UsdSnapshotExtractionError(
                f"USD room Mesh points are time-varying: {prim_path}"
            )
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        if points is None:
            raise UsdSnapshotExtractionError(f"USD Mesh has no points: {prim_path}")
        world = xform_cache.GetLocalToWorldTransform(prim)
        world_vertices, world_matrix = _world_points(points, world, Gf)
        groups = _triangles_by_material(
            mesh,
            prim,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
        )
        category, object_name, scope = _semantic_category(prim_path)
        scope_mesh_counts[scope] += 1
        orientation = mesh.GetOrientationAttr().Get(Usd.TimeCode.Default())
        source_left_handed = orientation == UsdGeom.Tokens.leftHanded
        transform_reflects = float(np.linalg.det(world_matrix[:3, :3])) < 0

        for primitive_index, (material_path, local_triangles) in enumerate(
            sorted(groups.items())
        ):
            if int(local_triangles.max(initial=0)) >= len(world_vertices):
                raise UsdSnapshotExtractionError(
                    f"USD Mesh has an out-of-range vertex index: {prim_path}"
                )
            adjusted = local_triangles.copy()
            if source_left_handed != transform_reflects:
                adjusted[:, [1, 2]] = adjusted[:, [2, 1]]
            adjusted = np.ascontiguousarray(
                adjusted.astype(np.uint64) + vertex_offset,
                dtype="<u4",
            )
            material_slot = (
                Path(material_path).name if material_path != "<unbound>" else "unbound"
            )
            source_material_name = f"usd::{category}::{material_slot}"
            identity_key = f"{room_id}/{category}/{material_slot}"
            surface = {
                "source_material_name": source_material_name,
                "semantic_category": category,
                "identity_key": identity_key,
                "material_slot": material_slot,
                "object_name": object_name,
            }
            previous = surface_records.setdefault(source_material_name, surface)
            if previous != surface:
                # Object names are only a hint.  The semantic category and
                # material slot define the shared acoustic surface identity.
                comparable_previous = dict(previous)
                comparable_surface = dict(surface)
                comparable_previous["object_name"] = category
                comparable_surface["object_name"] = category
                if comparable_previous != comparable_surface:
                    raise UsdSnapshotExtractionError(
                        f"conflicting USD surface identity: {source_material_name}"
                    )
                previous["object_name"] = category

            vertex_parts.append(world_vertices)
            triangle_parts.append(adjusted)
            triangle_count = len(adjusted)
            category_triangle_counts[category] += triangle_count
            objects.append(
                {
                    "object_id": f"{prim_path}#material={material_slot}",
                    "source_node_index": visible_mesh_count - 1,
                    "source_mesh_index": visible_mesh_count - 1,
                    "source_primitive_index": primitive_index,
                    "source_material_name": source_material_name,
                    "vertex_offset": vertex_offset,
                    "vertex_count": len(world_vertices),
                    "triangle_offset": triangle_offset,
                    "triangle_count": triangle_count,
                    "world_from_object": IDENTITY_MATRIX,
                    "source_world_matrix": _matrix_rows(world),
                    "transform_baked": True,
                }
            )
            vertex_offset += len(world_vertices)
            triangle_offset += triangle_count

    if not vertex_parts or not triangle_parts:
        raise UsdSnapshotExtractionError("USD stage produced no visible triangle Meshes")
    vertices = np.ascontiguousarray(np.concatenate(vertex_parts), dtype="<f4")
    triangles = np.ascontiguousarray(np.concatenate(triangle_parts), dtype="<u4")
    if len(vertices) > np.iinfo(np.uint32).max:
        raise UsdSnapshotExtractionError("USD acoustic snapshot exceeds uint32 indices")
    transform = TRANSFORM_PROFILES[transform_profile]
    metadata = {
        "schema": SNAPSHOT_SCHEMA,
        "room_id": room_id,
        "source_stage": str(source),
        "source_stage_sha256": _sha256(source),
        "source_stage_byte_size": source.stat().st_size,
        "stage_up_axis": "+Z",
        "stage_meters_per_unit": meters_per_unit,
        "source_to_canonical": transform,
        "source_mesh_prim_count": mesh_count,
        "visible_mesh_prim_count": visible_mesh_count,
        "hidden_mesh_prim_count": hidden_mesh_count,
        "source_primitive_count": len(objects),
        "source_node_instance_count": visible_mesh_count,
        "source_vertex_count": len(vertices),
        "source_triangle_count": len(triangles),
        "surface_identity_count": len(surface_records),
        "semantic_category_triangle_counts": dict(
            sorted(category_triangle_counts.items())
        ),
        "scope_visible_mesh_counts": dict(sorted(scope_mesh_counts.items())),
        "geometry_claim": "composed_visible_real_surface_mesh_no_hole_repair",
        "physical_material_claim": False,
    }
    return (
        vertices,
        triangles,
        objects,
        [surface_records[key] for key in sorted(surface_records)],
        metadata,
    )


def extract(
    *,
    source: Path,
    output: Path,
    room_id: str,
    transform_profile: str,
    interior_origins: Sequence[Sequence[float]],
    source_revision: str,
    dataset_id: str,
    source_license: str,
) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    if transform_profile not in TRANSFORM_PROFILES:
        raise UsdSnapshotExtractionError(
            f"unknown transform profile {transform_profile!r}"
        )
    if len(interior_origins) < 2:
        raise UsdSnapshotExtractionError(
            "at least two reviewed canonical interior origins are required"
        )
    origins = [[float(value) for value in item] for item in interior_origins]
    if any(
        len(item) != 3 or not all(math.isfinite(value) for value in item)
        for item in origins
    ):
        raise UsdSnapshotExtractionError(
            "every interior origin must contain three finite canonical coordinates"
        )

    (
        vertices,
        triangles,
        objects,
        surfaces,
        metadata,
    ) = _extract_stage(
        source=source,
        room_id=room_id,
        transform_profile=transform_profile,
    )
    metadata["reviewed_interior_origins_m"] = origins
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        snapshot_path = staging / "scene_snapshot.npz"
        np.savez(
            snapshot_path,
            vertices=vertices,
            triangles=triangles,
            metadata_json_utf8=np.frombuffer(_json_bytes(metadata), dtype="u1"),
            objects_json_utf8=np.frombuffer(_json_bytes(objects), dtype="u1"),
            surfaces_json_utf8=np.frombuffer(_json_bytes(surfaces), dtype="u1"),
        )
        snapshot_sha256 = _sha256(snapshot_path)
        transform = TRANSFORM_PROFILES[transform_profile]
        matrix = np.asarray(transform["matrix_row_major"], dtype=np.float64).reshape(
            4, 4
        )
        homogeneous = np.concatenate(
            [vertices.astype(np.float64), np.ones((len(vertices), 1))], axis=1
        )
        canonical = (matrix @ homogeneous.T).T[:, :3]
        report = {
            "schema": "avengine_m3_usd_acoustic_extraction_report_v1",
            "status": "research_candidate",
            "room_id": room_id,
            "source_stage": str(source),
            "source_stage_sha256": metadata["source_stage_sha256"],
            "snapshot": snapshot_path.name,
            "snapshot_sha256": snapshot_sha256,
            "snapshot_byte_size": snapshot_path.stat().st_size,
            "source_mesh_prim_count": metadata["source_mesh_prim_count"],
            "visible_mesh_prim_count": metadata["visible_mesh_prim_count"],
            "hidden_mesh_prim_count": metadata["hidden_mesh_prim_count"],
            "compiled_object_partition_count": len(objects),
            "surface_identity_count": len(surfaces),
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
            "source_bounds_xyz_m": {
                "min": vertices.min(axis=0).astype(float).tolist(),
                "max": vertices.max(axis=0).astype(float).tolist(),
            },
            "canonical_bounds_xyz_m": {
                "min": canonical.min(axis=0).astype(float).tolist(),
                "max": canonical.max(axis=0).astype(float).tolist(),
            },
            "semantic_category_triangle_counts": metadata[
                "semantic_category_triangle_counts"
            ],
            "reviewed_interior_origins_m": origins,
            "geometry_claim": metadata["geometry_claim"],
            "physical_material_claim": False,
            "hole_repair": "not_performed",
        }
        report_path = staging / "extraction_report.json"
        _write_json(report_path, report)
        room_manifest = {
            "schema": "avengine_room_package_v1",
            "room_id": room_id,
            "room_kind": "external_usd_real_surface",
            "geometry_representation": "real_surface_mesh",
            "coordinate_system": {
                "handedness": "right",
                "up_axis": "+Y",
                "forward_axis": "-Z",
                "linear_unit": "meter",
                "quaternion_order": "xyzw",
            },
            "scene": {
                "scene_id_kind": "path",
                "scene_id": "scene_snapshot.npz",
                "dataset_config_path": "not_applicable_external_usd",
                "navmesh_path": "not_applicable_external_usd",
                "navmesh_policy": "recompute_if_missing",
                "load_semantic_mesh": False,
                "enable_physics": False,
            },
            "assets": [
                {
                    "role": "render_surface_mesh",
                    "path": "scene_snapshot.npz",
                    "license": source_license,
                    "redistribution": "external_source_not_redistributed",
                },
                {
                    "role": "source_usd_stage",
                    "path": str(source),
                    "license": source_license,
                    "redistribution": "external_source_not_redistributed",
                },
                {
                    "role": "usd_acoustic_extraction_report",
                    "path": "extraction_report.json",
                    "license": "AVEngine generated metadata",
                    "redistribution": "generated_local",
                },
                {
                    "role": "scene_dataset_config",
                    "path": "not_applicable_external_usd",
                    "license": "not applicable to offline acoustic compilation",
                    "redistribution": "not_applicable",
                },
                {
                    "role": "navmesh",
                    "path": "not_applicable_external_usd",
                    "license": "not applicable to offline acoustic compilation",
                    "redistribution": "not_applicable",
                },
            ],
            "semantics": {
                "interpretation": (
                    "USD /Root/Meshes scope, object name and composed material binding"
                )
            },
            "navigation": {
                "agent_height_m": 1.5,
                "agent_radius_m": 0.2,
                "include_static_objects": True,
            },
            "openings": [],
            "connectivity_pairs": [
                {
                    "pair_id": "reviewed_interior_origin_pair_0",
                    "start_m": origins[0],
                    "end_m": origins[1],
                }
            ],
            "ray_checks": [],
            "acoustics": {
                "status": "deferred_to_m3",
                "reason": (
                    "USD geometry and semantic material proposals require M3 "
                    "compilation, leakage review and later physical calibration"
                ),
            },
            "provenance": {
                "source": dataset_id,
                "source_revision": source_revision,
                "source_stage": str(source),
                "source_stage_sha256": metadata["source_stage_sha256"],
                "derived_snapshot_sha256": snapshot_sha256,
                "license": source_license,
            },
            "surface_audit": {
                "method": (
                    "Pixar USD composed-stage traversal of visible real Mesh prims "
                    "with baked world transforms and bound material identities"
                ),
                "aabb_proxy": False,
                "hole_repair": False,
            },
        }
        _write_json(staging / "room_manifest.json", room_manifest)
        os.rename(staging, output)
        return output / "scene_snapshot.npz"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--room-id", required=True)
    parser.add_argument(
        "--transform-profile",
        choices=sorted(TRANSFORM_PROFILES),
        required=True,
    )
    parser.add_argument(
        "--interior-origin",
        nargs=3,
        action="append",
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Reviewed canonical interior point; provide at least two",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-license", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    snapshot = extract(
        source=Path(args.source),
        output=Path(args.output),
        room_id=args.room_id,
        transform_profile=args.transform_profile,
        interior_origins=args.interior_origin,
        source_revision=args.source_revision,
        dataset_id=args.dataset_id,
        source_license=args.source_license,
    )
    print(
        json.dumps(
            {
                "status": "research_candidate",
                "snapshot": str(snapshot),
                "room_manifest": str(snapshot.parent / "room_manifest.json"),
                "extraction_report": str(snapshot.parent / "extraction_report.json"),
                "physical_material_claim": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
