"""Dependency-free loading for externally expanded USD acoustic snapshots.

Pixar USD is an optional authoring dependency.  The extractor runs in an
environment that provides ``pxr`` and writes one immutable NPZ snapshot.  The
normal AVEngine/Habitat environment then validates and compiles that snapshot
without importing USD or copying the source dataset into the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import sha256_file
from avengine.acoustics.semantic import ExpandedSemanticScene
from avengine.acoustics.semantic_materials import SemanticSurfaceIdentity


USD_ACOUSTIC_SNAPSHOT_SCHEMA = "avengine_m3_usd_acoustic_snapshot_v1"


class UsdAcousticSnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedUsdAcousticSnapshot:
    scene: ExpandedSemanticScene
    surfaces: tuple[SemanticSurfaceIdentity, ...]
    metadata: dict[str, Any]


def _decode_json_array(
    archive: Mapping[str, np.ndarray], name: str
) -> Any:
    if name not in archive:
        raise UsdAcousticSnapshotError(f"USD snapshot is missing {name!r}")
    value = np.asarray(archive[name])
    if value.ndim != 1 or value.dtype != np.dtype("u1"):
        raise UsdAcousticSnapshotError(
            f"USD snapshot {name!r} must be a one-dimensional uint8 array"
        )
    try:
        return json.loads(value.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UsdAcousticSnapshotError(
            f"USD snapshot {name!r} is not valid UTF-8 JSON: {exc}"
        ) from exc


def _nonnegative_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UsdAcousticSnapshotError(f"{owner} must be a non-negative integer")
    return value


def _positive_integer(value: Any, *, owner: str) -> int:
    result = _nonnegative_integer(value, owner=owner)
    if result == 0:
        raise UsdAcousticSnapshotError(f"{owner} must be positive")
    return result


def _validate_source_to_canonical(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise UsdAcousticSnapshotError(
            "USD snapshot source_to_canonical must be an object"
        )
    matrix = value.get("matrix_row_major")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 16
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in matrix
        )
    ):
        raise UsdAcousticSnapshotError(
            "USD snapshot source_to_canonical.matrix_row_major must have 16 numbers"
        )
    matrix_array = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if (
        not np.isfinite(matrix_array).all()
        or not np.allclose(matrix_array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
        or abs(float(np.linalg.det(matrix_array[:3, :3]))) <= 1e-12
    ):
        raise UsdAcousticSnapshotError(
            "USD snapshot source_to_canonical must be a finite nonsingular affine matrix"
        )
    if not isinstance(value.get("source"), str) or not value["source"]:
        raise UsdAcousticSnapshotError(
            "USD snapshot source_to_canonical.source must be non-empty"
        )
    if value.get("reviewed") is not True:
        raise UsdAcousticSnapshotError(
            "USD snapshot source_to_canonical.reviewed must be true"
        )
    return {
        "matrix_row_major": [float(item) for item in matrix],
        "source": value["source"],
        "reviewed": True,
    }


def load_usd_acoustic_snapshot(path: str | Path) -> LoadedUsdAcousticSnapshot:
    """Load and structurally validate one exact USD-derived NPZ snapshot."""

    source = Path(path).resolve()
    if not source.is_file():
        raise UsdAcousticSnapshotError(f"USD acoustic snapshot does not exist: {source}")
    source_sha256 = sha256_file(source)
    source_byte_size = source.stat().st_size
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "vertices",
                "triangles",
                "metadata_json_utf8",
                "objects_json_utf8",
                "surfaces_json_utf8",
            }
            missing = required - set(archive.files)
            if missing:
                raise UsdAcousticSnapshotError(
                    f"USD snapshot is missing arrays: {sorted(missing)}"
                )
            vertices = np.ascontiguousarray(archive["vertices"], dtype="<f4")
            triangles = np.ascontiguousarray(archive["triangles"], dtype="<u4")
            metadata = _decode_json_array(archive, "metadata_json_utf8")
            objects = _decode_json_array(archive, "objects_json_utf8")
            surfaces_raw = _decode_json_array(archive, "surfaces_json_utf8")
    except (OSError, ValueError) as exc:
        if isinstance(exc, UsdAcousticSnapshotError):
            raise
        raise UsdAcousticSnapshotError(
            f"unable to load USD acoustic snapshot {source}: {exc}"
        ) from exc

    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise UsdAcousticSnapshotError(
            "USD snapshot vertices must have shape [N, 3] with N >= 3"
        )
    if not np.isfinite(vertices).all():
        raise UsdAcousticSnapshotError("USD snapshot vertices contain non-finite values")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) < 1:
        raise UsdAcousticSnapshotError(
            "USD snapshot triangles must have shape [M, 3] with M >= 1"
        )
    if int(triangles.max(initial=0)) >= len(vertices):
        raise UsdAcousticSnapshotError(
            "USD snapshot contains an out-of-range triangle index"
        )
    if not isinstance(metadata, dict):
        raise UsdAcousticSnapshotError("USD snapshot metadata must be an object")
    if metadata.get("schema") != USD_ACOUSTIC_SNAPSHOT_SCHEMA:
        raise UsdAcousticSnapshotError(
            f"USD snapshot metadata.schema must be {USD_ACOUSTIC_SNAPSHOT_SCHEMA!r}"
        )
    _validate_source_to_canonical(metadata.get("source_to_canonical"))
    primitive_count = _positive_integer(
        metadata.get("source_primitive_count"),
        owner="metadata.source_primitive_count",
    )
    node_count = _positive_integer(
        metadata.get("source_node_instance_count"),
        owner="metadata.source_node_instance_count",
    )

    if not isinstance(objects, list) or len(objects) != primitive_count:
        raise UsdAcousticSnapshotError(
            "USD snapshot objects must match metadata.source_primitive_count"
        )
    expected_triangle_offset = 0
    used_material_names: set[str] = set()
    normalized_objects: list[dict[str, Any]] = []
    required_object_fields = {
        "object_id",
        "source_node_index",
        "source_mesh_index",
        "source_primitive_index",
        "source_material_name",
        "vertex_offset",
        "vertex_count",
        "triangle_offset",
        "triangle_count",
        "world_from_object",
        "source_world_matrix",
        "transform_baked",
    }
    for index, raw in enumerate(objects):
        if not isinstance(raw, dict) or set(raw) != required_object_fields:
            raise UsdAcousticSnapshotError(
                f"USD snapshot objects[{index}] has invalid fields"
            )
        for field in ("object_id", "source_material_name"):
            if not isinstance(raw.get(field), str) or not raw[field]:
                raise UsdAcousticSnapshotError(
                    f"USD snapshot objects[{index}].{field} must be non-empty"
                )
        for field in (
            "source_node_index",
            "source_mesh_index",
            "source_primitive_index",
            "vertex_offset",
            "triangle_offset",
        ):
            _nonnegative_integer(raw.get(field), owner=f"objects[{index}].{field}")
        vertex_count = _positive_integer(
            raw.get("vertex_count"), owner=f"objects[{index}].vertex_count"
        )
        triangle_count = _positive_integer(
            raw.get("triangle_count"), owner=f"objects[{index}].triangle_count"
        )
        if raw["vertex_offset"] + vertex_count > len(vertices):
            raise UsdAcousticSnapshotError(
                f"USD snapshot objects[{index}] vertex range is out of bounds"
            )
        if raw["triangle_offset"] != expected_triangle_offset:
            raise UsdAcousticSnapshotError(
                "USD snapshot object triangle ranges must be contiguous and ordered"
            )
        expected_triangle_offset += triangle_count
        for field in ("world_from_object", "source_world_matrix"):
            matrix = raw.get(field)
            if (
                not isinstance(matrix, list)
                or len(matrix) != 16
                or not np.isfinite(np.asarray(matrix, dtype=np.float64)).all()
            ):
                raise UsdAcousticSnapshotError(
                    f"USD snapshot objects[{index}].{field} must have 16 finite numbers"
                )
        if raw.get("transform_baked") is not True:
            raise UsdAcousticSnapshotError(
                f"USD snapshot objects[{index}].transform_baked must be true"
            )
        used_material_names.add(raw["source_material_name"])
        normalized_objects.append(dict(raw))
    if expected_triangle_offset != len(triangles):
        raise UsdAcousticSnapshotError(
            "USD snapshot object ranges do not cover every triangle"
        )

    if not isinstance(surfaces_raw, list) or not surfaces_raw:
        raise UsdAcousticSnapshotError(
            "USD snapshot surfaces must be a non-empty array"
        )
    surfaces: list[SemanticSurfaceIdentity] = []
    surface_names: set[str] = set()
    for index, raw in enumerate(surfaces_raw):
        if not isinstance(raw, dict):
            raise UsdAcousticSnapshotError(
                f"USD snapshot surfaces[{index}] must be an object"
            )
        values: dict[str, str] = {}
        for field in (
            "source_material_name",
            "semantic_category",
            "identity_key",
            "material_slot",
            "object_name",
        ):
            value = raw.get(field, "")
            if not isinstance(value, str):
                raise UsdAcousticSnapshotError(
                    f"USD snapshot surfaces[{index}].{field} must be a string"
                )
            values[field] = value
        if not values["source_material_name"] or not values["semantic_category"]:
            raise UsdAcousticSnapshotError(
                f"USD snapshot surfaces[{index}] lacks source material/category"
            )
        if values["source_material_name"] in surface_names:
            raise UsdAcousticSnapshotError(
                "USD snapshot surface source_material_name values must be unique"
            )
        surface_names.add(values["source_material_name"])
        surfaces.append(SemanticSurfaceIdentity(**values))
    if surface_names != used_material_names:
        raise UsdAcousticSnapshotError(
            "USD snapshot surfaces must exactly cover object source material names"
        )

    category_counts: dict[str, int] = {}
    surface_by_name = {
        item.source_material_name: item.semantic_category for item in surfaces
    }
    for item in normalized_objects:
        category = surface_by_name[item["source_material_name"]]
        category_counts[category] = (
            category_counts.get(category, 0) + int(item["triangle_count"])
        )
    scene = ExpandedSemanticScene(
        vertices=vertices,
        triangles=triangles,
        objects=tuple(normalized_objects),
        source_primitive_count=primitive_count,
        source_node_instance_count=node_count,
        source_sha256=source_sha256,
        source_byte_size=source_byte_size,
        descriptor_sha256=source_sha256,
        descriptor_byte_size=source_byte_size,
        source_vertex_count=len(vertices),
        source_triangle_count=len(triangles),
        semantic_categories=tuple(sorted(category_counts)),
        category_triangle_counts=dict(sorted(category_counts.items())),
    )
    return LoadedUsdAcousticSnapshot(
        scene=scene,
        surfaces=tuple(surfaces),
        metadata=metadata,
    )
