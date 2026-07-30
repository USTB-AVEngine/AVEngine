"""Semantic surface extraction for research acoustic-scene compilation.

The first supported source is Matterport3D's binary semantic PLY paired with
its ``.house`` descriptor.  The parser deliberately accepts only the fixed
triangle encoding used by the dataset sample: accepting an unknown PLY layout
would make per-face semantic identity unauditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np


class SemanticSceneError(ValueError):
    pass


@dataclass(frozen=True)
class Mp3dHouseSemantics:
    category_index_to_label: dict[int, str]
    category_index_to_raw_label: dict[int, str]
    object_id_to_category_index: dict[int, int]
    source_sha256: str
    source_byte_size: int


@dataclass(frozen=True)
class ExpandedSemanticScene:
    vertices: np.ndarray
    triangles: np.ndarray
    objects: tuple[dict[str, Any], ...]
    source_primitive_count: int
    source_node_instance_count: int
    source_sha256: str
    source_byte_size: int
    descriptor_sha256: str
    descriptor_byte_size: int
    source_vertex_count: int
    source_triangle_count: int
    semantic_categories: tuple[str, ...]
    category_triangle_counts: dict[str, int]
    raw_semantic_category_labels: tuple[str, ...] = ()


_IDENTITY_MATRIX = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def _semantic_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "unknown_category"


def parse_mp3d_house_bytes(payload: bytes) -> Mp3dHouseSemantics:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SemanticSceneError(f"MP3D .house descriptor is not UTF-8: {exc}") from exc

    categories: dict[int, str] = {}
    raw_categories: dict[int, str] = {}
    objects: dict[int, int] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "C":
            if len(fields) < 6:
                raise SemanticSceneError(
                    f"MP3D .house C record at line {line_number} is truncated"
                )
            try:
                category_index = int(fields[1])
            except ValueError as exc:
                raise SemanticSceneError(
                    f"MP3D .house C record at line {line_number} has invalid index"
                ) from exc
            raw_label = fields[5]
            canonical_label = _semantic_label(raw_label)
            previous = categories.setdefault(category_index, canonical_label)
            if previous != canonical_label:
                raise SemanticSceneError(
                    f"MP3D category index {category_index} has conflicting labels"
                )
            previous_raw = raw_categories.setdefault(category_index, raw_label)
            if previous_raw != raw_label:
                raise SemanticSceneError(
                    f"MP3D category index {category_index} has conflicting raw labels"
                )
        elif fields[0] == "O":
            if len(fields) < 4:
                raise SemanticSceneError(
                    f"MP3D .house O record at line {line_number} is truncated"
                )
            try:
                object_id = int(fields[1])
                category_index = int(fields[3])
            except ValueError as exc:
                raise SemanticSceneError(
                    f"MP3D .house O record at line {line_number} has invalid IDs"
                ) from exc
            previous = objects.setdefault(object_id, category_index)
            if previous != category_index:
                raise SemanticSceneError(
                    f"MP3D object ID {object_id} has conflicting categories"
                )
    if not categories:
        raise SemanticSceneError("MP3D .house descriptor contains no C records")
    if not objects:
        raise SemanticSceneError("MP3D .house descriptor contains no O records")
    return Mp3dHouseSemantics(
        category_index_to_label=categories,
        category_index_to_raw_label=raw_categories,
        object_id_to_category_index=objects,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_byte_size=len(payload),
    )


def _parse_semantic_ply_bytes(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header_end_marker = b"end_header\n"
    header_end = payload.find(header_end_marker)
    if header_end < 0:
        raise SemanticSceneError("semantic PLY has no end_header line")
    data_offset = header_end + len(header_end_marker)
    try:
        header = payload[:data_offset].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise SemanticSceneError(f"semantic PLY header is not ASCII: {exc}") from exc
    if not header or header[0] != "ply":
        raise SemanticSceneError("semantic PLY magic is invalid")
    if "format binary_little_endian 1.0" not in header:
        raise SemanticSceneError(
            "semantic PLY must use binary_little_endian 1.0"
        )

    vertex_count: int | None = None
    face_count: int | None = None
    vertex_properties: list[str] = []
    face_properties: list[str] = []
    current_element: str | None = None
    for line in header:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "element" and len(fields) == 3:
            current_element = fields[1]
            try:
                count = int(fields[2])
            except ValueError as exc:
                raise SemanticSceneError(
                    f"semantic PLY element count is invalid: {line}"
                ) from exc
            if count < 0:
                raise SemanticSceneError("semantic PLY element count is negative")
            if current_element == "vertex":
                vertex_count = count
            elif current_element == "face":
                face_count = count
        elif fields[0] == "property":
            if current_element == "vertex":
                vertex_properties.append(" ".join(fields[1:]))
            elif current_element == "face":
                face_properties.append(" ".join(fields[1:]))

    expected_vertex_properties = [
        "float x",
        "float y",
        "float z",
        "uchar red",
        "uchar green",
        "uchar blue",
    ]
    expected_face_properties = [
        "list uchar int vertex_indices",
        "int object_id",
    ]
    if vertex_count is None or face_count is None:
        raise SemanticSceneError("semantic PLY must declare vertex and face elements")
    if vertex_properties != expected_vertex_properties:
        raise SemanticSceneError(
            "unsupported semantic PLY vertex layout; expected xyz float32 plus RGB uint8"
        )
    if face_properties != expected_face_properties:
        raise SemanticSceneError(
            "unsupported semantic PLY face layout; expected triangle indices plus object_id"
        )

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
        align=False,
    )
    face_dtype = np.dtype(
        [
            ("count", "u1"),
            ("indices", "<i4", (3,)),
            ("object_id", "<i4"),
        ],
        align=False,
    )
    expected_size = (
        data_offset + vertex_count * vertex_dtype.itemsize + face_count * face_dtype.itemsize
    )
    if len(payload) != expected_size:
        raise SemanticSceneError(
            "semantic PLY byte length does not match its fixed declared layout"
        )
    raw_vertices = np.frombuffer(
        payload, dtype=vertex_dtype, count=vertex_count, offset=data_offset
    )
    face_offset = data_offset + vertex_count * vertex_dtype.itemsize
    raw_faces = np.frombuffer(
        payload, dtype=face_dtype, count=face_count, offset=face_offset
    )
    if np.any(raw_faces["count"] != 3):
        raise SemanticSceneError("semantic PLY contains a non-triangle face")
    triangles_i64 = raw_faces["indices"].astype(np.int64, copy=False)
    if triangles_i64.size and (
        int(triangles_i64.min()) < 0 or int(triangles_i64.max()) >= vertex_count
    ):
        raise SemanticSceneError("semantic PLY contains an out-of-range vertex index")
    vertices = np.empty((vertex_count, 3), dtype="<f4")
    vertices[:, 0] = raw_vertices["x"]
    vertices[:, 1] = raw_vertices["y"]
    vertices[:, 2] = raw_vertices["z"]
    if not np.isfinite(vertices).all():
        raise SemanticSceneError("semantic PLY contains non-finite vertices")
    return (
        vertices,
        np.ascontiguousarray(triangles_i64, dtype="<u4"),
        np.ascontiguousarray(raw_faces["object_id"], dtype="<i4"),
    )


def load_mp3d_semantic_scene(
    semantic_ply: str | Path,
    house_descriptor: str | Path,
) -> ExpandedSemanticScene:
    """Load and group MP3D semantic triangles into auditable material ranges."""

    ply_path = Path(semantic_ply).resolve()
    house_path = Path(house_descriptor).resolve()
    ply_payload = ply_path.read_bytes()
    house_payload = house_path.read_bytes()
    house = parse_mp3d_house_bytes(house_payload)
    source_vertices, source_triangles, face_object_ids = _parse_semantic_ply_bytes(
        ply_payload
    )

    unique_object_ids = np.unique(face_object_ids)
    object_label: dict[int, str] = {}
    canonical_to_raw_label: dict[str, str] = {}
    for value in unique_object_ids:
        object_id = int(value)
        category_index = house.object_id_to_category_index.get(object_id)
        label = (
            house.category_index_to_label.get(category_index)
            if category_index is not None
            else None
        )
        raw_label = (
            house.category_index_to_raw_label.get(category_index)
            if category_index is not None
            else None
        )
        canonical_label = label or "unknown_object"
        effective_raw_label = raw_label or "unknown_object"
        previous_raw_label = canonical_to_raw_label.setdefault(
            canonical_label, effective_raw_label
        )
        if previous_raw_label != effective_raw_label:
            raise SemanticSceneError(
                "MP3D raw semantic labels collapse to the same canonical "
                f"category {canonical_label!r}: "
                f"{previous_raw_label!r}, {effective_raw_label!r}"
            )
        object_label[object_id] = canonical_label
    labels = sorted(set(object_label.values()))
    label_to_code = {label: index for index, label in enumerate(labels)}
    face_codes = np.empty(len(face_object_ids), dtype=np.int32)
    for object_id, label in object_label.items():
        face_codes[face_object_ids == object_id] = label_to_code[label]

    grouped_vertices: list[np.ndarray] = []
    grouped_triangles: list[np.ndarray] = []
    objects: list[dict[str, Any]] = []
    category_triangle_counts: dict[str, int] = {}
    vertex_offset = 0
    triangle_offset = 0
    for group_index, label in enumerate(labels):
        category_triangles = source_triangles[face_codes == group_index]
        if not len(category_triangles):
            continue
        source_vertex_indices, inverse = np.unique(
            category_triangles.reshape(-1), return_inverse=True
        )
        local_vertices = np.ascontiguousarray(
            source_vertices[source_vertex_indices], dtype="<f4"
        )
        local_triangles = np.ascontiguousarray(
            inverse.reshape(-1, 3) + vertex_offset, dtype="<u4"
        )
        grouped_vertices.append(local_vertices)
        grouped_triangles.append(local_triangles)
        triangle_count = int(len(local_triangles))
        category_triangle_counts[label] = triangle_count
        objects.append(
            {
                "object_id": f"semantic_category_{group_index:03d}_{label}",
                "source_node_index": group_index,
                "source_mesh_index": 0,
                "source_primitive_index": group_index,
                "source_material_name": label,
                "vertex_offset": vertex_offset,
                "vertex_count": int(len(local_vertices)),
                "triangle_offset": triangle_offset,
                "triangle_count": triangle_count,
                "world_from_object": list(_IDENTITY_MATRIX),
                "source_world_matrix": list(_IDENTITY_MATRIX),
                "transform_baked": True,
            }
        )
        vertex_offset += len(local_vertices)
        triangle_offset += triangle_count
    if not grouped_vertices or not grouped_triangles:
        raise SemanticSceneError("semantic PLY contains no usable triangles")
    vertices = np.ascontiguousarray(np.concatenate(grouped_vertices), dtype="<f4")
    triangles = np.ascontiguousarray(np.concatenate(grouped_triangles), dtype="<u4")
    return ExpandedSemanticScene(
        vertices=vertices,
        triangles=triangles,
        objects=tuple(objects),
        source_primitive_count=len(objects),
        source_node_instance_count=len(objects),
        source_sha256=hashlib.sha256(ply_payload).hexdigest(),
        source_byte_size=len(ply_payload),
        descriptor_sha256=house.source_sha256,
        descriptor_byte_size=house.source_byte_size,
        source_vertex_count=len(source_vertices),
        source_triangle_count=len(source_triangles),
        semantic_categories=tuple(item["source_material_name"] for item in objects),
        category_triangle_counts=category_triangle_counts,
        raw_semantic_category_labels=tuple(
            canonical_to_raw_label[item["source_material_name"]]
            for item in objects
        ),
    )
