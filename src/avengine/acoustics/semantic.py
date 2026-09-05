"""Semantic surface extraction for research acoustic-scene compilation.

Two sources are supported and they carry semantics in completely different
places.  Matterport3D pairs a binary semantic PLY with a ``.house``
descriptor, and per-face identity is an explicit ``object_id`` column.  HM3D
ships a ``.semantic.glb`` whose only machine-readable identity is painted into
COLOR_0, keyed by a ``.semantic.txt`` listing one sRGB hex colour per instance.

Both parsers deliberately accept only the fixed encoding each dataset actually
ships: accepting an unknown layout would make per-face semantic identity
unauditable, and a semantic mesh that silently mislabels its faces produces an
acoustic scene that looks healthy and is wrong everywhere.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np

from avengine.acoustics.gltf import (
    GltfError,
    extract_triangle_scene_document,
    load_glb_bytes,
    triangle_vertex_colours,
)


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
class Hm3dSemanticAnnotations:
    colour_to_category: dict[tuple[int, int, int], str]
    colour_to_raw_category: dict[tuple[int, int, int], str]
    colour_to_instance_id: dict[tuple[int, int, int], int]
    source_sha256: str
    source_byte_size: int
    defects: tuple[str, ...] = ()


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
    # Lines the annotation source itself got wrong, carried forward rather than
    # dropped: a parser that tolerates corruption silently is how corruption
    # stops being visible.
    source_defects: tuple[str, ...] = ()


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


def _group_triangles_by_category(
    source_vertices: np.ndarray,
    source_triangles: np.ndarray,
    face_codes: np.ndarray,
    labels: list[str],
    *,
    empty_message: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Regroup triangles into one contiguous range per semantic category.

    The acoustic package addresses materials by triangle range, so every
    category has to occupy one unbroken span and each span needs its own
    vertex block.  Both semantic sources share this step, and they share it in
    code rather than in parallel copies: the object records land verbatim in
    the compiled manifest, so two implementations drifting apart would show up
    as an unexplained package difference between datasets rather than as a bug.
    """

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
        raise SemanticSceneError(empty_message)
    vertices = np.ascontiguousarray(np.concatenate(grouped_vertices), dtype="<f4")
    triangles = np.ascontiguousarray(np.concatenate(grouped_triangles), dtype="<u4")
    return vertices, triangles, objects, category_triangle_counts


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

    (
        vertices,
        triangles,
        objects,
        category_triangle_counts,
    ) = _group_triangles_by_category(
        source_vertices,
        source_triangles,
        face_codes,
        labels,
        empty_message="semantic PLY contains no usable triangles",
    )
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


_HM3D_ANNOTATION_HEADER = "HM3D Semantic Annotations"
_HM3D_UNPAINTED_COLOUR = (0, 0, 0)
HM3D_UNANNOTATED_CATEGORY = "unannotated"


def parse_hm3d_annotation_bytes(
    payload: bytes, *, max_defect_fraction: float = 0.01
) -> Hm3dSemanticAnnotations:
    """Parse HM3D's ``.semantic.txt`` colour-to-category listing.

    One line per annotated instance: ``id,RRGGBB,"category",region``.  The
    colour is the key the mesh is painted with, so a colour appearing twice
    under two categories makes every face carrying it ambiguous; that is a
    corrupt annotation rather than something to resolve by preference, and it
    still raises.

    Individual malformed lines do not.  HM3D's released annotations contain
    typos - across the 145 annotated train scenes exactly one line is broken,
    ``474,c,"radiator",11`` in 00546-nS8T59Aw3sf, whose colour field is a
    single character - and rejecting a scene of a thousand instances over one
    upstream typo trades a whole room for a radiator.  Such a line is skipped
    and recorded: its faces then match no colour and fall out as unannotated,
    which is already handled and still reflects sound.  Above
    ``max_defect_fraction`` of the lines the file is broken rather than
    typo-ridden, and that does raise.
    """

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SemanticSceneError(
            f"HM3D semantic annotations are not UTF-8: {exc}"
        ) from exc
    lines = text.splitlines()
    if not lines or lines[0].strip() != _HM3D_ANNOTATION_HEADER:
        raise SemanticSceneError(
            "HM3D semantic annotations must open with "
            f"{_HM3D_ANNOTATION_HEADER!r}"
        )
    colour_to_category: dict[tuple[int, int, int], str] = {}
    colour_to_raw_category: dict[tuple[int, int, int], str] = {}
    colour_to_instance_id: dict[tuple[int, int, int], int] = {}
    defects: list[str] = []
    considered = 0
    # csv rather than str.split, because a quoted category is allowed to hold a
    # comma and splitting on the raw line would cut the name in half.
    for number, row in enumerate(csv.reader(lines[1:]), start=2):
        if not row or not any(field.strip() for field in row):
            continue
        considered += 1
        if len(row) != 4:
            defects.append(
                f"line {number}: {len(row)} fields, expected 4"
            )
            continue
        raw_id, raw_colour, raw_category, _region = (field.strip() for field in row)
        if not raw_id.isdigit():
            defects.append(f"line {number}: non-numeric instance id {raw_id!r}")
            continue
        colour_text = raw_colour.upper()
        if len(colour_text) != 6 or any(
            character not in "0123456789ABCDEF" for character in colour_text
        ):
            defects.append(
                f"line {number}: colour {raw_colour!r} is not six hex digits"
            )
            continue
        key = (
            int(colour_text[0:2], 16),
            int(colour_text[2:4], 16),
            int(colour_text[4:6], 16),
        )
        category = _semantic_label(raw_category)
        previous = colour_to_category.get(key)
        if previous is not None and previous != category:
            raise SemanticSceneError(
                f"HM3D annotation colour {colour_text} names two categories: "
                f"{previous!r}, {category!r}"
            )
        colour_to_category[key] = category
        colour_to_raw_category[key] = raw_category
        colour_to_instance_id[key] = int(raw_id)
    if not colour_to_category:
        raise SemanticSceneError("HM3D semantic annotations list no instances")
    if defects and considered and len(defects) > max_defect_fraction * considered:
        raise SemanticSceneError(
            f"{len(defects)} of {considered} HM3D annotation lines are malformed, "
            f"above the {max_defect_fraction:.1%} a typo would explain: "
            + "; ".join(defects[:5])
        )
    return Hm3dSemanticAnnotations(
        colour_to_category=colour_to_category,
        colour_to_raw_category=colour_to_raw_category,
        colour_to_instance_id=colour_to_instance_id,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_byte_size=len(payload),
        defects=tuple(defects),
    )


def _linear_to_srgb_bytes(linear: np.ndarray) -> np.ndarray:
    """Convert linear COLOR_0 values to the sRGB bytes the annotation lists.

    This transfer curve is the entire difference between a working lookup and a
    scene reporting every single face unannotated, and the failure is silent:
    hashing the linear bytes matched zero of 395018 faces in the first scene
    this was tried on.  glTF stores COLOR_0 linear; HM3D writes its keys as
    sRGB hex.
    """

    clipped = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.rint(encoded * 255.0).astype(np.int64)


def _match_hm3d_colours(
    colour_bytes: np.ndarray,
    annotations: Hm3dSemanticAnnotations,
    *,
    colour_tolerance: int,
) -> tuple[list[str], list[str], int]:
    """Resolve per-face sRGB bytes to categories, tolerating rounding drift.

    Two rules decide the awkward cases.  Pure black is HM3D's unpainted
    sentinel and is resolved before any tolerance search, because a
    near-black annotation colour sitting within tolerance of it would otherwise
    capture every unpainted face in the room.  And an unmatched colour is
    labelled rather than dropped: unannotated geometry is still a surface sound
    reflects off, and deleting it punches holes that leak rays out of the room.
    """

    if colour_tolerance < 0:
        raise SemanticSceneError("colour_tolerance must not be negative")
    palette = np.array(list(annotations.colour_to_category), dtype=np.int64)
    palette_categories = list(annotations.colour_to_category.values())
    palette_raw = list(annotations.colour_to_raw_category.values())
    unique, inverse = np.unique(colour_bytes, axis=0, return_inverse=True)
    resolved_category: list[str] = []
    resolved_raw: list[str] = []
    unmatched_unique = 0
    for row in unique:
        key = (int(row[0]), int(row[1]), int(row[2]))
        if key == _HM3D_UNPAINTED_COLOUR and key not in annotations.colour_to_category:
            resolved_category.append(HM3D_UNANNOTATED_CATEGORY)
            resolved_raw.append(HM3D_UNANNOTATED_CATEGORY)
            continue
        exact = annotations.colour_to_category.get(key)
        if exact is not None:
            resolved_category.append(exact)
            resolved_raw.append(annotations.colour_to_raw_category[key])
            continue
        distance = np.abs(palette - np.array(key, dtype=np.int64)).max(axis=1)
        nearest = int(np.argmin(distance))
        if int(distance[nearest]) <= colour_tolerance:
            resolved_category.append(palette_categories[nearest])
            resolved_raw.append(palette_raw[nearest])
            continue
        unmatched_unique += 1
        resolved_category.append(HM3D_UNANNOTATED_CATEGORY)
        resolved_raw.append(HM3D_UNANNOTATED_CATEGORY)
    face_categories = [resolved_category[index] for index in inverse.reshape(-1)]
    face_raw = [resolved_raw[index] for index in inverse.reshape(-1)]
    return face_categories, face_raw, unmatched_unique


def load_hm3d_semantic_scene(
    semantic_glb: str | Path,
    annotation_text: str | Path,
    *,
    colour_tolerance: int = 2,
) -> ExpandedSemanticScene:
    """Load and group HM3D semantic triangles into auditable material ranges.

    The result is shaped exactly like the MP3D one, so the compiler downstream
    does not learn a second geometry contract.  What differs is where identity
    comes from: HM3D's semantic GLB is the render mesh repainted, and the paint
    is the annotation.  Measured on 00800-TEEsavR23oF the two meshes agree to
    the last digit on all six raw bounding-box coordinates and carry the same
    395018 triangles.  The raw GLB remains in the dataset source frame;
    Habitat's dataset configuration applies the reviewed Z-up to Y-up
    canonicalization at runtime, and the HM3D compiler records that same
    transform in its package.
    """

    glb_path = Path(semantic_glb).resolve()
    text_path = Path(annotation_text).resolve()
    glb_payload = glb_path.read_bytes()
    text_payload = text_path.read_bytes()
    annotations = parse_hm3d_annotation_bytes(text_payload)
    try:
        document = load_glb_bytes(glb_payload, source_path=str(glb_path))
        expanded = extract_triangle_scene_document(document)
        linear_colours, mixed_triangle_count = triangle_vertex_colours(
            document, expanded
        )
    except GltfError as exc:
        raise SemanticSceneError(f"unable to read {glb_path.name}: {exc}") from exc
    colour_bytes = _linear_to_srgb_bytes(linear_colours)
    face_categories, face_raw_categories, _unmatched = _match_hm3d_colours(
        colour_bytes, annotations, colour_tolerance=colour_tolerance
    )

    raw_by_canonical: dict[str, set[str]] = {}
    for canonical, raw in zip(face_categories, face_raw_categories):
        raw_by_canonical.setdefault(canonical, set()).add(raw)
    labels = sorted(set(face_categories))
    label_to_code = {label: index for index, label in enumerate(labels)}
    face_codes = np.fromiter(
        (label_to_code[label] for label in face_categories),
        dtype=np.int32,
        count=len(face_categories),
    )
    (
        vertices,
        triangles,
        objects,
        category_triangle_counts,
    ) = _group_triangles_by_category(
        expanded.vertices,
        expanded.triangles,
        face_codes,
        labels,
        empty_message="HM3D semantic GLB contains no usable triangles",
    )
    return ExpandedSemanticScene(
        vertices=vertices,
        triangles=triangles,
        objects=tuple(objects),
        source_primitive_count=expanded.source_primitive_count,
        source_node_instance_count=expanded.source_node_instance_count,
        source_sha256=hashlib.sha256(glb_payload).hexdigest(),
        source_byte_size=len(glb_payload),
        descriptor_sha256=annotations.source_sha256,
        descriptor_byte_size=annotations.source_byte_size,
        source_vertex_count=int(len(expanded.vertices)),
        source_triangle_count=int(len(expanded.triangles)),
        semantic_categories=tuple(item["source_material_name"] for item in objects),
        category_triangle_counts=category_triangle_counts,
        # Several HM3D spellings collapse onto one canonical category - "TV"
        # and "tv" are the same surface acoustically - so the raw side records
        # every spelling that landed here rather than one arbitrary winner.
        raw_semantic_category_labels=tuple(
            "|".join(sorted(raw_by_canonical[item["source_material_name"]]))
            for item in objects
        ),
        source_defects=annotations.defects,
    )
