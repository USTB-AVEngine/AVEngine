"""Bounded, read-only mesh quality measurements.

The module measures geometry without changing it.  Connected components are
computed with a chunked union-find over the triangle index stream; no
vertices-by-vertices adjacency matrix is created.  A measurement has no
admission meaning by default.  An explicit, asset-category quality policy may
classify the measured result as ``pass`` or ``review_required``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np


_GLTF_TRIANGLES = 4
_COMPONENT_DTYPES = {
    5121: np.dtype("<u1"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_POLICY_SCHEMA = "avengine_mesh_quality_policy_v1"
_POLICY_LIMITS = {
    "max_tiny_faces",
    "max_small_component_count",
    "max_small_component_faces",
    "min_largest_component_fraction",
    "require_support_plane",
}


class MeshQualityError(ValueError):
    """The mesh or an explicit quality policy cannot be measured."""


@dataclass(frozen=True)
class MeshGeometry:
    """Decoded triangle geometry with all declared node transforms applied."""

    vertices: np.ndarray
    faces: np.ndarray
    primitive_count: int


@dataclass(frozen=True)
class _GlbPayload:
    document: dict[str, Any]
    binary: bytes


def _read_glb(path: Path) -> _GlbPayload:
    """Read the GLB container while ignoring unrelated glTF extensions.

    Geometry inspection does not need to decode texture extensions such as
    ``EXT_texture_webp``.  The parser still requires a valid GLB 2 container,
    one JSON chunk, at most one BIN chunk, and one embedded buffer.
    """

    try:
        data = path.read_bytes()
    except OSError as error:
        raise MeshQualityError(f"unable to read GLB {path}: {error}") from error
    if len(data) < 20:
        raise MeshQualityError("GLB is too short to contain a header and JSON chunk")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise MeshQualityError("unsupported GLB header; expected glTF version 2")
    if declared_length != len(data) or declared_length % 4:
        raise MeshQualityError("GLB declared length does not match its bytes")
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < declared_length:
        if offset + 8 > declared_length:
            raise MeshQualityError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + chunk_length
        if chunk_length % 4 or end > declared_length:
            raise MeshQualityError("invalid GLB chunk length")
        chunks.append((chunk_type, data[offset:end]))
        offset = end
    if not chunks or chunks[0][0] != 0x4E4F534A or len(chunks) > 2:
        raise MeshQualityError("GLB must contain one JSON chunk and at most one BIN chunk")
    if len(chunks) == 2 and chunks[1][0] != 0x004E4942:
        raise MeshQualityError("the optional second GLB chunk must be BIN")
    try:
        document = json.loads(chunks[0][1].rstrip(b" \t\r\n\x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeshQualityError(f"invalid GLB JSON chunk: {error}") from error
    if not isinstance(document, dict):
        raise MeshQualityError("GLB JSON root must be an object")
    buffers = document.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1 or not isinstance(buffers[0], dict):
        raise MeshQualityError("mesh quality requires exactly one embedded buffer")
    if "uri" in buffers[0]:
        raise MeshQualityError("external and data-URI buffers are unsupported")
    binary = chunks[1][1] if len(chunks) == 2 else b""
    declared_buffer_length = buffers[0].get("byteLength")
    if not isinstance(declared_buffer_length, int) or declared_buffer_length < 0:
        raise MeshQualityError("buffers[0].byteLength is invalid")
    if declared_buffer_length > len(binary):
        raise MeshQualityError("buffers[0] extends beyond the embedded BIN chunk")
    return _GlbPayload(document=document, binary=binary)


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeshQualityError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MeshQualityError(f"{name} must be a finite number")
    return number


def _vector(value: Any, *, length: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise MeshQualityError(f"{name} must contain {length} numbers")
    result = np.asarray(
        [_finite_number(item, name=name) for item in value], dtype=np.float64
    )
    return result


def _node_matrix(node: Mapping[str, Any], *, node_index: int) -> np.ndarray:
    if "matrix" in node:
        matrix = _vector(node["matrix"], length=16, name=f"nodes[{node_index}].matrix")
        return matrix.reshape(4, 4).T

    translation = _vector(
        node.get("translation", [0.0, 0.0, 0.0]),
        length=3,
        name=f"nodes[{node_index}].translation",
    )
    rotation = _vector(
        node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
        length=4,
        name=f"nodes[{node_index}].rotation",
    )
    scale = _vector(
        node.get("scale", [1.0, 1.0, 1.0]),
        length=3,
        name=f"nodes[{node_index}].scale",
    )
    norm = float(np.linalg.norm(rotation))
    if abs(norm - 1.0) > 1e-5:
        raise MeshQualityError(f"nodes[{node_index}].rotation must be a unit quaternion")
    x, y, z, w = rotation
    rotation_matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_matrix @ np.diag(scale)
    result[:3, 3] = translation
    return result


def _decode_accessor(
    gltf: Mapping[str, Any],
    binary: bytes,
    accessor_index: int,
    *,
    expected_type: str,
    expected_components: tuple[int, ...],
) -> np.ndarray:
    accessors = gltf.get("accessors")
    views = gltf.get("bufferViews")
    buffers = gltf.get("buffers")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise MeshQualityError("GLB has no usable accessors or bufferViews")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise MeshQualityError("mesh quality supports exactly one embedded GLB buffer")
    if not isinstance(accessor_index, int) or not 0 <= accessor_index < len(accessors):
        raise MeshQualityError(f"accessor index is out of range: {accessor_index!r}")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict):
        raise MeshQualityError(f"accessors[{accessor_index}] must be an object")
    if accessor.get("type") != expected_type:
        raise MeshQualityError(
            f"accessors[{accessor_index}] must be {expected_type}, "
            f"got {accessor.get('type')!r}"
        )
    if "sparse" in accessor or accessor.get("normalized", False):
        raise MeshQualityError(
            f"accessors[{accessor_index}] uses unsupported sparse/normalized storage"
        )
    component_type = accessor.get("componentType")
    dtype = _COMPONENT_DTYPES.get(component_type)
    if dtype is None or component_type not in expected_components:
        raise MeshQualityError(
            f"accessors[{accessor_index}] componentType {component_type!r} is unsupported"
        )
    count = accessor.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise MeshQualityError(f"accessors[{accessor_index}].count must be positive")
    view_index = accessor.get("bufferView")
    if isinstance(view_index, bool) or not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise MeshQualityError(f"accessors[{accessor_index}].bufferView is out of range")
    view = views[view_index]
    if not isinstance(view, dict) or view.get("buffer") != 0:
        raise MeshQualityError(f"bufferViews[{view_index}] must reference buffer 0")
    view_offset = view.get("byteOffset", 0)
    view_length = view.get("byteLength")
    accessor_offset = accessor.get("byteOffset", 0)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (view_offset, view_length, accessor_offset)
    ):
        raise MeshQualityError(f"accessors[{accessor_index}] has invalid byte offsets")
    if view_length <= 0:
        raise MeshQualityError(f"bufferViews[{view_index}].byteLength must be positive")
    declared_buffer_length = buffers[0].get("byteLength")
    if not isinstance(declared_buffer_length, int) or declared_buffer_length < 0:
        raise MeshQualityError("buffers[0].byteLength is invalid")
    if view_offset + view_length > declared_buffer_length or declared_buffer_length > len(binary):
        raise MeshQualityError(f"bufferViews[{view_index}] exceeds the embedded buffer")
    width = _TYPE_WIDTHS[expected_type]
    item_size = dtype.itemsize * width
    stride = view.get("byteStride", item_size)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < item_size:
        raise MeshQualityError(f"bufferViews[{view_index}].byteStride is invalid")
    required = accessor_offset + (count - 1) * stride + item_size
    if required > view_length:
        raise MeshQualityError(f"accessors[{accessor_index}] exceeds its bufferView")
    start = view_offset + accessor_offset
    if start % dtype.itemsize:
        raise MeshQualityError(f"accessors[{accessor_index}] is not aligned")
    if stride == item_size:
        values = np.frombuffer(
            binary,
            dtype=dtype,
            count=count * width,
            offset=start,
        ).reshape(count, width)
    else:
        values = np.ndarray(
            shape=(count, width),
            dtype=dtype,
            buffer=binary,
            offset=start,
            strides=(stride, dtype.itemsize),
        ).copy()
    return values


def load_glb_mesh(path: str | Path) -> MeshGeometry:
    """Decode triangle POSITION/indices accessors from one embedded GLB."""

    source = Path(path).resolve()
    document = _read_glb(source)
    gltf = document.document
    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])
    scenes = gltf.get("scenes", [])
    if not isinstance(nodes, list) or not isinstance(meshes, list):
        raise MeshQualityError("GLB nodes/meshes must be arrays")
    if scenes and isinstance(gltf.get("scene", 0), int):
        scene_index = gltf.get("scene", 0)
        if not 0 <= scene_index < len(scenes) or not isinstance(scenes[scene_index], dict):
            raise MeshQualityError("GLB default scene is invalid")
        roots = scenes[scene_index].get("nodes", [])
    else:
        roots = list(range(len(nodes)))
    if not isinstance(roots, list):
        raise MeshQualityError("GLB scene roots must be an array")

    vertex_chunks: list[np.ndarray] = []
    face_chunks: list[np.ndarray] = []
    primitive_count = 0
    vertex_count = 0
    active: set[int] = set()

    def walk(node_index: int, parent: np.ndarray) -> None:
        nonlocal primitive_count, vertex_count
        if not isinstance(node_index, int) or not 0 <= node_index < len(nodes):
            raise MeshQualityError(f"node index is out of range: {node_index!r}")
        if node_index in active:
            raise MeshQualityError("GLB node hierarchy contains a cycle")
        node = nodes[node_index]
        if not isinstance(node, dict):
            raise MeshQualityError(f"nodes[{node_index}] must be an object")
        active.add(node_index)
        world = parent @ _node_matrix(node, node_index=node_index)
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
                raise MeshQualityError(f"nodes[{node_index}].mesh is out of range")
            mesh = meshes[mesh_index]
            if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list):
                raise MeshQualityError(f"meshes[{mesh_index}].primitives must be an array")
            for primitive_index, primitive in enumerate(mesh["primitives"]):
                if not isinstance(primitive, dict):
                    raise MeshQualityError(f"meshes[{mesh_index}].primitives[{primitive_index}] must be an object")
                if primitive.get("mode", _GLTF_TRIANGLES) != _GLTF_TRIANGLES:
                    raise MeshQualityError("mesh quality supports TRIANGLES primitives only")
                attributes = primitive.get("attributes")
                if not isinstance(attributes, dict) or not isinstance(attributes.get("POSITION"), int):
                    raise MeshQualityError("triangle primitive has no POSITION accessor")
                positions = _decode_accessor(
                    gltf,
                    document.binary,
                    attributes["POSITION"],
                    expected_type="VEC3",
                    expected_components=(5126,),
                ).astype(np.float64, copy=False)
                if "indices" in primitive:
                    indices = _decode_accessor(
                        gltf,
                        document.binary,
                        primitive["indices"],
                        expected_type="SCALAR",
                        expected_components=(5121, 5123, 5125),
                    ).reshape(-1).astype(np.int64, copy=False)
                else:
                    indices = np.arange(len(positions), dtype=np.int64)
                if len(indices) == 0 or len(indices) % 3:
                    raise MeshQualityError("triangle primitive has an invalid index count")
                if int(indices.min()) < 0 or int(indices.max()) >= len(positions):
                    raise MeshQualityError("triangle primitive index exceeds POSITION accessor")
                transformed = (world[:3, :3] @ positions.T).T + world[:3, 3]
                vertex_chunks.append(transformed)
                face_chunks.append(indices.reshape(-1, 3) + vertex_count)
                vertex_count += len(positions)
                primitive_count += 1
        children = node.get("children", [])
        if not isinstance(children, list):
            raise MeshQualityError(f"nodes[{node_index}].children must be an array")
        for child in children:
            walk(child, world)
        active.remove(node_index)

    for root in roots:
        walk(root, np.eye(4, dtype=np.float64))
    if not vertex_chunks or not face_chunks or primitive_count == 0:
        raise MeshQualityError(f"GLB has no triangle mesh: {source}")
    return MeshGeometry(
        vertices=np.concatenate(vertex_chunks, axis=0),
        faces=np.concatenate(face_chunks, axis=0).astype(np.int32, copy=False),
        primitive_count=primitive_count,
    )


def _validate_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise MeshQualityError("vertices must be a non-empty [N, 3] array")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise MeshQualityError("faces must be a non-empty [M, 3] triangle array")
    if not np.isfinite(points).all():
        raise MeshQualityError("vertices contain non-finite coordinates")
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(points):
        raise MeshQualityError("faces contain an out-of-range vertex index")
    if int(triangles.max()) > np.iinfo(np.int32).max:
        raise MeshQualityError("face indices exceed the bounded int32 union-find domain")
    return points, triangles.astype(np.int32, copy=False)


def _find(parent: np.ndarray, value: int) -> int:
    root = value
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[value]) != value:
        next_value = int(parent[value])
        parent[value] = root
        value = next_value
    return root


def _union(parent: np.ndarray, rank: np.ndarray, left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if rank[left_root] < rank[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    if rank[left_root] == rank[right_root]:
        rank[left_root] = np.uint8(rank[left_root] + 1)


def _component_metrics(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tiny_area_threshold: float,
    small_component_max_faces: int,
    chunk_size: int,
) -> dict[str, Any]:
    parent = np.arange(len(points), dtype=np.int32)
    rank = np.zeros(len(points), dtype=np.uint8)
    tiny_faces = 0
    total_area = 0.0
    for start in range(0, len(triangles), chunk_size):
        chunk = triangles[start : start + chunk_size]
        tri_points = points[chunk]
        cross = np.cross(
            tri_points[:, 1] - tri_points[:, 0],
            tri_points[:, 2] - tri_points[:, 0],
        )
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        tiny_faces += int(np.count_nonzero(areas <= tiny_area_threshold))
        total_area += float(areas.sum())
        for left, middle, right in chunk:
            _union(parent, rank, int(left), int(middle))
            _union(parent, rank, int(middle), int(right))
            _union(parent, rank, int(right), int(left))

    roots = np.fromiter(
        (_find(parent, index) for index in range(len(points))),
        dtype=np.int32,
        count=len(points),
    )
    face_roots = roots[triangles[:, 0]]
    face_component_roots, face_inverse = np.unique(face_roots, return_inverse=True)
    component_count = len(face_component_roots)
    face_counts = np.bincount(face_inverse, minlength=component_count)
    root_to_component = np.full(len(points), -1, dtype=np.int32)
    root_to_component[face_component_roots] = np.arange(component_count, dtype=np.int32)
    used_vertices = root_to_component[roots] >= 0
    vertex_components = root_to_component[roots[used_vertices]]
    vertex_counts = np.bincount(vertex_components, minlength=component_count)
    component_min = np.full((component_count, 3), np.inf, dtype=np.float64)
    component_max = np.full((component_count, 3), -np.inf, dtype=np.float64)
    np.minimum.at(component_min, vertex_components, points[used_vertices])
    np.maximum.at(component_max, vertex_components, points[used_vertices])
    face_component_consistent = bool(
        np.all(
            (roots[triangles[:, 0]] == roots[triangles[:, 1]])
            & (roots[triangles[:, 1]] == roots[triangles[:, 2]])
        )
    )
    largest_index = int(np.argmax(face_counts))
    small_mask = face_counts <= small_component_max_faces
    largest_faces = int(face_counts[largest_index])
    largest_vertices = int(vertex_counts[largest_index])
    components = [
        {
            "vertices": int(vertex_counts[index]),
            "faces": int(face_counts[index]),
            "bounds_min": [float(value) for value in component_min[index]],
            "bounds_max": [float(value) for value in component_max[index]],
        }
        for index in np.argsort(face_counts)[::-1][: min(10, component_count)]
    ]
    return {
        "vertices": int(len(points)),
        "faces": int(len(triangles)),
        "finite_vertices": True,
        "valid_indices": True,
        "tiny_face_area_threshold": float(tiny_area_threshold),
        "tiny_face_count": int(tiny_faces),
        "total_surface_area": total_area,
        "connected_component_count": int(component_count),
        "small_component_max_faces": int(small_component_max_faces),
        "small_component_count": int(np.count_nonzero(small_mask)),
        "small_component_faces": int(face_counts[small_mask].sum()),
        "largest_component_vertices": largest_vertices,
        "largest_component_faces": largest_faces,
        "largest_component_fraction": largest_faces / len(triangles),
        "largest_components": components,
        "face_component_consistent": face_component_consistent,
        "bounds_min": [float(value) for value in points.min(axis=0)],
        "bounds_max": [float(value) for value in points.max(axis=0)],
        "chunk_size": int(chunk_size),
    }


def _validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if policy.get("schema") != _POLICY_SCHEMA:
        raise MeshQualityError(f"quality policy schema must be {_POLICY_SCHEMA!r}")
    category = policy.get("asset_category")
    if not isinstance(category, str) or not category.strip():
        raise MeshQualityError("quality policy asset_category must be non-empty")
    limits = policy.get("limits")
    if not isinstance(limits, dict):
        raise MeshQualityError("quality policy limits must be an object")
    unknown = sorted(set(limits) - _POLICY_LIMITS)
    if unknown:
        raise MeshQualityError(f"quality policy has unknown limits: {unknown}")
    if not limits:
        raise MeshQualityError("quality policy must declare at least one limit")
    normalized: dict[str, Any] = {
        "schema": _POLICY_SCHEMA,
        "asset_category": category,
        "limits": {},
    }
    for key, value in limits.items():
        if key in {"max_tiny_faces", "max_small_component_count", "max_small_component_faces"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MeshQualityError(f"quality policy {key} must be a non-negative integer")
            normalized["limits"][key] = value
        elif key == "min_largest_component_fraction":
            number = _finite_number(value, name=f"quality policy {key}")
            if not 0.0 <= number <= 1.0:
                raise MeshQualityError(f"quality policy {key} must be within [0, 1]")
            normalized["limits"][key] = number
        elif key == "require_support_plane":
            if not isinstance(value, bool):
                raise MeshQualityError("quality policy require_support_plane must be boolean")
            normalized["limits"][key] = value
    measurement = policy.get("measurement", {})
    if measurement is not None:
        if not isinstance(measurement, dict):
            raise MeshQualityError("quality policy measurement must be an object")
        normalized_measurement: dict[str, Any] = {}
        if "tiny_face_area_threshold" in measurement:
            threshold = _finite_number(
                measurement["tiny_face_area_threshold"],
                name="quality policy tiny_face_area_threshold",
            )
            if threshold < 0.0:
                raise MeshQualityError("quality policy tiny_face_area_threshold must be non-negative")
            normalized_measurement["tiny_face_area_threshold"] = threshold
        if "small_component_max_faces" in measurement:
            value = measurement["small_component_max_faces"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MeshQualityError("quality policy small_component_max_faces must be positive")
            normalized_measurement["small_component_max_faces"] = value
        unknown_measurement = sorted(
            set(measurement) - {"tiny_face_area_threshold", "small_component_max_faces"}
        )
        if unknown_measurement:
            raise MeshQualityError(f"quality policy has unknown measurement fields: {unknown_measurement}")
        normalized["measurement"] = normalized_measurement
    return normalized


def _classify(metrics: Mapping[str, Any], support_plane: Mapping[str, Any], policy: Mapping[str, Any] | None) -> tuple[str, dict[str, Any] | None, dict[str, bool]]:
    if policy is None:
        return "measured_unclassified", None, {}
    normalized = _validate_policy(policy)
    limits = normalized["limits"]
    checks: dict[str, bool] = {}
    if "max_tiny_faces" in limits:
        checks["max_tiny_faces"] = metrics["tiny_face_count"] <= limits["max_tiny_faces"]
    if "max_small_component_count" in limits:
        checks["max_small_component_count"] = metrics["small_component_count"] <= limits["max_small_component_count"]
    if "max_small_component_faces" in limits:
        checks["max_small_component_faces"] = metrics["small_component_faces"] <= limits["max_small_component_faces"]
    if "min_largest_component_fraction" in limits:
        checks["min_largest_component_fraction"] = metrics["largest_component_fraction"] >= limits["min_largest_component_fraction"]
    if "require_support_plane" in limits:
        checks["require_support_plane"] = support_plane["present"] == limits["require_support_plane"]
    return ("pass" if all(checks.values()) else "review_required"), normalized, checks


def measure_mesh_quality(
    vertices: Any,
    faces: Any,
    *,
    tiny_area_threshold: float = 1.0e-12,
    small_component_max_faces: int = 10,
    chunk_size: int = 100_000,
    support_plane_path: str | Path | None = None,
    quality_policy: Mapping[str, Any] | None = None,
    source_path: str | Path | None = None,
    primitive_count: int | None = None,
) -> dict[str, Any]:
    """Measure one triangle mesh and optionally classify it by explicit policy."""

    normalized_policy = _validate_policy(quality_policy) if quality_policy is not None else None
    measurement = normalized_policy.get("measurement", {}) if normalized_policy else {}
    if "tiny_face_area_threshold" in measurement:
        tiny_area_threshold = measurement["tiny_face_area_threshold"]
    if "small_component_max_faces" in measurement:
        small_component_max_faces = measurement["small_component_max_faces"]
    threshold = _finite_number(tiny_area_threshold, name="tiny_area_threshold")
    if threshold < 0.0:
        raise MeshQualityError("tiny_area_threshold must be non-negative")
    if isinstance(small_component_max_faces, bool) or not isinstance(small_component_max_faces, int) or small_component_max_faces < 1:
        raise MeshQualityError("small_component_max_faces must be positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise MeshQualityError("chunk_size must be positive")
    points, triangles = _validate_arrays(vertices, faces)
    support_plane = {
        "present": support_plane_path is not None,
        "path": str(Path(support_plane_path).resolve()) if support_plane_path is not None else None,
    }
    metrics = _component_metrics(
        points,
        triangles,
        tiny_area_threshold=threshold,
        small_component_max_faces=small_component_max_faces,
        chunk_size=chunk_size,
    )
    status, normalized_policy, checks = _classify(metrics, support_plane, normalized_policy)
    report: dict[str, Any] = {
        "schema": "avengine_mesh_quality_measurement_v1",
        "status": status,
        "input": {
            "path": str(Path(source_path).resolve()) if source_path is not None else None,
            "primitive_count": primitive_count,
        },
        "metrics": metrics,
        "support_plane": support_plane,
        "policy": normalized_policy,
        "policy_checks": checks,
        "mutation": {"input_modified": False, "components_deleted": False},
    }
    return report


def measure_glb(
    path: str | Path,
    *,
    tiny_area_threshold: float = 1.0e-12,
    small_component_max_faces: int = 10,
    chunk_size: int = 100_000,
    support_plane_path: str | Path | None = None,
    quality_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one GLB read-only and return bounded quality metrics."""

    source = Path(path).resolve()
    geometry = load_glb_mesh(source)
    return measure_mesh_quality(
        geometry.vertices,
        geometry.faces,
        tiny_area_threshold=tiny_area_threshold,
        small_component_max_faces=small_component_max_faces,
        chunk_size=chunk_size,
        support_plane_path=support_plane_path,
        quality_policy=quality_policy,
        source_path=source,
        primitive_count=geometry.primitive_count,
    )


def load_quality_policy(path: str | Path) -> dict[str, Any]:
    """Read and validate an explicit policy JSON without modifying it."""

    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeshQualityError(f"invalid quality policy {source}: {error}") from error
    if not isinstance(payload, dict):
        raise MeshQualityError("quality policy JSON must be an object")
    return _validate_policy(payload)
