"""Prepare ReplicaCAD GLBs for deterministic UE StaticMesh spawning.

ReplicaCAD stores important scale/rotation/translation on glTF nodes.  UE's
asset-only Interchange import does not provide a stable runtime actor hierarchy
contract, so importing the meshes and then spawning every StaticMesh at an
identity local transform can silently scramble the room.  This module bakes
each mesh node's complete world transform into POSITION/NORMAL/TANGENT data and
resets the glTF node transforms to identity.  Materials, textures, UVs, indices,
primitive topology, and mesh count are retained.

The operation is deliberately offline and standard-library only.  It does not
import Unreal, SPEAR, Blender, or Habitat-Sim.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, MutableMapping, Sequence


PREPARED_GLB_SCHEMA = "avengine_optional_spear_replicacad_prepared_glbs_v1"
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT32 = 5126


class ReplicaCADGLBError(ValueError):
    """A source GLB cannot be normalized without changing scene semantics."""


Matrix4 = tuple[tuple[float, float, float, float], ...]
Matrix3 = tuple[tuple[float, float, float], ...]


def _identity4() -> Matrix4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mul4(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(4))
            for column in range(4)
        )
        for row in range(4)
    )


def _node_matrix(node: Mapping[str, Any]) -> Matrix4:
    if "matrix" in node:
        value = node["matrix"]
        if not isinstance(value, list) or len(value) != 16:
            raise ReplicaCADGLBError("glTF node matrix must contain 16 numbers")
        numbers = [float(item) for item in value]
        if not all(math.isfinite(item) for item in numbers):
            raise ReplicaCADGLBError("glTF node matrix contains a non-finite value")
        # glTF serializes matrices column-major.
        return tuple(
            tuple(numbers[column * 4 + row] for column in range(4))
            for row in range(4)
        )

    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    if not isinstance(translation, list) or len(translation) != 3:
        raise ReplicaCADGLBError("glTF node translation must contain 3 numbers")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise ReplicaCADGLBError("glTF node rotation must contain 4 numbers")
    if not isinstance(scale, list) or len(scale) != 3:
        raise ReplicaCADGLBError("glTF node scale must contain 3 numbers")
    tx, ty, tz = (float(item) for item in translation)
    x, y, z, w = (float(item) for item in rotation)
    sx, sy, sz = (float(item) for item in scale)
    values = (tx, ty, tz, x, y, z, w, sx, sy, sz)
    if not all(math.isfinite(item) for item in values):
        raise ReplicaCADGLBError("glTF node TRS contains a non-finite value")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12 or min(abs(sx), abs(sy), abs(sz)) <= 1.0e-12:
        raise ReplicaCADGLBError("glTF node rotation/scale is singular")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    # T * R * S; each scale multiplies one rotation column.
    return (
        ((1 - 2 * (yy + zz)) * sx, 2 * (xy - wz) * sy, 2 * (xz + wy) * sz, tx),
        (2 * (xy + wz) * sx, (1 - 2 * (xx + zz)) * sy, 2 * (yz - wx) * sz, ty),
        (2 * (xz - wy) * sx, 2 * (yz + wx) * sy, (1 - 2 * (xx + yy)) * sz, tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _node_world_matrices(document: Mapping[str, Any]) -> list[Matrix4]:
    nodes = document.get("nodes", [])
    if not isinstance(nodes, list):
        raise ReplicaCADGLBError("glTF nodes must be an array")
    parents: list[int | None] = [None] * len(nodes)
    for parent_index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise ReplicaCADGLBError("glTF node must be an object")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ReplicaCADGLBError("glTF node children must be an array")
        for child_value in children:
            child = int(child_value)
            if not 0 <= child < len(nodes) or parents[child] is not None:
                raise ReplicaCADGLBError("glTF node graph is invalid or multiply parented")
            parents[child] = parent_index

    cache: list[Matrix4 | None] = [None] * len(nodes)
    visiting: set[int] = set()

    def world(index: int) -> Matrix4:
        cached = cache[index]
        if cached is not None:
            return cached
        if index in visiting:
            raise ReplicaCADGLBError("glTF node graph contains a cycle")
        visiting.add(index)
        local = _node_matrix(nodes[index])
        parent = parents[index]
        value = local if parent is None else _mul4(world(parent), local)
        visiting.remove(index)
        cache[index] = value
        return value

    return [world(index) for index in range(len(nodes))]


def _linear3(matrix: Matrix4) -> Matrix3:
    return tuple(tuple(matrix[row][column] for column in range(3)) for row in range(3))


def _det3(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _normal_matrix(matrix: Matrix4) -> tuple[Matrix3, float]:
    linear = _linear3(matrix)
    determinant = _det3(linear)
    if abs(determinant) <= 1.0e-15:
        raise ReplicaCADGLBError("glTF mesh node transform is singular")
    a, b, c = linear
    inverse: Matrix3 = (
        (
            (b[1] * c[2] - b[2] * c[1]) / determinant,
            (a[2] * c[1] - a[1] * c[2]) / determinant,
            (a[1] * b[2] - a[2] * b[1]) / determinant,
        ),
        (
            (b[2] * c[0] - b[0] * c[2]) / determinant,
            (a[0] * c[2] - a[2] * c[0]) / determinant,
            (a[2] * b[0] - a[0] * b[2]) / determinant,
        ),
        (
            (b[0] * c[1] - b[1] * c[0]) / determinant,
            (a[1] * c[0] - a[0] * c[1]) / determinant,
            (a[0] * b[1] - a[1] * b[0]) / determinant,
        ),
    )
    transpose = tuple(
        tuple(inverse[column][row] for column in range(3)) for row in range(3)
    )
    return transpose, determinant


def _transform_point(matrix: Matrix4, value: Sequence[float]) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * float(value[column]) for column in range(3))
        + matrix[row][3]
        for row in range(3)
    )


def _transform_direction(matrix: Matrix3, value: Sequence[float]) -> tuple[float, float, float]:
    result = tuple(
        sum(matrix[row][column] * float(value[column]) for column in range(3))
        for row in range(3)
    )
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= 1.0e-15:
        raise ReplicaCADGLBError("normal/tangent transformed to a zero vector")
    return tuple(item / norm for item in result)


def _load_glb(path: Path) -> tuple[dict[str, Any], bytearray]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReplicaCADGLBError(f"cannot read GLB: {path}") from exc
    if len(payload) < 28:
        raise ReplicaCADGLBError(f"GLB is truncated: {path}")
    magic, version, declared = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared != len(payload):
        raise ReplicaCADGLBError(f"expected complete GLB 2.0: {path}")
    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ReplicaCADGLBError(f"truncated GLB chunk header: {path}")
        length, kind = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + length
        if end > len(payload):
            raise ReplicaCADGLBError(f"truncated GLB chunk: {path}")
        chunks.append((kind, payload[offset:end]))
        offset = end
    if [kind for kind, _ in chunks] != [JSON_CHUNK, BIN_CHUNK]:
        raise ReplicaCADGLBError("expected one JSON chunk followed by one BIN chunk")
    try:
        document = json.loads(chunks[0][1].rstrip(b" \t\r\n\x00").decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplicaCADGLBError(f"invalid GLB JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ReplicaCADGLBError("GLB JSON root must be an object")
    buffers = document.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1 or buffers[0].get("uri"):
        raise ReplicaCADGLBError("only one embedded GLB buffer is supported")
    declared_binary = int(buffers[0].get("byteLength", -1))
    if declared_binary < 0 or not declared_binary <= len(chunks[1][1]) < declared_binary + 4:
        raise ReplicaCADGLBError("GLB embedded buffer length is invalid")
    return document, bytearray(chunks[1][1])


def _build_glb(document: Mapping[str, Any], binary: bytes) -> bytes:
    encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    padded = bytes(binary) + b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(padded)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<II", len(encoded), JSON_CHUNK),
            encoded,
            struct.pack("<II", len(padded), BIN_CHUNK),
            padded,
        )
    )


def _accessor_layout(
    document: Mapping[str, Any], accessor_index: int, expected_type: str
) -> tuple[MutableMapping[str, Any], int, int, int]:
    accessors = document.get("accessors")
    views = document.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise ReplicaCADGLBError("GLB lacks accessor/bufferView arrays")
    if not 0 <= accessor_index < len(accessors):
        raise ReplicaCADGLBError("primitive references an invalid accessor")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, MutableMapping):
        raise ReplicaCADGLBError("GLB accessor must be an object")
    if (
        accessor.get("componentType") != FLOAT32
        or accessor.get("type") != expected_type
        or accessor.get("normalized", False)
        or "sparse" in accessor
    ):
        raise ReplicaCADGLBError(
            f"{expected_type} geometry accessor must be dense float32"
        )
    view_index = accessor.get("bufferView")
    if isinstance(view_index, bool) or not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise ReplicaCADGLBError("geometry accessor has no valid bufferView")
    view = views[view_index]
    if not isinstance(view, Mapping) or int(view.get("buffer", 0)) != 0:
        raise ReplicaCADGLBError("geometry accessor must use embedded buffer 0")
    components = 4 if expected_type == "VEC4" else 3
    element_size = 4 * components
    stride = int(view.get("byteStride", element_size))
    if stride < element_size or stride % 4:
        raise ReplicaCADGLBError("geometry accessor byteStride is invalid")
    count = int(accessor.get("count", -1))
    if count <= 0:
        raise ReplicaCADGLBError("geometry accessor must be non-empty")
    first = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    view_end = int(view.get("byteOffset", 0)) + int(view.get("byteLength", -1))
    if first < 0 or first + (count - 1) * stride + element_size > view_end:
        raise ReplicaCADGLBError("geometry accessor exceeds its bufferView")
    return accessor, first, stride, count


def normalize_replicacad_glb(source: Path, destination: Path) -> dict[str, Any]:
    """Bake every glTF mesh-node world transform and write an identity-node GLB."""

    source = source.resolve()
    destination = destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace prepared GLB: {destination}")
    document, binary = _load_glb(source)
    if document.get("skins") or document.get("animations"):
        raise ReplicaCADGLBError("ReplicaCAD visual GLB must be static")
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    if not isinstance(nodes, list) or not isinstance(meshes, list) or not meshes:
        raise ReplicaCADGLBError("ReplicaCAD GLB has no meshes")
    worlds = _node_world_matrices(document)
    node_for_mesh: dict[int, int] = {}
    for node_index, node in enumerate(nodes):
        if not isinstance(node, Mapping) or "mesh" not in node:
            continue
        mesh_index = int(node["mesh"])
        if not 0 <= mesh_index < len(meshes) or mesh_index in node_for_mesh:
            raise ReplicaCADGLBError("each GLB mesh must have exactly one mesh node")
        node_for_mesh[mesh_index] = node_index
    if set(node_for_mesh) != set(range(len(meshes))):
        raise ReplicaCADGLBError("every GLB mesh must have exactly one mesh node")

    transforms: dict[tuple[int, str], Matrix4] = {}
    for mesh_index, mesh in enumerate(meshes):
        if not isinstance(mesh, Mapping):
            raise ReplicaCADGLBError("glTF mesh must be an object")
        matrix = worlds[node_for_mesh[mesh_index]]
        if _det3(_linear3(matrix)) <= 0.0:
            # A reflection also reverses triangle winding.  This preparer does
            # not mutate primitive indices, so fail closed instead of emitting
            # inside-out meshes with broken back-face culling/shadows.
            raise ReplicaCADGLBError(
                "negative-determinant mesh transform requires index winding reversal"
            )
        primitives = mesh.get("primitives", [])
        if not isinstance(primitives, list) or not primitives:
            raise ReplicaCADGLBError("glTF mesh has no primitives")
        for primitive in primitives:
            if not isinstance(primitive, Mapping) or primitive.get("targets"):
                raise ReplicaCADGLBError("morph targets are not supported")
            attributes = primitive.get("attributes")
            if not isinstance(attributes, Mapping) or "POSITION" not in attributes:
                raise ReplicaCADGLBError("mesh primitive lacks POSITION")
            for semantic in ("POSITION", "NORMAL", "TANGENT"):
                if semantic not in attributes:
                    continue
                accessor_index = int(attributes[semantic])
                key = (accessor_index, semantic)
                previous = transforms.setdefault(key, matrix)
                if previous != matrix:
                    raise ReplicaCADGLBError(
                        "one geometry accessor is shared by differently transformed nodes"
                    )

    transformed_accessors: set[int] = set()
    position_minimum = [math.inf, math.inf, math.inf]
    position_maximum = [-math.inf, -math.inf, -math.inf]
    for (accessor_index, semantic), matrix in transforms.items():
        if accessor_index in transformed_accessors:
            raise ReplicaCADGLBError(
                "one accessor is reused for incompatible geometry semantics"
            )
        transformed_accessors.add(accessor_index)
        expected_type = "VEC4" if semantic == "TANGENT" else "VEC3"
        accessor, first, stride, count = _accessor_layout(
            document, accessor_index, expected_type
        )
        normal_matrix, determinant = _normal_matrix(matrix)
        local_minimum = [math.inf, math.inf, math.inf]
        local_maximum = [-math.inf, -math.inf, -math.inf]
        for element in range(count):
            offset = first + element * stride
            values = struct.unpack_from("<4f" if expected_type == "VEC4" else "<3f", binary, offset)
            if not all(math.isfinite(item) for item in values):
                raise ReplicaCADGLBError("geometry accessor contains non-finite data")
            if semantic == "POSITION":
                result = _transform_point(matrix, values)
                struct.pack_into("<3f", binary, offset, *result)
                for axis in range(3):
                    local_minimum[axis] = min(local_minimum[axis], result[axis])
                    local_maximum[axis] = max(local_maximum[axis], result[axis])
                    position_minimum[axis] = min(position_minimum[axis], result[axis])
                    position_maximum[axis] = max(position_maximum[axis], result[axis])
            else:
                direction = _transform_direction(normal_matrix, values[:3])
                if semantic == "TANGENT":
                    handedness = float(values[3]) * (-1.0 if determinant < 0.0 else 1.0)
                    struct.pack_into("<4f", binary, offset, *direction, handedness)
                else:
                    struct.pack_into("<3f", binary, offset, *direction)
        if semantic == "POSITION":
            accessor["min"] = local_minimum
            accessor["max"] = local_maximum

    for node in nodes:
        if not isinstance(node, MutableMapping):
            raise ReplicaCADGLBError("glTF node must be mutable object")
        for key in ("matrix", "translation", "rotation", "scale"):
            node.pop(key, None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_build_glb(document, binary))

    # Re-open the result so an incomplete write cannot be reported as ready.
    reread_document, _ = _load_glb(destination)
    identity_worlds = _node_world_matrices(reread_document)
    if any(matrix != _identity4() for matrix in identity_worlds):
        raise ReplicaCADGLBError("prepared GLB nodes are not identity transforms")
    return {
        "status": "pass",
        "source_glb_path": str(source),
        "prepared_glb_path": str(destination),
        "mesh_count": len(meshes),
        "transformed_accessor_count": len(transformed_accessors),
        "position_bounds_gltf_m": {
            "minimum": position_minimum,
            "maximum": position_maximum,
        },
        "retained_semantics": [
            "mesh_count",
            "primitive_topology",
            "indices",
            "materials",
            "textures",
            "uvs",
        ],
        "baked_semantics": "complete glTF mesh-node world transforms",
    }


def prepare_replicacad_source_glbs(
    request: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Prepare all request GLBs and return a request ready for editor import."""

    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to replace prepared GLB directory: {output_dir}")
    output_dir.mkdir(parents=True)
    result = copy.deepcopy(dict(request))
    pbr = result.get("pbr_import")
    sources = pbr.get("source_meshes") if isinstance(pbr, MutableMapping) else None
    if not isinstance(sources, list) or not sources:
        raise ReplicaCADGLBError("execution request has no source GLBs")
    records = []
    for item in sources:
        if not isinstance(item, MutableMapping):
            raise ReplicaCADGLBError("execution request source GLB is invalid")
        mesh_id = item.get("mesh_source_id")
        if not isinstance(mesh_id, str) or not mesh_id:
            raise ReplicaCADGLBError("execution request source GLB has no ID")
        destination = output_dir / f"{mesh_id}.glb"
        evidence = normalize_replicacad_glb(
            Path(str(item.get("source_glb_path"))), destination
        )
        if evidence["mesh_count"] != item.get("source_inventory", {}).get("mesh_count"):
            raise ReplicaCADGLBError(f"prepared mesh count differs for {mesh_id}")
        item["editor_import_source_glb_path"] = str(destination)
        item["node_transform_policy"] = "baked_into_geometry_then_identity_nodes"
        records.append(evidence)
    result["glb_preparation"] = {
        "schema": PREPARED_GLB_SCHEMA,
        "status": "pass",
        "prepared_directory": str(output_dir),
        "source_glb_count": len(records),
        "prepared_mesh_asset_count": sum(item["mesh_count"] for item in records),
        "records": records,
    }
    return result


__all__ = [
    "PREPARED_GLB_SCHEMA",
    "ReplicaCADGLBError",
    "normalize_replicacad_glb",
    "prepare_replicacad_source_glbs",
]
