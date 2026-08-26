"""Fail-closed GLB surface extraction for static acoustic-room compilation.

The extractor intentionally accepts only geometry encodings whose triangle and
material identities can be audited without a renderer.  Visual-only material
extensions may be present, but compressed geometry, sparse accessors, morph
targets, skins, non-triangle primitives, and external buffers are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np


_GLB_HEADER = struct.Struct("<4sII")
_CHUNK_HEADER = struct.Struct("<II")
_MAGIC = b"glTF"
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_COMPONENT_DTYPES = {
    5121: np.dtype("u1"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class GltfError(ValueError):
    pass


@dataclass(frozen=True)
class GlbDocument:
    document: dict[str, Any]
    binary: bytes
    sha256: str
    byte_size: int
    source_path: Path


@dataclass(frozen=True)
class ExpandedGltfScene:
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_source_material_names: tuple[str, ...]
    objects: tuple[dict[str, Any], ...]
    source_primitive_count: int
    source_node_instance_count: int
    source_sha256: str
    source_byte_size: int


def _required_array(document: Mapping[str, Any], field: str) -> list[Any]:
    value = document.get(field)
    if not isinstance(value, list):
        raise GltfError(f"GLB document field {field!r} must be an array")
    return value


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GltfError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise GltfError(f"{name} must be an integer >= {minimum}")
    return value


def _index(value: Any, values: list[Any], name: str) -> int:
    index = _integer(value, name)
    if index >= len(values):
        raise GltfError(f"{name} index {index} is out of range")
    return index


def load_glb_bytes(
    payload: bytes, *, source_path: str | Path = "<memory>.glb"
) -> GlbDocument:
    """Parse one immutable GLB byte snapshot without reopening its source path."""

    source = Path(source_path)
    if len(payload) < _GLB_HEADER.size + _CHUNK_HEADER.size:
        raise GltfError("GLB is shorter than its header and JSON chunk")
    magic, version, declared_length = _GLB_HEADER.unpack_from(payload)
    if magic != _MAGIC:
        raise GltfError("invalid GLB magic")
    if version != 2:
        raise GltfError(f"unsupported GLB version {version}")
    if declared_length != len(payload):
        raise GltfError("GLB declared length does not match its byte length")
    if len(payload) % 4:
        raise GltfError("GLB byte length must be four-byte aligned")

    chunks: list[tuple[int, bytes]] = []
    offset = _GLB_HEADER.size
    while offset < len(payload):
        if offset + _CHUNK_HEADER.size > len(payload):
            raise GltfError("truncated GLB chunk header")
        length, kind = _CHUNK_HEADER.unpack_from(payload, offset)
        offset += _CHUNK_HEADER.size
        if length % 4:
            raise GltfError("GLB chunk length must be four-byte aligned")
        end = offset + length
        if end > len(payload):
            raise GltfError("GLB chunk extends past the container")
        chunks.append((kind, payload[offset:end]))
        offset = end
    if not chunks or chunks[0][0] != _JSON_CHUNK:
        raise GltfError("GLB first chunk must be JSON")
    if len(chunks) != 2 or chunks[1][0] != _BIN_CHUNK:
        raise GltfError("acoustic GLB input must contain one JSON and one BIN chunk")
    try:
        document = json.loads(chunks[0][1].rstrip(b" \t\r\n\x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GltfError(f"invalid GLB JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise GltfError("GLB JSON root must be an object")
    asset = _object(document.get("asset"), "asset")
    if asset.get("version") != "2.0":
        raise GltfError("GLB asset.version must be '2.0'")
    required_extensions = document.get("extensionsRequired", [])
    if not isinstance(required_extensions, list):
        raise GltfError("extensionsRequired must be an array")
    if required_extensions:
        raise GltfError(
            "required GLB extensions are unsupported for auditable acoustic extraction: "
            + ", ".join(sorted(map(str, required_extensions)))
        )
    buffers = _required_array(document, "buffers")
    if len(buffers) != 1:
        raise GltfError("acoustic GLB input must declare exactly one embedded buffer")
    buffer = _object(buffers[0], "buffers[0]")
    if "uri" in buffer:
        raise GltfError("external or data-URI GLB buffers are not accepted")
    declared_buffer_length = _integer(buffer.get("byteLength"), "buffers[0].byteLength")
    binary_chunk = chunks[1][1]
    if declared_buffer_length > len(binary_chunk):
        raise GltfError("embedded BIN chunk is shorter than buffers[0].byteLength")
    padding = binary_chunk[declared_buffer_length:]
    if len(padding) > 3 or any(padding):
        raise GltfError("embedded BIN padding must be at most three zero bytes")
    binary = binary_chunk[:declared_buffer_length]
    return GlbDocument(
        document=document,
        binary=binary,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        source_path=source,
    )


def load_glb(path: str | Path) -> GlbDocument:
    source = Path(path).resolve()
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise GltfError(f"unable to read GLB {source}: {exc}") from exc
    return load_glb_bytes(payload, source_path=source)


def _decode_accessor(
    glb: GlbDocument,
    accessor_index: int,
    *,
    expected_type: str,
    allowed_component_types: set[int],
    allow_normalized: bool = False,
) -> np.ndarray:
    document = glb.document
    accessors = _required_array(document, "accessors")
    views = _required_array(document, "bufferViews")
    index = _index(accessor_index, accessors, "accessor")
    accessor = _object(accessors[index], f"accessors[{index}]")
    if "sparse" in accessor:
        raise GltfError(f"accessors[{index}] uses unsupported sparse storage")
    if accessor.get("normalized", False) is not False and not allow_normalized:
        raise GltfError(f"accessors[{index}] uses unsupported normalization")
    if accessor.get("type") != expected_type:
        raise GltfError(f"accessors[{index}].type must be {expected_type}")
    component_type = _integer(
        accessor.get("componentType"), f"accessors[{index}].componentType"
    )
    if component_type not in allowed_component_types:
        raise GltfError(
            f"accessors[{index}] componentType {component_type} is unsupported"
        )
    count = _integer(accessor.get("count"), f"accessors[{index}].count", minimum=1)
    if "bufferView" not in accessor:
        raise GltfError(f"accessors[{index}] has no bufferView")
    view_index = _index(accessor["bufferView"], views, f"accessors[{index}].bufferView")
    view = _object(views[view_index], f"bufferViews[{view_index}]")
    if view.get("buffer") != 0:
        raise GltfError(f"bufferViews[{view_index}] must reference embedded buffer 0")
    if view.get("extensions"):
        raise GltfError(f"bufferViews[{view_index}] uses unsupported extensions")
    view_offset = _integer(view.get("byteOffset", 0), f"bufferViews[{view_index}].byteOffset")
    view_length = _integer(view.get("byteLength"), f"bufferViews[{view_index}].byteLength", minimum=1)
    accessor_offset = _integer(
        accessor.get("byteOffset", 0), f"accessors[{index}].byteOffset"
    )
    dtype = _COMPONENT_DTYPES[component_type]
    component_count = _TYPE_COMPONENTS[expected_type]
    element_size = dtype.itemsize * component_count
    stride = view.get("byteStride", element_size)
    stride = _integer(stride, f"bufferViews[{view_index}].byteStride", minimum=1)
    if stride < element_size or stride % dtype.itemsize:
        raise GltfError(f"bufferViews[{view_index}].byteStride is invalid")
    required = accessor_offset + (count - 1) * stride + element_size
    if required > view_length:
        raise GltfError(f"accessors[{index}] extends beyond its bufferView")
    absolute_offset = view_offset + accessor_offset
    if absolute_offset % dtype.itemsize:
        raise GltfError(f"accessors[{index}] is not component-aligned")
    if view_offset + view_length > len(glb.binary):
        raise GltfError(f"bufferViews[{view_index}] extends beyond the BIN chunk")
    shape = (count,) if component_count == 1 else (count, component_count)
    strides = (stride,) if component_count == 1 else (stride, dtype.itemsize)
    try:
        values = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=glb.binary,
            offset=absolute_offset,
            strides=strides,
        ).copy(order="C")
    except (TypeError, ValueError) as exc:
        raise GltfError(f"unable to decode accessors[{index}]: {exc}") from exc
    return values


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise GltfError(f"{name} must contain {length} numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise GltfError(f"{name} must contain {length} numbers")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise GltfError(f"{name} must contain finite numbers")
    return result


def _quaternion_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion_xyzw))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-5:
        raise GltfError("node rotation quaternion must be unit normalized")
    x, y, z, w = quaternion_xyzw / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _node_local_matrix(node: Mapping[str, Any], node_index: int) -> np.ndarray:
    trs_fields = {"translation", "rotation", "scale"}
    if "matrix" in node:
        if trs_fields.intersection(node):
            raise GltfError(f"nodes[{node_index}] defines both matrix and TRS")
        values = _finite_vector(node["matrix"], 16, f"nodes[{node_index}].matrix")
        matrix = values.reshape((4, 4), order="F")
    else:
        translation = _finite_vector(
            node.get("translation", [0.0, 0.0, 0.0]),
            3,
            f"nodes[{node_index}].translation",
        )
        rotation = _finite_vector(
            node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
            4,
            f"nodes[{node_index}].rotation",
        )
        scale = _finite_vector(
            node.get("scale", [1.0, 1.0, 1.0]),
            3,
            f"nodes[{node_index}].scale",
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _quaternion_matrix(rotation) @ np.diag(scale)
        matrix[:3, 3] = translation
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise GltfError(f"nodes[{node_index}] matrix must be affine")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise GltfError(f"nodes[{node_index}] has a singular transform")
    return matrix


def _scene_node_instances(glb: GlbDocument) -> list[tuple[int, np.ndarray]]:
    document = glb.document
    nodes = _required_array(document, "nodes")
    scenes = _required_array(document, "scenes")
    scene_index = _index(document.get("scene", 0), scenes, "scene")
    scene = _object(scenes[scene_index], f"scenes[{scene_index}]")
    roots = scene.get("nodes")
    if not isinstance(roots, list) or not roots:
        raise GltfError("selected scene must declare at least one root node")

    parents: dict[int, int] = {}
    for parent_index, raw_node in enumerate(nodes):
        node = _object(raw_node, f"nodes[{parent_index}]")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise GltfError(f"nodes[{parent_index}].children must be an array")
        for raw_child in children:
            child = _index(raw_child, nodes, f"nodes[{parent_index}].children")
            if child in parents:
                raise GltfError(f"nodes[{child}] has multiple parents")
            parents[child] = parent_index

    result: list[tuple[int, np.ndarray]] = []
    active: set[int] = set()
    visited: set[int] = set()

    def visit(node_index: int, parent_world: np.ndarray) -> None:
        if node_index in active:
            raise GltfError(f"node hierarchy contains a cycle at nodes[{node_index}]")
        if node_index in visited:
            raise GltfError(f"selected scene references nodes[{node_index}] more than once")
        active.add(node_index)
        visited.add(node_index)
        node = _object(nodes[node_index], f"nodes[{node_index}]")
        if node.get("extensions"):
            raise GltfError(
                f"nodes[{node_index}] uses unsupported instance-affecting extensions"
            )
        if "skin" in node:
            raise GltfError(f"nodes[{node_index}] uses a skin in a static room")
        world = parent_world @ _node_local_matrix(node, node_index)
        if "mesh" in node:
            result.append((node_index, world))
        for raw_child in node.get("children", []):
            visit(_index(raw_child, nodes, f"nodes[{node_index}].children"), world)
        active.remove(node_index)

    for raw_root in roots:
        root = _index(raw_root, nodes, f"scenes[{scene_index}].nodes")
        if root in parents:
            raise GltfError(f"selected root nodes[{root}] also has a parent")
        visit(root, np.eye(4, dtype=np.float64))
    if not result:
        raise GltfError("selected scene contains no mesh node instances")
    return result


def _extract_triangle_scene_from_document(glb: GlbDocument) -> ExpandedGltfScene:
    document = glb.document
    nodes = _required_array(document, "nodes")
    meshes = _required_array(document, "meshes")
    materials = _required_array(document, "materials")
    vertex_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    triangle_material_names: list[str] = []
    objects: list[dict[str, Any]] = []
    vertex_offset = 0
    triangle_offset = 0
    node_instances = _scene_node_instances(glb)

    for node_index, world in node_instances:
        node = _object(nodes[node_index], f"nodes[{node_index}]")
        mesh_index = _index(node.get("mesh"), meshes, f"nodes[{node_index}].mesh")
        mesh = _object(meshes[mesh_index], f"meshes[{mesh_index}]")
        if mesh.get("weights"):
            raise GltfError(f"meshes[{mesh_index}] has morph weights")
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise GltfError(f"meshes[{mesh_index}].primitives must be non-empty")
        for primitive_index, raw_primitive in enumerate(primitives):
            prefix = f"meshes[{mesh_index}].primitives[{primitive_index}]"
            primitive = _object(raw_primitive, prefix)
            if primitive.get("mode", 4) != 4:
                raise GltfError(f"{prefix} must use TRIANGLES mode 4")
            if primitive.get("targets"):
                raise GltfError(f"{prefix} uses morph targets")
            primitive_extensions = primitive.get("extensions", {})
            if isinstance(primitive_extensions, Mapping) and (
                "KHR_draco_mesh_compression" in primitive_extensions
                or "EXT_meshopt_compression" in primitive_extensions
            ):
                raise GltfError(f"{prefix} uses compressed geometry")
            attributes = _object(primitive.get("attributes"), f"{prefix}.attributes")
            if "POSITION" not in attributes:
                raise GltfError(f"{prefix} has no POSITION accessor")
            positions = _decode_accessor(
                glb,
                _integer(attributes["POSITION"], f"{prefix}.attributes.POSITION"),
                expected_type="VEC3",
                allowed_component_types={5126},
            ).astype(np.float64, copy=False)
            if not np.isfinite(positions).all():
                raise GltfError(f"{prefix} POSITION contains non-finite values")
            if "indices" in primitive:
                flat_indices = _decode_accessor(
                    glb,
                    _integer(primitive["indices"], f"{prefix}.indices"),
                    expected_type="SCALAR",
                    allowed_component_types={5121, 5123, 5125},
                ).astype(np.uint64, copy=False)
            else:
                flat_indices = np.arange(len(positions), dtype=np.uint64)
            if len(flat_indices) % 3:
                raise GltfError(f"{prefix} index count is not divisible by three")
            local_triangles = flat_indices.reshape((-1, 3))
            if int(local_triangles.max(initial=0)) >= len(positions):
                raise GltfError(f"{prefix} contains an out-of-range vertex index")
            if len(positions) + vertex_offset > np.iinfo(np.uint32).max:
                raise GltfError("expanded scene exceeds uint32 vertex index capacity")
            material_index = _index(
                primitive.get("material"), materials, f"{prefix}.material"
            )
            material = _object(materials[material_index], f"materials[{material_index}]")
            material_name = material.get("name")
            if not isinstance(material_name, str) or not material_name:
                raise GltfError(f"materials[{material_index}].name must be non-empty")

            homogeneous = np.concatenate(
                (positions, np.ones((len(positions), 1), dtype=np.float64)), axis=1
            )
            world_positions = (world @ homogeneous.T).T[:, :3]
            if not np.isfinite(world_positions).all():
                raise GltfError(f"{prefix} transform produced non-finite vertices")
            adjusted = local_triangles.copy()
            if float(np.linalg.det(world[:3, :3])) < 0:
                adjusted[:, [1, 2]] = adjusted[:, [2, 1]]
            adjusted = (adjusted + vertex_offset).astype("<u4", copy=False)
            with np.errstate(over="ignore", invalid="ignore"):
                world_vertices = world_positions.astype("<f4", copy=False)
            if not np.isfinite(world_vertices).all():
                raise GltfError(
                    f"{prefix} vertices overflow float32 after canonical encoding"
                )
            vertex_parts.append(world_vertices)
            triangle_parts.append(adjusted)
            triangle_material_names.extend([material_name] * len(adjusted))
            object_id = f"node{node_index}_mesh{mesh_index}_primitive{primitive_index}"
            objects.append(
                {
                    "object_id": object_id,
                    "source_node_index": node_index,
                    "source_mesh_index": mesh_index,
                    "source_primitive_index": primitive_index,
                    "source_material_name": material_name,
                    "vertex_offset": vertex_offset,
                    "vertex_count": len(world_vertices),
                    "triangle_offset": triangle_offset,
                    "triangle_count": len(adjusted),
                    "world_from_object": np.eye(4, dtype=float).reshape(-1).tolist(),
                    "source_world_matrix": world.reshape(-1).tolist(),
                    "transform_baked": True,
                }
            )
            vertex_offset += len(world_vertices)
            triangle_offset += len(adjusted)

    if not vertex_parts or not triangle_parts:
        raise GltfError("selected scene produced no triangle geometry")
    vertices = np.ascontiguousarray(np.concatenate(vertex_parts), dtype="<f4")
    triangles = np.ascontiguousarray(np.concatenate(triangle_parts), dtype="<u4")
    return ExpandedGltfScene(
        vertices=vertices,
        triangles=triangles,
        triangle_source_material_names=tuple(triangle_material_names),
        objects=tuple(objects),
        source_primitive_count=len(objects),
        source_node_instance_count=len(node_instances),
        source_sha256=glb.sha256,
        source_byte_size=glb.byte_size,
    )


def extract_triangle_scene_bytes(
    payload: bytes, *, source_path: str | Path = "<memory>.glb"
) -> ExpandedGltfScene:
    """Extract triangles from the exact bytes whose size/hash were validated."""

    return _extract_triangle_scene_from_document(
        load_glb_bytes(payload, source_path=source_path)
    )


def extract_triangle_scene(path: str | Path) -> ExpandedGltfScene:
    return _extract_triangle_scene_from_document(load_glb(path))


def extract_triangle_scene_document(glb: GlbDocument) -> ExpandedGltfScene:
    """Expand an already-parsed document, for callers that need both."""

    return _extract_triangle_scene_from_document(glb)


def triangle_vertex_colours(
    glb: GlbDocument, scene: ExpandedGltfScene
) -> tuple[np.ndarray, int]:
    """Read one linear RGB colour per triangle of ``scene`` from COLOR_0.

    Datasets that carry semantics as painted vertex colour - HM3D is the one
    this exists for - put the only machine-readable instance identity in
    COLOR_0, so the acoustic compiler has to see it. Two things about this are
    load-bearing.

    Alignment is derived, not assumed. Each triangle range is taken from the
    object record that names its node, mesh and primitive index, so this cannot
    silently drift out of step with the extraction that produced ``scene`` -
    which is exactly the failure that would misassign every material in a room
    while looking perfectly healthy.

    The values stay linear. glTF defines COLOR_0 as linear, and whether a
    consumer needs sRGB is a fact about that consumer's annotation file, not
    about the mesh; converting here would bury a dataset quirk in a format
    reader. The second return value counts triangles whose three vertices did
    not agree on a colour, which is the honest measure of how much of the
    result rests on the majority rule below.
    """

    document = glb.document
    meshes = _required_array(document, "meshes")
    triangle_count = len(scene.triangles)
    colours = np.zeros((triangle_count, 3), dtype=np.float64)
    mixed = 0
    for record in scene.objects:
        count = int(record["triangle_count"])
        if not count:
            continue
        offset = int(record["triangle_offset"])
        mesh = _object(
            meshes[int(record["source_mesh_index"])],
            f"meshes[{record['source_mesh_index']}]",
        )
        primitive = _object(
            mesh["primitives"][int(record["source_primitive_index"])],
            "primitive",
        )
        attributes = _object(primitive.get("attributes"), "primitive.attributes")
        if "COLOR_0" not in attributes:
            raise GltfError(
                f"{record['object_id']} has no COLOR_0 accessor; this mesh "
                "carries no vertex-painted semantics"
            )
        accessor_index = _integer(
            attributes["COLOR_0"], "primitive.attributes.COLOR_0"
        )
        accessor = _object(
            _required_array(document, "accessors")[accessor_index],
            f"accessors[{accessor_index}]",
        )
        component_type = _integer(
            accessor.get("componentType"), f"accessors[{accessor_index}].componentType"
        )
        # A normalized integer colour is a fraction of its own full scale; a
        # float colour is already the fraction. Dividing a float attribute by
        # 255 would darken the whole room by that factor and match nothing.
        scale = {5121: 255.0, 5123: 65535.0, 5126: 1.0}.get(component_type)
        if scale is None:
            raise GltfError(
                f"accessors[{accessor_index}] COLOR_0 componentType "
                f"{component_type} is unsupported"
            )
        raw = _decode_accessor(
            glb,
            accessor_index,
            expected_type=accessor.get("type", "VEC4"),
            allowed_component_types={5121, 5123, 5126},
            allow_normalized=True,
        )
        vertex_colours = np.asarray(raw, dtype=np.float64)[:, :3] / scale
        local = (
            scene.triangles[offset : offset + count].astype(np.int64, copy=False)
            - int(record["vertex_offset"])
        )
        if local.size and (local.min() < 0 or local.max() >= len(vertex_colours)):
            raise GltfError(
                f"{record['object_id']} triangle range falls outside its COLOR_0 accessor"
            )
        face = vertex_colours[local]
        first = face[:, 0, :]
        second = face[:, 1, :]
        third = face[:, 2, :]
        agree_all = np.all(first == second, axis=1) & np.all(second == third, axis=1)
        # Majority rule: two vertices agreeing decides the triangle. When all
        # three differ the first vertex decides, which is arbitrary but counted.
        chosen = first.copy()
        pick_second = ~agree_all & np.all(second == third, axis=1)
        chosen[pick_second] = second[pick_second]
        colours[offset : offset + count] = chosen
        mixed += int(np.count_nonzero(~agree_all))
    return colours, mixed
