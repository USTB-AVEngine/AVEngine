"""Strict, read-only GLB 2.0 inspection primitives for M2.

This module deliberately implements only the GLB and glTF features needed to
inspect a deterministic skinned animation input.  Unsupported encodings are
errors rather than implicit conversions.  In particular, it never follows an
external URI, applies an extension, normalizes authored values, or writes an
asset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence


_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_GLB_HEADER = struct.Struct("<4sII")
_CHUNK_HEADER = struct.Struct("<II")
_JSON_CHUNK_TYPE = 0x4E4F534A
_BIN_CHUNK_TYPE = 0x004E4942
_FLOAT_COMPONENT_TYPE = 5126
_FLOAT_SIZE = 4
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_ANIMATION_PATH_TYPES = {
    "translation": "VEC3",
    "rotation": "VEC4",
    "scale": "VEC3",
}
_INTERPOLATIONS = {"LINEAR", "STEP", "CUBICSPLINE"}


class GlbError(ValueError):
    """A structural or unsupported-feature failure while inspecting a GLB."""


def _json_module_dumps(value: Mapping[str, Any]) -> bytes:
    """Serialize retained JSON so no mutable object aliases escape."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, init=False)
class GlbDocument:
    """Parsed GLB bytes and their immutable identity.

    ``json`` returns a detached decoded glTF object.  Mutating that copy cannot
    make the parsed content disagree with ``sha256`` or affect later extracts.
    """

    _json_bytes: bytes
    binary: bytes
    sha256: str
    byte_length: int
    source_path: Path | None = None

    def __init__(
        self,
        *,
        json: Mapping[str, Any],
        binary: bytes,
        sha256: str,
        byte_length: int,
        source_path: Path | None = None,
    ) -> None:
        try:
            json_bytes = _json_module_dumps(json)
        except (TypeError, ValueError) as exc:
            raise GlbError(f"glTF JSON cannot be retained immutably: {exc}") from exc
        object.__setattr__(self, "_json_bytes", json_bytes)
        object.__setattr__(self, "binary", bytes(binary))
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "byte_length", byte_length)
        object.__setattr__(self, "source_path", source_path)

    @property
    def json(self) -> dict[str, Any]:
        value = json.loads(self._json_bytes)
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise GlbError("retained glTF JSON root is no longer an object")
        return value


@dataclass(frozen=True)
class AccessorData:
    accessor_index: int
    component_type: int
    element_type: str
    count: int
    values: tuple[tuple[float, ...], ...]

    @property
    def scalars(self) -> tuple[float, ...]:
        """Return SCALAR values without silently flattening another type."""

        if self.element_type != "SCALAR":
            raise GlbError(
                f"accessor {self.accessor_index} is {self.element_type}, not SCALAR"
            )
        return tuple(value[0] for value in self.values)


@dataclass(frozen=True)
class NodeLocalTRS:
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class NodeRecord:
    node_index: int
    name: str | None
    parent_node_index: int | None
    children_node_indices: tuple[int, ...]
    local_trs: NodeLocalTRS


@dataclass(frozen=True)
class JointNode:
    joint_ordinal: int
    node_index: int
    name: str | None
    parent_node_index: int | None
    parent_joint_node_index: int | None
    children_node_indices: tuple[int, ...]
    child_joint_node_indices: tuple[int, ...]
    local_trs: NodeLocalTRS


@dataclass(frozen=True)
class SkinRecord:
    skin_index: int
    name: str | None
    skeleton_node_index: int | None
    inverse_bind_matrices_accessor_index: int | None
    inverse_bind_matrices: tuple[tuple[float, ...], ...] | None
    joints: tuple[JointNode, ...]


@dataclass(frozen=True)
class AnimationChannel:
    channel_index: int
    sampler_index: int
    target_node_index: int
    target_node_name: str | None
    target_path: str
    interpolation: str
    input_accessor_index: int
    output_accessor_index: int
    timestamps_seconds: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class AnimationAction:
    animation_index: int
    name: str
    duration_seconds: float
    channels: tuple[AnimationChannel, ...]


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GlbError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise GlbError(f"non-finite JSON number is not supported: {value}")


def _decode_json_chunk(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GlbError("GLB JSON chunk is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except GlbError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GlbError(f"invalid GLB JSON chunk: {exc}") from exc
    if not isinstance(value, dict):
        raise GlbError("GLB JSON chunk root must be an object")
    return value


def _required_int(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GlbError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}"
        if maximum is not None:
            bounds += f" and <= {maximum}"
        raise GlbError(f"{name} must be {bounds}")
    return value


def _optional_name(value: Mapping[str, Any], *, owner: str) -> str | None:
    if "name" not in value:
        return None
    name = value["name"]
    if not isinstance(name, str):
        raise GlbError(f"{owner}.name must be a string")
    return name


def _object_array(document: GlbDocument, key: str) -> list[Mapping[str, Any]]:
    raw = document.json.get(key, [])
    if not isinstance(raw, list):
        raise GlbError(f"glTF {key} must be an array")
    result: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise GlbError(f"glTF {key}[{index}] must be an object")
        result.append(value)
    return result


def _validate_document_envelope(
    document_json: Mapping[str, Any], binary: bytes
) -> None:
    asset = document_json.get("asset")
    if not isinstance(asset, dict):
        raise GlbError("glTF asset must be an object")
    if asset.get("version") != "2.0":
        raise GlbError("glTF asset.version must be exactly '2.0'")
    min_version = asset.get("minVersion")
    if min_version is not None and min_version != "2.0":
        raise GlbError(f"unsupported glTF asset.minVersion: {min_version!r}")

    required_extensions = document_json.get("extensionsRequired", [])
    if not isinstance(required_extensions, list) or any(
        not isinstance(item, str) for item in required_extensions
    ):
        raise GlbError("glTF extensionsRequired must be an array of strings")
    if len(set(required_extensions)) != len(required_extensions):
        raise GlbError("glTF extensionsRequired contains duplicates")
    if required_extensions:
        raise GlbError(
            "required glTF extensions are unsupported: "
            + ", ".join(sorted(required_extensions))
        )

    buffers = document_json.get("buffers", [])
    if not isinstance(buffers, list):
        raise GlbError("glTF buffers must be an array")
    if len(buffers) > 1:
        raise GlbError("GLB audit supports exactly one embedded buffer")
    if not buffers:
        if binary:
            raise GlbError("GLB has a BIN chunk but no buffers[0] declaration")
        return

    buffer = buffers[0]
    if not isinstance(buffer, dict):
        raise GlbError("glTF buffers[0] must be an object")
    if "uri" in buffer:
        raise GlbError("external or data-URI buffers are unsupported in GLB audit")
    declared_length = _required_int(
        buffer.get("byteLength"), name="buffers[0].byteLength", minimum=1
    )
    if not binary:
        raise GlbError("buffers[0] is declared but the GLB has no BIN chunk")
    if len(binary) < declared_length or len(binary) > declared_length + 3:
        raise GlbError(
            "BIN chunk length must equal buffers[0].byteLength plus at most "
            "three padding bytes"
        )
    padding = binary[declared_length:]
    if any(padding):
        raise GlbError("BIN chunk padding bytes must be zero")


def parse_glb(data: bytes | bytearray | memoryview) -> GlbDocument:
    """Parse in-memory GLB 2.0 bytes without resolving any external resource."""

    try:
        payload = bytes(data)
    except (TypeError, ValueError) as exc:
        raise GlbError("GLB input must be bytes-like") from exc
    if len(payload) < _GLB_HEADER.size + _CHUNK_HEADER.size:
        raise GlbError("GLB is too short to contain a header and JSON chunk")

    magic, version, declared_length = _GLB_HEADER.unpack_from(payload)
    if magic != _GLB_MAGIC:
        raise GlbError("invalid GLB magic")
    if version != _GLB_VERSION:
        raise GlbError(f"unsupported GLB container version: {version}")
    if declared_length != len(payload):
        raise GlbError(
            f"GLB declared length {declared_length} does not match "
            f"actual length {len(payload)}"
        )
    if declared_length % 4:
        raise GlbError("GLB total length must be 4-byte aligned")

    chunks: list[tuple[int, bytes]] = []
    offset = _GLB_HEADER.size
    while offset < len(payload):
        if len(payload) - offset < _CHUNK_HEADER.size:
            raise GlbError("truncated GLB chunk header")
        chunk_length, chunk_type = _CHUNK_HEADER.unpack_from(payload, offset)
        offset += _CHUNK_HEADER.size
        if chunk_length % 4:
            raise GlbError("GLB chunk length must be 4-byte aligned")
        end = offset + chunk_length
        if end > len(payload):
            raise GlbError("GLB chunk extends beyond the declared container length")
        chunks.append((chunk_type, payload[offset:end]))
        offset = end

    if not chunks or chunks[0][0] != _JSON_CHUNK_TYPE:
        raise GlbError("the first GLB chunk must be JSON")
    if len(chunks) > 2:
        raise GlbError("GLB audit supports only one JSON and one optional BIN chunk")
    if len(chunks) == 2 and chunks[1][0] != _BIN_CHUNK_TYPE:
        raise GlbError("the optional second GLB chunk must be BIN")
    if not chunks[0][1]:
        raise GlbError("GLB JSON chunk must not be empty")

    document_json = _decode_json_chunk(chunks[0][1])
    binary = chunks[1][1] if len(chunks) == 2 else b""
    _validate_document_envelope(document_json, binary)
    return GlbDocument(
        json=document_json,
        binary=binary,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_length=len(payload),
    )


def load_glb(path: str | Path) -> GlbDocument:
    """Read and parse one GLB while retaining its resolved source identity."""

    source_path = Path(path).resolve()
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise GlbError(f"unable to read GLB {source_path}: {exc}") from exc
    document = parse_glb(payload)
    return GlbDocument(
        json=document.json,
        binary=document.binary,
        sha256=document.sha256,
        byte_length=document.byte_length,
        source_path=source_path,
    )


def _validate_buffer_view(
    document: GlbDocument, index: int
) -> tuple[int, int, int | None]:
    views = _object_array(document, "bufferViews")
    if index >= len(views):
        raise GlbError(f"bufferView index {index} is out of range")
    view = views[index]
    buffer_index = _required_int(
        view.get("buffer"), name=f"bufferViews[{index}].buffer"
    )
    if buffer_index != 0:
        raise GlbError(
            f"bufferViews[{index}] references unsupported buffer {buffer_index}"
        )
    byte_offset = _required_int(
        view.get("byteOffset", 0), name=f"bufferViews[{index}].byteOffset"
    )
    byte_length = _required_int(
        view.get("byteLength"), name=f"bufferViews[{index}].byteLength", minimum=1
    )
    buffers = _object_array(document, "buffers")
    if len(buffers) != 1:
        raise GlbError("accessor decoding requires exactly one embedded buffer")
    declared_buffer_length = _required_int(
        buffers[0].get("byteLength"), name="buffers[0].byteLength", minimum=1
    )
    if byte_offset + byte_length > declared_buffer_length:
        raise GlbError(f"bufferViews[{index}] extends beyond buffers[0].byteLength")
    stride: int | None = None
    if "byteStride" in view:
        stride = _required_int(
            view["byteStride"],
            name=f"bufferViews[{index}].byteStride",
            minimum=4,
            maximum=252,
        )
        if stride % 4:
            raise GlbError(f"bufferViews[{index}].byteStride must be a multiple of 4")
    return byte_offset, byte_length, stride


def decode_accessor(document: GlbDocument, accessor_index: int) -> AccessorData:
    """Decode a non-sparse FLOAT accessor from the embedded BIN chunk.

    SCALAR, vector and square matrix element shapes are accepted.  The M2
    animation extractor further constrains inputs and outputs to SCALAR,
    VEC3, and VEC4 as required by their target channel.
    """

    index = _required_int(accessor_index, name="accessor_index")
    accessors = _object_array(document, "accessors")
    if index >= len(accessors):
        raise GlbError(f"accessor index {index} is out of range")
    accessor = accessors[index]
    if "sparse" in accessor:
        raise GlbError(f"accessors[{index}] uses unsupported sparse storage")
    if "bufferView" not in accessor:
        raise GlbError(f"accessors[{index}] has no bufferView")
    if accessor.get("normalized", False) is not False:
        raise GlbError(f"accessors[{index}] uses unsupported normalization")
    component_type = _required_int(
        accessor.get("componentType"), name=f"accessors[{index}].componentType"
    )
    if component_type != _FLOAT_COMPONENT_TYPE:
        raise GlbError(
            f"accessors[{index}] componentType {component_type} is unsupported; "
            "expected FLOAT (5126)"
        )
    element_type = accessor.get("type")
    if not isinstance(element_type, str) or element_type not in _TYPE_COMPONENTS:
        raise GlbError(f"accessors[{index}].type is unsupported: {element_type!r}")
    count = _required_int(
        accessor.get("count"), name=f"accessors[{index}].count", minimum=1
    )
    buffer_view_index = _required_int(
        accessor["bufferView"], name=f"accessors[{index}].bufferView"
    )
    view_offset, view_length, byte_stride = _validate_buffer_view(
        document, buffer_view_index
    )
    accessor_offset = _required_int(
        accessor.get("byteOffset", 0), name=f"accessors[{index}].byteOffset"
    )
    if (view_offset + accessor_offset) % _FLOAT_SIZE:
        raise GlbError(f"accessors[{index}] is not aligned to a FLOAT boundary")

    component_count = _TYPE_COMPONENTS[element_type]
    element_size = component_count * _FLOAT_SIZE
    stride = element_size if byte_stride is None else byte_stride
    if stride < element_size:
        raise GlbError(
            f"bufferViews[{buffer_view_index}].byteStride is smaller than one "
            f"{element_type} element"
        )
    required_length = accessor_offset + (count - 1) * stride + element_size
    if required_length > view_length:
        raise GlbError(f"accessors[{index}] extends beyond its bufferView")

    unpacker = struct.Struct("<" + "f" * component_count)
    values: list[tuple[float, ...]] = []
    first = view_offset + accessor_offset
    for item_index in range(count):
        value = unpacker.unpack_from(document.binary, first + item_index * stride)
        if not all(math.isfinite(component) for component in value):
            raise GlbError(
                f"accessors[{index}] contains a non-finite FLOAT at item {item_index}"
            )
        values.append(tuple(float(component) for component in value))
    return AccessorData(
        accessor_index=index,
        component_type=component_type,
        element_type=element_type,
        count=count,
        values=tuple(values),
    )


def _finite_vector(value: Any, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise GlbError(f"{name} must be an array of {length} numbers")
    result: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise GlbError(f"{name} must be an array of {length} numbers")
        number = float(component)
        if not math.isfinite(number):
            raise GlbError(f"{name} must contain only finite numbers")
        result.append(number)
    return tuple(result)


def _node_local_trs(node: Mapping[str, Any], *, node_index: int) -> NodeLocalTRS:
    trs_keys = {"translation", "rotation", "scale"}
    if "matrix" in node:
        if trs_keys.intersection(node):
            raise GlbError(
                f"nodes[{node_index}] ambiguously defines both matrix and TRS"
            )
        raise GlbError(
            f"nodes[{node_index}] uses an unsupported local matrix; explicit TRS required"
        )
    translation = _finite_vector(
        node.get("translation", [0.0, 0.0, 0.0]),
        length=3,
        name=f"nodes[{node_index}].translation",
    )
    rotation = _finite_vector(
        node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
        length=4,
        name=f"nodes[{node_index}].rotation",
    )
    scale = _finite_vector(
        node.get("scale", [1.0, 1.0, 1.0]),
        length=3,
        name=f"nodes[{node_index}].scale",
    )
    rotation_norm = math.sqrt(sum(component * component for component in rotation))
    if abs(rotation_norm - 1.0) > 1e-5:
        raise GlbError(f"nodes[{node_index}].rotation must be a unit quaternion")
    return NodeLocalTRS(
        translation=(translation[0], translation[1], translation[2]),
        rotation_xyzw=(rotation[0], rotation[1], rotation[2], rotation[3]),
        scale=(scale[0], scale[1], scale[2]),
    )


def _node_graph(
    document: GlbDocument,
) -> tuple[
    list[Mapping[str, Any]], tuple[int | None, ...], tuple[tuple[int, ...], ...]
]:
    nodes = _object_array(document, "nodes")
    parents: list[int | None] = [None] * len(nodes)
    children_by_node: list[tuple[int, ...]] = []
    for node_index, node in enumerate(nodes):
        raw_children = node.get("children", [])
        if not isinstance(raw_children, list):
            raise GlbError(f"nodes[{node_index}].children must be an array")
        children: list[int] = []
        for child_ordinal, raw_child in enumerate(raw_children):
            child = _required_int(
                raw_child,
                name=f"nodes[{node_index}].children[{child_ordinal}]",
            )
            if child >= len(nodes):
                raise GlbError(
                    f"nodes[{node_index}].children[{child_ordinal}] is out of range"
                )
            if child == node_index:
                raise GlbError(f"nodes[{node_index}] cannot be its own child")
            if child in children:
                raise GlbError(
                    f"nodes[{node_index}].children contains duplicate {child}"
                )
            if parents[child] is not None:
                raise GlbError(
                    f"nodes[{child}] has multiple parents: "
                    f"{parents[child]} and {node_index}"
                )
            parents[child] = node_index
            children.append(child)
        children_by_node.append(tuple(children))

    state = [0] * len(nodes)

    def visit(node_index: int) -> None:
        if state[node_index] == 1:
            raise GlbError(f"node hierarchy contains a cycle at nodes[{node_index}]")
        if state[node_index] == 2:
            return
        state[node_index] = 1
        for child in children_by_node[node_index]:
            visit(child)
        state[node_index] = 2

    for node_index in range(len(nodes)):
        visit(node_index)
    return nodes, tuple(parents), tuple(children_by_node)


def extract_node_hierarchy(document: GlbDocument) -> tuple[NodeRecord, ...]:
    """Extract the complete node tree, requiring explicit/decomposable TRS."""

    nodes, parents, children = _node_graph(document)
    return tuple(
        NodeRecord(
            node_index=index,
            name=_optional_name(node, owner=f"nodes[{index}]"),
            parent_node_index=parents[index],
            children_node_indices=children[index],
            local_trs=_node_local_trs(node, node_index=index),
        )
        for index, node in enumerate(nodes)
    )


def _is_ancestor_or_self(
    ancestor: int, node: int, parents: Sequence[int | None]
) -> bool:
    current: int | None = node
    while current is not None:
        if current == ancestor:
            return True
        current = parents[current]
    return False


def extract_skins(document: GlbDocument) -> tuple[SkinRecord, ...]:
    """Extract skin joint order, raw node hierarchy, local TRS, and bind matrices."""

    nodes, parents, children = _node_graph(document)
    skins = _object_array(document, "skins")
    result: list[SkinRecord] = []
    for skin_index, skin in enumerate(skins):
        raw_joints = skin.get("joints")
        if not isinstance(raw_joints, list) or not raw_joints:
            raise GlbError(f"skins[{skin_index}].joints must be a non-empty array")
        joints: list[int] = []
        for ordinal, raw_joint in enumerate(raw_joints):
            joint = _required_int(
                raw_joint, name=f"skins[{skin_index}].joints[{ordinal}]"
            )
            if joint >= len(nodes):
                raise GlbError(f"skins[{skin_index}].joints[{ordinal}] is out of range")
            if joint in joints:
                raise GlbError(
                    f"skins[{skin_index}].joints contains duplicate node {joint}"
                )
            joints.append(joint)
        joint_set = set(joints)

        skeleton_node: int | None = None
        if "skeleton" in skin:
            skeleton_node = _required_int(
                skin["skeleton"], name=f"skins[{skin_index}].skeleton"
            )
            if skeleton_node >= len(nodes):
                raise GlbError(f"skins[{skin_index}].skeleton is out of range")
            if any(
                not _is_ancestor_or_self(skeleton_node, joint, parents)
                for joint in joints
            ):
                raise GlbError(
                    f"skins[{skin_index}].skeleton is not an ancestor of every joint"
                )

        inverse_accessor: int | None = None
        inverse_matrices: tuple[tuple[float, ...], ...] | None = None
        if "inverseBindMatrices" in skin:
            inverse_accessor = _required_int(
                skin["inverseBindMatrices"],
                name=f"skins[{skin_index}].inverseBindMatrices",
            )
            decoded = decode_accessor(document, inverse_accessor)
            if decoded.element_type != "MAT4":
                raise GlbError(
                    f"skins[{skin_index}].inverseBindMatrices must be FLOAT MAT4"
                )
            if decoded.count != len(joints):
                raise GlbError(
                    f"skins[{skin_index}].inverseBindMatrices count "
                    "must equal joints count"
                )
            inverse_matrices = decoded.values

        joint_records: list[JointNode] = []
        for ordinal, joint in enumerate(joints):
            parent = parents[joint]
            joint_records.append(
                JointNode(
                    joint_ordinal=ordinal,
                    node_index=joint,
                    name=_optional_name(nodes[joint], owner=f"nodes[{joint}]"),
                    parent_node_index=parent,
                    parent_joint_node_index=parent if parent in joint_set else None,
                    children_node_indices=children[joint],
                    child_joint_node_indices=tuple(
                        child for child in children[joint] if child in joint_set
                    ),
                    local_trs=_node_local_trs(nodes[joint], node_index=joint),
                )
            )
        result.append(
            SkinRecord(
                skin_index=skin_index,
                name=_optional_name(skin, owner=f"skins[{skin_index}]"),
                skeleton_node_index=skeleton_node,
                inverse_bind_matrices_accessor_index=inverse_accessor,
                inverse_bind_matrices=inverse_matrices,
                joints=tuple(joint_records),
            )
        )
    return tuple(result)


def _require_accessor_type(
    accessor: AccessorData, *, expected_type: str, owner: str
) -> None:
    if accessor.element_type != expected_type:
        raise GlbError(
            f"{owner} must reference FLOAT {expected_type}, got "
            f"FLOAT {accessor.element_type}"
        )


def _strict_timestamps(accessor: AccessorData, *, owner: str) -> tuple[float, ...]:
    _require_accessor_type(accessor, expected_type="SCALAR", owner=owner)
    values = accessor.scalars
    if values[0] < 0.0:
        raise GlbError(f"{owner} timestamps must be non-negative")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise GlbError(f"{owner} timestamps must be strictly increasing")
    return values


def extract_actions(document: GlbDocument) -> tuple[AnimationAction, ...]:
    """Extract named transform animation channels and decoded sampler values.

    Action names must be unique and non-empty.  Only node translation,
    rotation, and scale targets are accepted; morph-weight and extension-based
    targets intentionally fail closed.
    """

    nodes, _, _ = _node_graph(document)
    animations = _object_array(document, "animations")
    names: set[str] = set()
    actions: list[AnimationAction] = []
    accessor_cache: dict[int, AccessorData] = {}

    def accessor(accessor_index: int) -> AccessorData:
        if accessor_index not in accessor_cache:
            accessor_cache[accessor_index] = decode_accessor(document, accessor_index)
        return accessor_cache[accessor_index]

    for animation_index, animation in enumerate(animations):
        name = _optional_name(animation, owner=f"animations[{animation_index}]")
        if name is None or not name.strip():
            raise GlbError(f"animations[{animation_index}] must have a non-empty name")
        if name in names:
            raise GlbError(
                f"animation name is ambiguous because it is duplicated: {name!r}"
            )
        names.add(name)

        raw_samplers = animation.get("samplers")
        raw_channels = animation.get("channels")
        if not isinstance(raw_samplers, list) or not raw_samplers:
            raise GlbError(
                f"animations[{animation_index}].samplers must be a non-empty array"
            )
        if not isinstance(raw_channels, list) or not raw_channels:
            raise GlbError(
                f"animations[{animation_index}].channels must be a non-empty array"
            )
        samplers: list[Mapping[str, Any]] = []
        for sampler_index, sampler in enumerate(raw_samplers):
            if not isinstance(sampler, dict):
                raise GlbError(
                    f"animations[{animation_index}].samplers[{sampler_index}] "
                    "must be an object"
                )
            samplers.append(sampler)

        channels: list[AnimationChannel] = []
        targets: set[tuple[int, str]] = set()
        for channel_index, raw_channel in enumerate(raw_channels):
            channel_owner = f"animations[{animation_index}].channels[{channel_index}]"
            if not isinstance(raw_channel, dict):
                raise GlbError(f"{channel_owner} must be an object")
            sampler_index = _required_int(
                raw_channel.get("sampler"), name=f"{channel_owner}.sampler"
            )
            if sampler_index >= len(samplers):
                raise GlbError(f"{channel_owner}.sampler is out of range")
            target = raw_channel.get("target")
            if not isinstance(target, dict):
                raise GlbError(f"{channel_owner}.target must be an object")
            if "extensions" in target:
                raise GlbError(f"{channel_owner}.target extensions are unsupported")
            target_node = _required_int(
                target.get("node"), name=f"{channel_owner}.target.node"
            )
            if target_node >= len(nodes):
                raise GlbError(f"{channel_owner}.target.node is out of range")
            target_path = target.get("path")
            if target_path not in _ANIMATION_PATH_TYPES:
                raise GlbError(
                    f"{channel_owner}.target.path is unsupported: {target_path!r}"
                )
            target_key = (target_node, target_path)
            if target_key in targets:
                raise GlbError(
                    f"animations[{animation_index}] has duplicate channels for "
                    f"node {target_node} {target_path}"
                )
            targets.add(target_key)
            _node_local_trs(nodes[target_node], node_index=target_node)

            sampler = samplers[sampler_index]
            sampler_owner = f"animations[{animation_index}].samplers[{sampler_index}]"
            input_index = _required_int(
                sampler.get("input"), name=f"{sampler_owner}.input"
            )
            output_index = _required_int(
                sampler.get("output"), name=f"{sampler_owner}.output"
            )
            interpolation = sampler.get("interpolation", "LINEAR")
            if (
                not isinstance(interpolation, str)
                or interpolation not in _INTERPOLATIONS
            ):
                raise GlbError(
                    f"{sampler_owner}.interpolation is unsupported: {interpolation!r}"
                )
            timestamps = _strict_timestamps(
                accessor(input_index), owner=f"{sampler_owner}.input"
            )
            output = accessor(output_index)
            _require_accessor_type(
                output,
                expected_type=_ANIMATION_PATH_TYPES[target_path],
                owner=f"{sampler_owner}.output for {target_path}",
            )
            expected_output_count = len(timestamps) * (
                3 if interpolation == "CUBICSPLINE" else 1
            )
            if output.count != expected_output_count:
                raise GlbError(
                    f"{sampler_owner}.output count {output.count} does not match "
                    f"{interpolation} input count {len(timestamps)}"
                )
            channels.append(
                AnimationChannel(
                    channel_index=channel_index,
                    sampler_index=sampler_index,
                    target_node_index=target_node,
                    target_node_name=_optional_name(
                        nodes[target_node], owner=f"nodes[{target_node}]"
                    ),
                    target_path=target_path,
                    interpolation=interpolation,
                    input_accessor_index=input_index,
                    output_accessor_index=output_index,
                    timestamps_seconds=timestamps,
                    values=output.values,
                )
            )
        duration = max(channel.timestamps_seconds[-1] for channel in channels)
        actions.append(
            AnimationAction(
                animation_index=animation_index,
                name=name,
                duration_seconds=duration,
                channels=tuple(channels),
            )
        )
    return tuple(actions)
