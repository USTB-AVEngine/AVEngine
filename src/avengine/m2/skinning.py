"""Strict CPU skinning over the complete glTF node graph.

The implementation is deliberately planning-only.  It is intended to compute
conservative actor envelopes before a live renderer is launched; it does not
claim that CPU vertices are runtime readback evidence.  Unsupported animation
or deformation features fail closed instead of being silently omitted.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import math
from pathlib import Path
import struct
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from avengine.m2.glb import (
    AnimationAction,
    AnimationChannel,
    GlbDocument,
    GlbError,
    decode_accessor,
    extract_actions,
    extract_node_hierarchy,
    extract_skins,
    load_glb,
)


SKINNING_SCHEMA = "avengine_compiled_actor_skinning_v1"
_COMPILER_REVISION = 1
_WEIGHT_SUM_TOLERANCE = 1.0e-5
_QUATERNION_NORM_TOLERANCE = 1.0e-5


class SkinningError(ValueError):
    """The asset is outside the strict actor-skinning boundary."""


@dataclass(frozen=True)
class SkinningCacheKey:
    """Path-local planning cache identity."""

    source_path: str
    skin_index: int
    compiler_revision: int = _COMPILER_REVISION


@dataclass(frozen=True)
class CompiledNode:
    node_index: int
    parent_node_index: int | None
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class CompiledPrimitive:
    mesh_node_index: int
    mesh_index: int
    primitive_index: int
    position_accessor_index: int
    joints_accessor_index: int
    weights_accessor_index: int
    positions: np.ndarray = field(repr=False, compare=False)
    joint_ordinals: np.ndarray = field(repr=False, compare=False)
    weights: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class CompiledSkinning:
    """Immutable decoded inputs needed for repeated action sampling."""

    source_path: Path | None
    skin_index: int
    skin_name: str | None
    nodes: tuple[CompiledNode, ...]
    joint_node_indices: tuple[int, ...]
    inverse_bind_matrices: np.ndarray = field(repr=False, compare=False)
    primitives: tuple[CompiledPrimitive, ...]
    actions: tuple[AnimationAction, ...]
    schema: str = field(default=SKINNING_SCHEMA, init=False)
    qualification_state: str = field(default="planning_only", init=False)
    qualification_claim: bool = field(default=False, init=False)
    formal_eligible: bool = field(default=False, init=False)

    @property
    def cache_key(self) -> SkinningCacheKey:
        if self.source_path is None:
            raise SkinningError("in-memory skinning has no path-local cache key")
        return SkinningCacheKey(str(self.source_path), self.skin_index)

    def action(self, action_name: str) -> AnimationAction:
        for action in self.actions:
            if action.name == action_name:
                return action
        raise SkinningError(f"unknown source action: {action_name!r}")


def _objects(value: Any, *, owner: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SkinningError(f"{owner} must be an array of objects")
    return value


def _index(value: Any, *, owner: str, count: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= count
    ):
        raise SkinningError(f"{owner} must be an in-range integer")
    return value


def _readonly(array: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _decode_unsigned_vec4(
    document: GlbDocument, accessor_index: Any, *, owner: str
) -> np.ndarray:
    accessors = _objects(document.json.get("accessors", []), owner="accessors")
    views = _objects(document.json.get("bufferViews", []), owner="bufferViews")
    index = _index(accessor_index, owner=f"{owner} accessor", count=len(accessors))
    accessor = accessors[index]
    if (
        accessor.get("type") != "VEC4"
        or accessor.get("normalized", False) is not False
        or "sparse" in accessor
    ):
        raise SkinningError(
            f"{owner} must be a non-sparse, non-normalized VEC4 accessor"
        )
    formats = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4)}
    try:
        component_format, component_size = formats[accessor.get("componentType")]
    except KeyError as exc:
        raise SkinningError(
            f"{owner} must use an unsigned integer component type"
        ) from exc
    count = accessor.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise SkinningError(f"{owner}.count must be a positive integer")
    view_index = _index(
        accessor.get("bufferView"), owner=f"{owner}.bufferView", count=len(views)
    )
    view = views[view_index]
    if view.get("buffer") != 0:
        raise SkinningError(f"{owner} must reference embedded buffer 0")
    values = (view.get("byteOffset", 0), accessor.get("byteOffset", 0))
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise SkinningError(f"{owner} offsets must be non-negative integers")
    view_offset, accessor_offset = values
    view_length = view.get("byteLength")
    if (
        isinstance(view_length, bool)
        or not isinstance(view_length, int)
        or view_length < 0
    ):
        raise SkinningError(f"{owner} bufferView length must be non-negative")
    packer = struct.Struct("<" + component_format * 4)
    stride = view.get("byteStride", packer.size)
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < packer.size
        or stride % component_size
    ):
        raise SkinningError(f"{owner} has an invalid byte stride")
    required = accessor_offset + (count - 1) * stride + packer.size
    declared_buffers = _objects(document.json.get("buffers", []), owner="buffers")
    if len(declared_buffers) != 1:
        raise SkinningError("actor skinning requires exactly one embedded buffer")
    declared_length = declared_buffers[0].get("byteLength")
    if (
        isinstance(declared_length, bool)
        or not isinstance(declared_length, int)
        or declared_length < 0
    ):
        raise SkinningError("buffers[0].byteLength must be non-negative")
    if (
        required > view_length
        or view_offset + required > declared_length
        or view_offset + required > len(document.binary)
    ):
        raise SkinningError(f"{owner} extends beyond its declared bufferView")
    start = view_offset + accessor_offset
    result = np.empty((count, 4), dtype=np.int64)
    for item_index in range(count):
        result[item_index] = packer.unpack_from(
            document.binary, start + item_index * stride
        )
    return result


def _rotation_matrix(quaternion_xyzw: Sequence[float], *, owner: str) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise SkinningError(f"{owner} must contain four finite xyzw components")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        raise SkinningError(f"{owner} must already be a unit quaternion")
    quaternion /= norm
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _local_matrix(
    translation: Sequence[float],
    rotation_xyzw: Sequence[float],
    scale: Sequence[float],
    *,
    owner: str,
) -> np.ndarray:
    translation_array = np.asarray(translation, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if (
        translation_array.shape != (3,)
        or scale_array.shape != (3,)
        or not np.all(np.isfinite(translation_array))
        or not np.all(np.isfinite(scale_array))
    ):
        raise SkinningError(f"{owner} translation and scale must be finite VEC3")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation_matrix(rotation_xyzw, owner=owner) @ np.diag(scale_array)
    result[:3, 3] = translation_array
    return result


def _normalised_quaternion(value: Sequence[float], *, owner: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise SkinningError(f"{owner} must contain four finite xyzw components")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        raise SkinningError(f"{owner} must already be a unit quaternion")
    return quaternion / norm


def _sample_channel(channel: AnimationChannel, time_seconds: float) -> np.ndarray:
    if channel.interpolation == "CUBICSPLINE":
        raise SkinningError(
            f"animation channel {channel.channel_index} uses unsupported CUBICSPLINE"
        )
    if channel.interpolation not in {"LINEAR", "STEP"}:
        raise SkinningError(
            f"animation channel {channel.channel_index} has unsupported interpolation"
        )
    timestamps = channel.timestamps_seconds
    if time_seconds <= timestamps[0] or len(timestamps) == 1:
        left_index = right_index = 0
        fraction = 0.0
    elif time_seconds >= timestamps[-1]:
        left_index = right_index = len(timestamps) - 1
        fraction = 0.0
    else:
        left_index = bisect_right(timestamps, time_seconds) - 1
        if channel.interpolation == "STEP":
            right_index = left_index
            fraction = 0.0
        else:
            right_index = left_index + 1
            fraction = (time_seconds - timestamps[left_index]) / (
                timestamps[right_index] - timestamps[left_index]
            )
    left = np.asarray(channel.values[left_index], dtype=np.float64)
    right = np.asarray(channel.values[right_index], dtype=np.float64)
    if channel.target_path != "rotation":
        return left + fraction * (right - left)

    first = _normalised_quaternion(left, owner="animation rotation key")
    second = _normalised_quaternion(right, owner="animation rotation key")
    if fraction == 0.0:
        return first
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    orthogonal = second - dot * first
    sine = float(np.linalg.norm(orthogonal))
    if sine <= 8.0 * np.finfo(np.float64).eps:
        return first
    tangent = orthogonal / sine
    angle = math.atan2(sine, dot)
    return _normalised_quaternion(
        math.cos(fraction * angle) * first + math.sin(fraction * angle) * tangent,
        owner="interpolated animation rotation",
    )


def _validate_fail_closed_features(document: GlbDocument) -> None:
    raw = document.json
    nodes = _objects(raw.get("nodes", []), owner="nodes")
    meshes = _objects(raw.get("meshes", []), owner="meshes")
    for node_index, node in enumerate(nodes):
        if "weights" in node:
            raise SkinningError(
                f"nodes[{node_index}].weights morph state is unsupported"
            )
    for mesh_index, mesh in enumerate(meshes):
        if "weights" in mesh:
            raise SkinningError(
                f"meshes[{mesh_index}].weights morph state is unsupported"
            )
        primitives = _objects(
            mesh.get("primitives", []), owner=f"meshes[{mesh_index}].primitives"
        )
        for primitive_index, primitive in enumerate(primitives):
            owner = f"meshes[{mesh_index}].primitives[{primitive_index}]"
            if "targets" in primitive:
                raise SkinningError(f"{owner} morph targets are unsupported")
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict):
                raise SkinningError(f"{owner}.attributes must be an object")
            for attribute in attributes:
                if attribute.startswith("JOINTS_") and attribute != "JOINTS_0":
                    raise SkinningError(f"{owner}.{attribute} is unsupported")
                if attribute.startswith("WEIGHTS_") and attribute != "WEIGHTS_0":
                    raise SkinningError(f"{owner}.{attribute} is unsupported")


def compile_skinning(document: GlbDocument, *, skin_index: int = 0) -> CompiledSkinning:
    """Compile one glTF skin without discarding non-skin graph ancestors."""

    _validate_fail_closed_features(document)
    try:
        hierarchy = extract_node_hierarchy(document)
        skins = extract_skins(document)
        actions = extract_actions(document)
    except GlbError as exc:
        raise SkinningError(f"unable to compile actor GLB: {exc}") from exc
    selected_skin_index = _index(skin_index, owner="skin_index", count=len(skins))
    skin = skins[selected_skin_index]
    if skin.inverse_bind_matrices is None:
        raise SkinningError("actor skinning requires explicit inverse bind matrices")
    if any(
        channel.interpolation == "CUBICSPLINE"
        for action in actions
        for channel in action.channels
    ):
        raise SkinningError("CUBICSPLINE animation channels are unsupported")
    for action in actions:
        for channel in action.channels:
            if channel.target_path == "rotation":
                for value in channel.values:
                    _normalised_quaternion(
                        value,
                        owner=f"action {action.name!r} rotation key",
                    )

    compiled_nodes = tuple(
        CompiledNode(
            node_index=node.node_index,
            parent_node_index=node.parent_node_index,
            translation=node.local_trs.translation,
            rotation_xyzw=node.local_trs.rotation_xyzw,
            scale=node.local_trs.scale,
        )
        for node in hierarchy
    )
    inverse_bind = _readonly(
        np.stack(
            [
                np.asarray(value, dtype=np.float64).reshape(4, 4).T
                for value in skin.inverse_bind_matrices
            ]
        ),
        dtype=np.dtype(np.float64),
    )
    if inverse_bind.shape != (len(skin.joints), 4, 4) or not np.all(
        np.isfinite(inverse_bind)
    ):
        raise SkinningError("inverse bind matrices are malformed or non-finite")

    raw = document.json
    raw_nodes = _objects(raw.get("nodes", []), owner="nodes")
    meshes = _objects(raw.get("meshes", []), owner="meshes")
    primitives: list[CompiledPrimitive] = []
    for node_index, node in enumerate(raw_nodes):
        if node.get("skin") != selected_skin_index:
            continue
        mesh_index = _index(
            node.get("mesh"), owner=f"nodes[{node_index}].mesh", count=len(meshes)
        )
        raw_primitives = _objects(
            meshes[mesh_index].get("primitives", []),
            owner=f"meshes[{mesh_index}].primitives",
        )
        if not raw_primitives:
            raise SkinningError(f"meshes[{mesh_index}] contains no primitives")
        for primitive_index, primitive in enumerate(raw_primitives):
            owner = f"meshes[{mesh_index}].primitives[{primitive_index}]"
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict):
                raise SkinningError(f"{owner}.attributes must be an object")
            required = {"POSITION", "JOINTS_0", "WEIGHTS_0"}
            if not required.issubset(attributes):
                raise SkinningError(f"{owner} lacks POSITION, JOINTS_0, or WEIGHTS_0")
            try:
                positions_accessor = decode_accessor(document, attributes["POSITION"])
                weights_accessor = decode_accessor(document, attributes["WEIGHTS_0"])
            except GlbError as exc:
                raise SkinningError(f"unable to decode {owner}: {exc}") from exc
            if positions_accessor.element_type != "VEC3":
                raise SkinningError(f"{owner}.POSITION must be FLOAT VEC3")
            if weights_accessor.element_type != "VEC4":
                raise SkinningError(f"{owner}.WEIGHTS_0 must be FLOAT VEC4")
            positions = np.asarray(positions_accessor.values, dtype=np.float64)
            weights = np.asarray(weights_accessor.values, dtype=np.float64)
            joints = _decode_unsigned_vec4(
                document, attributes["JOINTS_0"], owner=f"{owner}.JOINTS_0"
            )
            if not (len(positions) == len(weights) == len(joints)) or not len(
                positions
            ):
                raise SkinningError(
                    f"{owner} skin attribute counts differ or are empty"
                )
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(weights)):
                raise SkinningError(f"{owner} contains non-finite positions or weights")
            if np.min(weights) < -1.0e-7:
                raise SkinningError(f"{owner} contains negative skin weights")
            if (
                float(np.max(np.abs(np.sum(weights, axis=1) - 1.0)))
                > _WEIGHT_SUM_TOLERANCE
            ):
                raise SkinningError(f"{owner} skin weights do not sum to one")
            if int(np.max(joints)) >= len(skin.joints):
                raise SkinningError(
                    f"{owner}.JOINTS_0 references an out-of-range joint"
                )
            primitives.append(
                CompiledPrimitive(
                    mesh_node_index=node_index,
                    mesh_index=mesh_index,
                    primitive_index=primitive_index,
                    position_accessor_index=int(attributes["POSITION"]),
                    joints_accessor_index=int(attributes["JOINTS_0"]),
                    weights_accessor_index=int(attributes["WEIGHTS_0"]),
                    positions=_readonly(positions, dtype=np.dtype(np.float64)),
                    joint_ordinals=_readonly(joints, dtype=np.dtype(np.int64)),
                    weights=_readonly(weights, dtype=np.dtype(np.float64)),
                )
            )
    if not primitives:
        raise SkinningError(
            f"no mesh node references selected skin {selected_skin_index}"
        )

    return CompiledSkinning(
        source_path=document.source_path,
        skin_index=selected_skin_index,
        skin_name=skin.name,
        nodes=compiled_nodes,
        joint_node_indices=tuple(joint.node_index for joint in skin.joints),
        inverse_bind_matrices=inverse_bind,
        primitives=tuple(primitives),
        actions=actions,
    )


def action_time_bounds(
    compiled: CompiledSkinning, action_name: str
) -> tuple[float, float]:
    action = compiled.action(action_name)
    return (
        min(channel.timestamps_seconds[0] for channel in action.channels),
        max(channel.timestamps_seconds[-1] for channel in action.channels),
    )


def sample_action_global_matrices(
    compiled: CompiledSkinning, action_name: str, time_seconds: float
) -> np.ndarray:
    """Sample globals for every node, including animated non-skin ancestors."""

    if isinstance(time_seconds, bool) or not isinstance(time_seconds, (int, float)):
        raise SkinningError("time_seconds must be a finite number")
    time = float(time_seconds)
    if not math.isfinite(time):
        raise SkinningError("time_seconds must be a finite number")
    action = compiled.action(action_name)
    start, end = action_time_bounds(compiled, action_name)
    tolerance = 1.0e-12
    if time < start - tolerance or time > end + tolerance:
        raise SkinningError(
            f"time_seconds {time:.17g} is outside action bounds [{start:.17g}, {end:.17g}]"
        )
    time = min(end, max(start, time))

    translations = [
        np.asarray(node.translation, dtype=np.float64) for node in compiled.nodes
    ]
    rotations = [
        np.asarray(node.rotation_xyzw, dtype=np.float64) for node in compiled.nodes
    ]
    scales = [np.asarray(node.scale, dtype=np.float64) for node in compiled.nodes]
    for channel in action.channels:
        value = _sample_channel(channel, time)
        if channel.target_path == "translation":
            translations[channel.target_node_index] = value
        elif channel.target_path == "rotation":
            rotations[channel.target_node_index] = value
        elif channel.target_path == "scale":
            scales[channel.target_node_index] = value
        else:  # pragma: no cover - extract_actions invariant
            raise SkinningError(f"unsupported target path: {channel.target_path!r}")

    locals_ = [
        _local_matrix(
            translations[node.node_index],
            rotations[node.node_index],
            scales[node.node_index],
            owner=f"nodes[{node.node_index}] sampled TRS",
        )
        for node in compiled.nodes
    ]
    globals_: list[np.ndarray | None] = [None] * len(compiled.nodes)

    def resolve(node_index: int) -> np.ndarray:
        cached = globals_[node_index]
        if cached is not None:
            return cached
        parent = compiled.nodes[node_index].parent_node_index
        result = (
            locals_[node_index]
            if parent is None
            else resolve(parent) @ locals_[node_index]
        )
        globals_[node_index] = result
        return result

    result = np.stack([resolve(index) for index in range(len(compiled.nodes))])
    if not np.all(np.isfinite(result)):
        raise SkinningError("sampled node globals contain non-finite values")
    return result


def sample_action_vertices(
    compiled: CompiledSkinning, action_name: str, time_seconds: float
) -> np.ndarray:
    """Return all selected-skin vertices in the actor GLB's global coordinates."""

    globals_ = sample_action_global_matrices(compiled, action_name, time_seconds)
    joints = globals_[list(compiled.joint_node_indices)]
    joint_matrices = joints @ compiled.inverse_bind_matrices
    outputs: list[np.ndarray] = []
    for primitive in compiled.primitives:
        selected = joint_matrices[primitive.joint_ordinals]
        homogeneous = np.concatenate(
            [
                primitive.positions,
                np.ones((len(primitive.positions), 1), dtype=np.float64),
            ],
            axis=1,
        )
        transformed = np.einsum("vjab,vb->vja", selected, homogeneous)
        vertices = np.sum(transformed[:, :, :3] * primitive.weights[:, :, None], axis=1)
        outputs.append(vertices)
    result = np.concatenate(outputs, axis=0)
    if not np.all(np.isfinite(result)):
        raise SkinningError("CPU-skinned vertices contain non-finite values")
    return result


class SkinningCompileCache:
    """Explicit in-process cache for repeated loads of one resolved source path."""

    def __init__(self) -> None:
        self._compiled: MutableMapping[SkinningCacheKey, CompiledSkinning] = {}

    def compile(
        self, document: GlbDocument, *, skin_index: int = 0
    ) -> CompiledSkinning:
        if document.source_path is None:
            return compile_skinning(document, skin_index=skin_index)
        key = SkinningCacheKey(str(document.source_path.resolve()), skin_index)
        cached = self._compiled.get(key)
        if cached is not None:
            return cached
        compiled = compile_skinning(document, skin_index=skin_index)
        self._compiled[key] = compiled
        return compiled

    def load(self, path: str | Path, *, skin_index: int = 0) -> CompiledSkinning:
        raw_path = Path(path)
        try:
            resolved = raw_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SkinningError(f"actor GLB does not exist: {raw_path}") from exc
        if not resolved.is_file():
            raise SkinningError(f"actor GLB is not a regular file: {resolved}")
        try:
            document = load_glb(resolved)
        except GlbError as exc:
            raise SkinningError(f"unable to load actor GLB: {exc}") from exc
        return self.compile(document, skin_index=skin_index)

    def __len__(self) -> int:
        return len(self._compiled)
