"""Bounded automatic QA for the single M2 skinned-dog canary.

These checks intentionally separate numerical safety from visual approval.
They prove that the canonical GLB has coherent topology, UVs, weights and bind
matrices; that every baked frame skins to finite, non-collapsed geometry; and
that the declared no-mouth contract is respected.  They do not decide whether
the motion looks anatomically convincing.  In particular, measured hind-leg
under-articulation is retained as a human-review warning rather than silently
"fixed" or promoted to a stronger claim.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m2.actions import BakedActionSet, baked_actions_content_sha256
from avengine.m2.glb import (
    AnimationChannel,
    GlbDocument,
    GlbError,
    decode_accessor,
    extract_actions,
    extract_skins,
)
from avengine.m2.habitat import HabitatAssetMapping


_TRIANGLES_MODE = 4
_MAX_WEIGHT_SUM_ERROR = 1.0e-5
_MAX_BIND_CLOSURE_ERROR_M = 1.0e-4
_MAX_LOOP_CLOSURE_ERROR_M = 1.0e-4
_MAX_LOOP_ROTATION_ERROR = 1.0e-5
_MAX_LOOP_SCALE_ERROR = 1.0e-5
_MAX_VERTEX_STEP_REST_DIAGONAL_RATIO = 0.10
_MIN_TRIANGLE_AREA_M2 = 1.0e-12
_LANDMARK_MARGIN_M = 0.02
_MOUTH_ANGLE_TOLERANCE_DEGREES = 1.0e-6
_SOURCE_QUATERNION_NORM_TOLERANCE = 1.0e-5
_SOURCE_CLIP_TIME_TOLERANCE_SECONDS = 1.0e-12


class M2QaError(ValueError):
    """The candidate is outside the strict automatic-QA boundary."""


@dataclass(frozen=True)
class M2AutomaticQa:
    static_geometry: dict[str, Any]
    deformation: dict[str, Any]
    animation: dict[str, Any]


@dataclass(frozen=True)
class _PrimitiveData:
    primitive_index: int
    positions: np.ndarray
    normals: np.ndarray
    texcoords: np.ndarray
    joint_ordinals: np.ndarray
    weights: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class _SourceLoopEndpoints:
    source_action_name: str
    clip_start_seconds: float
    clip_end_seconds: float
    start_global_joints: np.ndarray
    end_global_joints: np.ndarray
    maximum_joint_rotation_error: float
    maximum_joint_translation_error_m: float
    maximum_joint_scale_error: float


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _objects(value: Any, *, owner: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise M2QaError(f"{owner} must be an array of objects")
    return value


def _decode_unsigned_accessor(
    document: GlbDocument,
    accessor_index: Any,
    *,
    element_type: str,
    owner: str,
) -> np.ndarray:
    if isinstance(accessor_index, bool) or not isinstance(accessor_index, int):
        raise M2QaError(f"{owner} accessor index must be an integer")
    accessors = _objects(document.json.get("accessors", []), owner="accessors")
    views = _objects(document.json.get("bufferViews", []), owner="bufferViews")
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise M2QaError(f"{owner} accessor is out of range")
    accessor = accessors[accessor_index]
    if (
        accessor.get("type") != element_type
        or accessor.get("normalized", False) is not False
        or "sparse" in accessor
    ):
        raise M2QaError(
            f"{owner} must be a non-sparse, non-normalized {element_type} accessor"
        )
    formats = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4)}
    component = formats.get(accessor.get("componentType"))
    if component is None:
        raise M2QaError(f"{owner} must use an unsigned integer component type")
    component_format, component_size = component
    component_count = {"SCALAR": 1, "VEC4": 4}[element_type]
    count = accessor.get("count")
    view_index = accessor.get("bufferView")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or view_index < 0
        or view_index >= len(views)
    ):
        raise M2QaError(f"{owner} has an invalid layout")
    view = views[view_index]
    if view.get("buffer") != 0:
        raise M2QaError(f"{owner} must reference embedded buffer 0")
    view_offset = view.get("byteOffset", 0)
    accessor_offset = accessor.get("byteOffset", 0)
    view_length = view.get("byteLength")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (view_offset, accessor_offset, view_length)
    ):
        raise M2QaError(f"{owner} offsets and lengths must be non-negative integers")
    packer = struct.Struct("<" + component_format * component_count)
    stride = view.get("byteStride", packer.size)
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < packer.size
        or stride % component_size
    ):
        raise M2QaError(f"{owner} has an invalid byte stride")
    required = accessor_offset + (count - 1) * stride + packer.size
    if required > view_length or view_offset + required > len(document.binary):
        raise M2QaError(f"{owner} extends beyond its bufferView")
    start = view_offset + accessor_offset
    result = np.empty((count, component_count), dtype=np.int64)
    for index in range(count):
        result[index] = packer.unpack_from(document.binary, start + index * stride)
    return result[:, 0] if component_count == 1 else result


def _rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise M2QaError("joint quaternion must contain four finite components")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise M2QaError("joint quaternion must already be unit normalized")
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _normalise_source_quaternion(
    quaternion_xyzw: Sequence[float], *, owner: str
) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise M2QaError(f"{owner} must contain four finite components")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > _SOURCE_QUATERNION_NORM_TOLERANCE:
        raise M2QaError(f"{owner} is not a unit quaternion")
    return quaternion / norm


def _quaternion_equivalence_error(
    left_xyzw: Sequence[float], right_xyzw: Sequence[float]
) -> float:
    left = _normalise_source_quaternion(left_xyzw, owner="source loop start rotation")
    right = _normalise_source_quaternion(right_xyzw, owner="source loop end rotation")
    return min(
        float(np.max(np.abs(left - right))),
        float(np.max(np.abs(left + right))),
    )


def _local_matrix(
    translation: Sequence[float],
    rotation: Sequence[float],
    scale: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    translation_array = np.asarray(translation, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if (
        translation_array.shape != (3,)
        or scale_array.shape != (3,)
        or not np.all(np.isfinite(translation_array))
        or not np.all(np.isfinite(scale_array))
    ):
        raise M2QaError("joint translation and scale must contain finite VEC3 values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation_matrix(rotation) @ np.diag(scale_array)
    result[:3, 3] = translation_array
    return result


def _globals_from_local_matrices(
    mapping: HabitatAssetMapping, local_matrices: Mapping[str, np.ndarray]
) -> np.ndarray:
    by_name: dict[str, np.ndarray] = {}
    ordered: list[np.ndarray] = []
    for joint in mapping.joints:
        try:
            local = local_matrices[joint.joint_id]
        except KeyError as exc:
            raise M2QaError(
                f"source action has no local transform for joint {joint.joint_id!r}"
            ) from exc
        if joint.parent_joint_id is None:
            global_matrix = local
        else:
            try:
                global_matrix = by_name[joint.parent_joint_id] @ local
            except KeyError as exc:
                raise M2QaError(
                    "mapping joints must be in parent-before-child order"
                ) from exc
        by_name[joint.joint_id] = global_matrix
        ordered.append(global_matrix)
    return np.stack(ordered)


def _joint_globals(
    mapping: HabitatAssetMapping,
    rotations_xyzw: Sequence[Sequence[float]] | None,
) -> np.ndarray:
    pose_by_name: dict[str, Sequence[float]] = {}
    if rotations_xyzw is not None:
        if len(rotations_xyzw) != len(mapping.runtime_joint_order):
            raise M2QaError("runtime pose joint count differs from the mapping")
        pose_by_name = dict(
            zip(mapping.runtime_joint_order, rotations_xyzw, strict=True)
        )
    by_name: dict[str, np.ndarray] = {}
    ordered: list[np.ndarray] = []
    for joint in mapping.joints:
        rotation = pose_by_name.get(joint.joint_id, joint.rest_rotation_xyzw)
        local = _local_matrix(joint.local_translation_m, rotation)
        if joint.parent_joint_id is None:
            global_matrix = local
        else:
            try:
                global_matrix = by_name[joint.parent_joint_id] @ local
            except KeyError as exc:
                raise M2QaError(
                    "mapping joints must be in parent-before-child order"
                ) from exc
        by_name[joint.joint_id] = global_matrix
        ordered.append(global_matrix)
    return np.stack(ordered)


def _sample_source_channel(
    channel: AnimationChannel, source_time_seconds: float
) -> np.ndarray:
    if channel.interpolation == "CUBICSPLINE":
        raise M2QaError(
            f"source action channel {channel.channel_index} uses unsupported CUBICSPLINE"
        )
    if channel.interpolation not in {"STEP", "LINEAR"}:
        raise M2QaError(
            f"source action channel {channel.channel_index} has unsupported interpolation"
        )
    timestamps = channel.timestamps_seconds
    values = channel.values
    if source_time_seconds <= timestamps[0] or len(timestamps) == 1:
        left_index = right_index = 0
        fraction = 0.0
    elif source_time_seconds >= timestamps[-1]:
        left_index = right_index = len(timestamps) - 1
        fraction = 0.0
    else:
        left_index = bisect_right(timestamps, source_time_seconds) - 1
        if channel.interpolation == "STEP":
            right_index = left_index
            fraction = 0.0
        else:
            right_index = left_index + 1
            left_time = timestamps[left_index]
            right_time = timestamps[right_index]
            fraction = (source_time_seconds - left_time) / (right_time - left_time)

    left = np.asarray(values[left_index], dtype=np.float64)
    right = np.asarray(values[right_index], dtype=np.float64)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise M2QaError("source action channel contains non-finite values")
    if channel.target_path != "rotation":
        return left + fraction * (right - left)

    first = _normalise_source_quaternion(left, owner="source action rotation key")
    second = _normalise_source_quaternion(right, owner="source action rotation key")
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
    return _normalise_source_quaternion(
        math.cos(fraction * angle) * first + math.sin(fraction * angle) * tangent,
        owner="interpolated source action rotation",
    )


def _extract_source_loop_endpoints(
    document: GlbDocument,
    actions: BakedActionSet,
    mapping: HabitatAssetMapping,
) -> dict[str, _SourceLoopEndpoints]:
    """Read and sample each GLB source action at its true clip endpoints."""

    try:
        source_actions = extract_actions(document)
    except GlbError as exc:
        raise M2QaError(f"unable to extract source GLB actions: {exc}") from exc
    source_by_name = {action.name: action for action in source_actions}
    result: dict[str, _SourceLoopEndpoints] = {}
    expected_joint_names = set(mapping.joint_order)
    for baked_action in actions.actions:
        try:
            source_action = source_by_name[baked_action.source_action_name]
        except KeyError as exc:
            raise M2QaError(
                f"source GLB lacks action {baked_action.source_action_name!r}"
            ) from exc
        source_start = min(
            channel.timestamps_seconds[0] for channel in source_action.channels
        )
        source_end = max(
            channel.timestamps_seconds[-1] for channel in source_action.channels
        )
        if not math.isclose(
            source_start,
            baked_action.clip_start_seconds,
            rel_tol=0.0,
            abs_tol=_SOURCE_CLIP_TIME_TOLERANCE_SECONDS,
        ) or not math.isclose(
            source_end,
            baked_action.clip_end_seconds,
            rel_tol=0.0,
            abs_tol=_SOURCE_CLIP_TIME_TOLERANCE_SECONDS,
        ):
            raise M2QaError(
                f"source GLB action {source_action.name!r} clip bounds differ from "
                "the baked action"
            )

        channels_by_joint: dict[str, dict[str, AnimationChannel]] = {}
        for channel in source_action.channels:
            joint_name = channel.target_node_name
            if joint_name not in expected_joint_names:
                raise M2QaError(
                    f"source GLB action {source_action.name!r} targets a non-skin "
                    f"or unnamed node: {joint_name!r}"
                )
            joint_channels = channels_by_joint.setdefault(joint_name, {})
            if channel.target_path in joint_channels:
                raise M2QaError(
                    f"source GLB action {source_action.name!r} duplicates "
                    f"{joint_name!r} {channel.target_path}"
                )
            joint_channels[channel.target_path] = channel
        missing_rotations = sorted(
            expected_joint_names
            - {
                name
                for name, channels in channels_by_joint.items()
                if "rotation" in channels
            }
        )
        if missing_rotations:
            raise M2QaError(
                f"source GLB action {source_action.name!r} lacks joint rotations: "
                f"{missing_rotations}"
            )

        endpoint_locals: list[dict[str, np.ndarray]] = [{}, {}]
        rotations: list[dict[str, np.ndarray]] = [{}, {}]
        translations: list[dict[str, np.ndarray]] = [{}, {}]
        scales: list[dict[str, np.ndarray]] = [{}, {}]
        for joint in mapping.joints:
            channels = channels_by_joint[joint.joint_id]
            for endpoint_index, source_time in enumerate((source_start, source_end)):
                translation = (
                    _sample_source_channel(channels["translation"], source_time)
                    if "translation" in channels
                    else np.asarray(joint.local_translation_m, dtype=np.float64)
                )
                rotation = _sample_source_channel(channels["rotation"], source_time)
                scale = (
                    _sample_source_channel(channels["scale"], source_time)
                    if "scale" in channels
                    else np.asarray(joint.local_scale, dtype=np.float64)
                )
                endpoint_locals[endpoint_index][joint.joint_id] = _local_matrix(
                    translation, rotation, scale
                )
                rotations[endpoint_index][joint.joint_id] = rotation
                translations[endpoint_index][joint.joint_id] = translation
                scales[endpoint_index][joint.joint_id] = scale

        maximum_rotation_error = max(
            _quaternion_equivalence_error(
                rotations[0][joint.joint_id], rotations[1][joint.joint_id]
            )
            for joint in mapping.joints
        )
        maximum_translation_error = max(
            float(
                np.linalg.norm(
                    translations[1][joint.joint_id] - translations[0][joint.joint_id]
                )
            )
            for joint in mapping.joints
        )
        maximum_scale_error = max(
            float(np.max(np.abs(scales[1][joint.joint_id] - scales[0][joint.joint_id])))
            for joint in mapping.joints
        )
        result[baked_action.semantic_action_id] = _SourceLoopEndpoints(
            source_action_name=source_action.name,
            clip_start_seconds=float(source_start),
            clip_end_seconds=float(source_end),
            start_global_joints=_globals_from_local_matrices(
                mapping, endpoint_locals[0]
            ),
            end_global_joints=_globals_from_local_matrices(mapping, endpoint_locals[1]),
            maximum_joint_rotation_error=maximum_rotation_error,
            maximum_joint_translation_error_m=maximum_translation_error,
            maximum_joint_scale_error=maximum_scale_error,
        )
    return result


def _inverse_bind_matrices(document: GlbDocument) -> np.ndarray:
    skins = extract_skins(document)
    if len(skins) != 1 or skins[0].inverse_bind_matrices is None:
        raise M2QaError("M2 QA requires exactly one skin with inverse bind matrices")
    raw = skins[0].inverse_bind_matrices
    matrices = np.stack(
        [np.asarray(value, dtype=np.float64).reshape(4, 4).T for value in raw]
    )
    if not np.all(np.isfinite(matrices)):
        raise M2QaError("inverse bind matrices contain non-finite values")
    return matrices


def _primitive_data(document: GlbDocument, *, joint_count: int) -> list[_PrimitiveData]:
    meshes = _objects(document.json.get("meshes", []), owner="meshes")
    if len(meshes) != 1:
        raise M2QaError(f"M2 canary requires exactly one mesh, found {len(meshes)}")
    primitives = _objects(meshes[0].get("primitives", []), owner="mesh.primitives")
    if not primitives:
        raise M2QaError("M2 mesh contains no primitives")
    result: list[_PrimitiveData] = []
    for primitive_index, primitive in enumerate(primitives):
        if primitive.get("mode", _TRIANGLES_MODE) != _TRIANGLES_MODE:
            raise M2QaError("M2 mesh primitives must use TRIANGLES mode")
        if "targets" in primitive:
            raise M2QaError("morph targets violate the M2 no-mouth boundary")
        attributes = primitive.get("attributes")
        required = {"POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
        if not isinstance(attributes, dict) or not required.issubset(attributes):
            raise M2QaError(
                f"mesh primitive {primitive_index} lacks required attributes"
            )
        positions_accessor = decode_accessor(document, attributes["POSITION"])
        normals_accessor = decode_accessor(document, attributes["NORMAL"])
        texcoords_accessor = decode_accessor(document, attributes["TEXCOORD_0"])
        weights_accessor = decode_accessor(document, attributes["WEIGHTS_0"])
        if (
            positions_accessor.element_type != "VEC3"
            or normals_accessor.element_type != "VEC3"
            or texcoords_accessor.element_type != "VEC2"
            or weights_accessor.element_type != "VEC4"
        ):
            raise M2QaError("mesh attribute accessor types do not match M2")
        positions = np.asarray(positions_accessor.values, dtype=np.float64)
        normals = np.asarray(normals_accessor.values, dtype=np.float64)
        texcoords = np.asarray(texcoords_accessor.values, dtype=np.float64)
        weights = np.asarray(weights_accessor.values, dtype=np.float64)
        joints = _decode_unsigned_accessor(
            document,
            attributes["JOINTS_0"],
            element_type="VEC4",
            owner=f"mesh primitive {primitive_index} JOINTS_0",
        )
        indices = _decode_unsigned_accessor(
            document,
            primitive.get("indices"),
            element_type="SCALAR",
            owner=f"mesh primitive {primitive_index} indices",
        )
        vertex_count = len(positions)
        if any(
            len(value) != vertex_count
            for value in (normals, texcoords, weights, joints)
        ):
            raise M2QaError("mesh attribute vertex counts differ")
        if len(indices) % 3 or np.any(indices >= vertex_count):
            raise M2QaError("triangle indices are malformed or out of range")
        if int(np.max(joints)) >= joint_count:
            raise M2QaError("JOINTS_0 references an out-of-range skin joint")
        if np.min(weights) < -1.0e-7:
            raise M2QaError("skin weights contain a negative value")
        weight_sum_error = float(np.max(np.abs(np.sum(weights, axis=1) - 1.0)))
        if weight_sum_error > _MAX_WEIGHT_SUM_ERROR:
            raise M2QaError("skin weights do not sum to one")
        normal_lengths = np.linalg.norm(normals, axis=1)
        if np.any(normal_lengths <= 1.0e-12):
            raise M2QaError("mesh normals contain a zero vector")
        if not all(
            np.all(np.isfinite(value))
            for value in (positions, normals, texcoords, weights)
        ):
            raise M2QaError("mesh attributes contain non-finite values")
        result.append(
            _PrimitiveData(
                primitive_index=primitive_index,
                positions=positions,
                normals=normals,
                texcoords=texcoords,
                joint_ordinals=joints,
                weights=weights,
                triangles=indices.reshape(-1, 3),
            )
        )
    return result


def _skin_vertices(
    primitive: _PrimitiveData,
    global_joints: np.ndarray,
    inverse_bind_matrices: np.ndarray,
) -> np.ndarray:
    joint_matrices = global_joints @ inverse_bind_matrices
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
    if not np.all(np.isfinite(vertices)):
        raise M2QaError("CPU-skinned vertices contain non-finite values")
    return vertices


def _triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points = vertices[triangles]
    return 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        axis=1,
    )


def _outside_bbox_distance(
    points: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> float:
    below = np.maximum(minimum - points, 0.0)
    above = np.maximum(points - maximum, 0.0)
    return float(np.max(np.linalg.norm(below + above, axis=1)))


def _quaternion_excursion_degrees(values: Sequence[Sequence[float]]) -> float:
    array = np.asarray(values, dtype=np.float64)
    reference = array[0] / np.linalg.norm(array[0])
    normalized = array / np.linalg.norm(array, axis=1)[:, None]
    dots = np.clip(np.abs(normalized @ reference), 0.0, 1.0)
    return float(np.degrees(2.0 * np.max(np.arccos(dots))))


def _semantic_motion_metrics(
    mapping: HabitatAssetMapping,
    actions: BakedActionSet,
    semantic_joint_map: Mapping[str, str],
) -> tuple[dict[str, Any], list[str]]:
    required = {
        "paw_front_left",
        "paw_front_right",
        "paw_hind_left",
        "paw_hind_right",
        "muzzle",
    }
    if not required.issubset(semantic_joint_map):
        raise M2QaError("semantic_joint_map lacks paws or muzzle")
    joint_index = {joint.joint_id: index for index, joint in enumerate(mapping.joints)}
    if any(name not in joint_index for name in semantic_joint_map.values()):
        raise M2QaError("semantic_joint_map references an unknown joint")
    actor_from_skin = np.asarray(mapping.actor_from_skin_root, dtype=np.float64)
    clips: dict[str, Any] = {}
    for action in actions.actions:
        positions: dict[str, list[np.ndarray]] = {
            semantic_id: [] for semantic_id in required
        }
        for pose in action.rotations_xyzw:
            global_joints = _joint_globals(mapping, pose)
            actor_joints = actor_from_skin @ global_joints
            for semantic_id in required:
                matrix = actor_joints[joint_index[semantic_joint_map[semantic_id]]]
                positions[semantic_id].append(matrix[:3, 3])
        clips[action.semantic_action_id] = {
            semantic_id: {
                "minimum_actor_m": np.min(values, axis=0).tolist(),
                "maximum_actor_m": np.max(values, axis=0).tolist(),
                "range_actor_m": np.ptp(values, axis=0).tolist(),
                "path_length_m": float(
                    np.sum(np.linalg.norm(np.diff(values, axis=0), axis=1))
                ),
            }
            for semantic_id, values in positions.items()
        }

    walk = clips["walk"]
    front_forward = np.mean(
        [
            walk[name]["range_actor_m"][0]
            for name in ("paw_front_left", "paw_front_right")
        ]
    )
    hind_forward = np.mean(
        [walk[name]["range_actor_m"][0] for name in ("paw_hind_left", "paw_hind_right")]
    )
    hind_lateral = np.mean(
        [walk[name]["range_actor_m"][2] for name in ("paw_hind_left", "paw_hind_right")]
    )
    measured_hind_limitation = bool(
        hind_forward < 0.25 * front_forward and hind_lateral > hind_forward
    )
    limitations = [
        "Known legacy gait limitation carried forward for review: the hind legs "
        "show limited whole-leg articulation and motion can be concentrated at the "
        "toe/terminal joints. Temporary research-canary use does not claim that this "
        "gait defect is fixed or that the asset is qualified."
    ]
    if measured_hind_limitation:
        limitations.append(
            "The current terminal-joint metrics also detect much less hind-paw "
            "forward excursion than front-paw excursion, with hind motion dominated "
            "by the lateral/toe axis."
        )
    return (
        {
            "source_facing_axis_in_actor_frame": "+X",
            "actor_up_axis": "+Y",
            "actor_lateral_axis": "+Z",
            "clips": clips,
            "walking_summary": {
                "mean_front_paw_forward_range_m": float(front_forward),
                "mean_hind_paw_forward_range_m": float(hind_forward),
                "mean_hind_paw_lateral_range_m": float(hind_lateral),
                "legacy_hind_gait_metric_triggered": measured_hind_limitation,
            },
        },
        limitations,
    )


def audit_m2_candidate(
    document: GlbDocument,
    actions: BakedActionSet,
    mapping: HabitatAssetMapping,
    *,
    semantic_joint_map: Mapping[str, str],
) -> M2AutomaticQa:
    """Run the bounded automatic gates and return three package-role reports."""

    if document.sha256 != actions.source_glb_sha256:
        raise M2QaError("baked actions are not bound to the candidate GLB")
    if document.sha256 != mapping.source_glb_sha256:
        raise M2QaError("Habitat mapping is not bound to the candidate GLB")
    if actions.runtime_joint_order != mapping.runtime_joint_order:
        raise M2QaError("action and Habitat mapping joint orders differ")
    inverse_bind = _inverse_bind_matrices(document)
    if len(inverse_bind) != len(mapping.joints):
        raise M2QaError("inverse bind matrix count differs from joint count")
    primitives = _primitive_data(document, joint_count=len(mapping.joints))
    rest_globals = _joint_globals(mapping, None)
    bind_matrices = rest_globals @ inverse_bind
    maximum_bind_closure = float(
        np.max(np.abs(bind_matrices - np.eye(4, dtype=np.float64)))
    )
    if maximum_bind_closure > _MAX_BIND_CLOSURE_ERROR_M:
        raise M2QaError("canonical joint/inverse-bind closure exceeds threshold")

    static_primitive_records: list[dict[str, Any]] = []
    topology_payload: list[dict[str, Any]] = []
    uv_payload: list[dict[str, Any]] = []
    weights_payload: list[dict[str, Any]] = []
    rest_minima: list[np.ndarray] = []
    rest_maxima: list[np.ndarray] = []
    rest_diagonal = 0.0
    for primitive in primitives:
        repeated = np.any(
            (primitive.triangles[:, 0] == primitive.triangles[:, 1])
            | (primitive.triangles[:, 1] == primitive.triangles[:, 2])
            | (primitive.triangles[:, 0] == primitive.triangles[:, 2])
        )
        if repeated:
            raise M2QaError("topology contains a triangle with repeated indices")
        areas = _triangle_areas(primitive.positions, primitive.triangles)
        minimum_area = float(np.min(areas))
        if minimum_area <= _MIN_TRIANGLE_AREA_M2:
            raise M2QaError("rest topology contains a collapsed triangle")
        skinned_bind = _skin_vertices(primitive, rest_globals, inverse_bind)
        bind_vertex_error = float(np.max(np.abs(skinned_bind - primitive.positions)))
        if bind_vertex_error > _MAX_BIND_CLOSURE_ERROR_M:
            raise M2QaError("weighted bind skinning does not reproduce POSITION")
        minimum = np.min(primitive.positions, axis=0)
        maximum = np.max(primitive.positions, axis=0)
        diagonal = float(np.linalg.norm(maximum - minimum))
        rest_diagonal = max(rest_diagonal, diagonal)
        rest_minima.append(minimum)
        rest_maxima.append(maximum)
        weight_sum_error = float(
            np.max(np.abs(np.sum(primitive.weights, axis=1) - 1.0))
        )
        static_primitive_records.append(
            {
                "primitive_index": primitive.primitive_index,
                "vertex_count": len(primitive.positions),
                "triangle_count": len(primitive.triangles),
                "minimum_triangle_area_m2": minimum_area,
                "bbox_min_m": minimum.tolist(),
                "bbox_max_m": maximum.tolist(),
                "bbox_diagonal_m": diagonal,
                "minimum_weight": float(np.min(primitive.weights)),
                "maximum_weight_sum_error": weight_sum_error,
                "maximum_weighted_bind_vertex_error_m": bind_vertex_error,
            }
        )
        topology_payload.append(
            {
                "primitive_index": primitive.primitive_index,
                "vertex_count": len(primitive.positions),
                "triangles": primitive.triangles.tolist(),
            }
        )
        uv_payload.append(
            {
                "primitive_index": primitive.primitive_index,
                "texcoord_0": primitive.texcoords.tolist(),
            }
        )
        weights_payload.append(
            {
                "primitive_index": primitive.primitive_index,
                "joint_ordinals": primitive.joint_ordinals.tolist(),
                "weights": primitive.weights.tolist(),
            }
        )

    global_minimum = np.min(rest_minima, axis=0)
    global_maximum = np.max(rest_maxima, axis=0)
    rest_landmarks = rest_globals[:, :3, 3]
    rest_landmark_outside = _outside_bbox_distance(
        rest_landmarks, global_minimum, global_maximum
    )
    if rest_landmark_outside > _LANDMARK_MARGIN_M:
        raise M2QaError("a rest joint landmark lies outside the mesh bbox margin")
    topology_sha256 = canonical_json_sha256(
        {"schema": "avengine_m2_topology_identity_v1", "primitives": topology_payload}
    )
    uv_sha256 = canonical_json_sha256(
        {"schema": "avengine_m2_uv_identity_v1", "primitives": uv_payload}
    )
    weights_sha256 = canonical_json_sha256(
        {"schema": "avengine_m2_weight_identity_v1", "primitives": weights_payload}
    )
    static_geometry = {
        "schema": "avengine_m2_static_geometry_qa_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_glb_sha256": document.sha256,
        "joint_count": len(mapping.joints),
        "primitive_count": len(primitives),
        "primitives": static_primitive_records,
        "maximum_bind_closure_error": maximum_bind_closure,
        "maximum_rest_landmark_bbox_outside_distance_m": rest_landmark_outside,
        "topology_sha256": topology_sha256,
        "uv_sha256": uv_sha256,
        "weights_sha256": weights_sha256,
        "thresholds": {
            "maximum_weight_sum_error": _MAX_WEIGHT_SUM_ERROR,
            "maximum_bind_closure_error_m": _MAX_BIND_CLOSURE_ERROR_M,
            "minimum_triangle_area_m2_exclusive": _MIN_TRIANGLE_AREA_M2,
            "maximum_landmark_bbox_outside_distance_m": _LANDMARK_MARGIN_M,
        },
        "notes": [
            "This automatic gate proves bounded geometry/skin consistency, not visual plausibility."
        ],
    }

    source_loop_endpoints = _extract_source_loop_endpoints(document, actions, mapping)
    per_action: list[dict[str, Any]] = []
    overall_maximum_step = 0.0
    overall_maximum_source_loop_error = 0.0
    overall_minimum_area = math.inf
    overall_landmark_outside = 0.0
    for action in actions.actions:
        frames: list[np.ndarray] = []
        minimum_areas: list[float] = []
        landmark_outside: list[float] = []
        for pose in action.rotations_xyzw:
            globals_at_frame = _joint_globals(mapping, pose)
            deformed_parts = [
                _skin_vertices(primitive, globals_at_frame, inverse_bind)
                for primitive in primitives
            ]
            combined = np.concatenate(deformed_parts, axis=0)
            minimum = np.min(combined, axis=0)
            maximum = np.max(combined, axis=0)
            outside = _outside_bbox_distance(
                globals_at_frame[:, :3, 3], minimum, maximum
            )
            landmark_outside.append(outside)
            frame_minimum_area = min(
                float(np.min(_triangle_areas(vertices, primitive.triangles)))
                for vertices, primitive in zip(deformed_parts, primitives, strict=True)
            )
            minimum_areas.append(frame_minimum_area)
            frames.append(combined)
        steps = [
            float(np.max(np.linalg.norm(right - left, axis=1)))
            for left, right in zip(frames, frames[1:], strict=False)
        ]
        maximum_step = max(steps, default=0.0)
        minimum_area = min(minimum_areas)
        maximum_outside = max(landmark_outside)
        if minimum_area <= _MIN_TRIANGLE_AREA_M2:
            raise M2QaError(f"{action.semantic_action_id} collapses a triangle")
        if maximum_outside > _LANDMARK_MARGIN_M:
            raise M2QaError(
                f"{action.semantic_action_id} moves a joint landmark outside the mesh"
            )
        if maximum_step > _MAX_VERTEX_STEP_REST_DIAGONAL_RATIO * rest_diagonal:
            raise M2QaError(
                f"{action.semantic_action_id} has an excessive one-frame deformation"
            )
        source_endpoints = source_loop_endpoints[action.semantic_action_id]
        source_start_parts = [
            _skin_vertices(
                primitive, source_endpoints.start_global_joints, inverse_bind
            )
            for primitive in primitives
        ]
        source_end_parts = [
            _skin_vertices(primitive, source_endpoints.end_global_joints, inverse_bind)
            for primitive in primitives
        ]
        source_loop_boundary_error = max(
            float(np.max(np.linalg.norm(end - start, axis=1)))
            for start, end in zip(source_start_parts, source_end_parts, strict=True)
        )
        if (
            source_loop_boundary_error > _MAX_LOOP_CLOSURE_ERROR_M
            or source_endpoints.maximum_joint_translation_error_m
            > _MAX_LOOP_CLOSURE_ERROR_M
            or source_endpoints.maximum_joint_rotation_error > _MAX_LOOP_ROTATION_ERROR
            or source_endpoints.maximum_joint_scale_error > _MAX_LOOP_SCALE_ERROR
        ):
            raise M2QaError(
                f"source GLB action {action.source_action_name!r} does not close at "
                "its true loop endpoint"
            )
        per_action.append(
            {
                "semantic_action_id": action.semantic_action_id,
                "source_action_name": action.source_action_name,
                "sample_count": action.sample_count,
                "minimum_triangle_area_m2": minimum_area,
                "maximum_joint_landmark_bbox_outside_distance_m": maximum_outside,
                "maximum_vertex_step_m": maximum_step,
                "maximum_vertex_step_rest_diagonal_ratio": maximum_step / rest_diagonal,
                "source_clip_start_seconds": source_endpoints.clip_start_seconds,
                "source_clip_end_seconds": source_endpoints.clip_end_seconds,
                "source_loop_endpoint_vertex_error_m": source_loop_boundary_error,
                "source_loop_endpoint_maximum_joint_rotation_error": (
                    source_endpoints.maximum_joint_rotation_error
                ),
                "source_loop_endpoint_maximum_joint_translation_error_m": (
                    source_endpoints.maximum_joint_translation_error_m
                ),
                "source_loop_endpoint_maximum_joint_scale_error": (
                    source_endpoints.maximum_joint_scale_error
                ),
            }
        )
        overall_maximum_step = max(overall_maximum_step, maximum_step)
        overall_maximum_source_loop_error = max(
            overall_maximum_source_loop_error, source_loop_boundary_error
        )
        overall_minimum_area = min(overall_minimum_area, minimum_area)
        overall_landmark_outside = max(overall_landmark_outside, maximum_outside)

    baked_actions_sha256 = baked_actions_content_sha256(actions)
    deformation = {
        "schema": "avengine_m2_deformation_qa_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_glb_sha256": document.sha256,
        "baked_actions_sha256": baked_actions_sha256,
        "rest_bbox_diagonal_m": rest_diagonal,
        "maximum_vertex_step_m": overall_maximum_step,
        "maximum_source_loop_endpoint_vertex_error_m": (
            overall_maximum_source_loop_error
        ),
        "minimum_animated_triangle_area_m2": overall_minimum_area,
        "maximum_joint_landmark_bbox_outside_distance_m": overall_landmark_outside,
        "actions": per_action,
        "thresholds": {
            "maximum_vertex_step_rest_diagonal_ratio": _MAX_VERTEX_STEP_REST_DIAGONAL_RATIO,
            "maximum_source_loop_endpoint_vertex_error_m": _MAX_LOOP_CLOSURE_ERROR_M,
            "maximum_source_loop_endpoint_joint_translation_error_m": _MAX_LOOP_CLOSURE_ERROR_M,
            "maximum_source_loop_endpoint_joint_rotation_error": _MAX_LOOP_ROTATION_ERROR,
            "maximum_source_loop_endpoint_joint_scale_error": _MAX_LOOP_SCALE_ERROR,
            "minimum_triangle_area_m2_exclusive": _MIN_TRIANGLE_AREA_M2,
            "maximum_landmark_bbox_outside_distance_m": _LANDMARK_MARGIN_M,
        },
        "notes": [
            "The baked loops are endpoint-exclusive; loop closure was measured independently from the true first and final source-GLB action states.",
            "Visual mesh/action plausibility remains a separate human-review gate.",
        ],
    }

    mouth_name = semantic_joint_map["muzzle"]
    try:
        mouth_index = actions.runtime_joint_order.index(mouth_name)
    except ValueError as exc:
        raise M2QaError("muzzle joint is not a runtime action joint") from exc
    mouth_excursions = {
        action.semantic_action_id: _quaternion_excursion_degrees(
            [frame[mouth_index] for frame in action.rotations_xyzw]
        )
        for action in actions.actions
    }
    maximum_mouth_excursion = max(mouth_excursions.values())
    if maximum_mouth_excursion > _MOUTH_ANGLE_TOLERANCE_DEGREES:
        raise M2QaError("muzzle joint rotation violates the no-mouth M2 contract")
    motion_metrics, limitations = _semantic_motion_metrics(
        mapping, actions, semantic_joint_map
    )
    animation = {
        "schema": "avengine_m2_animation_qa_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_glb_sha256": document.sha256,
        "baked_actions_sha256": baked_actions_sha256,
        "sample_rate_hz": actions.sample_rate_hz,
        "time_base_hz": actions.time_base_hz,
        "runtime_joint_order": list(actions.runtime_joint_order),
        "actions": [
            {
                "semantic_action_id": action.semantic_action_id,
                "source_action_name": action.source_action_name,
                "sample_count": action.sample_count,
                "loop_duration_ticks": action.loop_duration_ticks,
                "first_sample_tick": action.sample_ticks[0],
                "last_sample_tick": action.sample_ticks[-1],
            }
            for action in actions.actions
        ],
        "mouth": {
            "joint_id": mouth_name,
            "open_ratio_policy": "exactly_zero",
            "rotation_excursion_degrees_by_action": mouth_excursions,
            "maximum_rotation_excursion_degrees": maximum_mouth_excursion,
            "threshold_degrees": _MOUTH_ANGLE_TOLERANCE_DEGREES,
        },
        "semantic_terminal_motion": motion_metrics,
        "known_limitations": limitations,
        "human_visual_review_required": True,
        "notes": [
            "Automatic pass covers deterministic playback and numerical safety only.",
            "Known gait limitations remain visible and require the hash-bound human review before canary qualification.",
        ],
    }
    return M2AutomaticQa(
        static_geometry=static_geometry,
        deformation=deformation,
        animation=animation,
    )


__all__ = ["M2AutomaticQa", "M2QaError", "audit_m2_candidate"]
