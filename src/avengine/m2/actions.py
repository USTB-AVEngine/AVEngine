"""Deterministic, research-only M2 articulated action baking.

The baker intentionally accepts a narrow input contract: one GLB skin, the
two explicitly mapped source clips ``Idle`` and ``Walking``, and one rotation
channel for every skin joint.  The skin root is validated but omitted from the
runtime quaternion array because Habitat applies it through the articulated
object transform.  Every remaining quaternion is an absolute child-local
glTF/Habitat spherical rotation in ``xyzw`` order, not a bind-relative delta.

This module does not infer contacts, qualify an asset, or write a package
manifest.  Its output is always a research candidate.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from avengine.m2.glb import (
    AnimationAction,
    AnimationChannel,
    GlbDocument,
    GlbError,
    JointNode,
    extract_actions,
    extract_skins,
)


SAMPLE_RATE_HZ = 15
TIME_BASE_HZ = 48_000
TICKS_PER_SAMPLE = TIME_BASE_HZ // SAMPLE_RATE_HZ
SEMANTIC_ACTION_SOURCES: tuple[tuple[str, str], ...] = (
    ("idle", "Idle"),
    ("walk", "Walking"),
)
BAKED_ACTIONS_NPZ_SCHEMA = "avengine_m2_baked_actions_v1"

_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_QUATERNION_INPUT_NORM_TOLERANCE = 1.0e-5
_QUATERNION_OUTPUT_NORM_TOLERANCE = 1.0e-12
_STATIC_CHANNEL_TOLERANCE = 5.0e-5
_ROOT_ROTATION_TOLERANCE = 5.0e-5
_TICK_ROUNDING_TOLERANCE = 1.0e-2
_HEMISPHERE_ZERO_TOLERANCE = 1.0e-15

QuaternionXYZW = tuple[float, float, float, float]


class ActionBakeError(ValueError):
    """The source animation or baked artifact violates the M2 action boundary."""


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionBakeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ActionBakeError(f"{name} must be a finite number")
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise ActionBakeError(f"{name} must use canonical positive zero")
    return result


def _quaternion_sign_component(quaternion: Sequence[float]) -> float:
    scalar = float(quaternion[3])
    if math.isclose(scalar, 0.0, rel_tol=0.0, abs_tol=_HEMISPHERE_ZERO_TOLERANCE):
        for component in quaternion[:3]:
            number = float(component)
            if not math.isclose(
                number, 0.0, rel_tol=0.0, abs_tol=_HEMISPHERE_ZERO_TOLERANCE
            ):
                return number
    return scalar


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _normalise_quaternion(
    value: Sequence[float],
    *,
    owner: str,
    validate_input_norm: bool,
) -> QuaternionXYZW:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ActionBakeError(f"{owner} must contain four finite xyzw components")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ActionBakeError(f"{owner} cannot be a zero quaternion")
    if validate_input_norm and abs(norm - 1.0) > _QUATERNION_INPUT_NORM_TOLERANCE:
        raise ActionBakeError(
            f"{owner} must already be a unit quaternion; norm={norm:.17g}"
        )
    quaternion /= norm
    if _quaternion_sign_component(quaternion) < 0.0:
        quaternion = -quaternion
    return tuple(_canonical_float(component) for component in quaternion)  # type: ignore[return-value]


def _quaternion_equivalence_error(
    value: Sequence[float], reference: Sequence[float]
) -> float:
    left = np.asarray(value, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return min(
        float(np.max(np.abs(left - right))),
        float(np.max(np.abs(left + right))),
    )


@dataclass(frozen=True)
class BakedActionClip:
    """One endpoint-exclusive loop sampled on the M2 48 kHz clock."""

    semantic_action_id: str
    source_action_name: str
    clip_start_seconds: float
    clip_end_seconds: float
    loop_duration_ticks: int
    sample_ticks: tuple[int, ...]
    source_times_seconds: tuple[float, ...]
    rotations_xyzw: tuple[tuple[QuaternionXYZW, ...], ...]

    @property
    def sample_count(self) -> int:
        return len(self.sample_ticks)


@dataclass(frozen=True)
class BakedActionSet:
    """The two required M2 action loops and their immutable source identity."""

    source_glb_sha256: str
    runtime_joint_order: tuple[str, ...]
    actions: tuple[BakedActionClip, ...]
    sample_rate_hz: int = SAMPLE_RATE_HZ
    time_base_hz: int = TIME_BASE_HZ
    qualification_state: str = field(default="research_candidate", init=False)
    qualification_claim: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _validate_baked_action_set(self)

    def action(self, semantic_action_id: str) -> BakedActionClip:
        """Return one explicitly mapped action without a fallback heuristic."""

        for action in self.actions:
            if action.semantic_action_id == semantic_action_id:
                return action
        raise KeyError(f"unknown semantic action: {semantic_action_id!r}")


def _validate_baked_action_set(value: BakedActionSet) -> None:
    if not _is_lower_sha256(value.source_glb_sha256):
        raise ActionBakeError("source_glb_sha256 must be lowercase SHA-256")
    if value.sample_rate_hz != SAMPLE_RATE_HZ:
        raise ActionBakeError(f"sample_rate_hz must equal {SAMPLE_RATE_HZ}")
    if value.time_base_hz != TIME_BASE_HZ:
        raise ActionBakeError(f"time_base_hz must equal {TIME_BASE_HZ}")
    if TIME_BASE_HZ % SAMPLE_RATE_HZ:
        raise ActionBakeError("time base must divide exactly by sample rate")
    if (
        not isinstance(value.runtime_joint_order, tuple)
        or not value.runtime_joint_order
        or any(
            not isinstance(name, str) or not name for name in value.runtime_joint_order
        )
        or len(set(value.runtime_joint_order)) != len(value.runtime_joint_order)
    ):
        raise ActionBakeError(
            "runtime_joint_order must be a non-empty tuple of unique names"
        )
    if not isinstance(value.actions, tuple) or any(
        not isinstance(action, BakedActionClip) for action in value.actions
    ):
        raise ActionBakeError("actions must be an immutable tuple of baked clips")
    expected_actions = SEMANTIC_ACTION_SOURCES
    actual_actions = tuple(
        (action.semantic_action_id, action.source_action_name)
        for action in value.actions
    )
    if actual_actions != expected_actions:
        raise ActionBakeError(
            "actions must follow the exact idle->Idle, walk->Walking mapping"
        )

    joint_count = len(value.runtime_joint_order)
    for action in value.actions:
        start = _finite_float(
            action.clip_start_seconds,
            name=f"{action.semantic_action_id}.clip_start_seconds",
        )
        end = _finite_float(
            action.clip_end_seconds,
            name=f"{action.semantic_action_id}.clip_end_seconds",
        )
        if start < 0.0 or end <= start:
            raise ActionBakeError(
                f"{action.semantic_action_id} clip bounds must satisfy 0 <= start < end"
            )
        if (
            isinstance(action.loop_duration_ticks, bool)
            or not isinstance(action.loop_duration_ticks, int)
            or action.loop_duration_ticks <= 0
            or action.loop_duration_ticks % TICKS_PER_SAMPLE
        ):
            raise ActionBakeError(
                f"{action.semantic_action_id}.loop_duration_ticks must be a positive "
                f"multiple of {TICKS_PER_SAMPLE}"
            )
        exact_duration_ticks = (end - start) * TIME_BASE_HZ
        if (
            abs(exact_duration_ticks - action.loop_duration_ticks)
            > _TICK_ROUNDING_TOLERANCE
        ):
            raise ActionBakeError(
                f"{action.semantic_action_id}.loop_duration_ticks does not match "
                "its explicit clip bounds"
            )
        expected_ticks = tuple(range(0, action.loop_duration_ticks, TICKS_PER_SAMPLE))
        if (
            not isinstance(action.sample_ticks, tuple)
            or any(
                isinstance(tick, bool) or not isinstance(tick, int)
                for tick in action.sample_ticks
            )
            or action.sample_ticks != expected_ticks
        ):
            raise ActionBakeError(
                f"{action.semantic_action_id}.sample_ticks must be endpoint-exclusive "
                f"{TICKS_PER_SAMPLE}-tick samples"
            )
        if not isinstance(action.source_times_seconds, tuple) or len(
            action.source_times_seconds
        ) != len(expected_ticks):
            raise ActionBakeError(
                f"{action.semantic_action_id} source time count differs from samples"
            )
        if not isinstance(action.rotations_xyzw, tuple) or len(
            action.rotations_xyzw
        ) != len(expected_ticks):
            raise ActionBakeError(
                f"{action.semantic_action_id} rotation frame count differs from samples"
            )
        for sample_index, (tick, source_time, frame) in enumerate(
            zip(
                expected_ticks,
                action.source_times_seconds,
                action.rotations_xyzw,
                strict=True,
            )
        ):
            expected_time = start + tick / TIME_BASE_HZ
            actual_time = _finite_float(
                source_time,
                name=(
                    f"{action.semantic_action_id}.source_times_seconds[{sample_index}]"
                ),
            )
            if not math.isclose(
                actual_time, expected_time, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ActionBakeError(
                    f"{action.semantic_action_id} source time is off the integer tick grid"
                )
            if actual_time >= end:
                raise ActionBakeError(
                    f"{action.semantic_action_id} stores the duplicate loop endpoint"
                )
            if not isinstance(frame, tuple) or len(frame) != joint_count:
                raise ActionBakeError(
                    f"{action.semantic_action_id} frame {sample_index} must contain "
                    f"{joint_count} joint quaternions"
                )
            for joint_index, quaternion in enumerate(frame):
                if not isinstance(quaternion, tuple) or len(quaternion) != 4:
                    raise ActionBakeError(
                        f"{action.semantic_action_id} frame {sample_index} joint "
                        f"{joint_index} must be an immutable xyzw quaternion"
                    )
                array = np.asarray(quaternion, dtype=np.float64)
                if not np.all(np.isfinite(array)):
                    raise ActionBakeError("baked quaternion contains non-finite values")
                if any(
                    component == 0.0 and math.copysign(1.0, component) < 0.0
                    for component in array
                ):
                    raise ActionBakeError(
                        "baked quaternion must use canonical positive zero"
                    )
                norm = float(np.linalg.norm(array))
                if abs(norm - 1.0) > _QUATERNION_OUTPUT_NORM_TOLERANCE:
                    raise ActionBakeError(
                        "baked quaternion must be float64 unit normalized"
                    )
                if _quaternion_sign_component(array) < 0.0:
                    raise ActionBakeError(
                        "baked quaternion must use the canonical hemisphere"
                    )


@dataclass(frozen=True)
class _RotationTrack:
    interpolation: str
    timestamps_seconds: tuple[float, ...]
    quaternions_xyzw: tuple[QuaternionXYZW, ...]


def _prepare_rotation_track(channel: AnimationChannel) -> _RotationTrack:
    if channel.interpolation == "CUBICSPLINE":
        raise ActionBakeError(
            f"{channel.target_node_name or channel.target_node_index} rotation uses "
            "unsupported CUBICSPLINE interpolation"
        )
    if channel.interpolation not in {"STEP", "LINEAR"}:
        raise ActionBakeError(
            f"unsupported rotation interpolation: {channel.interpolation!r}"
        )
    quaternions = [
        _normalise_quaternion(
            value,
            owner=(
                f"{channel.target_node_name or channel.target_node_index} rotation "
                f"key {index}"
            ),
            validate_input_norm=True,
        )
        for index, value in enumerate(channel.values)
    ]
    if channel.interpolation == "LINEAR":
        continuous: list[QuaternionXYZW] = []
        for quaternion in quaternions:
            candidate = np.asarray(quaternion, dtype=np.float64)
            if (
                continuous
                and float(
                    np.dot(np.asarray(continuous[-1], dtype=np.float64), candidate)
                )
                < 0.0
            ):
                candidate = -candidate
            continuous.append(tuple(float(item) for item in candidate))  # type: ignore[arg-type]
        quaternions = continuous
    return _RotationTrack(
        interpolation=channel.interpolation,
        timestamps_seconds=channel.timestamps_seconds,
        quaternions_xyzw=tuple(quaternions),
    )


def _slerp_shortest(
    left: Sequence[float], right: Sequence[float], fraction: float
) -> QuaternionXYZW:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    # Resolve the great-circle tangent explicitly.  This remains stable for
    # small angles without replacing SLERP with an nlerp approximation.
    orthogonal = second - dot * first
    sine = float(np.linalg.norm(orthogonal))
    if sine <= 8.0 * np.finfo(np.float64).eps:
        result = first
    else:
        tangent = orthogonal / sine
        angle = math.atan2(sine, dot)
        result = (
            math.cos(fraction * angle) * first + math.sin(fraction * angle) * tangent
        )
    return _normalise_quaternion(
        result,
        owner="interpolated rotation",
        validate_input_norm=False,
    )


def _sample_rotation(track: _RotationTrack, source_time: float) -> QuaternionXYZW:
    timestamps = track.timestamps_seconds
    values = track.quaternions_xyzw
    if source_time <= timestamps[0] or len(timestamps) == 1:
        return _normalise_quaternion(
            values[0], owner="sampled rotation", validate_input_norm=False
        )
    if source_time >= timestamps[-1]:
        return _normalise_quaternion(
            values[-1], owner="sampled rotation", validate_input_norm=False
        )
    left_index = bisect_right(timestamps, source_time) - 1
    if track.interpolation == "STEP":
        return _normalise_quaternion(
            values[left_index], owner="sampled STEP rotation", validate_input_norm=False
        )
    left_time = timestamps[left_index]
    right_time = timestamps[left_index + 1]
    fraction = (source_time - left_time) / (right_time - left_time)
    return _slerp_shortest(values[left_index], values[left_index + 1], fraction)


def _validate_static_channel(channel: AnimationChannel, joint: JointNode) -> None:
    if channel.interpolation == "CUBICSPLINE":
        raise ActionBakeError(
            f"{joint.name} {channel.target_path} uses unsupported CUBICSPLINE"
        )
    values = np.asarray(channel.values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ActionBakeError(f"{joint.name} {channel.target_path} is non-finite")
    temporal_error = float(np.max(np.abs(values - values[0])))
    default = np.asarray(
        joint.local_trs.translation
        if channel.target_path == "translation"
        else joint.local_trs.scale,
        dtype=np.float64,
    )
    default_error = float(np.max(np.abs(values[0] - default)))
    if max(temporal_error, default_error) > _STATIC_CHANNEL_TOLERANCE:
        raise ActionBakeError(
            f"{joint.name} {channel.target_path} must be constant at its node default"
        )


def _action_tracks(
    action: AnimationAction,
    *,
    joints: tuple[JointNode, ...],
    root: JointNode,
) -> tuple[dict[int, _RotationTrack], float, float]:
    joints_by_node = {joint.node_index: joint for joint in joints}
    rotation_channels: dict[int, AnimationChannel] = {}
    all_timestamps: list[float] = []
    for channel in action.channels:
        joint = joints_by_node.get(channel.target_node_index)
        if joint is None:
            raise ActionBakeError(
                f"action {action.name!r} targets non-skin node "
                f"{channel.target_node_index}"
            )
        if channel.interpolation == "CUBICSPLINE":
            raise ActionBakeError(
                f"action {action.name!r} uses unsupported CUBICSPLINE"
            )
        all_timestamps.extend(channel.timestamps_seconds)
        if channel.target_path == "rotation":
            if channel.target_node_index in rotation_channels:
                raise ActionBakeError(
                    f"action {action.name!r} has duplicate rotation target "
                    f"{joint.name!r}"
                )
            rotation_channels[channel.target_node_index] = channel
        elif channel.target_path in {"translation", "scale"}:
            _validate_static_channel(channel, joint)
        else:  # extract_actions() currently prevents this; retain fail-closedness.
            raise ActionBakeError(
                f"action {action.name!r} has unsupported target path "
                f"{channel.target_path!r}"
            )

    missing = [
        joint.name for joint in joints if joint.node_index not in rotation_channels
    ]
    if missing:
        raise ActionBakeError(
            f"action {action.name!r} is missing rotation targets: {missing}"
        )
    if not all_timestamps:
        raise ActionBakeError(f"action {action.name!r} has no timestamps")

    tracks = {
        node_index: _prepare_rotation_track(channel)
        for node_index, channel in rotation_channels.items()
    }
    root_channel = rotation_channels[root.node_index]
    root_reference = root.local_trs.rotation_xyzw
    root_error = max(
        _quaternion_equivalence_error(value, root_reference)
        for value in root_channel.values
    )
    if root_error > _ROOT_ROTATION_TOLERANCE:
        raise ActionBakeError(
            f"action {action.name!r} has dynamic/non-default root rotation"
        )
    clip_start = min(all_timestamps)
    clip_end = max(all_timestamps)
    if clip_end <= clip_start:
        raise ActionBakeError(f"action {action.name!r} has a zero-duration clip")
    return tracks, clip_start, clip_end


def _loop_duration_ticks(
    *, clip_start_seconds: float, clip_end_seconds: float, action_name: str
) -> int:
    exact = (clip_end_seconds - clip_start_seconds) * TIME_BASE_HZ
    rounded = int(round(exact))
    if abs(exact - rounded) > _TICK_ROUNDING_TOLERANCE:
        raise ActionBakeError(
            f"action {action_name!r} duration does not resolve to an integer "
            f"{TIME_BASE_HZ} Hz tick: {exact:.17g}"
        )
    if rounded <= 0 or rounded % TICKS_PER_SAMPLE:
        raise ActionBakeError(
            f"action {action_name!r} duration must be a positive multiple of "
            f"{TICKS_PER_SAMPLE} ticks for endpoint-exclusive {SAMPLE_RATE_HZ} fps"
        )
    return rounded


def bake_required_actions(document: GlbDocument) -> BakedActionSet:
    """Bake exactly ``idle->Idle`` and ``walk->Walking`` from one GLB skin."""

    if not isinstance(document, GlbDocument):
        raise ActionBakeError("document must be a GlbDocument")
    if not _is_lower_sha256(document.sha256):
        raise ActionBakeError("GlbDocument has an invalid source SHA-256")
    try:
        skins = extract_skins(document)
        source_actions = extract_actions(document)
    except GlbError as exc:
        raise ActionBakeError(f"invalid GLB action input: {exc}") from exc
    if len(skins) != 1:
        raise ActionBakeError(f"expected exactly one skin, found {len(skins)}")
    skin = skins[0]
    if len(skin.joints) < 2:
        raise ActionBakeError("skin must contain a root and at least one runtime joint")
    names = [joint.name for joint in skin.joints]
    if any(name is None or not name for name in names) or len(set(names)) != len(names):
        raise ActionBakeError("skin joints must have unique non-empty names")
    roots = [joint for joint in skin.joints if joint.parent_joint_node_index is None]
    if len(roots) != 1:
        raise ActionBakeError(
            f"skin must be one direct joint tree with one root, found {len(roots)}"
        )
    root = roots[0]
    runtime_joints = tuple(joint for joint in skin.joints if joint != root)
    runtime_joint_order = tuple(joint.name for joint in runtime_joints)
    assert all(name is not None for name in runtime_joint_order)

    source_by_name = {action.name: action for action in source_actions}
    required_names = {source_name for _, source_name in SEMANTIC_ACTION_SOURCES}
    if set(source_by_name) != required_names:
        missing = sorted(required_names - set(source_by_name))
        extra = sorted(set(source_by_name) - required_names)
        raise ActionBakeError(
            "GLB actions must match the explicit idle->Idle, walk->Walking map; "
            f"missing={missing}, extra={extra}"
        )

    baked: list[BakedActionClip] = []
    for semantic_id, source_name in SEMANTIC_ACTION_SOURCES:
        source_action = source_by_name[source_name]
        tracks, clip_start, clip_end = _action_tracks(
            source_action,
            joints=skin.joints,
            root=root,
        )
        duration_ticks = _loop_duration_ticks(
            clip_start_seconds=clip_start,
            clip_end_seconds=clip_end,
            action_name=source_name,
        )
        sample_ticks = tuple(range(0, duration_ticks, TICKS_PER_SAMPLE))
        source_times = tuple(clip_start + tick / TIME_BASE_HZ for tick in sample_ticks)
        rotations = tuple(
            tuple(
                _sample_rotation(tracks[joint.node_index], source_time)
                for joint in runtime_joints
            )
            for source_time in source_times
        )
        baked.append(
            BakedActionClip(
                semantic_action_id=semantic_id,
                source_action_name=source_name,
                clip_start_seconds=float(clip_start),
                clip_end_seconds=float(clip_end),
                loop_duration_ticks=duration_ticks,
                sample_ticks=sample_ticks,
                source_times_seconds=source_times,
                rotations_xyzw=rotations,
            )
        )
    return BakedActionSet(
        source_glb_sha256=document.sha256,
        runtime_joint_order=runtime_joint_order,  # type: ignore[arg-type]
        actions=tuple(baked),
    )


def bake_actions(document: GlbDocument) -> BakedActionSet:
    """Compatibility spelling for :func:`bake_required_actions`."""

    return bake_required_actions(document)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(array),
        version=(1, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _metadata(value: BakedActionSet) -> dict[str, Any]:
    return {
        "schema": BAKED_ACTIONS_NPZ_SCHEMA,
        "qualification_state": value.qualification_state,
        "qualification_claim": value.qualification_claim,
        "source_glb_sha256": value.source_glb_sha256,
        "sample_rate_hz": value.sample_rate_hz,
        "time_base_hz": value.time_base_hz,
        "quaternion_order": "xyzw",
        "quaternion_semantics": "absolute_child_local",
        "runtime_joint_order": list(value.runtime_joint_order),
        "actions": [
            {
                "semantic_action_id": action.semantic_action_id,
                "source_action_name": action.source_action_name,
                "clip_start_seconds": action.clip_start_seconds,
                "clip_end_seconds": action.clip_end_seconds,
                "loop_duration_ticks": action.loop_duration_ticks,
                "sample_count": action.sample_count,
                "sample_ticks_member": f"{action.semantic_action_id}.sample_ticks.npy",
                "source_times_member": (
                    f"{action.semantic_action_id}.source_times_seconds.npy"
                ),
                "rotations_member": f"{action.semantic_action_id}.rotations_xyzw.npy",
            }
            for action in value.actions
        ],
    }


def _zip_member(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info, payload


def serialize_baked_actions_npz(value: BakedActionSet) -> bytes:
    """Serialize a canonical NPZ-compatible ZIP with fixed member metadata."""

    _validate_baked_action_set(value)
    metadata_payload = json.dumps(
        _metadata(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    members: list[tuple[str, bytes]] = [("metadata.json", metadata_payload)]
    for action in value.actions:
        members.extend(
            [
                (
                    f"{action.semantic_action_id}.sample_ticks.npy",
                    _npy_bytes(np.asarray(action.sample_ticks, dtype=np.dtype("<i8"))),
                ),
                (
                    f"{action.semantic_action_id}.source_times_seconds.npy",
                    _npy_bytes(
                        np.asarray(action.source_times_seconds, dtype=np.dtype("<f8"))
                    ),
                ),
                (
                    f"{action.semantic_action_id}.rotations_xyzw.npy",
                    _npy_bytes(
                        np.asarray(action.rotations_xyzw, dtype=np.dtype("<f8"))
                    ),
                ),
            ]
        )
    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = b""
        for name, payload in members:
            info, content = _zip_member(name, payload)
            archive.writestr(info, content)
    return output.getvalue()


def baked_actions_content_sha256(value: BakedActionSet) -> str:
    """Hash the canonical NPZ bytes for the complete baked action content."""

    return hashlib.sha256(serialize_baked_actions_npz(value)).hexdigest()


def write_baked_actions_npz(value: BakedActionSet, path: str | Path) -> str:
    """Write canonical bytes and return their lowercase SHA-256 identity."""

    payload = serialize_baked_actions_npz(value)
    destination = Path(path)
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _json_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise ActionBakeError(f"metadata contains duplicate key {key!r}")
        result[key] = item
    return result


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ActionBakeError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], *, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ActionBakeError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _read_npy(
    payload: bytes, *, name: str, dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
    try:
        array = np.load(BytesIO(payload), allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise ActionBakeError(f"unable to decode {name}: {exc}") from exc
    expected_dtype = np.dtype(dtype)
    if not isinstance(array, np.ndarray) or array.dtype != expected_dtype:
        raise ActionBakeError(f"{name} must have dtype {expected_dtype.str}")
    if array.shape != shape:
        raise ActionBakeError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ActionBakeError(f"{name} contains non-finite values")
    return array


def _read_zip_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ActionBakeError(f"unable to read NPZ member {name!r}: {exc}") from exc


def parse_baked_actions_npz(data: bytes | bytearray | memoryview) -> BakedActionSet:
    """Read and verify canonical deterministic baked-action NPZ bytes."""

    try:
        payload = bytes(data)
    except (TypeError, ValueError) as exc:
        raise ActionBakeError("baked NPZ input must be bytes-like") from exc
    try:
        archive = zipfile.ZipFile(BytesIO(payload), mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ActionBakeError(f"invalid baked action NPZ: {exc}") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ActionBakeError("baked action NPZ contains duplicate members")
        for info in infos:
            if (
                info.date_time != _FIXED_ZIP_TIME
                or info.compress_type != zipfile.ZIP_STORED
                or info.extra
            ):
                raise ActionBakeError(
                    f"NPZ member {info.filename!r} is not canonically encoded"
                )
        if "metadata.json" not in names:
            raise ActionBakeError("baked action NPZ lacks metadata.json")
        try:
            metadata = json.loads(
                _read_zip_member(archive, "metadata.json").decode("utf-8"),
                object_pairs_hook=_json_without_duplicates,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ActionBakeError(f"non-finite metadata number: {constant}")
                ),
            )
        except UnicodeDecodeError as exc:
            raise ActionBakeError("metadata.json is not UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise ActionBakeError(f"metadata.json is invalid: {exc}") from exc
        root = _object(metadata, name="metadata")
        _exact_keys(
            root,
            name="metadata",
            expected={
                "schema",
                "qualification_state",
                "qualification_claim",
                "source_glb_sha256",
                "sample_rate_hz",
                "time_base_hz",
                "quaternion_order",
                "quaternion_semantics",
                "runtime_joint_order",
                "actions",
            },
        )
        if root["schema"] != BAKED_ACTIONS_NPZ_SCHEMA:
            raise ActionBakeError("metadata schema is unsupported")
        if (
            root["qualification_state"] != "research_candidate"
            or root["qualification_claim"] is not False
        ):
            raise ActionBakeError("baked actions cannot claim qualification")
        if (
            root["sample_rate_hz"] != SAMPLE_RATE_HZ
            or root["time_base_hz"] != TIME_BASE_HZ
        ):
            raise ActionBakeError("metadata clock does not match M2")
        if (
            root["quaternion_order"] != "xyzw"
            or root["quaternion_semantics"] != "absolute_child_local"
        ):
            raise ActionBakeError("metadata quaternion contract is unsupported")
        raw_joint_order = root["runtime_joint_order"]
        if not isinstance(raw_joint_order, list):
            raise ActionBakeError("runtime_joint_order must be an array")
        joint_order = tuple(raw_joint_order)
        raw_actions = root["actions"]
        if not isinstance(raw_actions, list) or len(raw_actions) != len(
            SEMANTIC_ACTION_SOURCES
        ):
            raise ActionBakeError("metadata must contain exactly idle and walk")

        action_values: list[BakedActionClip] = []
        expected_members = ["metadata.json"]
        action_keys = {
            "semantic_action_id",
            "source_action_name",
            "clip_start_seconds",
            "clip_end_seconds",
            "loop_duration_ticks",
            "sample_count",
            "sample_ticks_member",
            "source_times_member",
            "rotations_member",
        }
        for ordinal, ((semantic_id, source_name), raw_action) in enumerate(
            zip(SEMANTIC_ACTION_SOURCES, raw_actions, strict=True)
        ):
            item = _object(raw_action, name=f"actions[{ordinal}]")
            _exact_keys(item, expected=action_keys, name=f"actions[{ordinal}]")
            if (
                item["semantic_action_id"] != semantic_id
                or item["source_action_name"] != source_name
            ):
                raise ActionBakeError("metadata action mapping is not canonical")
            sample_count = item["sample_count"]
            duration_ticks = item["loop_duration_ticks"]
            if (
                isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count <= 0
                or isinstance(duration_ticks, bool)
                or not isinstance(duration_ticks, int)
            ):
                raise ActionBakeError("sample_count/duration ticks must be integers")
            member_names = (
                item["sample_ticks_member"],
                item["source_times_member"],
                item["rotations_member"],
            )
            canonical_names = (
                f"{semantic_id}.sample_ticks.npy",
                f"{semantic_id}.source_times_seconds.npy",
                f"{semantic_id}.rotations_xyzw.npy",
            )
            if member_names != canonical_names:
                raise ActionBakeError("metadata array member names are not canonical")
            if any(not isinstance(name, str) for name in member_names):
                raise ActionBakeError("metadata member name must be a string")
            expected_members.extend(canonical_names)
            ticks = _read_npy(
                _read_zip_member(archive, canonical_names[0]),
                name=canonical_names[0],
                dtype="<i8",
                shape=(sample_count,),
            )
            times = _read_npy(
                _read_zip_member(archive, canonical_names[1]),
                name=canonical_names[1],
                dtype="<f8",
                shape=(sample_count,),
            )
            rotations = _read_npy(
                _read_zip_member(archive, canonical_names[2]),
                name=canonical_names[2],
                dtype="<f8",
                shape=(sample_count, len(joint_order), 4),
            )
            frames = tuple(
                tuple(
                    tuple(float(component) for component in quaternion)  # type: ignore[misc]
                    for quaternion in frame
                )
                for frame in rotations
            )
            action_values.append(
                BakedActionClip(
                    semantic_action_id=semantic_id,
                    source_action_name=source_name,
                    clip_start_seconds=_finite_float(
                        item["clip_start_seconds"],
                        name=f"{semantic_id}.clip_start_seconds",
                    ),
                    clip_end_seconds=_finite_float(
                        item["clip_end_seconds"],
                        name=f"{semantic_id}.clip_end_seconds",
                    ),
                    loop_duration_ticks=duration_ticks,
                    sample_ticks=tuple(int(tick) for tick in ticks),
                    source_times_seconds=tuple(float(time) for time in times),
                    rotations_xyzw=frames,
                )
            )
        if names != expected_members:
            raise ActionBakeError(
                "NPZ members/order differ from the canonical deterministic layout"
            )
        result = BakedActionSet(
            source_glb_sha256=root["source_glb_sha256"],
            runtime_joint_order=joint_order,
            actions=tuple(action_values),
        )
    if serialize_baked_actions_npz(result) != payload:
        raise ActionBakeError("baked action NPZ bytes are not canonical")
    return result


def read_baked_actions_npz(path: str | Path) -> BakedActionSet:
    """Load a baked action artifact without allowing NumPy pickle payloads."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ActionBakeError(
            f"unable to read baked action NPZ {source}: {exc}"
        ) from exc
    return parse_baked_actions_npz(payload)


__all__ = [
    "ActionBakeError",
    "BAKED_ACTIONS_NPZ_SCHEMA",
    "BakedActionClip",
    "BakedActionSet",
    "SAMPLE_RATE_HZ",
    "SEMANTIC_ACTION_SOURCES",
    "TICKS_PER_SAMPLE",
    "TIME_BASE_HZ",
    "bake_actions",
    "bake_required_actions",
    "baked_actions_content_sha256",
    "parse_baked_actions_npz",
    "read_baked_actions_npz",
    "serialize_baked_actions_npz",
    "write_baked_actions_npz",
]
