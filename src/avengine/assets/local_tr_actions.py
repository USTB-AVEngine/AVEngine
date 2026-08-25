"""Deterministic, research-only local-TR action baking for M2.

``actions.py`` deliberately admits only rotation-driven articulated motion.
This sibling format is an isolated research route for source rigs (notably the
Quaternius horse) whose authored deformation also depends on non-root joint
translations.  It does not change, extend, or qualify the formal v1 artifact.

Every baked pose stores absolute child-local glTF/Habitat translation and
rotation for every non-root skin joint.  The skin root remains owned by the
articulated-object transform and therefore must stay at its authored rest TR.
Scale animation is intentionally unsupported even when Habitat could represent
some of it: any authored scale channel must be static at the node rest value.
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

from avengine.assets.glb import (
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
LOCAL_TR_ACTIONS_NPZ_SCHEMA = "avengine_m2_local_tr_actions_v2"

_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_INPUT_QUATERNION_NORM_TOLERANCE = 1.0e-5
_OUTPUT_QUATERNION_NORM_TOLERANCE = 1.0e-12
_STATIC_CHANNEL_TOLERANCE = 5.0e-5
_TICK_ROUNDING_TOLERANCE = 1.0e-2
_HEMISPHERE_ZERO_TOLERANCE = 1.0e-15

Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


class LocalTRActionBakeError(ValueError):
    """The source action or local-TR artifact violates the research boundary."""


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalTRActionBakeError(f"{owner} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise LocalTRActionBakeError(f"{owner} must be a finite number")
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise LocalTRActionBakeError(f"{owner} must use canonical positive zero")
    return number


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _canonical_vector3(value: Sequence[float], *, owner: str) -> Vector3:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise LocalTRActionBakeError(f"{owner} must contain three finite values")
    return tuple(_canonical_float(component) for component in array)  # type: ignore[return-value]


def _quaternion_sign_component(value: Sequence[float]) -> float:
    scalar = float(value[3])
    if math.isclose(scalar, 0.0, rel_tol=0.0, abs_tol=_HEMISPHERE_ZERO_TOLERANCE):
        for component in value[:3]:
            number = float(component)
            if not math.isclose(
                number, 0.0, rel_tol=0.0, abs_tol=_HEMISPHERE_ZERO_TOLERANCE
            ):
                return number
    return scalar


def _normalise_quaternion(
    value: Sequence[float],
    *,
    owner: str,
    validate_input_norm: bool,
) -> QuaternionXYZW:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise LocalTRActionBakeError(
            f"{owner} must contain four finite xyzw components"
        )
    norm = float(np.linalg.norm(array))
    if norm < 1.0e-12:
        raise LocalTRActionBakeError(f"{owner} cannot be a zero quaternion")
    if validate_input_norm and abs(norm - 1.0) > _INPUT_QUATERNION_NORM_TOLERANCE:
        raise LocalTRActionBakeError(
            f"{owner} must already be a unit quaternion; norm={norm:.17g}"
        )
    array /= norm
    if _quaternion_sign_component(array) < 0.0:
        array = -array
    return tuple(_canonical_float(component) for component in array)  # type: ignore[return-value]


def _quaternion_equivalence_error(
    value: Sequence[float], reference: Sequence[float]
) -> float:
    left = np.asarray(value, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return math.inf
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1.0e-12 or right_norm < 1.0e-12:
        return math.inf
    left /= left_norm
    right /= right_norm
    return min(
        float(np.max(np.abs(left - right))),
        float(np.max(np.abs(left + right))),
    )


@dataclass(frozen=True)
class LocalTRActionClip:
    """One endpoint-exclusive loop on the M2 clock, with absolute local TR."""

    semantic_action_id: str
    source_action_name: str
    clip_start_seconds: float
    clip_end_seconds: float
    loop_duration_ticks: int
    sample_ticks: tuple[int, ...]
    source_times_seconds: tuple[float, ...]
    translations_m: tuple[tuple[Vector3, ...], ...]
    rotations_xyzw: tuple[tuple[QuaternionXYZW, ...], ...]

    @property
    def sample_count(self) -> int:
        return len(self.sample_ticks)


@dataclass(frozen=True)
class LocalTRActionSet:
    """The two research clips and the local-translation contract they require."""

    source_glb_sha256: str
    runtime_joint_order: tuple[str, ...]
    rest_translations_m: tuple[Vector3, ...]
    translation_driven_joint_ids: tuple[str, ...]
    actions: tuple[LocalTRActionClip, ...]
    sample_rate_hz: int = SAMPLE_RATE_HZ
    time_base_hz: int = TIME_BASE_HZ
    qualification_state: str = field(default="research_candidate", init=False)
    qualification_claim: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _validate_local_tr_action_set(self)

    def action(self, semantic_action_id: str) -> LocalTRActionClip:
        for action in self.actions:
            if action.semantic_action_id == semantic_action_id:
                return action
        raise KeyError(f"unknown semantic action: {semantic_action_id!r}")


def _validate_local_tr_action_set(value: LocalTRActionSet) -> None:
    if not _is_lower_sha256(value.source_glb_sha256):
        raise LocalTRActionBakeError("source_glb_sha256 must be lowercase SHA-256")
    if value.sample_rate_hz != SAMPLE_RATE_HZ:
        raise LocalTRActionBakeError(f"sample_rate_hz must equal {SAMPLE_RATE_HZ}")
    if value.time_base_hz != TIME_BASE_HZ:
        raise LocalTRActionBakeError(f"time_base_hz must equal {TIME_BASE_HZ}")
    if TIME_BASE_HZ % SAMPLE_RATE_HZ:
        raise LocalTRActionBakeError("time base must divide exactly by sample rate")
    if (
        not isinstance(value.runtime_joint_order, tuple)
        or not value.runtime_joint_order
        or any(
            not isinstance(name, str) or not name for name in value.runtime_joint_order
        )
        or len(set(value.runtime_joint_order)) != len(value.runtime_joint_order)
    ):
        raise LocalTRActionBakeError(
            "runtime_joint_order must be a non-empty tuple of unique names"
        )
    joint_count = len(value.runtime_joint_order)
    if (
        not isinstance(value.rest_translations_m, tuple)
        or len(value.rest_translations_m) != joint_count
    ):
        raise LocalTRActionBakeError(
            "rest_translations_m must align exactly with runtime_joint_order"
        )
    for index, translation in enumerate(value.rest_translations_m):
        if not isinstance(translation, tuple) or len(translation) != 3:
            raise LocalTRActionBakeError(
                f"rest_translations_m[{index}] must be an immutable vec3"
            )
        for component_index, component in enumerate(translation):
            _finite_float(
                component,
                owner=f"rest_translations_m[{index}][{component_index}]",
            )
    if (
        not isinstance(value.translation_driven_joint_ids, tuple)
        or any(
            not isinstance(name, str) or not name
            for name in value.translation_driven_joint_ids
        )
        or len(set(value.translation_driven_joint_ids))
        != len(value.translation_driven_joint_ids)
    ):
        raise LocalTRActionBakeError(
            "translation_driven_joint_ids must be an immutable unique tuple"
        )
    driven_set = set(value.translation_driven_joint_ids)
    if tuple(name for name in value.runtime_joint_order if name in driven_set) != (
        value.translation_driven_joint_ids
    ):
        raise LocalTRActionBakeError(
            "translation_driven_joint_ids must be an ordered subset of "
            "runtime_joint_order"
        )
    if not isinstance(value.actions, tuple) or any(
        not isinstance(action, LocalTRActionClip) for action in value.actions
    ):
        raise LocalTRActionBakeError("actions must be an immutable tuple")
    actual_mapping = tuple(
        (action.semantic_action_id, action.source_action_name)
        for action in value.actions
    )
    if actual_mapping != SEMANTIC_ACTION_SOURCES:
        raise LocalTRActionBakeError(
            "actions must follow the exact idle->Idle, walk->Walking mapping"
        )

    driven_indices = {
        index
        for index, name in enumerate(value.runtime_joint_order)
        if name in driven_set
    }
    for action in value.actions:
        start = _finite_float(
            action.clip_start_seconds,
            owner=f"{action.semantic_action_id}.clip_start_seconds",
        )
        end = _finite_float(
            action.clip_end_seconds,
            owner=f"{action.semantic_action_id}.clip_end_seconds",
        )
        if start < 0.0 or end <= start:
            raise LocalTRActionBakeError(
                f"{action.semantic_action_id} bounds must satisfy 0 <= start < end"
            )
        if (
            isinstance(action.loop_duration_ticks, bool)
            or not isinstance(action.loop_duration_ticks, int)
            or action.loop_duration_ticks <= 0
            or action.loop_duration_ticks % TICKS_PER_SAMPLE
        ):
            raise LocalTRActionBakeError(
                f"{action.semantic_action_id}.loop_duration_ticks must be a positive "
                f"multiple of {TICKS_PER_SAMPLE}"
            )
        exact_duration_ticks = (end - start) * TIME_BASE_HZ
        if (
            abs(exact_duration_ticks - action.loop_duration_ticks)
            > _TICK_ROUNDING_TOLERANCE
        ):
            raise LocalTRActionBakeError(
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
            raise LocalTRActionBakeError(
                f"{action.semantic_action_id}.sample_ticks must be endpoint-exclusive "
                f"{TICKS_PER_SAMPLE}-tick samples"
            )
        sample_count = len(expected_ticks)
        if (
            not isinstance(action.source_times_seconds, tuple)
            or len(action.source_times_seconds) != sample_count
            or not isinstance(action.translations_m, tuple)
            or len(action.translations_m) != sample_count
            or not isinstance(action.rotations_xyzw, tuple)
            or len(action.rotations_xyzw) != sample_count
        ):
            raise LocalTRActionBakeError(
                f"{action.semantic_action_id} frame arrays must match sample count"
            )
        for sample_index, (tick, source_time, translations, rotations) in enumerate(
            zip(
                expected_ticks,
                action.source_times_seconds,
                action.translations_m,
                action.rotations_xyzw,
                strict=True,
            )
        ):
            actual_time = _finite_float(
                source_time,
                owner=(
                    f"{action.semantic_action_id}.source_times_seconds[{sample_index}]"
                ),
            )
            expected_time = start + tick / TIME_BASE_HZ
            if not math.isclose(
                actual_time, expected_time, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise LocalTRActionBakeError(
                    f"{action.semantic_action_id} source time is off the tick grid"
                )
            if actual_time >= end:
                raise LocalTRActionBakeError(
                    f"{action.semantic_action_id} stores the duplicate loop endpoint"
                )
            if not isinstance(translations, tuple) or len(translations) != joint_count:
                raise LocalTRActionBakeError(
                    f"{action.semantic_action_id} frame {sample_index} translations "
                    f"must contain {joint_count} joints"
                )
            if not isinstance(rotations, tuple) or len(rotations) != joint_count:
                raise LocalTRActionBakeError(
                    f"{action.semantic_action_id} frame {sample_index} rotations "
                    f"must contain {joint_count} joints"
                )
            for joint_index, translation in enumerate(translations):
                if not isinstance(translation, tuple) or len(translation) != 3:
                    raise LocalTRActionBakeError(
                        "baked translation must be an immutable vec3"
                    )
                array = np.asarray(translation, dtype=np.float64)
                if not np.all(np.isfinite(array)):
                    raise LocalTRActionBakeError(
                        "baked translation contains non-finite values"
                    )
                if any(
                    component == 0.0 and math.copysign(1.0, component) < 0.0
                    for component in array
                ):
                    raise LocalTRActionBakeError(
                        "baked translation must use canonical positive zero"
                    )
                if joint_index not in driven_indices:
                    rest = np.asarray(
                        value.rest_translations_m[joint_index], dtype=np.float64
                    )
                    if float(np.max(np.abs(array - rest))) > (
                        _STATIC_CHANNEL_TOLERANCE
                    ):
                        joint_id = value.runtime_joint_order[joint_index]
                        raise LocalTRActionBakeError(
                            f"non-driven joint {joint_id!r} departs from rest translation"
                        )
            for quaternion in rotations:
                if not isinstance(quaternion, tuple) or len(quaternion) != 4:
                    raise LocalTRActionBakeError(
                        "baked rotation must be an immutable xyzw quaternion"
                    )
                array = np.asarray(quaternion, dtype=np.float64)
                if not np.all(np.isfinite(array)):
                    raise LocalTRActionBakeError(
                        "baked quaternion contains non-finite values"
                    )
                if any(
                    component == 0.0 and math.copysign(1.0, component) < 0.0
                    for component in array
                ):
                    raise LocalTRActionBakeError(
                        "baked quaternion must use canonical positive zero"
                    )
                norm = float(np.linalg.norm(array))
                if abs(norm - 1.0) > _OUTPUT_QUATERNION_NORM_TOLERANCE:
                    raise LocalTRActionBakeError(
                        "baked quaternion must be float64 unit normalized"
                    )
                if _quaternion_sign_component(array) < 0.0:
                    raise LocalTRActionBakeError(
                        "baked quaternion must use the canonical hemisphere"
                    )


@dataclass(frozen=True)
class _TranslationTrack:
    interpolation: str
    timestamps_seconds: tuple[float, ...]
    values: tuple[Vector3, ...]


@dataclass(frozen=True)
class _RotationTrack:
    interpolation: str
    timestamps_seconds: tuple[float, ...]
    values: tuple[QuaternionXYZW, ...]


def _prepare_translation_track(channel: AnimationChannel) -> _TranslationTrack:
    if channel.interpolation == "CUBICSPLINE":
        raise LocalTRActionBakeError(
            f"{channel.target_node_name or channel.target_node_index} translation "
            "uses unsupported CUBICSPLINE"
        )
    if channel.interpolation not in {"STEP", "LINEAR"}:
        raise LocalTRActionBakeError(
            f"unsupported translation interpolation: {channel.interpolation!r}"
        )
    return _TranslationTrack(
        interpolation=channel.interpolation,
        timestamps_seconds=channel.timestamps_seconds,
        values=tuple(
            _canonical_vector3(value, owner="translation key")
            for value in channel.values
        ),
    )


def _prepare_rotation_track(channel: AnimationChannel) -> _RotationTrack:
    if channel.interpolation == "CUBICSPLINE":
        raise LocalTRActionBakeError(
            f"{channel.target_node_name or channel.target_node_index} rotation uses "
            "unsupported CUBICSPLINE"
        )
    if channel.interpolation not in {"STEP", "LINEAR"}:
        raise LocalTRActionBakeError(
            f"unsupported rotation interpolation: {channel.interpolation!r}"
        )
    values = [
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
        for value in values:
            candidate = np.asarray(value, dtype=np.float64)
            if (
                continuous
                and float(
                    np.dot(np.asarray(continuous[-1], dtype=np.float64), candidate)
                )
                < 0.0
            ):
                candidate = -candidate
            continuous.append(tuple(float(item) for item in candidate))  # type: ignore[arg-type]
        values = continuous
    return _RotationTrack(
        interpolation=channel.interpolation,
        timestamps_seconds=channel.timestamps_seconds,
        values=tuple(values),
    )


def _sample_translation(track: _TranslationTrack, source_time: float) -> Vector3:
    times = track.timestamps_seconds
    values = track.values
    if source_time <= times[0] or len(times) == 1:
        return _canonical_vector3(values[0], owner="sampled translation")
    if source_time >= times[-1]:
        return _canonical_vector3(values[-1], owner="sampled translation")
    left_index = bisect_right(times, source_time) - 1
    if track.interpolation == "STEP":
        return _canonical_vector3(values[left_index], owner="sampled STEP translation")
    fraction = (source_time - times[left_index]) / (
        times[left_index + 1] - times[left_index]
    )
    left = np.asarray(values[left_index], dtype=np.float64)
    right = np.asarray(values[left_index + 1], dtype=np.float64)
    return _canonical_vector3(
        left + fraction * (right - left), owner="sampled LINEAR translation"
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
        result, owner="interpolated rotation", validate_input_norm=False
    )


def _sample_rotation(track: _RotationTrack, source_time: float) -> QuaternionXYZW:
    times = track.timestamps_seconds
    values = track.values
    if source_time <= times[0] or len(times) == 1:
        return _normalise_quaternion(
            values[0], owner="sampled rotation", validate_input_norm=False
        )
    if source_time >= times[-1]:
        return _normalise_quaternion(
            values[-1], owner="sampled rotation", validate_input_norm=False
        )
    left_index = bisect_right(times, source_time) - 1
    if track.interpolation == "STEP":
        return _normalise_quaternion(
            values[left_index],
            owner="sampled STEP rotation",
            validate_input_norm=False,
        )
    fraction = (source_time - times[left_index]) / (
        times[left_index + 1] - times[left_index]
    )
    return _slerp_shortest(values[left_index], values[left_index + 1], fraction)


def _validate_static_vector_channel(
    channel: AnimationChannel,
    *,
    joint: JointNode,
    expected: Sequence[float],
) -> None:
    if channel.interpolation == "CUBICSPLINE":
        raise LocalTRActionBakeError(
            f"{joint.name} {channel.target_path} uses unsupported CUBICSPLINE"
        )
    if channel.interpolation not in {"STEP", "LINEAR"}:
        raise LocalTRActionBakeError(
            f"unsupported {channel.target_path} interpolation: "
            f"{channel.interpolation!r}"
        )
    values = np.asarray(channel.values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise LocalTRActionBakeError(
            f"{joint.name} {channel.target_path} is non-finite"
        )
    temporal_error = float(np.max(np.abs(values - values[0])))
    rest_error = float(np.max(np.abs(values[0] - np.asarray(expected))))
    if max(temporal_error, rest_error) > _STATIC_CHANNEL_TOLERANCE:
        raise LocalTRActionBakeError(
            f"{joint.name} {channel.target_path} must be static at its node rest value"
        )


def _validate_static_root_rotation(
    channel: AnimationChannel, *, root: JointNode
) -> None:
    if channel.interpolation == "CUBICSPLINE":
        raise LocalTRActionBakeError(
            f"{root.name} rotation uses unsupported CUBICSPLINE"
        )
    if channel.interpolation not in {"STEP", "LINEAR"}:
        raise LocalTRActionBakeError(
            f"unsupported root rotation interpolation: {channel.interpolation!r}"
        )
    error = max(
        _quaternion_equivalence_error(value, root.local_trs.rotation_xyzw)
        for value in channel.values
    )
    if error > _STATIC_CHANNEL_TOLERANCE:
        raise LocalTRActionBakeError(
            f"action root {root.name!r} rotation must be static at rest"
        )


def _action_tracks(
    action: AnimationAction,
    *,
    joints: tuple[JointNode, ...],
    root: JointNode,
) -> tuple[
    dict[int, _TranslationTrack],
    dict[int, _RotationTrack],
    set[int],
    float,
    float,
]:
    joints_by_node = {joint.node_index: joint for joint in joints}
    translations: dict[int, AnimationChannel] = {}
    rotations: dict[int, AnimationChannel] = {}
    all_timestamps: list[float] = []
    for channel in action.channels:
        joint = joints_by_node.get(channel.target_node_index)
        if joint is None:
            raise LocalTRActionBakeError(
                f"action {action.name!r} targets non-skin node "
                f"{channel.target_node_index}"
            )
        if channel.interpolation == "CUBICSPLINE":
            raise LocalTRActionBakeError(
                f"action {action.name!r} uses unsupported CUBICSPLINE"
            )
        all_timestamps.extend(channel.timestamps_seconds)
        if channel.target_path == "translation":
            if channel.target_node_index in translations:
                raise LocalTRActionBakeError(
                    f"action {action.name!r} has duplicate translation target "
                    f"{joint.name!r}"
                )
            translations[channel.target_node_index] = channel
        elif channel.target_path == "rotation":
            if channel.target_node_index in rotations:
                raise LocalTRActionBakeError(
                    f"action {action.name!r} has duplicate rotation target "
                    f"{joint.name!r}"
                )
            rotations[channel.target_node_index] = channel
        elif channel.target_path == "scale":
            _validate_static_vector_channel(
                channel, joint=joint, expected=joint.local_trs.scale
            )
        else:
            raise LocalTRActionBakeError(
                f"action {action.name!r} has unsupported target path "
                f"{channel.target_path!r}"
            )
    missing_rotations = [
        joint.name for joint in joints if joint.node_index not in rotations
    ]
    if missing_rotations:
        raise LocalTRActionBakeError(
            f"action {action.name!r} is missing rotation targets: {missing_rotations}"
        )
    if not all_timestamps:
        raise LocalTRActionBakeError(f"action {action.name!r} has no timestamps")

    root_translation = translations.get(root.node_index)
    if root_translation is not None:
        _validate_static_vector_channel(
            root_translation, joint=root, expected=root.local_trs.translation
        )
    _validate_static_root_rotation(rotations[root.node_index], root=root)

    translation_tracks: dict[int, _TranslationTrack] = {}
    translation_driven_nodes: set[int] = set()
    for joint in joints:
        if joint == root:
            continue
        channel = translations.get(joint.node_index)
        if channel is None:
            continue
        track = _prepare_translation_track(channel)
        translation_tracks[joint.node_index] = track
        rest = np.asarray(joint.local_trs.translation, dtype=np.float64)
        maximum_rest_error = float(
            np.max(np.abs(np.asarray(track.values, dtype=np.float64) - rest))
        )
        if maximum_rest_error > _STATIC_CHANNEL_TOLERANCE:
            translation_driven_nodes.add(joint.node_index)
    rotation_tracks = {
        node_index: _prepare_rotation_track(channel)
        for node_index, channel in rotations.items()
    }
    clip_start = min(all_timestamps)
    clip_end = max(all_timestamps)
    if clip_end <= clip_start:
        raise LocalTRActionBakeError(f"action {action.name!r} has a zero-duration clip")
    return (
        translation_tracks,
        rotation_tracks,
        translation_driven_nodes,
        clip_start,
        clip_end,
    )


def _loop_duration_ticks(
    *, clip_start_seconds: float, clip_end_seconds: float, action_name: str
) -> int:
    exact = (clip_end_seconds - clip_start_seconds) * TIME_BASE_HZ
    rounded = int(round(exact))
    if abs(exact - rounded) > _TICK_ROUNDING_TOLERANCE:
        raise LocalTRActionBakeError(
            f"action {action_name!r} duration does not resolve to an integer "
            f"{TIME_BASE_HZ} Hz tick: {exact:.17g}"
        )
    if rounded <= 0 or rounded % TICKS_PER_SAMPLE:
        raise LocalTRActionBakeError(
            f"action {action_name!r} duration must be a positive multiple of "
            f"{TICKS_PER_SAMPLE} ticks for endpoint-exclusive {SAMPLE_RATE_HZ} fps"
        )
    return rounded


def bake_local_tr_actions(document: GlbDocument) -> LocalTRActionSet:
    """Bake ``idle->Idle`` and ``walk->Walking`` into local-TR v2 poses."""

    if not isinstance(document, GlbDocument):
        raise LocalTRActionBakeError("document must be a GlbDocument")
    if not _is_lower_sha256(document.sha256):
        raise LocalTRActionBakeError("GlbDocument has an invalid source SHA-256")
    try:
        skins = extract_skins(document)
        source_actions = extract_actions(document)
    except GlbError as exc:
        raise LocalTRActionBakeError(f"invalid GLB action input: {exc}") from exc
    if len(skins) != 1:
        raise LocalTRActionBakeError(f"expected exactly one skin, found {len(skins)}")
    skin = skins[0]
    if len(skin.joints) < 2:
        raise LocalTRActionBakeError(
            "skin must contain a root and at least one runtime joint"
        )
    names = [joint.name for joint in skin.joints]
    if any(name is None or not name for name in names) or len(set(names)) != len(names):
        raise LocalTRActionBakeError("skin joints must have unique non-empty names")
    roots = [joint for joint in skin.joints if joint.parent_joint_node_index is None]
    if len(roots) != 1:
        raise LocalTRActionBakeError(
            f"skin must be one direct joint tree with one root, found {len(roots)}"
        )
    root = roots[0]
    runtime_joints = tuple(joint for joint in skin.joints if joint != root)
    runtime_joint_order = tuple(joint.name for joint in runtime_joints)
    assert all(name is not None for name in runtime_joint_order)
    rest_translations = tuple(
        _canonical_vector3(
            joint.local_trs.translation, owner=f"{joint.name} rest translation"
        )
        for joint in runtime_joints
    )

    source_by_name = {action.name: action for action in source_actions}
    required_names = {source_name for _, source_name in SEMANTIC_ACTION_SOURCES}
    if set(source_by_name) != required_names:
        missing = sorted(required_names - set(source_by_name))
        extra = sorted(set(source_by_name) - required_names)
        raise LocalTRActionBakeError(
            "GLB actions must match the explicit idle->Idle, walk->Walking map; "
            f"missing={missing}, extra={extra}"
        )

    baked: list[LocalTRActionClip] = []
    driven_node_indices: set[int] = set()
    for semantic_id, source_name in SEMANTIC_ACTION_SOURCES:
        source_action = source_by_name[source_name]
        (
            translation_tracks,
            rotation_tracks,
            action_driven_nodes,
            clip_start,
            clip_end,
        ) = _action_tracks(source_action, joints=skin.joints, root=root)
        driven_node_indices.update(action_driven_nodes)
        duration_ticks = _loop_duration_ticks(
            clip_start_seconds=clip_start,
            clip_end_seconds=clip_end,
            action_name=source_name,
        )
        sample_ticks = tuple(range(0, duration_ticks, TICKS_PER_SAMPLE))
        source_times = tuple(clip_start + tick / TIME_BASE_HZ for tick in sample_ticks)
        translations = tuple(
            tuple(
                _sample_translation(translation_tracks[joint.node_index], source_time)
                if joint.node_index in action_driven_nodes
                else _canonical_vector3(
                    joint.local_trs.translation,
                    owner=f"{joint.name} static rest translation",
                )
                for joint in runtime_joints
            )
            for source_time in source_times
        )
        rotations = tuple(
            tuple(
                _sample_rotation(rotation_tracks[joint.node_index], source_time)
                for joint in runtime_joints
            )
            for source_time in source_times
        )
        baked.append(
            LocalTRActionClip(
                semantic_action_id=semantic_id,
                source_action_name=source_name,
                clip_start_seconds=float(clip_start),
                clip_end_seconds=float(clip_end),
                loop_duration_ticks=duration_ticks,
                sample_ticks=sample_ticks,
                source_times_seconds=source_times,
                translations_m=translations,
                rotations_xyzw=rotations,
            )
        )
    driven_ids = tuple(
        joint.name
        for joint in runtime_joints
        if joint.node_index in driven_node_indices
    )
    return LocalTRActionSet(
        source_glb_sha256=document.sha256,
        runtime_joint_order=runtime_joint_order,  # type: ignore[arg-type]
        rest_translations_m=rest_translations,
        translation_driven_joint_ids=driven_ids,  # type: ignore[arg-type]
        actions=tuple(baked),
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(array),
        version=(1, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _metadata(value: LocalTRActionSet) -> dict[str, Any]:
    return {
        "schema": LOCAL_TR_ACTIONS_NPZ_SCHEMA,
        "qualification_state": value.qualification_state,
        "qualification_claim": value.qualification_claim,
        "source_glb_sha256": value.source_glb_sha256,
        "sample_rate_hz": value.sample_rate_hz,
        "time_base_hz": value.time_base_hz,
        "translation_unit": "meter",
        "translation_semantics": "absolute_child_local",
        "quaternion_order": "xyzw",
        "quaternion_semantics": "absolute_child_local",
        "root_motion_contract": "static_rest_owned_by_articulated_object",
        "runtime_joint_order": list(value.runtime_joint_order),
        "rest_translations_member": "rest_translations_m.npy",
        "translation_driven_joint_ids": list(value.translation_driven_joint_ids),
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
                "translations_member": (
                    f"{action.semantic_action_id}.translations_m.npy"
                ),
                "rotations_member": (f"{action.semantic_action_id}.rotations_xyzw.npy"),
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


def serialize_local_tr_actions_npz(value: LocalTRActionSet) -> bytes:
    """Serialize canonical NPZ-compatible local-TR v2 bytes."""

    _validate_local_tr_action_set(value)
    metadata_payload = json.dumps(
        _metadata(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    members: list[tuple[str, bytes]] = [
        ("metadata.json", metadata_payload),
        (
            "rest_translations_m.npy",
            _npy_bytes(np.asarray(value.rest_translations_m, dtype=np.dtype("<f8"))),
        ),
    ]
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
                    f"{action.semantic_action_id}.translations_m.npy",
                    _npy_bytes(
                        np.asarray(action.translations_m, dtype=np.dtype("<f8"))
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


def local_tr_actions_content_sha256(value: LocalTRActionSet) -> str:
    """Hash the complete canonical local-TR v2 artifact."""

    return hashlib.sha256(serialize_local_tr_actions_npz(value)).hexdigest()


def write_local_tr_actions_npz(value: LocalTRActionSet, path: str | Path) -> str:
    """Write canonical local-TR v2 bytes and return their SHA-256."""

    payload = serialize_local_tr_actions_npz(value)
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _json_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LocalTRActionBakeError(f"metadata contains duplicate key {key!r}")
        result[key] = value
    return result


def _object(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise LocalTRActionBakeError(f"{owner} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], *, expected: set[str], owner: str) -> None:
    actual = set(value)
    if actual != expected:
        raise LocalTRActionBakeError(
            f"{owner} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _read_zip_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise LocalTRActionBakeError(
            f"unable to read local-TR NPZ member {name!r}: {exc}"
        ) from exc


def _read_npy(
    payload: bytes, *, owner: str, dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
    try:
        array = np.load(BytesIO(payload), allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise LocalTRActionBakeError(f"unable to decode {owner}: {exc}") from exc
    expected_dtype = np.dtype(dtype)
    if not isinstance(array, np.ndarray) or array.dtype != expected_dtype:
        raise LocalTRActionBakeError(f"{owner} must have dtype {expected_dtype.str}")
    if array.shape != shape:
        raise LocalTRActionBakeError(
            f"{owner} must have shape {shape}, got {array.shape}"
        )
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise LocalTRActionBakeError(f"{owner} contains non-finite values")
    return array


def parse_local_tr_actions_npz(
    data: bytes | bytearray | memoryview,
) -> LocalTRActionSet:
    """Parse and verify canonical deterministic local-TR v2 NPZ bytes."""

    try:
        payload = bytes(data)
    except (TypeError, ValueError) as exc:
        raise LocalTRActionBakeError("local-TR NPZ input must be bytes-like") from exc
    try:
        archive = zipfile.ZipFile(BytesIO(payload), mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise LocalTRActionBakeError(f"invalid local-TR action NPZ: {exc}") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise LocalTRActionBakeError("local-TR NPZ contains duplicate members")
        if archive.comment:
            raise LocalTRActionBakeError(
                "local-TR NPZ archive comment is not canonical"
            )
        for info in infos:
            if (
                info.date_time != _FIXED_ZIP_TIME
                or info.compress_type != zipfile.ZIP_STORED
                or info.extra
            ):
                raise LocalTRActionBakeError(
                    f"NPZ member {info.filename!r} is not canonically encoded"
                )
        if "metadata.json" not in names:
            raise LocalTRActionBakeError("local-TR NPZ lacks metadata.json")
        try:
            metadata = json.loads(
                _read_zip_member(archive, "metadata.json").decode("utf-8"),
                object_pairs_hook=_json_without_duplicates,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    LocalTRActionBakeError(f"non-finite metadata number: {constant}")
                ),
            )
        except UnicodeDecodeError as exc:
            raise LocalTRActionBakeError("metadata.json is not UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise LocalTRActionBakeError(f"metadata.json is invalid: {exc}") from exc
        root = _object(metadata, owner="metadata")
        _exact_keys(
            root,
            owner="metadata",
            expected={
                "schema",
                "qualification_state",
                "qualification_claim",
                "source_glb_sha256",
                "sample_rate_hz",
                "time_base_hz",
                "translation_unit",
                "translation_semantics",
                "quaternion_order",
                "quaternion_semantics",
                "root_motion_contract",
                "runtime_joint_order",
                "rest_translations_member",
                "translation_driven_joint_ids",
                "actions",
            },
        )
        if root["schema"] != LOCAL_TR_ACTIONS_NPZ_SCHEMA:
            raise LocalTRActionBakeError("metadata schema is unsupported")
        if (
            root["qualification_state"] != "research_candidate"
            or root["qualification_claim"] is not False
        ):
            raise LocalTRActionBakeError("local-TR actions cannot claim qualification")
        if (
            root["sample_rate_hz"] != SAMPLE_RATE_HZ
            or root["time_base_hz"] != TIME_BASE_HZ
        ):
            raise LocalTRActionBakeError("metadata clock does not match M2")
        if (
            root["translation_unit"] != "meter"
            or root["translation_semantics"] != "absolute_child_local"
            or root["quaternion_order"] != "xyzw"
            or root["quaternion_semantics"] != "absolute_child_local"
            or root["root_motion_contract"] != "static_rest_owned_by_articulated_object"
        ):
            raise LocalTRActionBakeError("metadata local-TR contract is unsupported")
        raw_joint_order = root["runtime_joint_order"]
        raw_driven_ids = root["translation_driven_joint_ids"]
        if not isinstance(raw_joint_order, list) or not isinstance(
            raw_driven_ids, list
        ):
            raise LocalTRActionBakeError(
                "runtime joint and translation-driven IDs must be arrays"
            )
        joint_order = tuple(raw_joint_order)
        driven_ids = tuple(raw_driven_ids)
        joint_count = len(joint_order)
        if root["rest_translations_member"] != "rest_translations_m.npy":
            raise LocalTRActionBakeError(
                "rest translation member name is not canonical"
            )
        rest_array = _read_npy(
            _read_zip_member(archive, "rest_translations_m.npy"),
            owner="rest_translations_m.npy",
            dtype="<f8",
            shape=(joint_count, 3),
        )
        rest_translations = tuple(
            tuple(float(component) for component in translation)
            for translation in rest_array
        )

        raw_actions = root["actions"]
        if not isinstance(raw_actions, list) or len(raw_actions) != len(
            SEMANTIC_ACTION_SOURCES
        ):
            raise LocalTRActionBakeError("metadata must contain exactly idle and walk")
        action_keys = {
            "semantic_action_id",
            "source_action_name",
            "clip_start_seconds",
            "clip_end_seconds",
            "loop_duration_ticks",
            "sample_count",
            "sample_ticks_member",
            "source_times_member",
            "translations_member",
            "rotations_member",
        }
        action_values: list[LocalTRActionClip] = []
        expected_members = ["metadata.json", "rest_translations_m.npy"]
        for ordinal, ((semantic_id, source_name), raw_action) in enumerate(
            zip(SEMANTIC_ACTION_SOURCES, raw_actions, strict=True)
        ):
            item = _object(raw_action, owner=f"actions[{ordinal}]")
            _exact_keys(item, expected=action_keys, owner=f"actions[{ordinal}]")
            if (
                item["semantic_action_id"] != semantic_id
                or item["source_action_name"] != source_name
            ):
                raise LocalTRActionBakeError("metadata action mapping is not canonical")
            sample_count = item["sample_count"]
            duration_ticks = item["loop_duration_ticks"]
            if (
                isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count <= 0
                or isinstance(duration_ticks, bool)
                or not isinstance(duration_ticks, int)
            ):
                raise LocalTRActionBakeError(
                    "sample_count/duration ticks must be integers"
                )
            member_names = (
                item["sample_ticks_member"],
                item["source_times_member"],
                item["translations_member"],
                item["rotations_member"],
            )
            canonical_names = (
                f"{semantic_id}.sample_ticks.npy",
                f"{semantic_id}.source_times_seconds.npy",
                f"{semantic_id}.translations_m.npy",
                f"{semantic_id}.rotations_xyzw.npy",
            )
            if member_names != canonical_names:
                raise LocalTRActionBakeError(
                    "metadata array member names are not canonical"
                )
            expected_members.extend(canonical_names)
            ticks = _read_npy(
                _read_zip_member(archive, canonical_names[0]),
                owner=canonical_names[0],
                dtype="<i8",
                shape=(sample_count,),
            )
            times = _read_npy(
                _read_zip_member(archive, canonical_names[1]),
                owner=canonical_names[1],
                dtype="<f8",
                shape=(sample_count,),
            )
            translations_array = _read_npy(
                _read_zip_member(archive, canonical_names[2]),
                owner=canonical_names[2],
                dtype="<f8",
                shape=(sample_count, joint_count, 3),
            )
            rotations_array = _read_npy(
                _read_zip_member(archive, canonical_names[3]),
                owner=canonical_names[3],
                dtype="<f8",
                shape=(sample_count, joint_count, 4),
            )
            translations = tuple(
                tuple(
                    tuple(float(component) for component in translation)
                    for translation in frame
                )
                for frame in translations_array
            )
            rotations = tuple(
                tuple(
                    tuple(float(component) for component in quaternion)
                    for quaternion in frame
                )
                for frame in rotations_array
            )
            action_values.append(
                LocalTRActionClip(
                    semantic_action_id=semantic_id,
                    source_action_name=source_name,
                    clip_start_seconds=_finite_float(
                        item["clip_start_seconds"],
                        owner=f"{semantic_id}.clip_start_seconds",
                    ),
                    clip_end_seconds=_finite_float(
                        item["clip_end_seconds"],
                        owner=f"{semantic_id}.clip_end_seconds",
                    ),
                    loop_duration_ticks=duration_ticks,
                    sample_ticks=tuple(int(tick) for tick in ticks),
                    source_times_seconds=tuple(float(time) for time in times),
                    translations_m=translations,  # type: ignore[arg-type]
                    rotations_xyzw=rotations,  # type: ignore[arg-type]
                )
            )
        if names != expected_members:
            raise LocalTRActionBakeError(
                "NPZ members/order differ from the canonical deterministic layout"
            )
        result = LocalTRActionSet(
            source_glb_sha256=root["source_glb_sha256"],
            runtime_joint_order=joint_order,
            rest_translations_m=rest_translations,  # type: ignore[arg-type]
            translation_driven_joint_ids=driven_ids,
            actions=tuple(action_values),
        )
    if serialize_local_tr_actions_npz(result) != payload:
        raise LocalTRActionBakeError("local-TR action NPZ bytes are not canonical")
    return result


def read_local_tr_actions_npz(path: str | Path) -> LocalTRActionSet:
    """Read a local-TR v2 artifact without allowing NumPy pickle payloads."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise LocalTRActionBakeError(
            f"unable to read local-TR action NPZ {source}: {exc}"
        ) from exc
    return parse_local_tr_actions_npz(payload)


__all__ = [
    "LOCAL_TR_ACTIONS_NPZ_SCHEMA",
    "LocalTRActionBakeError",
    "LocalTRActionClip",
    "LocalTRActionSet",
    "SAMPLE_RATE_HZ",
    "SEMANTIC_ACTION_SOURCES",
    "TICKS_PER_SAMPLE",
    "TIME_BASE_HZ",
    "bake_local_tr_actions",
    "local_tr_actions_content_sha256",
    "parse_local_tr_actions_npz",
    "read_local_tr_actions_npz",
    "serialize_local_tr_actions_npz",
    "write_local_tr_actions_npz",
]
