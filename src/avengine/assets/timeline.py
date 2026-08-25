"""Deterministic 75-frame M2 articulated-dog state construction.

The M2 runtime never asks Habitat or the GLB importer to advance an animation
clock.  This module expands the two endpoint-exclusive baked loops into the
exact states that a formal capture must apply at the 15 Hz video ticks.  The
default canary is intentionally small and explicit:

* 15 idle frames at the trajectory start;
* 45 walking frames on a straight 1.6 m path; and
* 15 idle frames at the trajectory end.

Contact phases are an input produced by the separate kinematic/contact QA
gate.  They are not guessed from action names here.  Formal request creation
remains fail-closed on ``canary_qualified`` package admission; callers may use
``build_m2_state_sequence`` to prepare review-only candidate states without
misrepresenting them as a valid formal request.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.assets.actions import BakedActionSet, TICKS_PER_SAMPLE
from avengine.assets.contracts import (
    APPLIED_STATE_HASH_ALGORITHM,
    CAPTURE_SCHEMA,
    CONTACT_ORDER,
    FORMAL_MODALITIES,
    FORMAL_VIEW_IDS,
    POSE_HASH_ALGORITHM,
    compute_applied_state_hash,
    compute_pose_hash,
    validate_capture_request,
)


FRAME_COUNT = 75
IDLE_LEAD_FRAME_COUNT = 15
WALK_FRAME_COUNT = 45
IDLE_TAIL_FRAME_COUNT = 15

_ZERO_TOLERANCE = 1.0e-15


class TimelineBuildError(ValueError):
    """An input cannot produce the canonical M2 canary state sequence."""


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _finite_vector3(value: Sequence[float], *, owner: str) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TimelineBuildError(f"{owner} must contain three finite numbers") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise TimelineBuildError(f"{owner} must contain three finite numbers")
    return tuple(_canonical_float(component) for component in array)


def _quaternion_sign_component(quaternion: np.ndarray) -> float:
    scalar = float(quaternion[3])
    if math.isclose(scalar, 0.0, rel_tol=0.0, abs_tol=_ZERO_TOLERANCE):
        for component in quaternion[:3]:
            if not math.isclose(
                float(component),
                0.0,
                rel_tol=0.0,
                abs_tol=_ZERO_TOLERANCE,
            ):
                return float(component)
    return scalar


def _canonical_unit_quaternion(
    value: Sequence[float], *, owner: str
) -> tuple[float, float, float, float]:
    try:
        quaternion = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TimelineBuildError(
            f"{owner} must contain four finite xyzw components"
        ) from exc
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise TimelineBuildError(f"{owner} must contain four finite xyzw components")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise TimelineBuildError(f"{owner} must already be unit normalized")
    if _quaternion_sign_component(quaternion) < 0.0:
        raise TimelineBuildError(f"{owner} must use the canonical hemisphere")
    return tuple(_canonical_float(component) for component in quaternion)  # type: ignore[return-value]


@dataclass(frozen=True)
class M2CanaryTrajectory:
    """The actor-space root path used by the bounded M2 custom-room canary."""

    # Keep the dog on the camera side of the custom room's x=0 partition.
    # The 2 cm lift clears the measured animated skin minimum (-12.98 mm)
    # without introducing a visually meaningful hover.
    start_translation_m: tuple[float, float, float] = (-0.15, 0.02, 0.8)
    end_translation_m: tuple[float, float, float] = (-0.15, 0.02, -0.8)
    # This reviewed Rocketbox source faces actor +X.  A +90 degree world yaw
    # maps that declared source-facing direction to AVEngine world -Z.
    rotation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.7071067811865475,
        0.0,
        0.7071067811865476,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_translation_m",
            _finite_vector3(
                self.start_translation_m, owner="trajectory.start_translation_m"
            ),
        )
        object.__setattr__(
            self,
            "end_translation_m",
            _finite_vector3(
                self.end_translation_m, owner="trajectory.end_translation_m"
            ),
        )
        object.__setattr__(
            self,
            "rotation_xyzw",
            _canonical_unit_quaternion(
                self.rotation_xyzw, owner="trajectory.rotation_xyzw"
            ),
        )


def _runtime_joint_order(asset: Mapping[str, Any]) -> tuple[str, ...]:
    skeleton = asset.get("skeleton")
    raw = skeleton.get("runtime_joint_order") if isinstance(skeleton, Mapping) else None
    if not (
        isinstance(raw, list)
        and raw
        and all(isinstance(name, str) and name for name in raw)
        and len(set(raw)) == len(raw)
    ):
        raise TimelineBuildError(
            "asset.skeleton.runtime_joint_order must contain unique joint names"
        )
    return tuple(raw)


def _validate_contact_phases(
    actions: BakedActionSet,
    value: Mapping[str, Sequence[Sequence[bool]]],
) -> dict[str, tuple[tuple[bool, ...], ...]]:
    if not isinstance(value, Mapping):
        raise TimelineBuildError("contact_phases must be a mapping")
    expected_ids = {action.semantic_action_id for action in actions.actions}
    if set(value) != expected_ids:
        raise TimelineBuildError(
            "contact_phases must contain exactly the baked idle and walk actions"
        )
    result: dict[str, tuple[tuple[bool, ...], ...]] = {}
    for action in actions.actions:
        raw_frames = value[action.semantic_action_id]
        if isinstance(raw_frames, (str, bytes)):
            raise TimelineBuildError(
                f"contact_phases[{action.semantic_action_id!r}] must be a sequence"
            )
        frames = tuple(tuple(frame) for frame in raw_frames)
        if len(frames) != action.sample_count:
            raise TimelineBuildError(
                f"contact_phases[{action.semantic_action_id!r}] must have "
                f"{action.sample_count} frames"
            )
        for frame_index, frame in enumerate(frames):
            if len(frame) != len(CONTACT_ORDER) or any(
                not isinstance(item, bool) for item in frame
            ):
                raise TimelineBuildError(
                    f"contact_phases[{action.semantic_action_id!r}]"
                    f"[{frame_index}] must contain four booleans"
                )
        result[action.semantic_action_id] = frames
    if any(not all(frame) for frame in result["idle"]):
        raise TimelineBuildError(
            "every idle contact frame must keep all four paws down"
        )
    return result


def _actor_translation(
    frame_index: int, trajectory: M2CanaryTrajectory
) -> tuple[float, float, float]:
    if frame_index < IDLE_LEAD_FRAME_COUNT:
        return trajectory.start_translation_m
    if frame_index >= IDLE_LEAD_FRAME_COUNT + WALK_FRAME_COUNT:
        return trajectory.end_translation_m
    walk_index = frame_index - IDLE_LEAD_FRAME_COUNT
    denominator = WALK_FRAME_COUNT - 1
    fraction = walk_index / denominator
    start = np.asarray(trajectory.start_translation_m, dtype=np.float64)
    end = np.asarray(trajectory.end_translation_m, dtype=np.float64)
    value = start + fraction * (end - start)
    return tuple(_canonical_float(component) for component in value)  # type: ignore[return-value]


def _segment_action(frame_index: int) -> tuple[str, int]:
    if frame_index < IDLE_LEAD_FRAME_COUNT:
        return "idle", frame_index
    if frame_index < IDLE_LEAD_FRAME_COUNT + WALK_FRAME_COUNT:
        return "walk", frame_index - IDLE_LEAD_FRAME_COUNT
    return "idle", frame_index - IDLE_LEAD_FRAME_COUNT - WALK_FRAME_COUNT


def build_m2_state_sequence(
    *,
    asset: Mapping[str, Any],
    asset_manifest_sha256: str,
    actions: BakedActionSet,
    contact_phases: Mapping[str, Sequence[Sequence[bool]]],
    trajectory: M2CanaryTrajectory | None = None,
) -> list[dict[str, Any]]:
    """Expand baked loops into the exact 75 hash-bound states.

    This function deliberately does not inspect ``asset.admission_state``.  It
    is useful for preparing human-review media for a research candidate.  Use
    :func:`build_m2_capture_request` for a formal request; that function
    requires and semantically validates ``canary_qualified`` admission.
    """

    if not isinstance(asset, Mapping):
        raise TimelineBuildError("asset must be a mapping")
    if not _is_lower_sha256(asset_manifest_sha256):
        raise TimelineBuildError("asset_manifest_sha256 must be lowercase SHA-256")
    if not isinstance(actions, BakedActionSet):
        raise TimelineBuildError("actions must be a BakedActionSet")
    joint_order = _runtime_joint_order(asset)
    if actions.runtime_joint_order != joint_order:
        raise TimelineBuildError(
            "baked action joint order does not match the animal package"
        )
    phases = _validate_contact_phases(actions, contact_phases)
    path = trajectory if trajectory is not None else M2CanaryTrajectory()
    if not isinstance(path, M2CanaryTrajectory):
        raise TimelineBuildError("trajectory must be an M2CanaryTrajectory")

    states: list[dict[str, Any]] = []
    for frame_index in range(FRAME_COUNT):
        action_id, segment_frame_index = _segment_action(frame_index)
        clip = actions.action(action_id)
        clip_frame_index = segment_frame_index % clip.sample_count
        action_time_ticks = clip.sample_ticks[clip_frame_index]
        rotations = clip.rotations_xyzw[clip_frame_index]
        state: dict[str, Any] = {
            "frame_index": frame_index,
            "pts_ticks": frame_index * TICKS_PER_SAMPLE,
            "action_id": action_id,
            "action_time_ticks": action_time_ticks,
            "root_transform": {
                "translation_m": list(_actor_translation(frame_index, path)),
                "rotation_xyzw": list(path.rotation_xyzw),
            },
            "joint_states": [
                {
                    "joint_id": joint_id,
                    "rotation_xyzw": [
                        _canonical_float(component) for component in quaternion
                    ],
                }
                for joint_id, quaternion in zip(joint_order, rotations, strict=True)
            ],
            "contact_states": [
                {"contact_id": contact_id, "in_contact": in_contact}
                for contact_id, in_contact in zip(
                    CONTACT_ORDER,
                    phases[action_id][clip_frame_index],
                    strict=True,
                )
            ],
            "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
        }
        # The hash helpers only consume fields already present above.  Add the
        # declarations afterwards so neither hash can accidentally self-bind.
        state["pose_hash"] = compute_pose_hash(dict(asset), state)
        state["applied_state_hash"] = compute_applied_state_hash(
            dict(asset),
            state,
            asset_manifest_sha256=asset_manifest_sha256,
        )
        states.append(state)
    return states


def build_m2_capture_request(
    *,
    asset: dict[str, Any],
    asset_manifest_sha256: str,
    actions: BakedActionSet,
    contact_phases: Mapping[str, Sequence[Sequence[bool]]],
    request_id: str,
    room_id: str,
    seed: int,
    trajectory: M2CanaryTrajectory | None = None,
) -> dict[str, Any]:
    """Build and independently validate one formal M2 capture request."""

    if asset.get("admission_state") != "canary_qualified":
        raise TimelineBuildError(
            "formal M2 request creation requires a canary_qualified asset"
        )
    request = _build_m2_request_data(
        asset=asset,
        asset_manifest_sha256=asset_manifest_sha256,
        actions=actions,
        contact_phases=contact_phases,
        request_id=request_id,
        room_id=room_id,
        seed=seed,
        trajectory=trajectory,
    )
    errors = validate_capture_request(
        request,
        asset=asset,
        asset_manifest_sha256=asset_manifest_sha256,
    )
    if errors:
        raise TimelineBuildError(
            "constructed M2 request failed validation: " + "; ".join(errors)
        )
    return request


def _build_m2_request_data(
    *,
    asset: dict[str, Any],
    asset_manifest_sha256: str,
    actions: BakedActionSet,
    contact_phases: Mapping[str, Sequence[Sequence[bool]]],
    request_id: str,
    room_id: str,
    seed: int,
    trajectory: M2CanaryTrajectory | None,
) -> dict[str, Any]:
    """Construct shared request bytes without weakening either admission gate."""

    if not isinstance(request_id, str) or not request_id:
        raise TimelineBuildError("request_id must be a non-empty string")
    if not isinstance(room_id, str) or not room_id:
        raise TimelineBuildError("room_id must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TimelineBuildError("seed must be an integer")
    runtime_joint_order = list(_runtime_joint_order(asset))
    states = build_m2_state_sequence(
        asset=asset,
        asset_manifest_sha256=asset_manifest_sha256,
        actions=actions,
        contact_phases=contact_phases,
        trajectory=trajectory,
    )
    return {
        "schema": CAPTURE_SCHEMA,
        "request_id": request_id,
        "room_id": room_id,
        "asset_id": asset["asset_id"],
        "asset_manifest_sha256": asset_manifest_sha256,
        "seed": seed,
        "camera_rig_id": "camera_rig_0",
        "listener_id": "listener0",
        "view_ids": list(FORMAL_VIEW_IDS),
        "modalities": list(FORMAL_MODALITIES),
        "runtime_joint_order": runtime_joint_order,
        "contact_order": list(CONTACT_ORDER),
        "pose_hash_algorithm": POSE_HASH_ALGORITHM,
        "applied_state_hash_algorithm": APPLIED_STATE_HASH_ALGORITHM,
        "capture_policy": {
            "state_evaluation": "explicit_fixed_state",
            "advance_clock_between_modalities": False,
            "free_running_animation": False,
        },
        "states": states,
    }


def build_m2_research_review_request(
    *,
    asset: dict[str, Any],
    asset_manifest_sha256: str,
    actions: BakedActionSet,
    contact_phases: Mapping[str, Sequence[Sequence[bool]]],
    request_id: str,
    room_id: str,
    seed: int,
    trajectory: M2CanaryTrajectory | None = None,
) -> dict[str, Any]:
    """Build a 75-state request whose only formal blocker is admission.

    The returned object deliberately uses the formal request schema so every
    timing, state, hash, single-view, and modality constraint is exercised.
    It is accepted only by the separately named review loader and cannot pass
    :func:`validate_capture_request` until a hash-bound human review promotes
    the package.
    """

    if asset.get("admission_state") != "research_candidate":
        raise TimelineBuildError(
            "research review request creation requires a research_candidate asset"
        )
    request = _build_m2_request_data(
        asset=asset,
        asset_manifest_sha256=asset_manifest_sha256,
        actions=actions,
        contact_phases=contact_phases,
        request_id=request_id,
        room_id=room_id,
        seed=seed,
        trajectory=trajectory,
    )
    errors = validate_capture_request(
        request,
        asset=asset,
        asset_manifest_sha256=asset_manifest_sha256,
    )
    expected = ["M2 capture accepts only a canary_qualified animal package"]
    if errors != expected:
        raise TimelineBuildError(
            "research review request has blockers beyond formal admission: "
            + "; ".join(errors)
        )
    return request


__all__ = [
    "FRAME_COUNT",
    "IDLE_LEAD_FRAME_COUNT",
    "IDLE_TAIL_FRAME_COUNT",
    "M2CanaryTrajectory",
    "TimelineBuildError",
    "WALK_FRAME_COUNT",
    "build_m2_capture_request",
    "build_m2_research_review_request",
    "build_m2_state_sequence",
]
