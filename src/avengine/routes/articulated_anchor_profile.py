"""Reusable articulated emitter anchors without per-episode visual rendering.

One validated articulated capture can reduce each action sample to an
actor-local emitter offset.  Later routes apply those offsets to the same
root/orientation and locomotion schedule used by Timeline v2.  The profile is
keyed by declared asset/action/sample identities, so it is body-plan agnostic:
dogs, cats, birds or people use the same math once their own profile exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.capture.mixed_capture import (
    MixedCaptureError,
    locomotion_schedule_from_root_trajectory,
    trajectory_world_matrices,
)


PROFILE_SCHEMA = "avengine_articulated_emitter_anchor_profile_v1"
DEFAULT_REPEAT_TOLERANCE_M = 2.0e-5


class ArticulatedAnchorProfileError(ValueError):
    """Captured action samples cannot form a reusable emitter profile."""


@dataclass(frozen=True)
class AnchorProfileSpec:
    source_endpoint_id: str
    actor_id: str
    asset_id: str
    record_key: str
    anchor_id: str
    anchor_record_key: str
    capture_matrix_index: int
    local_anatomical_forward_axis: tuple[float, float, float]
    action_sample_counts: Mapping[str, int]


def _points(value: Any, *, owner: str) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArticulatedAnchorProfileError(f"{owner} must be finite [N,3]") from exc
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise ArticulatedAnchorProfileError(f"{owner} must be finite [N,3]")
    return np.ascontiguousarray(points)


def _matrices(value: Any) -> np.ndarray:
    try:
        matrices = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArticulatedAnchorProfileError(
            "actor_world_matrices must be finite [N,A,4,4]"
        ) from exc
    if (
        matrices.ndim != 4
        or matrices.shape[0] < 2
        or matrices.shape[2:] != (4, 4)
        or not np.all(np.isfinite(matrices))
        or not np.allclose(
            matrices[:, :, 3, :],
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ArticulatedAnchorProfileError(
            "actor_world_matrices must be finite rigid [N,A,4,4] matrices"
        )
    rotations = matrices[:, :, :3, :3]
    orthogonality = np.matmul(np.swapaxes(rotations, -1, -2), rotations)
    if not np.allclose(
        orthogonality,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-7,
    ) or not np.allclose(np.linalg.det(rotations), 1.0, rtol=0.0, atol=1.0e-7):
        raise ArticulatedAnchorProfileError(
            "actor_world_matrices rotations must be proper orthonormal matrices"
        )
    return np.ascontiguousarray(matrices)


def _sample_counts(value: Mapping[str, int], *, owner: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ArticulatedAnchorProfileError(f"{owner} action_sample_counts are missing")
    result: dict[str, int] = {}
    for action_id, count in value.items():
        if (
            not isinstance(action_id, str)
            or not action_id
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ArticulatedAnchorProfileError(
                f"{owner} action_sample_counts are invalid"
            )
        result[action_id] = count
    return dict(sorted(result.items()))


def compile_articulated_anchor_profile(
    *,
    actor_world_matrices: Any,
    frame_records: Sequence[Mapping[str, Any]],
    specs: Sequence[AnchorProfileSpec],
    repeat_tolerance_m: float = DEFAULT_REPEAT_TOLERANCE_M,
) -> dict[str, Any]:
    """Compile actor-local emitter offsets and close repeated sample readback."""

    matrices = _matrices(actor_world_matrices)
    if len(frame_records) != matrices.shape[0]:
        raise ArticulatedAnchorProfileError(
            "frame_records and actor_world_matrices must have equal length"
        )
    if not math.isfinite(float(repeat_tolerance_m)) or repeat_tolerance_m <= 0.0:
        raise ArticulatedAnchorProfileError("repeat_tolerance_m must be positive")
    if not specs or len({spec.source_endpoint_id for spec in specs}) != len(specs):
        raise ArticulatedAnchorProfileError(
            "profile specs must have unique source_endpoint_id values"
        )

    actors: list[dict[str, Any]] = []
    for spec in specs:
        if not 0 <= spec.capture_matrix_index < matrices.shape[1]:
            raise ArticulatedAnchorProfileError(
                f"{spec.actor_id} capture_matrix_index is out of range"
            )
        counts = _sample_counts(spec.action_sample_counts, owner=spec.actor_id)
        axis = np.asarray(spec.local_anatomical_forward_axis, dtype=np.float64)
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or not math.isclose(float(np.linalg.norm(axis)), 1.0, abs_tol=1.0e-9)
            or math.hypot(float(axis[0]), float(axis[2])) <= 1.0e-12
        ):
            raise ArticulatedAnchorProfileError(
                f"{spec.actor_id} local anatomical forward axis is invalid"
            )
        grouped: dict[tuple[str, int], list[np.ndarray]] = {}
        for frame_index, record in enumerate(frame_records):
            actor = record.get(spec.record_key)
            if not isinstance(actor, Mapping):
                raise ArticulatedAnchorProfileError(
                    f"frame {frame_index} lacks {spec.record_key!r} actor data"
                )
            action_id = actor.get("action_id")
            sample_index = actor.get("action_sample_index")
            if (
                action_id not in counts
                or isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or not 0 <= sample_index < counts[action_id]
            ):
                raise ArticulatedAnchorProfileError(
                    f"frame {frame_index} {spec.actor_id} action sample is invalid"
                )
            anchor = np.asarray(actor.get(spec.anchor_record_key), dtype=np.float64)
            if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
                raise ArticulatedAnchorProfileError(
                    f"frame {frame_index} {spec.anchor_id} is invalid"
                )
            world_from_actor = matrices[frame_index, spec.capture_matrix_index]
            rotation = world_from_actor[:3, :3]
            actor_from_anchor = rotation.T @ (anchor - world_from_actor[:3, 3])
            grouped.setdefault((str(action_id), sample_index), []).append(
                actor_from_anchor
            )

        samples: list[dict[str, Any]] = []
        maximum_repeat_error = 0.0
        for (action_id, sample_index), values in sorted(grouped.items()):
            stack = np.stack(values, axis=0)
            offset = np.mean(stack, axis=0)
            error = float(np.max(np.linalg.norm(stack - offset, axis=1)))
            maximum_repeat_error = max(maximum_repeat_error, error)
            if error > repeat_tolerance_m:
                raise ArticulatedAnchorProfileError(
                    f"{spec.actor_id} {action_id}[{sample_index}] emitter repeat "
                    f"error {error} exceeds {repeat_tolerance_m} m"
                )
            samples.append(
                {
                    "action_id": action_id,
                    "sample_index": sample_index,
                    "actor_from_anchor_translation_m": offset.tolist(),
                    "observation_count": len(values),
                    "maximum_repeat_error_m": error,
                }
            )
        actors.append(
            {
                "source_endpoint_id": spec.source_endpoint_id,
                "actor_id": spec.actor_id,
                "asset_id": spec.asset_id,
                "anchor_id": spec.anchor_id,
                "local_anatomical_forward_axis": axis.tolist(),
                "action_sample_counts": counts,
                "samples": samples,
                "maximum_repeat_error_m": maximum_repeat_error,
            }
        )
    return {
        "schema": PROFILE_SCHEMA,
        "status": "pass",
        "profile_semantics": (
            "validated actor-local emitter translation by asset/action/sample"
        ),
        "body_plan_policy": "profile_driven_no_species_branch",
        "repeat_tolerance_m": float(repeat_tolerance_m),
        "actors": actors,
    }


def materialize_articulated_anchor_paths(
    profile: Mapping[str, Any],
    *,
    actor_root_paths: Mapping[str, Any],
    actor_fallback_forwards_xz: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Apply a validated anchor profile to new equal-clock actor routes."""

    if profile.get("schema") != PROFILE_SCHEMA or profile.get("status") != "pass":
        raise ArticulatedAnchorProfileError(
            "anchor profile is not a passing v1 profile"
        )
    actors = profile.get("actors")
    if not isinstance(actors, list) or not actors:
        raise ArticulatedAnchorProfileError("anchor profile actors are missing")
    output: dict[str, np.ndarray] = {}
    frame_count: int | None = None
    for actor in actors:
        if not isinstance(actor, Mapping):
            raise ArticulatedAnchorProfileError("anchor profile actor is invalid")
        actor_id = actor.get("actor_id")
        source_id = actor.get("source_endpoint_id")
        if not isinstance(actor_id, str) or not isinstance(source_id, str):
            raise ArticulatedAnchorProfileError("anchor profile identity is invalid")
        try:
            roots = _points(actor_root_paths[actor_id], owner=f"{actor_id} root path")
            fallback = np.asarray(
                actor_fallback_forwards_xz[actor_id], dtype=np.float64
            )
        except KeyError as exc:
            raise ArticulatedAnchorProfileError(
                f"new route lacks actor {actor_id!r}"
            ) from exc
        if fallback.shape != (2,) or not np.all(np.isfinite(fallback)):
            raise ArticulatedAnchorProfileError(
                f"{actor_id} fallback forward must be finite [2]"
            )
        if frame_count is None:
            frame_count = len(roots)
        elif len(roots) != frame_count:
            raise ArticulatedAnchorProfileError("actor routes must have equal length")
        counts = _sample_counts(actor.get("action_sample_counts"), owner=actor_id)
        try:
            schedule = locomotion_schedule_from_root_trajectory(
                roots, action_sample_counts=counts
            )
            matrices = trajectory_world_matrices(
                roots,
                local_forward_axis=actor["local_anatomical_forward_axis"],
                fallback_forward_xz=fallback,
            )
        except (MixedCaptureError, KeyError) as exc:
            raise ArticulatedAnchorProfileError(str(exc)) from exc
        sample_offsets: dict[tuple[str, int], np.ndarray] = {}
        for sample in actor.get("samples", ()):
            if not isinstance(sample, Mapping):
                raise ArticulatedAnchorProfileError(
                    f"{actor_id} profile sample is invalid"
                )
            key = (sample.get("action_id"), sample.get("sample_index"))
            if (
                not isinstance(key[0], str)
                or isinstance(key[1], bool)
                or not isinstance(key[1], int)
                or key in sample_offsets
            ):
                raise ArticulatedAnchorProfileError(
                    f"{actor_id} profile sample identity is invalid"
                )
            sample_offsets[key] = np.asarray(
                sample.get("actor_from_anchor_translation_m"), dtype=np.float64
            )
        anchors = np.empty((len(roots), 3), dtype=np.float64)
        for frame_index, state in enumerate(schedule):
            key = (state.action_id, state.action_sample_index)
            offset = sample_offsets.get(key)
            if (
                offset is None
                or offset.shape != (3,)
                or not np.all(np.isfinite(offset))
            ):
                raise ArticulatedAnchorProfileError(
                    f"{actor_id} profile lacks required {key[0]}[{key[1]}] sample"
                )
            anchors[frame_index] = (
                matrices[frame_index, :3, :3] @ offset + matrices[frame_index, :3, 3]
            )
        output[source_id] = np.ascontiguousarray(anchors)
    return dict(sorted(output.items()))


__all__ = [
    "AnchorProfileSpec",
    "ArticulatedAnchorProfileError",
    "DEFAULT_REPEAT_TOLERANCE_M",
    "PROFILE_SCHEMA",
    "compile_articulated_anchor_profile",
    "materialize_articulated_anchor_paths",
]
