"""Cadence-bound world-space contact inference for the M2 canary.

The first M2 contact report deliberately operated in actor space.  That is a
useful animation diagnostic, but it cannot decide whether a translating actor
slides over the floor.  This module adds the missing join: it identifies the
low, rearward-moving stance portion of each paw cycle, fits one deterministic
root step to those samples, and measures the resulting world-space residual.

The implementation is intentionally body-plan neutral at the math boundary.
Callers provide ordered semantic contact anchors and the authored actor
forward axis; canine, feline and equine profiles may therefore use the same
solver without sharing joint names or anatomical thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from avengine.m2.actions import BakedActionSet, baked_actions_content_sha256
from avengine.m2.kinematics import (
    CONTACT_ORDER,
    AnchorDefinition,
    HabitatAssetMapping,
    resolve_actor_anchors,
)


WORLD_CONTACT_SCHEMA = "avengine_m2_world_contact_audit_v1"
CONTACT_PHASES_SCHEMA = "avengine_m2_contact_phases_v1"


class WorldContactError(ValueError):
    """Cadence/contact inputs cannot support a deterministic world audit."""


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _unit_quaternion(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise WorldContactError("root rotation must be finite xyzw")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise WorldContactError("root rotation must already be unit normalized")
    return quaternion


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_vector = quaternion[:3]
    first = np.cross(q_vector, vector)
    second = np.cross(q_vector, first)
    return vector + 2.0 * (quaternion[3] * first + second)


def infer_height_backward_stance(
    positions_actor_m: np.ndarray,
    *,
    forward_axis: int = 0,
    forward_sign: float = 1.0,
    contact_height_fraction: float = 0.35,
) -> tuple[bool, ...]:
    """Return cyclic stance labels supported by both height and cadence.

    A walking paw is in stance only while it is near the bottom of its cycle
    and moving rearward relative to the actor.  Centered cyclic velocity keeps
    the endpoint-exclusive loop symmetric and avoids classifying the forward
    swing as contact merely because it happens near the floor.
    """

    positions = np.asarray(positions_actor_m, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[0] < 3
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise WorldContactError("paw positions must be finite Nx3 with N >= 3")
    if forward_axis not in {0, 2}:
        raise WorldContactError("forward_axis must be actor X (0) or Z (2)")
    if forward_sign not in {-1.0, 1.0}:
        raise WorldContactError("forward_sign must be -1.0 or 1.0")
    if not 0.0 < float(contact_height_fraction) < 1.0:
        raise WorldContactError("contact_height_fraction must be in (0, 1)")

    heights = positions[:, 1]
    height_threshold = float(np.min(heights)) + float(contact_height_fraction) * float(
        np.ptp(heights)
    )
    forward = forward_sign * positions[:, forward_axis]
    centered_velocity = 0.5 * (np.roll(forward, -1) - np.roll(forward, 1))
    states = (heights <= height_threshold) & (centered_velocity < 0.0)
    if int(np.count_nonzero(states)) < 3:
        raise WorldContactError(
            "paw cycle has fewer than three supported stance frames"
        )
    if not np.any(states & np.roll(states, 1)):
        raise WorldContactError("paw cycle has no consecutive supported stance pair")
    return tuple(bool(value) for value in states)


@dataclass(frozen=True)
class RootStepFit:
    step_m: float
    direction_world: tuple[float, float, float]
    maximum_contact_horizontal_step_m: float
    mean_contact_horizontal_step_m: float
    contact_pair_count: int

    def to_json_data(self) -> dict[str, Any]:
        return {
            "step_m": _canonical_float(self.step_m),
            "direction_world": [
                _canonical_float(component) for component in self.direction_world
            ],
            "maximum_contact_horizontal_step_m": _canonical_float(
                self.maximum_contact_horizontal_step_m
            ),
            "mean_contact_horizontal_step_m": _canonical_float(
                self.mean_contact_horizontal_step_m
            ),
            "contact_pair_count": self.contact_pair_count,
        }


def fit_constant_root_step(
    positions_by_contact: Sequence[np.ndarray],
    states_by_contact: Sequence[Sequence[bool]],
    *,
    root_rotation_xyzw: Sequence[float],
    actor_forward: Sequence[float] = (1.0, 0.0, 0.0),
    minimum_step_m: float = 0.005,
    maximum_step_m: float = 0.04,
    grid_step_m: float = 0.0001,
) -> RootStepFit:
    """Fit a deterministic constant root step by minimax stance residual."""

    if len(positions_by_contact) != len(states_by_contact) or not positions_by_contact:
        raise WorldContactError(
            "positions and stance states must be non-empty and aligned"
        )
    quaternion = _unit_quaternion(root_rotation_xyzw)
    forward = np.asarray(actor_forward, dtype=np.float64)
    if forward.shape != (3,) or not np.all(np.isfinite(forward)):
        raise WorldContactError("actor_forward must be a finite vec3")
    forward_norm = float(np.linalg.norm(forward))
    if not math.isclose(forward_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise WorldContactError("actor_forward must already be unit normalized")
    direction = _rotate_vector(quaternion, forward)
    if (
        not 0.0 <= minimum_step_m <= maximum_step_m
        or grid_step_m <= 0.0
        or not all(
            math.isfinite(value)
            for value in (minimum_step_m, maximum_step_m, grid_step_m)
        )
    ):
        raise WorldContactError("root-step search bounds are invalid")

    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for positions_value, states_value in zip(
        positions_by_contact, states_by_contact, strict=True
    ):
        positions = np.asarray(positions_value, dtype=np.float64)
        states = np.asarray(states_value, dtype=np.bool_)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or states.shape != (positions.shape[0],)
            or not np.all(np.isfinite(positions))
        ):
            raise WorldContactError(
                "each contact trajectory/state pair must be aligned"
            )
        rotated = np.asarray(
            [_rotate_vector(quaternion, point) for point in positions],
            dtype=np.float64,
        )
        prepared.append((rotated, states))

    candidate_count = (
        int(math.floor((maximum_step_m - minimum_step_m) / grid_step_m + 1.0e-12)) + 1
    )
    best: tuple[float, float, float, int] | None = None
    for index in range(candidate_count):
        step = minimum_step_m + index * grid_step_m
        root_delta = direction * step
        residuals: list[float] = []
        for positions, states in prepared:
            for sample_index in range(len(states)):
                previous = (sample_index - 1) % len(states)
                if states[sample_index] and states[previous]:
                    delta = root_delta + positions[sample_index] - positions[previous]
                    residuals.append(float(np.linalg.norm(delta[[0, 2]])))
        if not residuals:
            raise WorldContactError("stance inference produced no cyclic contact pairs")
        key = (max(residuals), float(np.mean(residuals)), step, len(residuals))
        if best is None or key[:3] < best[:3]:
            best = key
    assert best is not None
    return RootStepFit(
        step_m=_canonical_float(best[2]),
        direction_world=tuple(_canonical_float(value) for value in direction),
        maximum_contact_horizontal_step_m=_canonical_float(best[0]),
        mean_contact_horizontal_step_m=_canonical_float(best[1]),
        contact_pair_count=best[3],
    )


def _anchor_trajectories(
    mapping: HabitatAssetMapping,
    actions: BakedActionSet,
    anchors: tuple[AnchorDefinition, ...],
) -> dict[str, np.ndarray]:
    if tuple(anchor.anchor_id for anchor in anchors) != CONTACT_ORDER:
        raise WorldContactError(f"anchors must follow CONTACT_ORDER {CONTACT_ORDER}")
    if actions.runtime_joint_order != mapping.runtime_joint_order:
        raise WorldContactError("actions and Habitat mapping joint orders differ")
    if actions.source_glb_sha256 != mapping.source_glb_sha256:
        raise WorldContactError("actions and Habitat mapping visual hashes differ")
    result: dict[str, np.ndarray] = {}
    for action in actions.actions:
        frames: list[list[tuple[float, float, float]]] = []
        for pose in action.rotations_xyzw:
            resolved = resolve_actor_anchors(mapping, pose, anchors)
            frames.append(
                [
                    resolved.anchor_transform(contact_id).translation_m
                    for contact_id in CONTACT_ORDER
                ]
            )
        result[action.semantic_action_id] = np.asarray(frames, dtype=np.float64)
    if set(result) != {"idle", "walk"}:
        raise WorldContactError("actions must contain exactly idle and walk")
    return result


def derive_cadence_locked_contact_artifacts(
    mapping: HabitatAssetMapping,
    actions: BakedActionSet,
    anchors: tuple[AnchorDefinition, ...],
    *,
    root_start_translation_m: Sequence[float] = (-0.15, 0.02, 0.8),
    root_rotation_xyzw: Sequence[float] = (
        0.0,
        0.7071067811865475,
        0.0,
        0.7071067811865476,
    ),
    walk_frame_count: int = 45,
    sample_rate_hz: int = 15,
    maximum_contact_horizontal_step_m: float = 0.015,
    contact_height_fraction: float = 0.35,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a package-compatible contact report and its world-space audit."""

    if walk_frame_count < 2 or sample_rate_hz <= 0:
        raise WorldContactError("walk frame count and sample rate must be positive")
    start = np.asarray(root_start_translation_m, dtype=np.float64)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise WorldContactError("root start translation must be a finite vec3")
    trajectories = _anchor_trajectories(mapping, actions, anchors)
    walk = trajectories["walk"]
    idle = trajectories["idle"]
    stance_by_contact = tuple(
        infer_height_backward_stance(
            walk[:, index], contact_height_fraction=contact_height_fraction
        )
        for index in range(len(CONTACT_ORDER))
    )
    fit = fit_constant_root_step(
        [walk[:, index] for index in range(len(CONTACT_ORDER))],
        stance_by_contact,
        root_rotation_xyzw=root_rotation_xyzw,
    )
    direction = np.asarray(fit.direction_world, dtype=np.float64)
    root_rotation = _unit_quaternion(root_rotation_xyzw)
    root_delta = direction * fit.step_m
    world_metrics: dict[str, dict[str, Any]] = {}
    for contact_index, (contact_id, states_value) in enumerate(
        zip(CONTACT_ORDER, stance_by_contact, strict=True)
    ):
        positions = np.asarray(
            [_rotate_vector(root_rotation, point) for point in walk[:, contact_index]],
            dtype=np.float64,
        )
        states = np.asarray(states_value, dtype=np.bool_)
        residuals = [
            float(
                np.linalg.norm(
                    (
                        root_delta
                        + positions[index]
                        - positions[(index - 1) % len(positions)]
                    )[[0, 2]]
                )
            )
            for index in range(len(states))
            if states[index] and states[index - 1]
        ]
        world_metrics[contact_id] = {
            "contact_pair_count": len(residuals),
            "maximum_contact_horizontal_step_m": _canonical_float(max(residuals)),
            "mean_contact_horizontal_step_m": _canonical_float(
                float(np.mean(residuals))
            ),
        }
    end = start + direction * fit.step_m * (walk_frame_count - 1)
    status = (
        "pass"
        if fit.maximum_contact_horizontal_step_m <= maximum_contact_horizontal_step_m
        else "fail"
    )

    action_records: list[dict[str, Any]] = []
    for action_id in ("idle", "walk"):
        clip = actions.action(action_id)
        states = (
            tuple((True,) * clip.sample_count for _ in CONTACT_ORDER)
            if action_id == "idle"
            else stance_by_contact
        )
        metrics: list[dict[str, Any]] = []
        positions = idle if action_id == "idle" else walk
        for contact_index, contact_id in enumerate(CONTACT_ORDER):
            values = positions[:, contact_index]
            heights = values[:, 1]
            state_array = np.asarray(states[contact_index], dtype=np.bool_)
            metrics.append(
                {
                    "contact_id": contact_id,
                    "inference_mode": (
                        "forced_idle_contact"
                        if action_id == "idle"
                        else "height_backward_velocity_world_locked"
                    ),
                    "confidence": "high",
                    "idle_reference_height_m": _canonical_float(
                        float(np.median(idle[:, contact_index, 1]))
                    ),
                    "contact_height_threshold_m": _canonical_float(
                        float(np.min(heights))
                        + contact_height_fraction * float(np.ptp(heights))
                    ),
                    "minimum_height_m": _canonical_float(float(np.min(heights))),
                    "maximum_height_m": _canonical_float(float(np.max(heights))),
                    "vertical_range_m": _canonical_float(float(np.ptp(heights))),
                    "maximum_step_displacement_m": _canonical_float(
                        float(
                            np.max(
                                np.linalg.norm(
                                    values - np.roll(values, 1, axis=0), axis=1
                                )
                            )
                        )
                    ),
                    "maximum_horizontal_step_m": _canonical_float(
                        float(
                            np.max(
                                np.linalg.norm(
                                    (values - np.roll(values, 1, axis=0))[:, [0, 2]],
                                    axis=1,
                                )
                            )
                        )
                    ),
                    "maximum_contact_horizontal_step_m": (
                        0.0
                        if action_id == "idle"
                        else world_metrics[contact_id][
                            "maximum_contact_horizontal_step_m"
                        ]
                    ),
                    "contact_frame_count": int(np.count_nonzero(state_array)),
                    "swing_frame_count": int(
                        len(state_array) - np.count_nonzero(state_array)
                    ),
                }
            )
        action_records.append(
            {
                "semantic_action_id": action_id,
                "source_action_name": clip.source_action_name,
                "sample_count": clip.sample_count,
                "frames": [
                    {
                        "sample_index": index,
                        "sample_tick": clip.sample_ticks[index],
                        "source_time_seconds": clip.source_times_seconds[index],
                        "contacts": [
                            {
                                "contact_id": contact_id,
                                "in_contact": bool(states[contact_index][index]),
                            }
                            for contact_index, contact_id in enumerate(CONTACT_ORDER)
                        ],
                    }
                    for index in range(clip.sample_count)
                ],
                "metrics": metrics,
            }
        )

    actions_sha256 = baked_actions_content_sha256(actions)
    contact_report = {
        "schema": CONTACT_PHASES_SCHEMA,
        "source_glb_sha256": mapping.source_glb_sha256,
        "baked_actions_sha256": actions_sha256,
        "runtime_joint_order": list(mapping.runtime_joint_order),
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "sample_rate_hz": actions.sample_rate_hz,
        "time_base_hz": actions.time_base_hz,
        "contact_order": list(CONTACT_ORDER),
        "anchor_definitions": [anchor.to_json_data() for anchor in anchors],
        "thresholds": {
            "minimum_dynamic_vertical_range_m": 0.005,
            "contact_height_fraction": contact_height_fraction,
            "maximum_idle_vertical_range_m": 0.015,
            "maximum_idle_step_displacement_m": 0.003,
            "maximum_contact_horizontal_step_m": maximum_contact_horizontal_step_m,
        },
        "actions": action_records,
        "warnings": [],
        "notes": [
            "Walk stance requires both low height and rearward actor-relative velocity.",
            "The hash-bound world audit fits root cadence and gates stance residuals.",
        ],
    }
    contact_payload = (
        json.dumps(
            contact_report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")
    audit = {
        "schema": WORLD_CONTACT_SCHEMA,
        "status": status,
        "qualification_claim": False,
        "source_glb_sha256": mapping.source_glb_sha256,
        "baked_actions_sha256": actions_sha256,
        "contact_phases_sha256": hashlib.sha256(contact_payload).hexdigest(),
        "solver": {
            "solver_id": "height_backward_velocity_constant_root_minimax_v1",
            "contact_height_fraction": contact_height_fraction,
            "root_step_search_m": {
                "minimum": 0.005,
                "maximum": 0.04,
                "increment": 0.0001,
            },
        },
        "root_step_fit": fit.to_json_data(),
        "contacts": world_metrics,
        "trajectory": {
            "start_translation_m": [_canonical_float(value) for value in start],
            "end_translation_m": [_canonical_float(value) for value in end],
            "rotation_xyzw": [
                _canonical_float(value)
                for value in _unit_quaternion(root_rotation_xyzw)
            ],
            "walk_frame_count": walk_frame_count,
            "sample_rate_hz": sample_rate_hz,
            "path_length_m": _canonical_float(fit.step_m * (walk_frame_count - 1)),
            "root_speed_m_per_second": _canonical_float(fit.step_m * sample_rate_hz),
        },
        "gate": {
            "maximum_contact_horizontal_step_m": maximum_contact_horizontal_step_m,
            "measured_maximum_contact_horizontal_step_m": (
                fit.maximum_contact_horizontal_step_m
            ),
            "passed": status == "pass",
        },
        "stance_frames_by_contact": {
            contact_id: [index for index, state in enumerate(states) if state]
            for contact_id, states in zip(CONTACT_ORDER, stance_by_contact, strict=True)
        },
    }
    return contact_report, audit


__all__ = [
    "CONTACT_PHASES_SCHEMA",
    "WORLD_CONTACT_SCHEMA",
    "RootStepFit",
    "WorldContactError",
    "derive_cadence_locked_contact_artifacts",
    "fit_constant_root_step",
    "infer_height_backward_stance",
]
