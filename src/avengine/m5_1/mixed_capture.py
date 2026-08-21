"""Fixed-state Habitat capture for Rocketbox-human + Beagle routes.

The public path entrypoint consumes two equal-length actor trajectories.  It
derives each actor's idle/walk state from its authored root trajectory, samples
the action loops declared by that actor's runtime package, writes both
articulated states explicitly for every frame, and makes exactly one
co-located RGB/depth/semantic observation call.  RGB and semantic are retained;
depth is observed only to preserve the M1 shared-view contract.

The legacy-route wrapper reads the two paths verbatim from the committed M5.1
route manifest.  Neither entrypoint advances Habitat physics or uses a static
sliding human fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    resolve_declared_path,
    sha256_file,
)
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.evidence import array_sha256
from avengine.m1.habitat_capture import (
    InstalledHabitatRuntime,
    PBR_BRDF_LUT_RELATIVE_PATH,
    PBR_CONFIG_FILENAME,
    PBR_ENVIRONMENT_MAP_RELATIVE_PATH,
    _make_configuration,
    _resolved_assets,
    discover_runtime_root,
    prepare_installed_habitat_runtime,
)
from avengine.m2.contracts import (
    FORMAL_MODALITIES,
    load_and_validate_inputs as load_m2_inputs,
)
from avengine.m2.habitat import HabitatLinkJointBlock
from avengine.m2.habitat_capture import (
    HabitatCaptureError,
    _apply_root_with_habitat,
    _quaternion_block_error,
    _runtime_snapshot as dog_runtime_snapshot,
    _validate_observation_arrays,
    compile_frame_applications,
    load_research_review_inputs,
    load_runtime_asset_bundle,
    quaternion_xyzw_to_matrix,
)
from avengine.m2.local_tr_habitat import bind_local_tr_habitat_layout
from avengine.m2.local_tr_review import (
    _runtime_snapshot as human_runtime_snapshot,
    mixed_joint_readback_errors,
)
from avengine.m5.visual import (
    _instantiate_actor_with_semantic_template,
    _link_id_by_name,
    _node_world_position,
    _set_scene_node_semantic_readback,
)
from avengine.m5_1.human_runtime import (
    HEAD_LINK_NAME,
    MOUTH_LINK_NAME,
    HumanRuntimePackage,
    prepare_rocketbox_habitat_runtime,
)
from avengine.m5_1.legacy_route import (
    FRAME_RATE_HZ,
    assert_valid_route_manifest,
)


MIXED_CAPTURE_SCHEMA = "avengine_m5_1_human_beagle_capture_v1"
MIXED_CAPTURE_INSTALLED_SCHEMA_V2 = "avengine_m5_1_human_beagle_capture_v2"
HEADING_ALIGNMENT_SCHEMA = "avengine_m5_1_actor_heading_gate_v1"
HUMAN_SEMANTIC_ID = 220
BEAGLE_SEMANTIC_ID = 221
BEAGLE_MOUTH_LINK_NAME = "beagle Xtra Mouth"
TIME_BASE_HZ = 48_000
TICKS_PER_FRAME = TIME_BASE_HZ // FRAME_RATE_HZ
LEGACY_CAMERA_POSITION_M = (-0.7, 1.471, 0.65)
LEGACY_CAMERA_YAW_DEG = 55.0
LEGACY_CAMERA_HFOV_DEG = 105.0
M5_1_LIGHT_SETUP_KEY = "avengine_m5_1_room_lighting"
M5_1_ACTOR_SHADER_TYPE = "pbr"
M5_1_PBR_CONFIG_HANDLE = "avengine_m5_1_external_brown_photostudio_v1"
_ROOT_READBACK_ATOL = 2.0e-6
_JOINT_READBACK_ATOL = 2.0e-6
_LINK_MATRIX_READBACK_ATOL = 2.0e-5
_HEADING_ALIGNMENT_MAX_ERROR_DEG = 1.0e-6
LOCOMOTION_POLICY_ID = "authored_root_horizontal_speed_hysteresis_v1"
LOCOMOTION_WALK_ENTER_SPEED_M_S = 0.03
LOCOMOTION_IDLE_ENTER_SPEED_M_S = 0.015


class MixedCaptureError(RuntimeError):
    """The mixed M5.1 fixed-state capture or readback failed."""


@dataclass(frozen=True)
class MixedCaptureResult:
    """Retained arrays and evidence for one mixed articulated capture."""

    output_dir: Path
    rgb: np.ndarray
    semantic: np.ndarray
    actor_world_matrices: np.ndarray
    skin_root_world_matrices: np.ndarray
    anchor_positions_m: np.ndarray
    semantic_visibility_pixels: np.ndarray
    records: tuple[Mapping[str, Any], ...]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class LocomotionFrameState:
    """One action selection derived only from an authored actor-root path."""

    action_id: str
    action_frame_index: int
    action_sample_index: int
    action_phase: float
    horizontal_speed_m_s: float
    state_transition: bool


def _points(value: Any, *, owner: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MixedCaptureError(f"{owner} must be a finite [N,3] array") from exc
    if (
        array.ndim != 2
        or array.shape[0] < 2
        or array.shape[1] != 3
        or not np.all(np.isfinite(array))
    ):
        raise MixedCaptureError(f"{owner} must be a finite [N,3] array with N >= 2")
    return np.ascontiguousarray(array)


def locomotion_schedule_from_root_trajectory(
    points_m: Any,
    *,
    action_sample_counts: Mapping[str, int],
    frame_rate_hz: int = FRAME_RATE_HZ,
    walk_enter_speed_m_s: float = LOCOMOTION_WALK_ENTER_SPEED_M_S,
    idle_enter_speed_m_s: float = LOCOMOTION_IDLE_ENTER_SPEED_M_S,
) -> tuple[LocomotionFrameState, ...]:
    """Select ``idle``/``walk`` from root speed with deterministic hysteresis.

    The speed at a visual frame is the larger of its incoming and outgoing
    horizontal segment speeds.  This makes the pose switch to ``walk`` on the
    first authored moving frame and remain ``walk`` through the final moving
    frame.  A change of semantic action resets that action's sample clock.
    """

    points = _points(points_m, owner="actor root trajectory")
    try:
        sample_counts = {
            action_id: int(action_sample_counts[action_id])
            for action_id in ("idle", "walk")
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MixedCaptureError(
            "locomotion actions must declare positive idle and walk sample counts"
        ) from exc
    if (
        any(value < 1 for value in sample_counts.values())
        or isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, int)
        or frame_rate_hz < 1
        or not math.isfinite(float(walk_enter_speed_m_s))
        or not math.isfinite(float(idle_enter_speed_m_s))
        or idle_enter_speed_m_s < 0.0
        or walk_enter_speed_m_s <= idle_enter_speed_m_s
    ):
        raise MixedCaptureError(
            "locomotion thresholds/counts must be finite with "
            "0 <= idle-enter < walk-enter and positive sample counts"
        )

    segment_speeds = (
        np.linalg.norm(np.diff(points[:, (0, 2)], axis=0), axis=1)
        * frame_rate_hz
    )
    frame_speeds = np.empty(points.shape[0], dtype=np.float64)
    frame_speeds[0] = segment_speeds[0]
    frame_speeds[-1] = segment_speeds[-1]
    frame_speeds[1:-1] = np.maximum(segment_speeds[:-1], segment_speeds[1:])

    action_id = (
        "walk" if frame_speeds[0] >= walk_enter_speed_m_s else "idle"
    )
    action_frame_index = 0
    schedule: list[LocomotionFrameState] = []
    for frame_index, speed in enumerate(frame_speeds):
        selected = action_id
        if action_id == "idle" and speed >= walk_enter_speed_m_s:
            selected = "walk"
        elif action_id == "walk" and speed <= idle_enter_speed_m_s:
            selected = "idle"
        transition = frame_index == 0 or selected != action_id
        if transition:
            action_frame_index = 0
        else:
            action_frame_index += 1
        action_id = selected
        sample_count = sample_counts[action_id]
        sample_index = action_frame_index % sample_count
        schedule.append(
            LocomotionFrameState(
                action_id=action_id,
                action_frame_index=action_frame_index,
                action_sample_index=sample_index,
                action_phase=sample_index / sample_count,
                horizontal_speed_m_s=float(speed),
                state_transition=transition,
            )
        )
    return tuple(schedule)


def _locomotion_schedule_summary(
    schedule: Sequence[LocomotionFrameState],
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for frame_index, state in enumerate(schedule):
        if frame_index == 0 or state.action_id != schedule[frame_index - 1].action_id:
            runs.append(
                {
                    "action_id": state.action_id,
                    "start_frame": frame_index,
                    "end_frame_exclusive": frame_index + 1,
                }
            )
        else:
            runs[-1]["end_frame_exclusive"] = frame_index + 1
    return {
        "policy_id": LOCOMOTION_POLICY_ID,
        "horizontal_speed_authority": "authored_actor_root_trajectory_xz",
        "frame_speed_estimator": "max_incoming_outgoing_segment_speed",
        "walk_enter_speed_m_s": LOCOMOTION_WALK_ENTER_SPEED_M_S,
        "idle_enter_speed_m_s": LOCOMOTION_IDLE_ENTER_SPEED_M_S,
        "action_clock_reset_on_transition": True,
        "runs": runs,
    }


def _validate_used_action_render_evidence(
    *,
    actor_id: str,
    schedule: Sequence[LocomotionFrameState],
    pose_hashes_by_action: Mapping[str, set[str]],
) -> None:
    """Require render evidence only for actions selected by this route."""

    used_actions = {state.action_id for state in schedule}
    missing = sorted(
        action_id
        for action_id in used_actions
        if not pose_hashes_by_action.get(action_id)
    )
    if missing:
        raise MixedCaptureError(
            f"{actor_id} did not render pose evidence for used actions {missing}"
        )


def _axis_label_vector(value: Any, *, owner: str) -> tuple[float, float, float]:
    axes = {
        "+X": (1.0, 0.0, 0.0),
        "-X": (-1.0, 0.0, 0.0),
        "+Z": (0.0, 0.0, 1.0),
        "-Z": (0.0, 0.0, -1.0),
    }
    try:
        return axes[value]
    except (KeyError, TypeError) as exc:
        raise MixedCaptureError(
            f"{owner} must be one of {sorted(axes)} in the +Y-up actor frame"
        ) from exc


def _horizontal_unit_axis(value: Any, *, owner: str) -> np.ndarray:
    try:
        axis = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MixedCaptureError(
            f"{owner} must be a finite horizontal 3-vector"
        ) from exc
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise MixedCaptureError(f"{owner} must be a finite horizontal 3-vector")
    if not math.isclose(float(axis[1]), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise MixedCaptureError(f"{owner} must be horizontal in the +Y-up actor frame")
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise MixedCaptureError(f"{owner} must be nonzero")
    return np.ascontiguousarray(axis / norm)


def _trajectory_tangents(
    points_m: Any, *, fallback_forward_xz: Any | None = None
) -> np.ndarray:
    points = _points(points_m, owner="actor trajectory")
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[:, 1] = 0.0
    norms = np.linalg.norm(tangents, axis=1)
    moving_indices = np.flatnonzero(norms > 1.0e-12)
    if not len(moving_indices):
        if fallback_forward_xz is None:
            raise MixedCaptureError(
                "stationary actor trajectory requires an authored fallback_forward_xz"
            )
        try:
            fallback_xz = np.asarray(fallback_forward_xz, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MixedCaptureError(
                "authored fallback_forward_xz must be a finite nonzero 2-vector"
            ) from exc
        if fallback_xz.shape != (2,) or not np.all(np.isfinite(fallback_xz)):
            raise MixedCaptureError(
                "authored fallback_forward_xz must be a finite nonzero 2-vector"
            )
        fallback_norm = float(np.linalg.norm(fallback_xz))
        if fallback_norm <= 1.0e-12:
            raise MixedCaptureError(
                "authored fallback_forward_xz must be a finite nonzero 2-vector"
            )
        fallback = np.asarray(
            [fallback_xz[0] / fallback_norm, 0.0, fallback_xz[1] / fallback_norm],
            dtype=np.float64,
        )
        tangents[:] = fallback
        return np.ascontiguousarray(tangents)
    tangents[moving_indices] /= norms[moving_indices, None]
    for index in np.flatnonzero(norms <= 1.0e-12):
        nearest = moving_indices[
            int(np.argmin(np.abs(moving_indices - int(index))))
        ]
        tangents[index] = tangents[nearest]
    return np.ascontiguousarray(tangents)


def trajectory_world_matrices(
    points_m: Any,
    *,
    local_forward_axis: Any,
    fallback_forward_xz: Any | None = None,
) -> np.ndarray:
    """Create actor transforms that align an asset's forward axis to its path."""

    points = _points(points_m, owner="actor trajectory")
    tangents = _trajectory_tangents(
        points, fallback_forward_xz=fallback_forward_xz
    )
    local_forward = _horizontal_unit_axis(
        local_forward_axis, owner="local anatomical forward axis"
    )
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    local_right = np.cross(local_forward, up)
    local_right /= np.linalg.norm(local_right)
    local_basis = np.stack((local_right, up, -local_forward), axis=1)
    if not math.isclose(float(np.linalg.det(local_basis)), 1.0, abs_tol=1.0e-12):
        raise MixedCaptureError("local anatomical frame is not a proper rotation")
    matrices = np.repeat(
        np.eye(4, dtype=np.float64)[None, :, :], points.shape[0], axis=0
    )
    for index, forward in enumerate(tangents):
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        world_basis = np.stack((right, up, -forward), axis=1)
        rotation = world_basis @ local_basis.T
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-12):
            raise MixedCaptureError("trajectory tangent produced an improper rotation")
        matrices[index, :3, :3] = rotation
        matrices[index, :3, 3] = points[index]
    return np.ascontiguousarray(matrices)


def _actor_heading_evidence(
    *,
    actor_id: str,
    points_m: Any,
    actor_world_matrices: Any,
    local_forward_axis: Any,
    binding_source: Mapping[str, Any],
    fallback_forward_xz: Any | None = None,
) -> dict[str, Any]:
    """Retain and gate every frame's anatomical-forward/path alignment."""

    points = _points(points_m, owner=f"{actor_id} root path")
    matrices = np.asarray(actor_world_matrices, dtype=np.float64)
    frame_count = points.shape[0]
    if matrices.shape != (frame_count, 4, 4) or not np.all(np.isfinite(matrices)):
        raise MixedCaptureError(
            f"{actor_id} actor_world_matrices must be finite [{frame_count},4,4]"
        )
    local_forward = _horizontal_unit_axis(
        local_forward_axis, owner=f"{actor_id} local anatomical forward axis"
    )
    tangents = _trajectory_tangents(
        points, fallback_forward_xz=fallback_forward_xz
    )
    world_forwards = np.einsum(
        "nij,j->ni", matrices[:, :3, :3], local_forward
    )
    forward_norms = np.linalg.norm(world_forwards, axis=1)
    if np.any(forward_norms <= 1.0e-12):
        raise MixedCaptureError(f"{actor_id} produced a zero world anatomical forward")
    world_forwards /= forward_norms[:, None]
    dots = np.clip(np.sum(world_forwards * tangents, axis=1), -1.0, 1.0)
    cross_norms = np.linalg.norm(np.cross(world_forwards, tangents), axis=1)
    errors_deg = np.degrees(np.arctan2(cross_norms, dots))
    passed = errors_deg <= _HEADING_ALIGNMENT_MAX_ERROR_DEG
    maximum_index = int(np.argmax(errors_deg))
    if not bool(np.all(passed)):
        raise MixedCaptureError(
            f"{actor_id} anatomical forward/path heading gate failed at frame "
            f"{maximum_index}: {float(errors_deg[maximum_index]):.9f} degrees"
        )

    frames = [
        {
            "frame_index": index,
            "path_tangent_world": [float(value) for value in tangents[index]],
            "anatomical_forward_world": [
                float(value) for value in world_forwards[index]
            ],
            "alignment_dot": float(dots[index]),
            "heading_error_degrees": float(errors_deg[index]),
            "passed": bool(passed[index]),
        }
        for index in range(frame_count)
    ]
    stationary_heading = bool(
        np.allclose(
            points[:, (0, 2)],
            points[0, (0, 2)],
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    return {
        "actor_id": actor_id,
        "status": "pass",
        "local_anatomical_forward_axis": [float(value) for value in local_forward],
        "heading_authority": (
            "authored_first_anchor_yaw_fallback"
            if stationary_heading
            else "authored_root_path_tangent"
        ),
        "stationary_fallback_forward_xz": (
            [float(tangents[0, 0]), float(tangents[0, 2])]
            if stationary_heading else None
        ),
        "binding_source": dict(binding_source),
        "gate": {
            "maximum_allowed_error_degrees": _HEADING_ALIGNMENT_MAX_ERROR_DEG,
            "maximum_observed_error_degrees": float(errors_deg[maximum_index]),
            "maximum_error_frame_index": maximum_index,
            "minimum_alignment_dot": float(np.min(dots)),
            "all_frames_passed": True,
        },
        "frames": frames,
    }


def _human_anatomical_forward_binding(
    manifest_path: str | Path,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    path = Path(manifest_path).resolve()
    manifest = load_json(path)
    if (
        manifest.get("schema") != "avengine_m5_1_rocketbox_human_runtime_v1"
        or manifest.get("status") != "pass"
    ):
        raise MixedCaptureError("human runtime manifest must be a passing M5.1 report")
    content = dict(manifest)
    declared_content_hash = content.pop("manifest_content_sha256", None)
    if declared_content_hash != canonical_json_sha256(content):
        raise MixedCaptureError("human runtime manifest content hash differs")
    anatomical_frame = manifest.get("anatomical_frame")
    if not isinstance(anatomical_frame, Mapping):
        raise MixedCaptureError("human runtime manifest lacks anatomical_frame")
    if anatomical_frame.get("actor_up_axis") != "+Y":
        raise MixedCaptureError("human anatomical frame must declare actor_up_axis +Y")
    source = anatomical_frame.get("source")
    if not isinstance(source, str) or not source.strip():
        raise MixedCaptureError("human anatomical forward source must be explicit")
    axis_label = anatomical_frame.get("actor_forward_axis")
    axis = _axis_label_vector(axis_label, owner="human actor_forward_axis")
    return axis, {
        "kind": "generated_human_runtime_manifest",
        "path": str(path),
        "sha256": sha256_file(path),
        "json_field": "anatomical_frame.actor_forward_axis",
        "declared_axis": axis_label,
        "declared_source": source,
    }


def _beagle_anatomical_forward_binding(
    *, asset: Mapping[str, Any], asset_manifest_path: str | Path
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    records = [
        record
        for record in asset.get("files", [])
        if isinstance(record, Mapping) and record.get("role") == "animation_qa"
    ]
    if len(records) != 1:
        raise MixedCaptureError(
            "Beagle package must bind exactly one animation_qa file"
        )
    record = records[0]
    raw_path = record.get("path")
    try:
        path = resolve_declared_path(
            raw_path,
            manifest_dir=Path(asset_manifest_path).resolve().parent,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise MixedCaptureError(f"Beagle animation_qa path is invalid: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise MixedCaptureError("Beagle animation_qa is not a regular package file")
    actual_sha256 = sha256_file(path)
    if path.stat().st_size != record.get("byte_size") or actual_sha256 != record.get(
        "sha256"
    ):
        raise MixedCaptureError("Beagle animation_qa bytes differ from its manifest")
    animation_qa = load_json(path)
    if (
        animation_qa.get("schema") != "avengine_m2_animation_qa_v1"
        or animation_qa.get("status") != "pass"
    ):
        raise MixedCaptureError("Beagle animation_qa must be a passing M2 report")
    terminal_motion = animation_qa.get("semantic_terminal_motion")
    if not isinstance(terminal_motion, Mapping):
        raise MixedCaptureError("Beagle animation_qa lacks semantic_terminal_motion")
    if terminal_motion.get("actor_up_axis") != "+Y":
        raise MixedCaptureError("Beagle animation QA must declare actor_up_axis +Y")
    axis_label = terminal_motion.get("source_facing_axis_in_actor_frame")
    axis = _axis_label_vector(
        axis_label,
        owner="Beagle semantic_terminal_motion.source_facing_axis_in_actor_frame",
    )
    return axis, {
        "kind": "m2_animation_qa_declared_axis",
        "path": str(path),
        "sha256": actual_sha256,
        "json_field": (
            "semantic_terminal_motion.source_facing_axis_in_actor_frame"
        ),
        "declared_axis": axis_label,
    }


def _continuous_beagle_walk_states(
    states: Sequence[Any],
) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    indices = tuple(
        index for index, state in enumerate(states) if state.action_id == "walk"
    )
    if (
        len(indices) != 45
        or indices
        != tuple(range(indices[0], indices[0] + len(indices)))
    ):
        raise MixedCaptureError(
            "Beagle M2 request must contain one continuous 45-state walk block"
        )
    return indices, tuple(states[index] for index in indices)


def _validate_legacy_camera(room_inputs: Any) -> None:
    rig = room_inputs.request.get("primary_camera_rig", {})
    calibration = rig.get("shared_calibration", {})
    transform = rig.get("world_from_rig", {})
    position = np.asarray(transform.get("translation_m"), dtype=np.float64)
    if position.shape != (3,) or not np.allclose(
        position, LEGACY_CAMERA_POSITION_M, rtol=0.0, atol=1.0e-12
    ):
        raise MixedCaptureError(
            f"legacy camera position must be {list(LEGACY_CAMERA_POSITION_M)}"
        )
    if not math.isclose(
        float(calibration.get("hfov_degrees", math.nan)),
        LEGACY_CAMERA_HFOV_DEG,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise MixedCaptureError("legacy camera horizontal FOV must be 105 degrees")
    actual = np.asarray(
        normalized_quaternion_xyzw(transform.get("rotation_xyzw")),
        dtype=np.float64,
    )
    half = math.radians(LEGACY_CAMERA_YAW_DEG) / 2.0
    expected = np.asarray([0.0, math.sin(half), 0.0, math.cos(half)])
    error = min(
        float(np.max(np.abs(actual - expected))),
        float(np.max(np.abs(actual + expected))),
    )
    if error > 1.0e-12:
        raise MixedCaptureError("legacy camera yaw must be +55 degrees about Habitat +Y")


def _shader_type_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _readback_m5_1_pbr_ibl(
    mediator: Any,
    *,
    config_path: Path,
    asset_root: Path,
    phase: str,
    registration_id: int | None = None,
) -> dict[str, Any]:
    manager = mediator.pbr_shader_template_manager
    current_handle = mediator.get_curr_default_pbr_attributes_handle()
    if current_handle != M5_1_PBR_CONFIG_HANDLE:
        raise MixedCaptureError(
            "M5.1 current PBR config handle changed from the explicit "
            f"external IBL config: {current_handle!r}"
        )
    attributes = manager.get_template_by_handle(current_handle)
    if attributes is None:
        raise MixedCaptureError("M5.1 explicit PBR config cannot be read back")

    expected_lut = (asset_root / PBR_BRDF_LUT_RELATIVE_PATH).resolve()
    expected_environment = (
        asset_root / PBR_ENVIRONMENT_MAP_RELATIVE_PATH
    ).resolve()
    actual_lut = Path(attributes.ibl_brdfLUT_filename).resolve()
    actual_environment = Path(
        attributes.ibl_environment_map_filename
    ).resolve()
    if actual_lut != expected_lut or actual_environment != expected_environment:
        raise MixedCaptureError(
            "M5.1 PBR config did not retain the explicit absolute IBL paths"
        )
    if not bool(attributes.enable_ibl):
        raise MixedCaptureError("M5.1 explicit PBR config did not enable IBL")

    expected_flags = {
        "enable_direct_lights": True,
        "map_mat_txtr_to_linear": True,
        "map_ibl_txtr_to_linear": True,
        "map_output_to_srgb": True,
        "use_direct_tonemap": False,
        "use_ibl_tonemap": True,
        "use_burley_diffuse": True,
    }
    observed_flags = {
        key: bool(getattr(attributes, key)) for key in expected_flags
    }
    if observed_flags != expected_flags:
        raise MixedCaptureError(
            "M5.1 explicit PBR config flags differ from the adopted "
            "Brown Photostudio configuration"
        )

    return {
        "status": "pass",
        "phase": phase,
        "config_handle": current_handle,
        "config_path": str(config_path),
        "asset_root": str(asset_root),
        "registration_id": registration_id,
        "enable_ibl": True,
        "absolute_brdf_lut_path": str(actual_lut),
        "absolute_environment_map_path": str(actual_environment),
        "config_flags": observed_flags,
    }


def _prepare_m5_1_installed_pbr_ibl(
    configuration: Any,
    *,
    installed_runtime: InstalledHabitatRuntime,
    habitat_sim: Any,
) -> dict[str, Any]:
    """Inject one explicit external IBL config before Simulator construction."""

    asset_root = installed_runtime.pbr_asset_root
    if asset_root is None:
        raise MixedCaptureError(
            "installed M5.1 PBR actors require an explicit PBR asset root"
        )
    config_path = (
        installed_runtime.prefix / "config" / PBR_CONFIG_FILENAME
    ).resolve()
    try:
        config_path.relative_to(installed_runtime.prefix)
    except ValueError as error:
        raise MixedCaptureError(
            "installed PBR config escaped the selected runtime prefix"
        ) from error
    if not config_path.is_file():
        raise MixedCaptureError(
            f"installed Habitat runtime is missing {PBR_CONFIG_FILENAME}"
        )

    mediator = habitat_sim.metadata.MetadataMediator(configuration.sim_cfg)
    manager = mediator.pbr_shader_template_manager
    attributes = manager.create_template(
        str(config_path), register_template=False
    )
    if attributes is None:
        raise MixedCaptureError("cannot load the installed M5.1 PBR config")
    attributes.ibl_brdfLUT_filename = str(
        (asset_root / PBR_BRDF_LUT_RELATIVE_PATH).resolve()
    )
    attributes.ibl_environment_map_filename = str(
        (asset_root / PBR_ENVIRONMENT_MAP_RELATIVE_PATH).resolve()
    )
    registration_id = int(
        manager.register_template(attributes, M5_1_PBR_CONFIG_HANDLE)
    )
    if registration_id < 0:
        raise MixedCaptureError("cannot register the explicit M5.1 PBR config")
    if not bool(
        mediator.set_curr_default_pbr_attributes_handle(
            M5_1_PBR_CONFIG_HANDLE
        )
    ):
        raise MixedCaptureError("cannot select the explicit M5.1 PBR config")
    configuration.metadata_mediator = mediator
    return _readback_m5_1_pbr_ibl(
        mediator,
        config_path=config_path,
        asset_root=asset_root,
        phase="before_simulator",
        registration_id=registration_id,
    )


def _bind_m5_1_scene_lighting(
    simulator: Any,
    configuration: Any,
    *,
    light_setup_key: str = M5_1_LIGHT_SETUP_KEY,
    require_zero_direct_lights: bool = False,
) -> dict[str, Any]:
    """Copy the stage's effective setup to the AO-specific M5.1 key."""

    if not isinstance(light_setup_key, str) or not light_setup_key:
        raise MixedCaptureError("M5.1 light setup key must be a non-empty string")
    configured_hbao = bool(configuration.sim_cfg.enable_hbao)
    simulator_hbao = bool(simulator.config.sim_cfg.enable_hbao)
    if not configured_hbao or not simulator_hbao:
        raise MixedCaptureError("M5.1 HBAO configuration did not read back enabled")

    current_setup = list(simulator.get_current_light_setup())
    simulator.set_light_setup(current_setup, light_setup_key)
    registered_setup = list(simulator.get_light_setup(light_setup_key))
    registered_matches_current = registered_setup == current_setup
    if not registered_matches_current:
        raise MixedCaptureError(
            "M5.1 registered actor light setup differs from the scene setup"
        )
    if require_zero_direct_lights and registered_setup:
        raise MixedCaptureError(
            "installed M5.1 actor capture preserves zero direct lights; "
            f"observed {len(registered_setup)}"
        )
    return {
        "status": "pass",
        "hbao": {
            "requested": True,
            "configuration_readback": configured_hbao,
            "simulator_readback": simulator_hbao,
            "effect_scope": (
                "screen-space ambient occlusion; not dynamic shadow-map evidence"
            ),
        },
        "scene_lighting": {
            "actor_light_setup_key": light_setup_key,
            "source_api": "Simulator.get_current_light_setup",
            "registration_api": "Simulator.set_light_setup",
            "current_light_count": len(current_setup),
            "registered_light_count": len(registered_setup),
            "registered_setup_matches_current": registered_matches_current,
            "required_zero_direct_lights": require_zero_direct_lights,
        },
    }


def _actor_render_creation_evidence(
    actor: Any,
    *,
    actor_id: str,
    requested_shader_type: str,
    light_setup_key: str,
) -> dict[str, Any]:
    creation_shader_type = _shader_type_name(actor.creation_attributes.shader_type)
    if creation_shader_type != requested_shader_type:
        raise MixedCaptureError(
            f"{actor_id} creation shader type is {creation_shader_type!r}, "
            f"not {requested_shader_type!r}"
        )
    return {
        "status": "pass",
        "requested_shader_type": requested_shader_type,
        "creation_shader_type_readback": creation_shader_type,
        "creation_light_setup_key_argument": light_setup_key,
        "creation_light_setup_binding_api": (
            "ArticulatedObjectManager."
            "add_articulated_object_by_template_handle(light_setup_key=...)"
        ),
        "native_per_actor_light_key_readback": (
            "not_exposed_by_pinned_habitat_binding"
        ),
    }


def _instantiate_human(
    simulator: Any,
    *,
    package: HumanRuntimePackage,
    habitat_sim: Any,
    semantic_id: int,
    light_setup_key: str | None = None,
    shader_type: str | None = None,
) -> tuple[Any, Any, tuple[HabitatLinkJointBlock, ...]]:
    if light_setup_key is not None and (
        not isinstance(light_setup_key, str) or not light_setup_key
    ):
        raise MixedCaptureError(
            "human light_setup_key must be a non-empty string when provided"
        )
    if shader_type is not None and shader_type not in {
        "material",
        "flat",
        "phong",
        "pbr",
    }:
        raise MixedCaptureError(
            "human shader_type must be material, flat, phong, or pbr"
        )
    manager = simulator.metadata_mediator.ao_template_manager
    loaded = manager.load_configs(str(package.habitat_ao_config))
    prefix = package.habitat_ao_config.stem.removesuffix(".ao_config")
    handles = manager.get_template_handles(prefix)
    if len(loaded) != 1 or len(handles) != 1:
        raise MixedCaptureError(
            f"expected one human AO template, got ids={loaded}, handles={handles}"
        )
    attributes = manager.get_template_by_handle(handles[0])
    if attributes is None:
        raise MixedCaptureError("cannot retrieve the loaded human AO template")
    attributes.semantic_id = int(semantic_id)
    if shader_type is not None:
        attributes.shader_type = shader_type
    handle = f"{handles[0]}.m5_1_semantic{semantic_id}"
    if int(manager.register_template(attributes, handle)) < 0:
        raise MixedCaptureError("failed to register the semantic human AO template")
    object_manager = simulator.get_articulated_object_manager()
    if light_setup_key is None:
        actor = object_manager.add_articulated_object_by_template_handle(handle)
    else:
        actor = object_manager.add_articulated_object_by_template_handle(
            handle,
            light_setup_key=light_setup_key,
        )
    if actor is None:
        raise MixedCaptureError("Habitat failed to instantiate the Rocketbox human")
    actor.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    blocks = tuple(
        HabitatLinkJointBlock(
            link_name=actor.get_link_name(link_id),
            joint_position_offset=int(actor.get_link_joint_pos_offset(link_id)),
            joint_position_count=int(actor.get_link_num_joint_pos(link_id)),
        )
        for link_id in actor.get_link_ids()
    )
    try:
        binding = bind_local_tr_habitat_layout(
            package.mapping,
            blocks,
            joint_position_count=len(actor.joint_positions),
        )
    except (TypeError, ValueError) as exc:
        raise MixedCaptureError(f"human AO link layout differs: {exc}") from exc
    if actor.get_link_name(-1) != package.mapping.root_joint_id:
        raise MixedCaptureError("human AO base link differs from the synthetic root")
    if int(actor.creation_attributes.semantic_id) != semantic_id:
        raise MixedCaptureError("human AO creation semantic ID differs")
    if shader_type is not None and _shader_type_name(
        actor.creation_attributes.shader_type
    ) != shader_type:
        raise MixedCaptureError("human AO creation shader type differs")
    _set_scene_node_semantic_readback(actor, semantic_id)
    return actor, binding, blocks


def _save_array(output: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = output / "arrays" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value)
    np.save(path, array, allow_pickle=False)
    readback = np.load(path, mmap_mode="r", allow_pickle=False)
    if readback.dtype != array.dtype or readback.shape != array.shape or not np.array_equal(
        readback, array
    ):
        raise MixedCaptureError(f"saved {name} array differs on readback")
    return {
        **file_record(path, relative_to=output),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "readback_verified": True,
    }


def _human_skin_link_readback_error(
    articulated_object: Any,
    package: HumanRuntimePackage,
    *,
    world_from_skin_root: np.ndarray,
    translations_m: np.ndarray,
    rotations_xyzw: np.ndarray,
) -> float:
    """Compare skin links with FK without assuming joint-list topological order.

    The two zero-weight ancestors are deliberately appended so original
    JOINTS_0 ordinals remain unchanged.  Their list order is consequently not
    parent-first even though the hierarchy itself is one valid tree.
    """

    mapping = package.mapping
    translations = dict(
        zip(mapping.runtime_joint_order, translations_m, strict=True)
    )
    rotations = dict(zip(mapping.runtime_joint_order, rotations_xyzw, strict=True))
    joint_by_name = {joint.joint_id: joint for joint in mapping.joints}
    link_by_name = {
        articulated_object.get_link_name(link_id): int(link_id)
        for link_id in articulated_object.get_link_ids()
    }
    world_by_name: dict[str, np.ndarray] = {
        mapping.root_joint_id: np.asarray(world_from_skin_root, dtype=np.float64)
    }
    visiting: set[str] = set()

    def world(joint_id: str) -> np.ndarray:
        retained = world_by_name.get(joint_id)
        if retained is not None:
            return retained
        if joint_id in visiting:
            raise MixedCaptureError("human skin hierarchy contains a cycle")
        visiting.add(joint_id)
        try:
            joint = joint_by_name[joint_id]
            if joint.parent_joint_id is None:
                raise MixedCaptureError("human skin contains a second root")
            local = np.eye(4, dtype=np.float64)
            local[:3, :3] = quaternion_xyzw_to_matrix(rotations[joint_id])
            local[:3, 3] = translations[joint_id]
            value = world(joint.parent_joint_id) @ local
            world_by_name[joint_id] = value
            return value
        except KeyError as exc:
            raise MixedCaptureError(
                f"human FK lacks joint/link pose for {joint_id!r}"
            ) from exc
        finally:
            visiting.remove(joint_id)

    maximum = 0.0
    for joint_id in mapping.runtime_joint_order:
        try:
            actual = np.asarray(
                articulated_object.get_link_scene_node(
                    link_by_name[joint_id]
                ).absolute_transformation(),
                dtype=np.float64,
            )
        except KeyError as exc:
            raise MixedCaptureError(
                f"Habitat lacks human skin link {joint_id!r}"
            ) from exc
        expected = world(joint_id)
        if actual.shape != (4, 4) or not np.all(np.isfinite(actual)):
            raise MixedCaptureError(
                f"Habitat human skin link {joint_id!r} is not finite 4x4"
            )
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    return maximum


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


_INSTALLED_RUNTIME_ENVIRONMENT_VARIABLES = (
    "AVENGINE_HABITAT_RUNTIME_PREFIX",
    "AVENGINE_MP3D_ROOT",
    "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE",
)


def _installed_runtime_inputs_requested(
    *,
    runtime_prefix: str | Path | None,
    mp3d_root: str | Path | None,
    magnum_python_site: str | Path | None,
    pbr_asset_root: str | Path | None = None,
) -> bool:
    """Whether a caller selected an installed runtime without its old alias.

    Empty exported values remain a selection.  They must fail through the
    installed resolver instead of accidentally falling back to a checkout.
    """

    return any(
        value is not None
        for value in (
            runtime_prefix,
            mp3d_root,
            magnum_python_site,
            pbr_asset_root,
        )
    ) or any(name in os.environ for name in _INSTALLED_RUNTIME_ENVIRONMENT_VARIABLES)


def _room_declares_external_mp3d_root(room: Mapping[str, Any]) -> bool:
    """Return whether a room is migrated to the external MP3D root."""

    scene = room.get("scene")
    declared_paths: list[object] = []
    if isinstance(scene, Mapping):
        declared_paths.extend(
            scene.get(key) for key in ("scene_id", "dataset_config_path", "navmesh_path")
        )
    assets = room.get("assets")
    if isinstance(assets, Sequence) and not isinstance(assets, (str, bytes)):
        for asset in assets:
            if isinstance(asset, Mapping):
                declared_paths.append(asset.get("path"))
    return any(
        isinstance(path, str) and "${AVENGINE_MP3D_ROOT}" in path
        for path in declared_paths
    )


def _select_mixed_capture_runtime(
    *,
    room: Mapping[str, Any],
    runtime_prefix: str | Path | None,
    runtime_root: str | Path | None,
    legacy_runtime_root: str | Path | None,
    mp3d_root: str | Path | None,
    magnum_python_site: str | Path | None,
    installed_runtime: InstalledHabitatRuntime | None,
    pbr_asset_root: str | Path | None = None,
) -> InstalledHabitatRuntime | None:
    """Resolve new MP3D callers before any checkout-oriented import.

    ``runtime_root`` is an installed-prefix compatibility alias only for a
    room already migrated to ``AVENGINE_MP3D_ROOT``.  Other retained routes
    keep the legacy branch only when no newer prefix, MP3D, or Magnum input
    exists.  The internal ``legacy_runtime_root`` argument is reserved for the
    retained legacy wrapper, whose room manifest remains checkout-based.
    """

    installed_requested = _installed_runtime_inputs_requested(
        runtime_prefix=runtime_prefix,
        mp3d_root=mp3d_root,
        magnum_python_site=magnum_python_site,
        pbr_asset_root=pbr_asset_root,
    )
    explicit_installed_inputs = any(
        value is not None
        for value in (
            runtime_prefix,
            runtime_root,
            mp3d_root,
            magnum_python_site,
            pbr_asset_root,
        )
    )
    if legacy_runtime_root is not None:
        if installed_runtime is not None or explicit_installed_inputs or installed_requested:
            raise MixedCaptureError(
                "legacy-runtime-root cannot be combined with installed runtime "
                "prefix, MP3D, Magnum, or PBR inputs"
            )
        return None
    if installed_runtime is not None:
        if explicit_installed_inputs:
            raise MixedCaptureError(
                "installed_runtime cannot be combined with runtime-root/prefix, "
                "mp3d-root, magnum-python-site, or pbr-asset-root"
            )
        return installed_runtime
    select_installed = installed_requested or (
        runtime_root is not None and _room_declares_external_mp3d_root(room)
    )
    if select_installed:
        if pbr_asset_root is None:
            raise MixedCaptureError(
                "installed MP3D PBR actor capture requires an explicit "
                "pbr_asset_root before runtime preparation"
            )
        return prepare_installed_habitat_runtime(
            runtime_prefix=runtime_prefix,
            runtime_root=runtime_root,
            mp3d_root=mp3d_root,
            pbr_asset_root=pbr_asset_root,
            magnum_python_site=magnum_python_site,
        )
    return None


def capture_human_beagle_paths(
    *,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    human_runtime_glb_path: str | Path,
    beagle_animal_manifest_path: str | Path,
    beagle_m2_request_path: str | Path,
    human_root_path_m: Any,
    beagle_root_path_m: Any,
    output_dir: str | Path,
    runtime_root: str | Path | None = None,
    runtime_prefix: str | Path | None = None,
    legacy_runtime_root: str | Path | None = None,
    mp3d_root: str | Path | None = None,
    pbr_asset_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    installed_runtime: InstalledHabitatRuntime | None = None,
    route_provenance: Mapping[str, Any] | None = None,
    require_legacy_camera: bool = False,
    human_semantic_id: int = HUMAN_SEMANTIC_ID,
    beagle_semantic_id: int = BEAGLE_SEMANTIC_ID,
    human_fallback_forward_xz: Any | None = None,
    beagle_fallback_forward_xz: Any | None = None,
    human_asset_id: str = "rocketbox_human_male_adult_01_m5_1_candidate",
    secondary_actor_id: str = "dog0",
    secondary_actor_class: str = "dog",
    secondary_record_key: str = "beagle",
    secondary_emitter_link_name: str = BEAGLE_MOUTH_LINK_NAME,
    secondary_research_candidate: bool = False,
    research_capture_schema: str = MIXED_CAPTURE_SCHEMA,
    review_configuration_hook: Callable[..., Mapping[str, Any]] | None = None,
    review_scene_hook: Callable[..., Mapping[str, Any]] | None = None,
    review_scene_readback_hook: Callable[..., Mapping[str, Any]] | None = None,
) -> MixedCaptureResult:
    """Capture a human and one M2 articulated animal on explicit root paths.

    The historical public name is retained for the M5.1 Beagle route.  The
    secondary runtime is deliberately parameterized, however: a research
    preview may bind another M2 animal package only when it declares its own
    actor identity, semantic class, record key and emitter-link name.  This
    prevents a Cat render from being silently labelled as a Beagle.
    """

    if (
        isinstance(human_semantic_id, bool)
        or not isinstance(human_semantic_id, int)
        or human_semantic_id < 0
        or isinstance(beagle_semantic_id, bool)
        or not isinstance(beagle_semantic_id, int)
        or beagle_semantic_id < 0
        or human_semantic_id == beagle_semantic_id
    ):
        raise MixedCaptureError("human and secondary semantic IDs must be distinct nonnegative integers")
    if (
        not isinstance(human_asset_id, str)
        or not human_asset_id
        or not isinstance(secondary_actor_id, str)
        or not secondary_actor_id
        or secondary_actor_id == "human0"
        or not isinstance(secondary_actor_class, str)
        or not secondary_actor_class
        or not isinstance(secondary_record_key, str)
        or not secondary_record_key
        or not secondary_record_key.isidentifier()
        or secondary_record_key == "human"
        or not isinstance(secondary_emitter_link_name, str)
        or not secondary_emitter_link_name
        or not isinstance(secondary_research_candidate, bool)
        or not isinstance(research_capture_schema, str)
        or not research_capture_schema
    ):
        raise MixedCaptureError(
            "human/secondary actor identity fields and research capture schema must be valid"
        )
    human_points = _points(human_root_path_m, owner="human root path")
    beagle_points = _points(beagle_root_path_m, owner="Beagle root path")
    if human_points.shape != beagle_points.shape:
        raise MixedCaptureError(
            "human and Beagle root paths must have the same frame count"
        )
    frame_count = int(human_points.shape[0])
    output = Path(output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise MixedCaptureError(f"refusing to replace capture output: {output}")
    installed_requested = _installed_runtime_inputs_requested(
        runtime_prefix=runtime_prefix,
        mp3d_root=mp3d_root,
        magnum_python_site=magnum_python_site,
        pbr_asset_root=pbr_asset_root,
    )
    if installed_runtime is not None:
        if getattr(installed_runtime, "pbr_asset_root", None) is None:
            raise MixedCaptureError(
                "installed MP3D PBR actor capture requires a prepared runtime "
                "with an explicit pbr_asset_root"
            )
    elif installed_requested and pbr_asset_root is None:
        raise MixedCaptureError(
            "installed MP3D PBR actor capture requires an explicit "
            "pbr_asset_root before runtime preparation"
        )

    try:
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
        if (
            installed_runtime is None
            and runtime_root is not None
            and _room_declares_external_mp3d_root(room_inputs.room)
            and pbr_asset_root is None
        ):
            raise MixedCaptureError(
                "installed MP3D PBR actor capture requires an explicit "
                "pbr_asset_root before runtime preparation"
            )
        installed_runtime = _select_mixed_capture_runtime(
            room=room_inputs.room,
            runtime_prefix=runtime_prefix,
            runtime_root=runtime_root,
            legacy_runtime_root=legacy_runtime_root,
            mp3d_root=mp3d_root,
            magnum_python_site=magnum_python_site,
            pbr_asset_root=pbr_asset_root,
            installed_runtime=installed_runtime,
        )
        if installed_runtime is not None:
            if installed_runtime.mp3d_root is None:
                raise MixedCaptureError(
                    "installed MP3D capture requires an explicit --mp3d-root or "
                    "AVENGINE_MP3D_ROOT"
                )
            if getattr(installed_runtime, "pbr_asset_root", None) is None:
                raise MixedCaptureError(
                    "installed MP3D PBR actor capture requires an explicit "
                    "pbr_asset_root"
                )
            if research_capture_schema != MIXED_CAPTURE_INSTALLED_SCHEMA_V2:
                raise MixedCaptureError(
                    "installed MP3D capture must use "
                    f"{MIXED_CAPTURE_INSTALLED_SCHEMA_V2}"
                )
        output.mkdir(parents=True)
        human_package = prepare_rocketbox_habitat_runtime(
            human_runtime_glb_path, output / "runtime" / "human"
        )
        if require_legacy_camera:
            _validate_legacy_camera(room_inputs)
        m2_inputs = (
            load_research_review_inputs(
                beagle_animal_manifest_path, beagle_m2_request_path
            )
            if secondary_research_candidate
            else load_m2_inputs(beagle_animal_manifest_path, beagle_m2_request_path)
        )
        beagle_bundle = load_runtime_asset_bundle(m2_inputs)
        human_forward_axis, human_forward_source = (
            _human_anatomical_forward_binding(human_package.package_manifest)
        )
        beagle_forward_axis, beagle_forward_source = (
            _beagle_anatomical_forward_binding(
                asset=m2_inputs.asset,
                asset_manifest_path=m2_inputs.asset_path,
            )
        )
        human_world = trajectory_world_matrices(
            human_points,
            local_forward_axis=human_forward_axis,
            fallback_forward_xz=human_fallback_forward_xz,
        )
        beagle_world = trajectory_world_matrices(
            beagle_points,
            local_forward_axis=beagle_forward_axis,
            fallback_forward_xz=beagle_fallback_forward_xz,
        )
        heading_records = [
            _actor_heading_evidence(
                actor_id="human0",
                points_m=human_points,
                actor_world_matrices=human_world,
                local_forward_axis=human_forward_axis,
                binding_source=human_forward_source,
                fallback_forward_xz=human_fallback_forward_xz,
            ),
            _actor_heading_evidence(
                actor_id=secondary_actor_id,
                points_m=beagle_points,
                actor_world_matrices=beagle_world,
                local_forward_axis=beagle_forward_axis,
                binding_source=beagle_forward_source,
                fallback_forward_xz=beagle_fallback_forward_xz,
            ),
        ]
        beagle_states = compile_frame_applications(m2_inputs, beagle_bundle)
        if len(beagle_states) != 75:
            raise MixedCaptureError("Beagle M2 request must provide 75 validated states")
        human_actions = {
            action_id: human_package.actions.action(action_id)
            for action_id in ("idle", "walk")
        }
        if human_actions["walk"].sample_count != 16:
            raise MixedCaptureError("human Walking loop must contain exactly 16 samples")
        beagle_actions = {
            action_id: beagle_bundle.action_sets_by_role[
                beagle_bundle.action_roles_by_id[action_id]
            ].action(action_id)
            for action_id in ("idle", "walk")
        }
        human_locomotion = locomotion_schedule_from_root_trajectory(
            human_points,
            action_sample_counts={
                key: value.sample_count for key, value in human_actions.items()
            },
        )
        beagle_locomotion = locomotion_schedule_from_root_trajectory(
            beagle_points,
            action_sample_counts={
                key: value.sample_count for key, value in beagle_actions.items()
            },
        )
        if installed_runtime is None:
            runtime = discover_runtime_root(
                legacy_runtime_root
                if legacy_runtime_root is not None
                else runtime_root
            )
            missing = [
                record
                for record in _resolved_assets(room_inputs, runtime)
                if not record["exists"]
            ]
            if missing:
                raise MixedCaptureError("validated legacy room has missing runtime assets")

            # Legacy callers retain the historical checkout-only branch until
            # their own writer slice is migrated.
            import quaternion as qt

            import habitat_sim
            import magnum as mn

            configuration, modality_to_uuid, _listener_uuid, resolved_scene = (
                _make_configuration(room_inputs, runtime, output / "scene_scratch")
            )
        else:
            missing = [
                record
                for record in _resolved_assets(
                    room_inputs, None, mp3d_root=installed_runtime.mp3d_root
                )
                if not record["exists"]
            ]
            if missing:
                raise MixedCaptureError("validated MP3D room has missing external assets")
            qt = installed_runtime.quaternion
            habitat_sim = installed_runtime.habitat_sim
            mn = installed_runtime.magnum
            configuration, modality_to_uuid, _listener_uuid, resolved_scene = (
                _make_configuration(
                    room_inputs,
                    None,
                    output / "scene_scratch",
                    mp3d_root=installed_runtime.mp3d_root,
                    include_audio_sensor=False,
                    physics_config_path=installed_runtime.physics_config_path,
                )
            )
        configuration.sim_cfg.enable_hbao = True
        if not bool(configuration.sim_cfg.enable_hbao):
            raise MixedCaptureError("M5.1 HBAO could not be enabled in configuration")
        room_declared_physics = bool(resolved_scene.get("enable_physics", False))
        physics_enabled_for_articulation = not room_declared_physics
        if physics_enabled_for_articulation:
            # Articulated-object creation requires Bullet even for KINEMATIC
            # actors.  The loop still performs zero physics steps.
            configuration.sim_cfg.enable_physics = True
        if list(modality_to_uuid) != list(FORMAL_MODALITIES):
            raise MixedCaptureError("M1 modality order changed from rgb/depth/semantic")

        rgb_frames: list[np.ndarray] = []
        semantic_frames: list[np.ndarray] = []
        actor_matrices: list[np.ndarray] = []
        skin_root_matrices: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        visibility: list[tuple[int, int]] = []
        records: list[dict[str, Any]] = []
        human_pose_hashes: dict[str, set[str]] = {
            action_id: set() for action_id in human_actions
        }
        beagle_state_hashes: dict[str, set[str]] = {
            action_id: set() for action_id in beagle_actions
        }
        maximum_errors = {
            "human_root": 0.0,
            "human_prismatic": 0.0,
            "human_spherical": 0.0,
            "human_skin_link_fk": 0.0,
            "beagle_root": 0.0,
            "beagle_spherical": 0.0,
        }

        review_visual_profile_evidence: Mapping[str, Any] | None = None
        review_configuration_evidence: Mapping[str, Any] | None = None
        pbr_ibl_preparation: Mapping[str, Any] | None = None
        if review_configuration_hook is not None:
            configured = review_configuration_hook(
                configuration=configuration,
                habitat_sim=habitat_sim,
            )
            if not isinstance(configured, Mapping) or configured.get("status") != "pass":
                raise MixedCaptureError(
                    "review configuration hook did not return pass evidence"
                )
            review_configuration_evidence = dict(configured)
        if installed_runtime is not None:
            pbr_ibl_preparation = _prepare_m5_1_installed_pbr_ibl(
                configuration,
                installed_runtime=installed_runtime,
                habitat_sim=habitat_sim,
            )
        with habitat_sim.Simulator(configuration) as simulator:
            navmesh_path = resolved_scene.get("navmesh")
            if navmesh_path is not None and Path(navmesh_path).is_file():
                simulator.pathfinder.load_nav_mesh(str(navmesh_path))
            if review_scene_hook is not None:
                realized = review_scene_hook(
                    simulator=simulator,
                    configuration=configuration,
                    camera_listener_position_m=room_inputs.request[
                        "primary_camera_rig"
                    ]["world_from_rig"]["translation_m"],
                    habitat_sim=habitat_sim,
                    mn=mn,
                )
                if not isinstance(realized, Mapping) or realized.get("status") != "pass":
                    raise MixedCaptureError(
                        "review scene hook did not return pass evidence"
                    )
                review_visual_profile_evidence = dict(realized)
                if review_configuration_evidence is not None:
                    review_visual_profile_evidence = {
                        **review_visual_profile_evidence,
                        "configuration": dict(review_configuration_evidence),
                    }
            rendering_evidence = _bind_m5_1_scene_lighting(
                simulator,
                configuration,
                require_zero_direct_lights=installed_runtime is not None,
            )
            if installed_runtime is not None:
                pbr_ibl_readback = _readback_m5_1_pbr_ibl(
                    simulator.metadata_mediator,
                    config_path=(
                        installed_runtime.prefix
                        / "config"
                        / PBR_CONFIG_FILENAME
                    ).resolve(),
                    asset_root=installed_runtime.pbr_asset_root,
                    phase="after_simulator",
                )
                rendering_evidence["pbr_ibl"] = {
                    "status": "pass",
                    "preparation": dict(pbr_ibl_preparation or {}),
                    "simulator_readback": pbr_ibl_readback,
                    "actual_direct_light_count": 0,
                    "direct_light_workaround_used": False,
                    "actor_shader_type": M5_1_ACTOR_SHADER_TYPE,
                }
            if review_scene_readback_hook is not None:
                if review_visual_profile_evidence is None:
                    raise MixedCaptureError(
                        "review scene readback requires a realized review scene"
                    )
                final_readback = review_scene_readback_hook(
                    simulator=simulator,
                    configuration=configuration,
                    habitat_sim=habitat_sim,
                    mn=mn,
                    actor_light_setup_key=M5_1_LIGHT_SETUP_KEY,
                )
                if (
                    not isinstance(final_readback, Mapping)
                    or final_readback.get("status") != "pass"
                ):
                    raise MixedCaptureError(
                        "review scene readback hook did not return pass evidence"
                    )
                review_visual_profile_evidence = {
                    **review_visual_profile_evidence,
                    "final_light_readback": dict(final_readback),
                }
            simulator.seed(int(m2_inputs.request["seed"]))
            rig = room_inputs.request["primary_camera_rig"]
            camera_state = habitat_sim.AgentState()
            camera_state.position = np.asarray(
                rig["world_from_rig"]["translation_m"], dtype=np.float64
            )
            x, y, z, w = normalized_quaternion_xyzw(
                rig["world_from_rig"]["rotation_xyzw"]
            )
            camera_state.rotation = qt.quaternion(w, x, y, z)
            simulator.initialize_agent(0, camera_state)

            human, human_binding, human_blocks = _instantiate_human(
                simulator,
                package=human_package,
                habitat_sim=habitat_sim,
                semantic_id=human_semantic_id,
                light_setup_key=M5_1_LIGHT_SETUP_KEY,
                shader_type=M5_1_ACTOR_SHADER_TYPE,
            )
            human_render_evidence = _actor_render_creation_evidence(
                human,
                actor_id="human0",
                requested_shader_type=M5_1_ACTOR_SHADER_TYPE,
                light_setup_key=M5_1_LIGHT_SETUP_KEY,
            )
            template_manager = simulator.metadata_mediator.ao_template_manager
            dog_config = beagle_bundle.paths_by_role["habitat_ao_config"]
            loaded_dog = template_manager.load_configs(str(dog_config))
            dog_prefix = dog_config.stem.removesuffix(".ao_config")
            dog_handles = template_manager.get_template_handles(dog_prefix)
            if len(loaded_dog) != 1 or len(dog_handles) != 1:
                raise MixedCaptureError("expected exactly one Beagle AO template")
            beagle, beagle_binding = _instantiate_actor_with_semantic_template(
                simulator,
                bundle=beagle_bundle,
                habitat_sim=habitat_sim,
                base_handle=dog_handles[0],
                semantic_id=beagle_semantic_id,
                actor_index=1,
                light_setup_key=M5_1_LIGHT_SETUP_KEY,
                shader_type=M5_1_ACTOR_SHADER_TYPE,
            )
            beagle_render_evidence = _actor_render_creation_evidence(
                beagle,
                actor_id=secondary_actor_id,
                requested_shader_type=M5_1_ACTOR_SHADER_TYPE,
                light_setup_key=M5_1_LIGHT_SETUP_KEY,
            )
            human_head_link = _link_id_by_name(human, HEAD_LINK_NAME)
            human_mouth_link = _link_id_by_name(human, MOUTH_LINK_NAME)
            beagle_mouth_link = _link_id_by_name(beagle, secondary_emitter_link_name)
            sensors = [
                simulator.sensors[modality_to_uuid[modality]]
                for modality in FORMAL_MODALITIES
            ]
            human_actor_from_skin = np.asarray(
                human_package.actor_from_skin_root, dtype=np.float64
            )
            beagle_actor_from_skin = np.asarray(
                beagle_bundle.actor_from_skin_root, dtype=np.float64
            )
            initial_world_time = float(simulator.get_world_time())

            for frame_index in range(frame_count):
                human_locomotion_state = human_locomotion[frame_index]
                human_action = human_actions[human_locomotion_state.action_id]
                human_sample_index = human_locomotion_state.action_sample_index
                human_translations = np.asarray(
                    human_action.translations_m[human_sample_index], dtype=np.float64
                )
                human_rotations = np.asarray(
                    human_action.rotations_xyzw[human_sample_index], dtype=np.float64
                )
                human_skin = human_world[frame_index] @ human_actor_from_skin
                human_joints = np.asarray(
                    human_binding.map_pose(human_translations, human_rotations),
                    dtype=np.float64,
                )
                beagle_locomotion_state = beagle_locomotion[frame_index]
                beagle_action = beagle_actions[beagle_locomotion_state.action_id]
                beagle_sample_index = beagle_locomotion_state.action_sample_index
                beagle_skin = beagle_world[frame_index] @ beagle_actor_from_skin
                beagle_joints = np.asarray(
                    beagle_binding.map_pose(
                        beagle_action.rotations_xyzw[beagle_sample_index]
                    ),
                    dtype=np.float64,
                )

                _apply_root_with_habitat(human, human_skin, qt=qt, mn=mn)
                human.joint_positions = human_joints.copy()
                _apply_root_with_habitat(beagle, beagle_skin, qt=qt, mn=mn)
                beagle.joint_positions = beagle_joints.copy()
                human_before = human_runtime_snapshot(simulator, human)
                beagle_before = dog_runtime_snapshot(simulator, beagle)

                human_root_error = float(
                    np.max(
                        np.abs(
                            np.asarray(human_before["world_from_skin_root"])
                            - human_skin
                        )
                    )
                )
                human_prismatic_error, human_spherical_error = (
                    mixed_joint_readback_errors(
                        np.asarray(human_before["mixed_joint_positions"]),
                        human_joints,
                        human_blocks,
                    )
                )
                human_fk_error = _human_skin_link_readback_error(
                    human,
                    human_package,
                    world_from_skin_root=human_skin,
                    translations_m=human_translations,
                    rotations_xyzw=human_rotations,
                )
                beagle_root_error = float(
                    np.max(
                        np.abs(
                            np.asarray(beagle_before["world_from_skin_root"])
                            - beagle_skin
                        )
                    )
                )
                beagle_joint_error = _quaternion_block_error(
                    np.asarray(beagle_before["joint_positions_xyzw"]),
                    beagle_joints,
                )
                frame_errors = {
                    "human_root": human_root_error,
                    "human_prismatic": human_prismatic_error,
                    "human_spherical": human_spherical_error,
                    "human_skin_link_fk": human_fk_error,
                    "beagle_root": beagle_root_error,
                    "beagle_spherical": beagle_joint_error,
                }
                for key, value in frame_errors.items():
                    maximum_errors[key] = max(maximum_errors[key], float(value))
                if (
                    human_root_error > _ROOT_READBACK_ATOL
                    or human_prismatic_error > _JOINT_READBACK_ATOL
                    or human_spherical_error > _JOINT_READBACK_ATOL
                    or human_fk_error > _LINK_MATRIX_READBACK_ATOL
                    or beagle_root_error > _ROOT_READBACK_ATOL
                    or beagle_joint_error > _JOINT_READBACK_ATOL
                ):
                    raise MixedCaptureError(
                        f"frame {frame_index} articulated readback failed: {frame_errors}"
                    )

                observation = simulator.render_sensors(sensors)
                arrays = _validate_observation_arrays(
                    observation, modality_to_uuid
                )
                frame_visibility = (
                    int(np.count_nonzero(arrays["semantic"] == human_semantic_id)),
                    int(np.count_nonzero(arrays["semantic"] == beagle_semantic_id)),
                )
                frame_anchors = np.stack(
                    (
                        _node_world_position(human, human_head_link),
                        _node_world_position(human, human_mouth_link),
                        _node_world_position(beagle, beagle_mouth_link),
                    ),
                    axis=0,
                )
                human_after = human_runtime_snapshot(simulator, human)
                beagle_after = dog_runtime_snapshot(simulator, beagle)
                if (
                    human_before["sha256"] != human_after["sha256"]
                    or beagle_before["sha256"] != beagle_after["sha256"]
                ):
                    raise MixedCaptureError(
                        f"frame {frame_index} render changed an articulated state"
                    )
                if float(simulator.get_world_time()) != initial_world_time:
                    raise MixedCaptureError(
                        f"frame {frame_index} advanced Habitat world time"
                    )

                pose_hash = canonical_json_sha256(
                    {
                        "action_id": human_locomotion_state.action_id,
                        "sample_index": human_sample_index,
                        "translations_m": human_translations.tolist(),
                        "rotations_xyzw": human_rotations.tolist(),
                    }
                )
                human_pose_hashes[human_locomotion_state.action_id].add(pose_hash)
                beagle_state_hashes[beagle_locomotion_state.action_id].add(
                    str(beagle_before["sha256"])
                )
                rgb = np.asarray(arrays["rgb"])[..., :3].astype(np.uint8, copy=True)
                semantic = np.asarray(arrays["semantic"]).copy()
                rgb_frames.append(rgb)
                semantic_frames.append(semantic)
                actor_matrices.append(
                    np.stack((human_world[frame_index], beagle_world[frame_index]))
                )
                skin_root_matrices.append(
                    np.stack(
                        (
                            np.asarray(human_before["world_from_skin_root"]),
                            np.asarray(beagle_before["world_from_skin_root"]),
                        )
                    )
                )
                anchors.append(frame_anchors)
                visibility.append(frame_visibility)
                records.append(
                    {
                        "frame_index": frame_index,
                        "pts_ticks": frame_index * TICKS_PER_FRAME,
                        "human": {
                            "action_id": human_locomotion_state.action_id,
                            "source_action_name": human_action.source_action_name,
                            "action_time_ticks": (
                                human_locomotion_state.action_frame_index
                                * TICKS_PER_FRAME
                            ),
                            "action_sample_index": human_sample_index,
                            "action_phase": human_locomotion_state.action_phase,
                            "root_horizontal_speed_m_s": (
                                human_locomotion_state.horizontal_speed_m_s
                            ),
                            "locomotion_state_transition": (
                                human_locomotion_state.state_transition
                            ),
                            "pose_sha256": pose_hash,
                            "actor_root_position_m": human_points[
                                frame_index
                            ].tolist(),
                            "skin_root_readback_position_m": np.asarray(
                                human_before["world_from_skin_root"]
                            )[:3, 3].tolist(),
                            "semantic_id": human_semantic_id,
                            "semantic_visible_pixels": frame_visibility[0],
                            "head_anchor_m": frame_anchors[0].tolist(),
                            "mouth_emitter_anchor_m": frame_anchors[1].tolist(),
                            "readback": {
                                "maximum_root_error": human_root_error,
                                "maximum_prismatic_error": human_prismatic_error,
                                "maximum_spherical_error": human_spherical_error,
                                "maximum_skin_link_fk_error": human_fk_error,
                                "state_sha256": human_before["sha256"],
                            },
                        },
                        secondary_record_key: {
                            "action_id": beagle_locomotion_state.action_id,
                            "source_action_name": beagle_action.source_action_name,
                            "action_time_ticks": (
                                beagle_locomotion_state.action_frame_index
                                * TICKS_PER_FRAME
                            ),
                            "action_sample_index": beagle_sample_index,
                            "action_phase": beagle_locomotion_state.action_phase,
                            "root_horizontal_speed_m_s": (
                                beagle_locomotion_state.horizontal_speed_m_s
                            ),
                            "locomotion_state_transition": (
                                beagle_locomotion_state.state_transition
                            ),
                            "actor_root_position_m": beagle_points[
                                frame_index
                            ].tolist(),
                            "skin_root_readback_position_m": np.asarray(
                                beagle_before["world_from_skin_root"]
                            )[:3, 3].tolist(),
                            "semantic_id": beagle_semantic_id,
                            "semantic_visible_pixels": frame_visibility[1],
                            "mouth_emitter_anchor_m": frame_anchors[2].tolist(),
                            "readback": {
                                "maximum_root_error": beagle_root_error,
                                "maximum_spherical_error": beagle_joint_error,
                                "state_sha256": beagle_before["sha256"],
                            },
                        },
                        "observation": {
                            "calls": 1,
                            "rgb_payload_sha256": array_sha256(
                                modality_to_uuid["rgb"], arrays["rgb"]
                            ),
                            "semantic_payload_sha256": array_sha256(
                                modality_to_uuid["semantic"], arrays["semantic"]
                            ),
                        },
                    }
                )

        rgb_array = np.ascontiguousarray(np.stack(rgb_frames, axis=0))
        semantic_array = np.ascontiguousarray(np.stack(semantic_frames, axis=0))
        actor_array = np.ascontiguousarray(np.stack(actor_matrices, axis=0))
        skin_root_array = np.ascontiguousarray(
            np.stack(skin_root_matrices, axis=0)
        )
        anchor_array = np.ascontiguousarray(np.stack(anchors, axis=0))
        visibility_array = np.asarray(visibility, dtype=np.int64)
        for actor_id, schedule, pose_hashes in (
            ("human0", human_locomotion, human_pose_hashes),
            (secondary_actor_id, beagle_locomotion, beagle_state_hashes),
        ):
            _validate_used_action_render_evidence(
                actor_id=actor_id,
                schedule=schedule,
                pose_hashes_by_action=pose_hashes,
            )
        if np.max(visibility_array, axis=0).min() <= 0:
            raise MixedCaptureError(
                "the fixed legacy camera never observed one of the two semantic IDs"
            )
        if not np.array_equal(actor_array[:, 0, :3, 3], human_points) or not np.array_equal(
            actor_array[:, 1, :3, 3], beagle_points
        ):
            raise MixedCaptureError("retained actor roots differ from input paths")

        artifacts = {
            "rgb": _save_array(output, "rgb", rgb_array),
            "semantic": _save_array(output, "semantic", semantic_array),
            "actor_world_matrices": _save_array(
                output, "actor_world_matrices", actor_array
            ),
            "skin_root_world_matrices": _save_array(
                output, "skin_root_world_matrices", skin_root_array
            ),
            "anchor_positions_m": _save_array(
                output, "anchor_positions_m", anchor_array
            ),
            "semantic_visibility_pixels": _save_array(
                output, "semantic_visibility_pixels", visibility_array
            ),
        }
        records_path = output / "frame_readback.json"
        _write_json(records_path, records)
        retained_rig = room_inputs.request["primary_camera_rig"]
        evidence: dict[str, Any] = {
            "schema": research_capture_schema,
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "frame_count": frame_count,
            "frame_rate_hz": FRAME_RATE_HZ,
            "time_base_hz": TIME_BASE_HZ,
            "physics_steps": 0,
            "physics_configuration": {
                "room_declared_enable_physics": room_declared_physics,
                "enabled_for_articulated_object_creation": (
                    physics_enabled_for_articulation
                ),
                "effective_enable_physics": True,
            },
            "observation_calls_per_frame": 1,
            "camera": {
                "position_m": list(
                    retained_rig["world_from_rig"]["translation_m"]
                ),
                "rotation_xyzw": list(
                    retained_rig["world_from_rig"]["rotation_xyzw"]
                ),
                "horizontal_fov_deg": retained_rig["shared_calibration"][
                    "hfov_degrees"
                ],
                "resolution_hw": list(
                    retained_rig["shared_calibration"]["resolution_hw"]
                ),
                "legacy_camera_contract_required": require_legacy_camera,
            },
            "actors": [
                {
                    "actor_id": "human0",
                    "actor_class": "human",
                    "asset_id": human_asset_id,
                    "semantic_id": human_semantic_id,
                    "actions": ["idle", "walk"],
                    "action_selection": LOCOMOTION_POLICY_ID,
                    "fixed_state_playback": True,
                    "head_link": HEAD_LINK_NAME,
                    "emitter_link": MOUTH_LINK_NAME,
                    "local_anatomical_forward_axis": list(human_forward_axis),
                    "rendering": human_render_evidence,
                },
                {
                    "actor_id": secondary_actor_id,
                    "actor_class": secondary_actor_class,
                    "asset_id": m2_inputs.asset["asset_id"],
                    "semantic_id": beagle_semantic_id,
                    "actions": ["idle", "walk"],
                    "action_selection": LOCOMOTION_POLICY_ID,
                    "state_source": "asset_role_bound_action_clips",
                    "emitter_link": secondary_emitter_link_name,
                    "local_anatomical_forward_axis": list(beagle_forward_axis),
                    "rendering": beagle_render_evidence,
                },
            ],
            "rendering": rendering_evidence,
            "review_visual_profile": review_visual_profile_evidence,
            "locomotion": {
                "policy_id": LOCOMOTION_POLICY_ID,
                "source": "authored_actor_root_trajectories",
                "actors": {
                    "human0": _locomotion_schedule_summary(human_locomotion),
                    secondary_actor_id: _locomotion_schedule_summary(beagle_locomotion),
                },
            },
            "heading_alignment": {
                "schema": HEADING_ALIGNMENT_SCHEMA,
                "status": "pass",
                "gate": {
                    "required_actor_ids": ["human0", secondary_actor_id],
                    "all_actors_all_frames_passed": True,
                },
                "actors": heading_records,
            },
            "inputs": {
                "room_manifest": {
                    "path": str(Path(room_manifest_path).resolve()),
                    "sha256": sha256_file(room_manifest_path),
                },
                "m1_request": {
                    "path": str(Path(m1_request_path).resolve()),
                    "sha256": sha256_file(m1_request_path),
                },
                "human_runtime_glb": {
                    "path": str(Path(human_runtime_glb_path).resolve()),
                    "sha256": sha256_file(human_runtime_glb_path),
                },
                f"{secondary_record_key}_manifest": {
                    "path": str(Path(beagle_animal_manifest_path).resolve()),
                    "sha256": sha256_file(beagle_animal_manifest_path),
                },
                f"{secondary_record_key}_m2_request": {
                    "path": str(Path(beagle_m2_request_path).resolve()),
                    "sha256": sha256_file(beagle_m2_request_path),
                },
                "route_provenance": dict(route_provenance or {}),
            },
            "runtime": {
                "human_package_manifest": file_record(
                    human_package.package_manifest, relative_to=output
                ),
                "human_action_sample_counts": {
                    action_id: action.sample_count
                    for action_id, action in sorted(human_actions.items())
                },
                "human_distinct_pose_count_by_action": {
                    action_id: len(values)
                    for action_id, values in sorted(human_pose_hashes.items())
                },
                f"{secondary_record_key}_declared_state_count": len(beagle_states),
                f"{secondary_record_key}_action_sample_counts": {
                    action_id: action.sample_count
                    for action_id, action in sorted(beagle_actions.items())
                },
                f"{secondary_record_key}_distinct_state_count_by_action": {
                    action_id: len(values)
                    for action_id, values in sorted(beagle_state_hashes.items())
                },
                f"{secondary_record_key}_validated_source_request_sha256": sha256_file(
                    beagle_m2_request_path
                ),
                **(
                    {
                        "installed_habitat_runtime": {
                            "kind": "installed_prefix",
                            "prefix": str(installed_runtime.prefix),
                            "mp3d_root": str(installed_runtime.mp3d_root),
                            "pbr_asset_root": str(
                                installed_runtime.pbr_asset_root
                            ),
                            "magnum_python_site": str(
                                installed_runtime.magnum_python_site
                            ),
                            "physics_config_path": str(
                                installed_runtime.physics_config_path
                            ),
                        }
                    }
                    if installed_runtime is not None
                    else {}
                ),
            },
            "readback": {
                "maximum_errors": maximum_errors,
                "semantic_visible_frame_count": {
                    "human0": int(np.count_nonzero(visibility_array[:, 0] > 0)),
                    secondary_actor_id: int(np.count_nonzero(visibility_array[:, 1] > 0)),
                },
                "semantic_maximum_visible_pixels": {
                    "human0": int(np.max(visibility_array[:, 0])),
                    secondary_actor_id: int(np.max(visibility_array[:, 1])),
                },
                "frame_records": file_record(records_path, relative_to=output),
            },
            "array_artifacts": artifacts,
            "anchor_order": [
                "human0.head",
                "human0.mouth_emitter",
                f"{secondary_actor_id}.mouth_emitter",
            ],
            "claim_boundary": (
                "M5.1 fixed-state mixed visual research canary; no asset, room, "
                "episode, or dataset admission claim"
            ),
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        evidence_path = output / "evidence.json"
        _write_json(evidence_path, evidence)
        return MixedCaptureResult(
            output_dir=output,
            rgb=rgb_array,
            semantic=semantic_array,
            actor_world_matrices=actor_array,
            skin_root_world_matrices=skin_root_array,
            anchor_positions_m=anchor_array,
            semantic_visibility_pixels=visibility_array,
            records=tuple(records),
            evidence=evidence,
        )
    except (HabitatCaptureError, OSError, ValueError) as exc:
        if isinstance(exc, MixedCaptureError):
            raise
        raise MixedCaptureError(str(exc)) from exc


def capture_legacy_route(
    *,
    route_manifest_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    human_runtime_glb_path: str | Path,
    beagle_animal_manifest_path: str | Path,
    beagle_m2_request_path: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path | None = None,
) -> MixedCaptureResult:
    """Read and capture the committed legacy route without re-deriving paths."""

    route_path = Path(route_manifest_path).resolve()
    route = load_json(route_path)
    assert_valid_route_manifest(route)
    routes = route["routes"]
    human_path = routes["human_path"]["habitat_trajectory_m"]
    beagle_path = routes["dog_path"]["habitat_trajectory_m"]
    provenance = {
        "route_manifest_path": str(route_path),
        "route_manifest_sha256": sha256_file(route_path),
        "route_id": route["route_id"],
        "human_habitat_trajectory_sha256": routes["human_path"][
            "habitat_trajectory_sha256"
        ],
        "dog_habitat_trajectory_sha256": routes["dog_path"][
            "habitat_trajectory_sha256"
        ],
        "path_consumption": "verbatim_manifest_routes_habitat_trajectory_m",
    }
    return capture_human_beagle_paths(
        room_manifest_path=room_manifest_path,
        m1_request_path=m1_request_path,
        human_runtime_glb_path=human_runtime_glb_path,
        beagle_animal_manifest_path=beagle_animal_manifest_path,
        beagle_m2_request_path=beagle_m2_request_path,
        human_root_path_m=human_path,
        beagle_root_path_m=beagle_path,
        output_dir=output_dir,
        legacy_runtime_root=runtime_root,
        route_provenance=provenance,
        require_legacy_camera=True,
    )


__all__ = [
    "BEAGLE_MOUTH_LINK_NAME",
    "BEAGLE_SEMANTIC_ID",
    "HUMAN_SEMANTIC_ID",
    "LOCOMOTION_IDLE_ENTER_SPEED_M_S",
    "LOCOMOTION_POLICY_ID",
    "LOCOMOTION_WALK_ENTER_SPEED_M_S",
    "MIXED_CAPTURE_SCHEMA",
    "MIXED_CAPTURE_INSTALLED_SCHEMA_V2",
    "LocomotionFrameState",
    "MixedCaptureError",
    "MixedCaptureResult",
    "capture_human_beagle_paths",
    "capture_legacy_route",
    "locomotion_schedule_from_root_trajectory",
    "trajectory_world_matrices",
]
