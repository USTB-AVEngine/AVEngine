"""Raw per-frame MP3D clues from current or legacy native captures.

This module is a read-only join.  It does not rank, select, reject, or promote
camera candidates.  Current captures must carry the complete post-render pose
evidence written by :mod:`avengine.m5.current_visual`; older captures remain
readable but every authority introduced after them stays explicitly pending.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json
from avengine.contracts.transforms import compose_transforms
from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix
from avengine.m5.current_visual_review import (
    CurrentVisualReviewError,
    _artifact_path,
    _frame_records,
    _load_npy,
    _optional_frame_array,
    _pose_readback_evidence,
    _require_completed_receipt,
    _require_rgb,
    _require_semantic,
    _object_id_readback_evidence,
)
from avengine.qa.motion_feasibility import (
    MotionFeasibilityError,
    describe_source_pair_tracks,
)


FRAME_COUNT = 75
TICKS_PER_FRAME = 3_200
TICKS_PER_SECOND = 48_000
_NEW_POSE_ARTIFACTS = {
    "planned_applied_skin_root_matrices",
    "actual_skin_root_readback_matrices",
    "actual_agent_camera_poses",
    "actual_sensor_poses",
    "actual_logical_listener_poses",
}
_PINHOLE_AUTHORITY = (
    "actual_post_render_semantic_sensor_pose_plus_capture_time_m1_intrinsics"
)
_PIXEL_CENTER_CONVENTION = (
    "continuous_zero_based_pixel_centers_x_0_to_width_minus_1_y_0_to_height_minus_1"
)
_OBJECT_ID_MODAL_AUTHORITY = (
    "same_renderer_native_object_id_raster_runtime_object_id_readback"
)
_OBJECT_ID_IN_FRAME_INTERPRETATION = (
    "modal_object_id_pixel_presence_only; zero_does_not_distinguish_off_screen_"
    "from_fully_occluded_or_no_modal_pixels"
)


class MP3DRawClueError(ValueError):
    """A dynamic capture cannot support an honest raw-clue join."""


@dataclass(frozen=True)
class LoadedMP3DDynamicClues:
    """Validated dynamic capture inputs and their JSON-ready clue record."""

    root: Path
    receipt_path: Path
    receipt: Mapping[str, Any]
    actor_ids: tuple[str, str]
    source_ids: tuple[str, str]
    source_positions: np.ndarray
    visible_pixels: np.ndarray
    has_current_pose_authority: bool
    raw_clues: Mapping[str, Any]


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DRawClueError(f"{owner} must be an object")
    return value


def _string_axis(value: Any, *, owner: str) -> tuple[str, str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != 2
    ):
        raise MP3DRawClueError(f"{owner} must contain two unique nonempty IDs")
    return str(value[0]), str(value[1])


def _declared_input_file(receipt_path: Path, value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MP3DRawClueError(f"{owner} must name an ordinary file")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else receipt_path.parent / raw
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise MP3DRawClueError(f"{owner} cannot be resolved") from exc
    if not path.is_file():
        raise MP3DRawClueError(f"{owner} is not an ordinary file")
    return path


def dynamic_capture_request_path(capture_root: Path) -> Path:
    """Resolve the M1 request named by one dynamic capture receipt."""

    try:
        root = capture_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise MP3DRawClueError(
            f"dynamic capture cannot be resolved: {capture_root}"
        ) from exc
    if not root.is_dir():
        raise MP3DRawClueError(f"dynamic capture is not a directory: {root}")
    receipt_path = root / "research_receipt.json"
    receipt = _mapping(load_json(receipt_path), owner="dynamic receipt")
    inputs = _mapping(receipt.get("inputs"), owner="dynamic receipt inputs")
    return _declared_input_file(
        receipt_path, inputs.get("m1_request"), owner="dynamic inputs.m1_request"
    )


def _pose(value: Sequence[float]) -> dict[str, list[float]]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "translation_m": array[:3].tolist(),
        "rotation_xyzw": array[3:7].tolist(),
    }


def _basis_tracks(poses: np.ndarray) -> tuple[np.ndarray, ...]:
    positions = np.asarray(poses[:, :3], dtype=np.float64)
    forward = []
    right = []
    up = []
    for pose in poses:
        rotation = quaternion_xyzw_to_matrix(pose[3:7])
        right.append(rotation @ np.asarray([1.0, 0.0, 0.0]))
        up.append(rotation @ np.asarray([0.0, 1.0, 0.0]))
        forward.append(rotation @ np.asarray([0.0, 0.0, -1.0]))
    return (
        positions,
        np.asarray(forward, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(up, dtype=np.float64),
    )


def _calibration(
    request: Mapping[str, Any], *, semantic_shape: tuple[int, int]
) -> tuple[float, int, int]:
    try:
        calibration = request["primary_camera_rig"]["shared_calibration"]
        projection = calibration["projection"]
        hfov = float(calibration["hfov_degrees"])
        height, width = calibration["resolution_hw"]
    except (KeyError, TypeError, ValueError) as exc:
        raise MP3DRawClueError("M1 request lacks a valid shared calibration") from exc
    if (
        projection != "pinhole"
        or not math.isfinite(hfov)
        or not 0.0 < hfov < 180.0
        or isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or (height, width) != semantic_shape
    ):
        raise MP3DRawClueError(
            "M1 pinhole calibration must match the native semantic array"
        )
    return hfov, height, width


def _pinhole_tracks(
    positions: np.ndarray,
    camera_poses: np.ndarray,
    *,
    hfov_degrees: float,
    height: int,
    width: int,
) -> list[list[dict[str, Any]]]:
    tan_half_h = math.tan(math.radians(hfov_degrees) * 0.5)
    tan_half_v = tan_half_h * height / width
    result: list[list[dict[str, Any]]] = [[], []]
    for frame_index in range(FRAME_COUNT):
        rotation = quaternion_xyzw_to_matrix(camera_poses[frame_index, 3:7])
        camera = camera_poses[frame_index, :3]
        for axis in range(2):
            local = (positions[frame_index, axis] - camera) @ rotation
            depth = float(-local[2])
            if depth <= 0.0:
                record = {
                    "authority": _PINHOLE_AUTHORITY,
                    "intrinsics_authority": "capture_time_m1_capture_identity",
                    "pixel_center_convention": _PIXEL_CENTER_CONVENTION,
                    "camera_forward_depth_m": depth,
                    "normalized_pinhole_x_right": None,
                    "normalized_pinhole_y_up": None,
                    "pixel_x_right": None,
                    "pixel_y_down": None,
                    "horizontal_frustum_margin_normalized": None,
                    "vertical_frustum_margin_normalized": None,
                    "frustum_margin_normalized": None,
                    "center_in_frustum": False,
                }
            else:
                x = float(local[0] / (depth * tan_half_h))
                y = float(local[1] / (depth * tan_half_v))
                horizontal_margin = float(1.0 - abs(x))
                vertical_margin = float(1.0 - abs(y))
                if abs(horizontal_margin) <= 1.0e-12:
                    x = math.copysign(1.0, x)
                    horizontal_margin = 0.0
                if abs(vertical_margin) <= 1.0e-12:
                    y = math.copysign(1.0, y)
                    vertical_margin = 0.0
                record = {
                    "authority": _PINHOLE_AUTHORITY,
                    "intrinsics_authority": "capture_time_m1_capture_identity",
                    "pixel_center_convention": _PIXEL_CENTER_CONVENTION,
                    "camera_forward_depth_m": depth,
                    "normalized_pinhole_x_right": x,
                    "normalized_pinhole_y_up": y,
                    "pixel_x_right": float((x + 1.0) * 0.5 * (width - 1)),
                    "pixel_y_down": float((1.0 - y) * 0.5 * (height - 1)),
                    "horizontal_frustum_margin_normalized": horizontal_margin,
                    "vertical_frustum_margin_normalized": vertical_margin,
                    "frustum_margin_normalized": min(
                        horizontal_margin, vertical_margin
                    ),
                    "center_in_frustum": bool(
                        horizontal_margin >= 0.0 and vertical_margin >= 0.0
                    ),
                }
            result[axis].append(record)
    return result


def _object_id_tracks(readback: Any) -> list[list[dict[str, Any]]] | None:
    """Describe only the modal pixels supplied by the native OBJECT_ID raster.

    A positive count means that this runtime object contributed modal pixels to
    the rendered raster.  A zero count is intentionally not promoted to an
    amodal, occlusion, or geometric off-screen conclusion.
    """

    if readback is None:
        return None
    result: list[list[dict[str, Any]]] = [[], []]
    runtime_object_ids = tuple(int(value) for value in readback.runtime_object_ids)
    for frame_index in range(FRAME_COUNT):
        frame_records = readback.per_frame_records[frame_index]
        for axis, runtime_object_id in enumerate(runtime_object_ids):
            source = frame_records[axis]
            count = int(source["pixel_count"])
            bbox = deepcopy(source["bbox_xyxy"])
            modal_presence = bool(count > 0)
            result[axis].append(
                {
                    "authority": _OBJECT_ID_MODAL_AUTHORITY,
                    "authority_status": str(readback.authority_status),
                    "runtime_object_id": runtime_object_id,
                    "object_id_pixel_count": count,
                    "object_id_bbox_xyxy": bbox,
                    "visible_pixels": count,
                    "actor_modal_pixel_presence": modal_presence,
                    "actor_modal_visibility": modal_presence,
                    "actor_modal_visibility_status": (
                        "native_object_id_positive_modal_pixel_presence"
                        if modal_presence
                        else "native_object_id_zero_modal_pixel_presence"
                    ),
                    "in_frame": modal_presence,
                    "in_frame_status": _OBJECT_ID_IN_FRAME_INTERPRETATION,
                    "occlusion": None,
                    "occlusion_status": (
                        "pending_no_same_backend_amodal_or_occlusion_authority"
                    ),
                    "off_screen": None,
                    "off_screen_status": (
                        "pending_object_id_zero_cannot_distinguish_off_screen_"
                        "from_full_occlusion"
                    ),
                }
            )
    return result


def _object_id_boundary(
    readback: Any,
    tracks: list[list[dict[str, Any]]] | None,
    *,
    actor_ids: Sequence[str],
) -> dict[str, Any]:
    if readback is None or tracks is None:
        return {
            "status": "pending_no_native_object_id_raster",
            "sensor_uuid": None,
            "runtime_object_ids": None,
            "authority_status": None,
            "modal_pixel_count_by_frame": None,
            "modal_bbox_by_frame": None,
            "actor_modal_visibility_by_frame": None,
            "in_frame_by_frame": None,
            "occlusion_by_frame": None,
            "off_screen_by_frame": None,
            "legality_effect": "none",
        }
    return {
        "status": "native_object_id_raster_modal_descriptors_recorded",
        "sensor_uuid": str(readback.sensor_uuid),
        "runtime_object_ids": [int(value) for value in readback.runtime_object_ids],
        "authority_status": str(readback.authority_status),
        "modal_pixel_count_by_frame": {
            str(actor_ids[axis]): [
                int(frame["object_id_pixel_count"]) for frame in tracks[axis]
            ]
            for axis in range(2)
        },
        "modal_bbox_by_frame": {
            str(actor_ids[axis]): [
                deepcopy(frame["object_id_bbox_xyxy"]) for frame in tracks[axis]
            ]
            for axis in range(2)
        },
        "actor_modal_visibility_by_frame": {
            str(actor_ids[axis]): [
                bool(frame["actor_modal_visibility"]) for frame in tracks[axis]
            ]
            for axis in range(2)
        },
        "in_frame_by_frame": {
            str(actor_ids[axis]): [bool(frame["in_frame"]) for frame in tracks[axis]]
            for axis in range(2)
        },
        "occlusion_by_frame": None,
        "off_screen_by_frame": None,
        "in_frame_interpretation": _OBJECT_ID_IN_FRAME_INTERPRETATION,
        "legality_effect": "none",
    }


def _semantic_id_tracks(
    semantic: np.ndarray,
    *,
    semantic_ids: Sequence[int],
    visible_pixels: np.ndarray,
) -> list[list[dict[str, Any]]]:
    height, width = semantic.shape[1:]
    result: list[list[dict[str, Any]]] = [[], []]
    for frame_index in range(FRAME_COUNT):
        for axis, semantic_id in enumerate(semantic_ids):
            rows, columns = np.nonzero(semantic[frame_index] == semantic_id)
            count = int(rows.size)
            if count != int(visible_pixels[frame_index, axis]):
                raise MP3DRawClueError(
                    "native semantic pixels differ from the declared actor-axis counts"
                )
            if count == 0:
                bbox = None
                border_margin = None
                normalized_border_margin = None
            else:
                x_min = int(columns.min())
                x_max = int(columns.max())
                y_min = int(rows.min())
                y_max = int(rows.max())
                bbox = {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width_pixels": x_max - x_min + 1,
                    "height_pixels": y_max - y_min + 1,
                }
                border_margin = min(
                    x_min,
                    y_min,
                    width - 1 - x_max,
                    height - 1 - y_max,
                )
                normalized_border_margin = float(
                    border_margin / max(min(height, width) - 1, 1)
                )
            result[axis].append(
                {
                    "authority": "same_renderer_native_semantic_id_raster",
                    "semantic_id": int(semantic_id),
                    "semantic_id_pixel_count": count,
                    "semantic_id_bbox_xyxy": bbox,
                    "semantic_id_bbox_border_margin_pixels": border_margin,
                    "semantic_id_bbox_border_margin_normalized": normalized_border_margin,
                    "semantic_id_pixel_presence": count > 0,
                    "semantic_id_pixel_presence_interpretation": (
                        "positive_count_proves_only_that_this_semantic_id_appears; "
                        "zero_count_does_not_prove_actor_absence"
                    ),
                    "semantic_id_coverage_completeness_status": (
                        "semantic_id_coverage_completeness_not_certified"
                    ),
                    "actor_modal_visibility": None,
                    "actor_modal_bbox_xyxy": None,
                    "occlusion": None,
                    "off_screen": None,
                    "actor_modal_visibility_status": (
                        "pending_semantic_id_coverage_completeness_not_certified"
                    ),
                    "occlusion_status": (
                        "pending_no_actor_modal_or_amodal_visibility_authority"
                    ),
                    "off_screen_status": (
                        "pending_semantic_id_presence_or_absence_is_not_actor_visibility"
                    ),
                }
            )
    return result


def _motion_series(
    positions: np.ndarray,
    *,
    identities: Sequence[str],
    pts_ticks: Sequence[int] | None,
    pts_authority: str | None,
) -> dict[str, Any]:
    by_identity: dict[str, Any] = {}
    for axis, identity in enumerate(identities):
        displacements: list[float | None] = [None]
        speeds: list[float | None] = [None]
        for index in range(1, FRAME_COUNT):
            displacement = float(
                np.linalg.norm(positions[index, axis] - positions[index - 1, axis])
            )
            displacements.append(displacement)
            if pts_ticks is None:
                speeds.append(None)
            else:
                seconds = (pts_ticks[index] - pts_ticks[index - 1]) / TICKS_PER_SECOND
                speeds.append(float(displacement / seconds))
        by_identity[str(identity)] = {
            "positions_m": positions[:, axis, :].tolist(),
            "frame_to_frame_displacement_m": displacements,
            "frame_to_frame_speed_mps": speeds,
            "classification_threshold_or_gate": None,
        }
    return {
        "status": (
            "actual_displacement_and_speed_recorded_without_motion_gate"
            if pts_ticks is not None
            else "actual_displacement_recorded_speed_pending_no_pts_authority"
        ),
        "pts_authority": pts_authority,
        "by_identity": by_identity,
    }


def _validated_pts_ticks(records: Sequence[Mapping[str, Any]]) -> list[int]:
    result = [int(record["pts_ticks"]) for record in records]
    if any(right <= left for left, right in zip(result, result[1:])):
        raise MP3DRawClueError("dynamic frame-record PTS must be strictly increasing")
    return result


def _planned_motion(
    receipt_path: Path,
    inputs: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    m2_path = _declared_input_file(
        receipt_path, inputs.get("m2_request"), owner="dynamic inputs.m2_request"
    )
    m2 = _mapping(load_json(m2_path), owner="dynamic M2 request")
    states = m2.get("states")
    if not isinstance(states, list) or len(states) != FRAME_COUNT:
        raise MP3DRawClueError("dynamic M2 request must contain exactly 75 states")
    labels = []
    action_time_comparison_statuses = []
    for index, (state_value, frame_value) in enumerate(
        zip(states, records, strict=True)
    ):
        state = _mapping(state_value, owner=f"M2 states[{index}]")
        action_id = frame_value.get("action_id")
        action_sample_index = frame_value.get("action_sample_index")
        m2_action_time = state.get("action_time_ticks")
        frame_action_time = frame_value.get("action_time_ticks")
        if m2_action_time is not None and (
            isinstance(m2_action_time, bool)
            or not isinstance(m2_action_time, int)
            or m2_action_time < 0
        ):
            raise MP3DRawClueError(f"M2 action_time_ticks is invalid at frame {index}")
        if frame_action_time is not None and (
            isinstance(frame_action_time, bool)
            or not isinstance(frame_action_time, int)
            or frame_action_time < 0
        ):
            raise MP3DRawClueError(
                f"frame-record action_time_ticks is invalid at frame {index}"
            )
        if frame_action_time is not None and m2_action_time != frame_action_time:
            raise MP3DRawClueError(
                f"M2/frame-record action_time_ticks differ at frame {index}"
            )
        if (
            state.get("frame_index") != index
            or state.get("pts_ticks") != frame_value.get("pts_ticks")
            or state.get("action_id") != action_id
            or not isinstance(action_id, str)
            or not action_id
            or isinstance(action_sample_index, bool)
            or not isinstance(action_sample_index, int)
            or action_sample_index < 0
        ):
            raise MP3DRawClueError(
                f"M2/frame-record planned motion labels differ at frame {index}"
            )
        if m2_action_time is None:
            action_time_status = "absent_in_m2_and_frame_record"
        elif frame_action_time is None:
            action_time_status = "retained_from_m2_frame_record_value_absent"
        else:
            action_time_status = "exact_match_between_m2_and_frame_record"
        action_time_comparison_statuses.append(action_time_status)
        labels.append(
            {
                "frame_index": index,
                "pts_ticks": int(frame_value["pts_ticks"]),
                "action_id": action_id,
                "action_time_ticks": m2_action_time,
                "action_time_ticks_comparison_status": action_time_status,
                "action_sample_index": action_sample_index,
                "action_sample_index_status": "retained_not_independently_rederived",
            }
        )
    return {
        "status": (
            "frame_index_pts_and_action_id_compared; action_time_ticks_retained_"
            "and_compared_when_frame_recorded; action_sample_index_retained_not_"
            "independently_rederived"
        ),
        "m2_request": str(m2_path),
        "m2_request_id": m2.get("request_id"),
        "action_time_ticks_comparison_statuses": action_time_comparison_statuses,
        "action_sample_index_status": "retained_not_independently_rederived",
        "labels_by_frame": labels,
    }


def _pending_audio() -> dict[str, Any]:
    return {
        "status": "pending_no_audio_program_or_native_audio_receipt",
        "audio_program": None,
        "audio_receipt": None,
        "source_event_timing_by_frame": None,
        "p1_or_rir_identity": None,
    }


def _capture_identity(
    receipt: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    sources = request.get("sources")
    if not isinstance(sources, list):
        raise MP3DRawClueError("joined M1 request lacks source records")
    expected = {
        "request_id": request.get("request_id"),
        "room_id": request.get("room_id"),
        "primary_camera_rig": request.get("primary_camera_rig"),
        "listener": request.get("listener"),
        "source_axis_order": [
            item.get("source_id") if isinstance(item, Mapping) else None
            for item in sources
        ],
        "qa_views": request.get("qa_views", []),
    }
    identity = receipt.get("m1_capture_identity")
    if not isinstance(identity, Mapping) or dict(identity) != expected:
        raise MP3DRawClueError(
            "capture-time m1_capture_identity differs from the joined M1 record"
        )
    return deepcopy(dict(identity))


def _legacy_raw_clues(
    *,
    request: Mapping[str, Any],
    source_ids: tuple[str, str],
    actor_ids: tuple[str, str],
    source_positions: np.ndarray,
    visible_pixels: np.ndarray,
    receipt_path: Path,
    inputs: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    rig = _mapping(request.get("primary_camera_rig"), owner="M1 camera rig")
    transform = deepcopy(
        dict(_mapping(rig.get("world_from_rig"), owner="M1 camera transform"))
    )
    listener = _mapping(request.get("listener"), owner="M1 listener")
    planned_listener = compose_transforms(
        dict(transform),
        dict(_mapping(listener.get("rig_from_listener"), owner="M1 rig_from_listener")),
    )
    if records is None:
        pts_ticks = None
        pts_status = "pending_no_legacy_frame_records_artifact"
        planned_motion: Mapping[str, Any] = {
            "status": "pending_no_legacy_frame_records_or_m2_comparison",
            "labels_by_frame": None,
        }
    else:
        pts_ticks = _validated_pts_ticks(records)
        pts_status = "validated_from_legacy_frame_records_artifact"
        if isinstance(inputs.get("m2_request"), str) and inputs.get("m2_request"):
            planned_motion = _planned_motion(receipt_path, inputs, records)
        else:
            planned_motion = {
                "status": "pending_no_m2_request_for_legacy_frame_comparison",
                "labels_by_frame": None,
            }
    frames = []
    for index in range(FRAME_COUNT):
        planned_label = (
            None
            if planned_motion.get("labels_by_frame") is None
            else planned_motion["labels_by_frame"][index]
        )
        sources = {}
        for axis, source_id in enumerate(source_ids):
            sources[source_id] = {
                "actor_id": actor_ids[axis],
                "planned_actor_world_matrix": None,
                "planned_applied_skin_root_matrix": None,
                "actual_skin_root_readback_matrix": None,
                "planned_emitter_position_m": None,
                "planned_emitter_status": "pending_no_same_backend_authority",
                "actual_emitter_position_m": source_positions[index, axis].tolist(),
                "actual_listener_local_geometry": None,
                "analytic_pinhole_center": None,
                "native_semantic_id_raster": {
                    "semantic_id_pixel_count": int(visible_pixels[index, axis]),
                    "semantic_id_bbox_xyxy": None,
                    "semantic_id_pixel_presence": (
                        int(visible_pixels[index, axis]) > 0
                    ),
                    "semantic_id_coverage_completeness_status": (
                        "semantic_id_coverage_completeness_not_certified"
                    ),
                    "actor_modal_visibility": None,
                    "occlusion": None,
                    "off_screen": None,
                    "status": "legacy_semantic_id_count_only_raster_not_joined",
                },
                "runtime_object_id": None,
                "native_object_id_raster": None,
            }
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": None if pts_ticks is None else pts_ticks[index],
                "pts_status": pts_status,
                "planned_motion_label": planned_label,
                "camera": {
                    "planned_world_from_camera": deepcopy(transform),
                    "actual_post_render_agent_camera": None,
                    "actual_post_render_sensors": None,
                    "actual_status": "pending_legacy_capture",
                },
                "listener": {
                    "planned_world_from_listener": deepcopy(planned_listener),
                    "actual_derived_world_from_listener": None,
                    "actual_status": "pending_legacy_capture",
                },
                "sources": sources,
            }
        )
    return {
        "status": "legacy_capture_pending_current_pose_authority",
        "has_current_post_render_pose_authority": False,
        "frames": frames,
        "pair_tracks": {
            "actual_emitter_relative_to_actual_listener": None,
            "actual_skin_root_relative_to_actual_listener": None,
        },
        "motion_timing": {
            "planned": planned_motion,
            "actual_skin_root": {
                "status": "pending_no_actual_skin_root_readback",
                "by_identity": None,
            },
            "actual_emitter": _motion_series(
                source_positions,
                identities=source_ids,
                pts_ticks=pts_ticks,
                pts_authority=(
                    "legacy_dynamic_capture.frame_records.pts_ticks"
                    if pts_ticks is not None
                    else None
                ),
            ),
        },
        "planned_continuous_polyline": {
            "status": "pending_no_explicit_continuous_polyline_authority",
            "paths": None,
        },
        "audio_timing": _pending_audio(),
        "p1_amodal_visibility": {
            "status": "pending_no_p1_or_amodal_receipt",
            "occlusion_by_frame": None,
            "off_screen_by_frame": None,
        },
        "semantic_id_raster_boundary": {
            "status": "semantic_id_coverage_completeness_not_certified",
            "positive_count_interpretation": "semantic_id_appears_in_raster",
            "zero_count_interpretation": "does_not_prove_actor_absence",
            "actor_modal_visibility": None,
            "occlusion": None,
            "off_screen": None,
        },
        "object_id_raster_boundary": _object_id_boundary(
            None, None, actor_ids=actor_ids
        ),
        "candidate_filtering": "none",
        "candidate_retained": True,
    }


def _current_raw_clues(
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    request: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    source_positions: np.ndarray,
    visible_pixels: np.ndarray,
    actor_ids: tuple[str, str],
    source_ids: tuple[str, str],
) -> dict[str, Any]:
    root = receipt_path.parent
    frame_count = _require_completed_receipt(receipt)
    if frame_count != FRAME_COUNT:
        raise MP3DRawClueError("current raw clues require exactly 75 frames")
    rgb_path = _artifact_path(root, artifacts, name="rgb", required=True)
    semantic_path = _artifact_path(root, artifacts, name="semantic", required=True)
    records_path = _artifact_path(root, artifacts, name="frame_records", required=True)
    assert (
        rgb_path is not None and semantic_path is not None and records_path is not None
    )
    rgb = _require_rgb(
        _load_npy(rgb_path, owner="RGB artifact"), frame_count=frame_count
    )
    semantic = _require_semantic(
        _load_npy(semantic_path, owner="semantic artifact"), rgb=rgb
    )
    records = _frame_records(records_path, frame_count=frame_count)
    actor_matrices = _optional_frame_array(
        _artifact_path(root, artifacts, name="actor_world_matrices", required=True),
        owner="actor_world_matrices artifact",
        frame_count=frame_count,
        suffix_shape=(2, 4, 4),
        dtype_kind="finite",
    )
    pose_evidence = _pose_readback_evidence(
        root,
        receipt,
        artifacts,
        frame_count=frame_count,
        records=records,
        actor_matrices=actor_matrices,
        source_positions=source_positions,
        visibility=visible_pixels,
    )
    if pose_evidence is None or actor_matrices is None:
        raise MP3DRawClueError(
            "current capture lacks complete post-render pose evidence"
        )
    if (
        pose_evidence.actor_axis_order != actor_ids
        or pose_evidence.source_axis_order != source_ids
    ):
        raise MP3DRawClueError(
            "dynamic receipt identity axes differ from actor/source lists"
        )
    object_id_readback = _object_id_readback_evidence(
        root,
        receipt,
        artifacts,
        rgb=rgb,
        records=records,
        pose_readback=pose_evidence,
    )
    object_id_tracks = _object_id_tracks(object_id_readback)
    object_id_boundary = _object_id_boundary(
        object_id_readback, object_id_tracks, actor_ids=actor_ids
    )
    inputs = _mapping(receipt.get("inputs"), owner="dynamic receipt inputs")
    capture_identity = _capture_identity(receipt, request)
    planned_motion = _planned_motion(receipt_path, inputs, records)
    pts_ticks = _validated_pts_ticks(records)
    if pts_ticks != [index * TICKS_PER_FRAME for index in range(FRAME_COUNT)]:
        raise MP3DRawClueError("dynamic frame PTS must be exact 15 fps ticks")

    actors = _mapping(receipt.get("actors"), owner="dynamic actors")
    semantic_ids = actors.get("semantic_ids")
    if (
        not isinstance(semantic_ids, list)
        or len(semantic_ids) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in semantic_ids
        )
        or len(set(semantic_ids)) != 2
    ):
        raise MP3DRawClueError("dynamic semantic ID axis is invalid")
    axis_bindings = [deepcopy(item) for item in pose_evidence.axis_bindings]

    hfov, height, width = _calibration(
        capture_identity, semantic_shape=(semantic.shape[1], semantic.shape[2])
    )
    sensor_descriptors = _mapping(
        receipt.get("pose_readback"), owner="pose_readback"
    ).get("sensor_order")
    assert isinstance(sensor_descriptors, list)
    semantic_sensor_axes = [
        index
        for index, item in enumerate(sensor_descriptors)
        if isinstance(item, Mapping) and item.get("modality") == "semantic"
    ]
    if len(semantic_sensor_axes) != 1:
        raise MP3DRawClueError("pose_readback must identify one semantic sensor axis")
    semantic_sensor_axis = semantic_sensor_axes[0]
    semantic_sensor_uuid = pose_evidence.sensor_axis_order[semantic_sensor_axis]
    semantic_sensor_poses = np.asarray(
        pose_evidence.actual_sensor_poses[:, semantic_sensor_axis, :], dtype=np.float64
    )
    listener_positions, listener_forward, listener_right, listener_up = _basis_tracks(
        pose_evidence.actual_logical_listener_poses
    )
    actual_skin_positions = np.asarray(
        pose_evidence.actual_skin_root_matrices[:, :, :3, 3], dtype=np.float64
    )
    planned_actor_positions = np.asarray(actor_matrices[:, :, :3, 3], dtype=np.float64)
    planned_skin_positions = np.asarray(
        pose_evidence.planned_skin_root_matrices[:, :, :3, 3], dtype=np.float64
    )

    try:
        emitter_track = describe_source_pair_tracks(
            source_positions[:, 0, :],
            source_positions[:, 1, :],
            source1_id=source_ids[0],
            source2_id=source_ids[1],
            listener_positions_xyz=listener_positions,
            listener_forward_xyz=listener_forward,
            listener_right_xyz=listener_right,
            listener_up_xyz=listener_up,
            horizontal_axis_indices=(0, 2),
            vertical_axis_index=1,
            coordinate_space="habitat_world_xyz_m_y_up",
            track_authority="dynamic_capture.actual_emitter_link_readback",
            frame_indices=tuple(range(FRAME_COUNT)),
            pts_ticks=pts_ticks,
        )
        skin_track = describe_source_pair_tracks(
            actual_skin_positions[:, 0, :],
            actual_skin_positions[:, 1, :],
            source1_id=actor_ids[0],
            source2_id=actor_ids[1],
            listener_positions_xyz=listener_positions,
            listener_forward_xyz=listener_forward,
            listener_right_xyz=listener_right,
            listener_up_xyz=listener_up,
            horizontal_axis_indices=(0, 2),
            vertical_axis_index=1,
            coordinate_space="habitat_world_xyz_m_y_up",
            track_authority="dynamic_capture.actual_skin_root_readback",
            frame_indices=tuple(range(FRAME_COUNT)),
            pts_ticks=pts_ticks,
        )
    except MotionFeasibilityError as exc:
        raise MP3DRawClueError(f"actual source-pair track is invalid: {exc}") from exc
    for track in (emitter_track, skin_track):
        track["runtime_track_claim"] = True
        track["runtime_readback_verification"] = (
            "complete_current_visual_post_render_receipt_and_arrays_cross_checked"
        )
        track["legality"]["runtime_pose_readback"] = True

    projections = _pinhole_tracks(
        source_positions,
        semantic_sensor_poses,
        hfov_degrees=hfov,
        height=height,
        width=width,
    )
    semantic_id_tracks = _semantic_id_tracks(
        semantic, semantic_ids=semantic_ids, visible_pixels=visible_pixels
    )
    for frame_index, frame in enumerate(emitter_track["frames"]):
        for axis, key in enumerate(("source1", "source2")):
            frame[key]["projection"] = deepcopy(projections[axis][frame_index])
            object_record = (
                None
                if object_id_tracks is None
                else object_id_tracks[axis][frame_index]
            )
            frame[key]["visible_pixels"] = (
                None
                if object_record is None
                else int(object_record["object_id_pixel_count"])
            )
            frame[key]["semantic_id_pixel_count"] = int(
                visible_pixels[frame_index, axis]
            )
            frame[key]["object_id_pixel_count"] = (
                None
                if object_record is None
                else int(object_record["object_id_pixel_count"])
            )
            frame[key]["object_id_bbox_xyxy"] = (
                None
                if object_record is None
                else deepcopy(object_record["object_id_bbox_xyxy"])
            )
            frame[key]["actor_modal_visibility"] = (
                None
                if object_record is None
                else bool(object_record["actor_modal_visibility"])
            )
            frame[key]["in_frame"] = (
                None if object_record is None else bool(object_record["in_frame"])
            )
            frame[key]["occlusion"] = None
            frame[key]["off_screen"] = None
    emitter_track["native_projection_visibility"] = {
        "status": (
            "analytic_center_and_native_object_id_modal_raster_described_"
            "semantic_id_diagnostic_only"
            if object_id_tracks is not None
            else "analytic_center_and_native_semantic_id_raster_described"
        ),
        "projection_authority": _PINHOLE_AUTHORITY,
        "semantic_id_raster_authority": "same_renderer_native_semantic_id_raster",
        "semantic_id_coverage_completeness_status": (
            "semantic_id_coverage_completeness_not_certified"
        ),
        "actor_modal_visibility_by_frame": (
            None
            if object_id_tracks is None
            else {
                str(source_ids[axis]): [
                    bool(frame["actor_modal_visibility"])
                    for frame in object_id_tracks[axis]
                ]
                for axis in range(2)
            }
        ),
        "in_frame_by_frame": (
            None
            if object_id_tracks is None
            else {
                str(source_ids[axis]): [
                    bool(frame["in_frame"]) for frame in object_id_tracks[axis]
                ]
                for axis in range(2)
            }
        ),
        "occlusion_by_frame": None,
        "off_screen_by_frame": None,
        "object_id_raster_boundary": deepcopy(object_id_boundary),
        "ambiguity": (
            "semantic ID presence or absence is not actor visibility authority; "
            "OBJECT_ID records modal pixel presence only"
        ),
    }

    rig = _mapping(capture_identity.get("primary_camera_rig"), owner="M1 camera rig")
    planned_camera = deepcopy(
        dict(_mapping(rig.get("world_from_rig"), owner="M1 camera transform"))
    )
    listener = _mapping(capture_identity.get("listener"), owner="M1 listener")
    planned_listener = compose_transforms(
        dict(planned_camera),
        dict(_mapping(listener.get("rig_from_listener"), owner="M1 rig_from_listener")),
    )
    frame_output = []
    for frame_index in range(FRAME_COUNT):
        sensors = {
            sensor_uuid: _pose(
                pose_evidence.actual_sensor_poses[frame_index, sensor_index]
            )
            for sensor_index, sensor_uuid in enumerate(pose_evidence.sensor_axis_order)
        }
        source_records = {}
        for axis, source_id in enumerate(source_ids):
            geometry_key = "source1" if axis == 0 else "source2"
            geometry = emitter_track["frames"][frame_index][geometry_key]
            source_records[source_id] = {
                "actor_id": actor_ids[axis],
                "semantic_id": int(semantic_ids[axis]),
                "runtime_object_id": (
                    None
                    if object_id_readback is None
                    else int(object_id_readback.runtime_object_ids[axis])
                ),
                "axis_binding": deepcopy(axis_bindings[axis]),
                "planned_actor_world_matrix": actor_matrices[
                    frame_index, axis
                ].tolist(),
                "planned_actor_world_position_m": planned_actor_positions[
                    frame_index, axis
                ].tolist(),
                "planned_applied_skin_root_matrix": pose_evidence.planned_skin_root_matrices[
                    frame_index, axis
                ].tolist(),
                "planned_applied_skin_root_position_m": planned_skin_positions[
                    frame_index, axis
                ].tolist(),
                "actual_skin_root_readback_matrix": pose_evidence.actual_skin_root_matrices[
                    frame_index, axis
                ].tolist(),
                "actual_skin_root_readback_position_m": actual_skin_positions[
                    frame_index, axis
                ].tolist(),
                "planned_emitter_position_m": None,
                "planned_emitter_status": "pending_no_explicit_planned_emitter_authority",
                "actual_emitter_position_m": source_positions[
                    frame_index, axis
                ].tolist(),
                "actual_listener_local_geometry": {
                    "azimuth_deg": geometry["listener_local_azimuth_deg"],
                    "elevation_deg": geometry["listener_local_elevation_deg"],
                    "distance_m": float(
                        np.linalg.norm(
                            source_positions[frame_index, axis]
                            - listener_positions[frame_index]
                        )
                    ),
                    "right_coordinate_m": geometry["listener_local_right_coordinate"],
                    "side": geometry["listener_side"],
                    "side_margin": geometry["listener_side_margin"],
                },
                "analytic_pinhole_center": deepcopy(projections[axis][frame_index]),
                "native_semantic_id_raster": deepcopy(
                    semantic_id_tracks[axis][frame_index]
                ),
                "native_object_id_raster": (
                    None
                    if object_id_tracks is None
                    else deepcopy(object_id_tracks[axis][frame_index])
                ),
            }
        frame_output.append(
            {
                "frame_index": frame_index,
                "pts_ticks": pts_ticks[frame_index],
                "planned_motion_label": deepcopy(
                    planned_motion["labels_by_frame"][frame_index]
                ),
                "camera": {
                    "planned_world_from_camera": deepcopy(planned_camera),
                    "actual_post_render_agent_camera": _pose(
                        pose_evidence.actual_agent_camera_poses[frame_index]
                    ),
                    "actual_post_render_sensors": sensors,
                    "projection_sensor_uuid": semantic_sensor_uuid,
                },
                "listener": {
                    "planned_world_from_listener": deepcopy(planned_listener),
                    "actual_derived_world_from_listener": _pose(
                        pose_evidence.actual_logical_listener_poses[frame_index]
                    ),
                    "derived_from_actual_post_render_agent_readback": True,
                    "native_listener_sensor_readback": False,
                },
                "sources": source_records,
            }
        )

    return {
        "status": "current_native_post_render_raw_clues_described",
        "has_current_post_render_pose_authority": True,
        "frame_count": FRAME_COUNT,
        "m1_capture_identity": capture_identity,
        "identity_axes": {
            "actor_axis_order": list(actor_ids),
            "source_axis_order": list(source_ids),
            "sensor_axis_order": list(pose_evidence.sensor_axis_order),
            "axis_bindings": axis_bindings,
            "runtime_object_id_axis_order": (
                None
                if object_id_readback is None
                else [int(value) for value in object_id_readback.runtime_object_ids]
            ),
        },
        "calibration": {
            "projection": "pinhole",
            "hfov_degrees": hfov,
            "resolution_hw": [height, width],
            "projection_sensor_uuid": semantic_sensor_uuid,
            "pose_authority": "actual_post_render_semantic_sensor_pose",
            "intrinsics_authority": "capture_time_m1_capture_identity",
            "pixel_center_convention": _PIXEL_CENTER_CONVENTION,
        },
        "frames": frame_output,
        "pair_tracks": {
            "actual_emitter_relative_to_actual_listener": emitter_track,
            "actual_skin_root_relative_to_actual_listener": skin_track,
        },
        "motion_timing": {
            "planned": planned_motion,
            "actual_skin_root": _motion_series(
                actual_skin_positions,
                identities=actor_ids,
                pts_ticks=pts_ticks,
                pts_authority="dynamic_capture.frame_records.pts_ticks",
            ),
            "actual_emitter": _motion_series(
                source_positions,
                identities=source_ids,
                pts_ticks=pts_ticks,
                pts_authority="dynamic_capture.frame_records.pts_ticks",
            ),
        },
        "planned_continuous_polyline": {
            "status": "pending_no_explicit_continuous_polyline_authority",
            "paths": None,
            "between_frame_interpolation_performed": False,
        },
        "audio_timing": _pending_audio(),
        "p1_amodal_visibility": {
            "status": "pending_no_p1_or_amodal_receipt",
            "occlusion_by_frame": None,
            "off_screen_by_frame": None,
        },
        "semantic_id_raster_boundary": {
            "status": "semantic_id_coverage_completeness_not_certified",
            "positive_count_interpretation": "semantic_id_appears_in_raster",
            "zero_count_interpretation": "does_not_prove_actor_absence",
            "actor_modal_visibility": None,
            "occlusion": None,
            "off_screen": None,
        },
        "object_id_raster_boundary": object_id_boundary,
        "legality": {
            "status": "not_redecided_by_raw_clue_bridge",
            "candidate_retained": True,
            "distance_side_visibility_composition_effect": "none",
        },
        "difficulty": {
            "status": "raw_descriptors_only",
            "candidate_filtering": "disabled",
        },
        "candidate_filtering": "none",
        "candidate_retained": True,
    }


def load_dynamic_capture_raw_clues(
    capture_root: Path,
    *,
    request: Mapping[str, Any],
    request_path: Path,
    expected_room_id: str,
) -> LoadedMP3DDynamicClues:
    """Validate one capture and describe it without changing candidate retention."""

    try:
        root = capture_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise MP3DRawClueError(
            f"dynamic capture cannot be resolved: {capture_root}"
        ) from exc
    if not root.is_dir():
        raise MP3DRawClueError(f"dynamic capture is not a directory: {root}")
    receipt_path = root / "research_receipt.json"
    receipt = _mapping(load_json(receipt_path), owner="dynamic receipt")
    inputs = _mapping(receipt.get("inputs"), owner="dynamic receipt inputs")
    selected_room = _mapping(receipt.get("selected_room"), owner="selected room")
    capture = _mapping(receipt.get("capture"), owner="dynamic capture metadata")
    actors = _mapping(receipt.get("actors"), owner="dynamic actors")
    artifacts = _mapping(receipt.get("artifacts"), owner="dynamic artifacts")
    visibility = _mapping(receipt.get("visibility"), owner="dynamic visibility")
    declared_request = _declared_input_file(
        receipt_path, inputs.get("m1_request"), owner="dynamic inputs.m1_request"
    )
    if declared_request != request_path.resolve(strict=True):
        raise MP3DRawClueError(
            "dynamic receipt M1 request differs from joined candidate"
        )
    if (
        receipt.get("status") != "research_only"
        or receipt.get("research_only") is not True
        or receipt.get("episode_counted") is not False
        or selected_room.get("room_id") != expected_room_id
        or capture.get("native_habitat_started") is not True
        or capture.get("frame_count") != FRAME_COUNT
        or visibility.get("candidate_filtering") != "disabled"
    ):
        raise MP3DRawClueError(
            "dynamic receipt must be one research-only, non-counted 75-frame native run"
        )
    actor_ids = _string_axis(actors.get("actor_ids"), owner="actor IDs")
    source_ids = _string_axis(actors.get("source_ids"), owner="source IDs")
    request_sources = request.get("sources")
    if isinstance(request_sources, Sequence) and not isinstance(
        request_sources, (str, bytes)
    ):
        request_source_ids = tuple(
            item.get("source_id") if isinstance(item, Mapping) else None
            for item in request_sources
        )
        if request_source_ids != source_ids:
            raise MP3DRawClueError(
                "dynamic source identity axis differs from the joined M1 request"
            )

    source_path = _artifact_path(
        root, artifacts, name="source_positions_m", required=True
    )
    visibility_path = _artifact_path(
        root, artifacts, name="semantic_visibility_pixels", required=True
    )
    assert source_path is not None and visibility_path is not None
    source_positions = _optional_frame_array(
        source_path,
        owner="source_positions_m artifact",
        frame_count=FRAME_COUNT,
        suffix_shape=(2, 3),
        dtype_kind="finite",
    )
    visible_pixels = _optional_frame_array(
        visibility_path,
        owner="semantic_visibility_pixels artifact",
        frame_count=FRAME_COUNT,
        suffix_shape=(2,),
        dtype_kind="integer",
    )
    assert source_positions is not None and visible_pixels is not None
    if np.any(visible_pixels < 0):
        raise MP3DRawClueError("native visible-pixel counts must be nonnegative")
    per_actor = _mapping(visibility.get("per_actor"), owner="per-actor visibility")
    for axis, actor_id in enumerate(actor_ids):
        record = _mapping(per_actor.get(actor_id), owner=f"visibility {actor_id}")
        if record.get("per_frame_visible_pixels") != visible_pixels[:, axis].tolist():
            raise MP3DRawClueError(
                f"dynamic receipt visibility differs from array for {actor_id}"
            )

    has_metadata = receipt.get("pose_readback") is not None
    present_new_artifacts = _NEW_POSE_ARTIFACTS.intersection(artifacts)
    object_id_declared = "object_id_visibility" in receipt or "object_id" in artifacts
    if object_id_declared and (
        not has_metadata or present_new_artifacts != _NEW_POSE_ARTIFACTS
    ):
        raise MP3DRawClueError(
            "object-ID evidence is partial; current pose/readback authority is required"
        )
    if has_metadata or present_new_artifacts:
        if not has_metadata or present_new_artifacts != _NEW_POSE_ARTIFACTS:
            raise MP3DRawClueError(
                "current pose evidence is partial; legacy captures cannot be upgraded"
            )
        try:
            raw_clues = _current_raw_clues(
                receipt_path=receipt_path,
                receipt=receipt,
                request=request,
                artifacts=artifacts,
                source_positions=np.asarray(source_positions),
                visible_pixels=np.asarray(visible_pixels),
                actor_ids=actor_ids,
                source_ids=source_ids,
            )
        except CurrentVisualReviewError as exc:
            raise MP3DRawClueError(str(exc)) from exc
        current = True
    else:
        legacy_records = None
        if artifacts.get("frame_records") is not None:
            frame_records_path = _artifact_path(
                root, artifacts, name="frame_records", required=True
            )
            assert frame_records_path is not None
            legacy_records = _frame_records(frame_records_path, frame_count=FRAME_COUNT)
        raw_clues = _legacy_raw_clues(
            request=request,
            source_ids=source_ids,
            actor_ids=actor_ids,
            source_positions=np.asarray(source_positions),
            visible_pixels=np.asarray(visible_pixels),
            receipt_path=receipt_path,
            inputs=inputs,
            records=legacy_records,
        )
        current = False
    return LoadedMP3DDynamicClues(
        root=root,
        receipt_path=receipt_path,
        receipt=deepcopy(dict(receipt)),
        actor_ids=actor_ids,
        source_ids=source_ids,
        source_positions=np.ascontiguousarray(source_positions),
        visible_pixels=np.ascontiguousarray(visible_pixels),
        has_current_pose_authority=current,
        raw_clues=raw_clues,
    )


__all__ = [
    "LoadedMP3DDynamicClues",
    "MP3DRawClueError",
    "dynamic_capture_request_path",
    "load_dynamic_capture_raw_clues",
]
