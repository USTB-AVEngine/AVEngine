"""Pixel-level modal/target-only visibility truth for QA facts.

The normal semantic pass and the target-only semantic pass must come from the
same renderer, RGB backend, camera contract and per-frame camera poses.  A
target-only mask is the target's in-view footprint with scene occluders
removed; it is not an assertion about geometry outside the camera frustum.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np


PIXEL_VISIBILITY_SCHEMA = "avengine_qa_pixel_visibility_truth_v1"
PIXEL_VISIBILITY_AUTHORITY = (
    "same_renderer_same_camera_modal_target_only_semantic_masks_v1"
)
PIXEL_VISIBILITY_DEPTH_AUTHORITY = (
    "same_renderer_same_camera_normal_vs_target_only_metric_depth_v1"
)
PIXEL_VISIBILITY_AUTHORITIES = (
    PIXEL_VISIBILITY_AUTHORITY,
    PIXEL_VISIBILITY_DEPTH_AUTHORITY,
)
PIXEL_VISIBILITY_STATES = (
    "out_of_view",
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
)
_COMMON_CONTEXT_FIELDS = (
    "renderer_backend",
    "rgb_renderer_backend",
    "camera_contract_id",
    "semantic_id_namespace",
    "resolution_hw",
    "frame_indices",
    "camera_pose_ids",
)


class PixelVisibilityError(ValueError):
    """Modal and target-only semantic evidence is inconsistent."""


def _non_empty_text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PixelVisibilityError(f"{owner} must be a non-empty string")
    return value


def _normalize_resolution(value: Any, *, owner: str) -> tuple[int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item <= 0 for item in value)
    ):
        raise PixelVisibilityError(f"{owner} must be positive [height, width]")
    return int(value[0]), int(value[1])


def _normalize_context(
    value: Any,
    *,
    expected_pass_kind: str,
    frame_count: int,
    target_instance_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PixelVisibilityError(f"{expected_pass_kind} context must be an object")
    pass_kind = value.get("pass_kind")
    if pass_kind != expected_pass_kind:
        raise PixelVisibilityError(
            f"context pass_kind must be {expected_pass_kind!r}, got {pass_kind!r}"
        )
    normalized: dict[str, Any] = {"pass_kind": pass_kind}
    for field in (
        "renderer_backend",
        "rgb_renderer_backend",
        "camera_contract_id",
        "semantic_id_namespace",
    ):
        normalized[field] = _non_empty_text(value.get(field), owner=field)
    if normalized["renderer_backend"] != normalized["rgb_renderer_backend"]:
        raise PixelVisibilityError(
            "semantic-mask renderer_backend must equal rgb_renderer_backend; "
            "Habitat labels cannot describe non-matching UE RGB"
        )
    resolution = _normalize_resolution(
        value.get("resolution_hw"), owner="resolution_hw"
    )
    normalized["resolution_hw"] = [resolution[0], resolution[1]]

    frame_indices = value.get("frame_indices")
    if (
        isinstance(frame_indices, (str, bytes))
        or not isinstance(frame_indices, Sequence)
        or len(frame_indices) != frame_count
        or any(
            isinstance(frame, bool) or not isinstance(frame, int)
            for frame in frame_indices
        )
        or any(frame < 0 for frame in frame_indices)
        or list(frame_indices) != sorted(set(frame_indices))
    ):
        raise PixelVisibilityError(
            "frame_indices must be a strictly increasing integer per mask frame"
        )
    normalized["frame_indices"] = [int(frame) for frame in frame_indices]

    camera_pose_ids = value.get("camera_pose_ids")
    if (
        isinstance(camera_pose_ids, (str, bytes))
        or not isinstance(camera_pose_ids, Sequence)
        or len(camera_pose_ids) != frame_count
    ):
        raise PixelVisibilityError("camera_pose_ids must contain one id per mask frame")
    normalized["camera_pose_ids"] = [
        _non_empty_text(pose_id, owner="camera_pose_ids[]")
        for pose_id in camera_pose_ids
    ]

    if expected_pass_kind == "target_only":
        observed_target = _non_empty_text(
            value.get("target_instance_id"), owner="target_instance_id"
        )
        if observed_target != target_instance_id:
            raise PixelVisibilityError(
                "target-only context targets a different instance"
            )
        normalized["target_instance_id"] = observed_target
    return normalized


def _normalize_mask(
    value: Any, *, resolution_hw: tuple[int, int], owner: str
) -> np.ndarray:
    try:
        mask = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise PixelVisibilityError(
            f"{owner} must be an integer semantic mask"
        ) from error
    if mask.shape != resolution_hw or mask.dtype.kind not in {"i", "u"}:
        raise PixelVisibilityError(
            f"{owner} must be an integer semantic mask with shape {resolution_hw}"
        )
    if np.any(mask < 0):
        raise PixelVisibilityError(f"{owner} may not contain negative semantic ids")
    return np.ascontiguousarray(mask)


def _normalize_depth_frames(
    value: Any,
    *,
    frame_count: int,
    resolution_hw: tuple[int, int],
    owner: str,
) -> list[np.ndarray]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != frame_count
    ):
        raise PixelVisibilityError(
            f"{owner} must contain one metric-depth frame per context frame"
        )
    frames: list[np.ndarray] = []
    for index, item in enumerate(value):
        try:
            frame = np.asarray(item)
        except (TypeError, ValueError) as error:
            raise PixelVisibilityError(
                f"{owner}[{index}] must be a metric-depth array"
            ) from error
        if frame.shape != resolution_hw or frame.dtype.kind not in {"f", "i", "u"}:
            raise PixelVisibilityError(
                f"{owner}[{index}] must have numeric shape {resolution_hw}"
            )
        normalized = np.asarray(frame, dtype=np.float32)
        if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
            raise PixelVisibilityError(
                f"{owner}[{index}] must contain finite positive metric depth"
            )
        frames.append(np.ascontiguousarray(normalized))
    return frames


def _finite_non_negative(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise PixelVisibilityError(f"{owner} must be finite and non-negative")
    return float(value)


def _frame_truth(
    *,
    modal_mask: np.ndarray,
    target_only_mask: np.ndarray,
    target_semantic_id: int,
    frame_index: int,
) -> dict[str, Any]:
    target_values = set(int(value) for value in np.unique(target_only_mask))
    unexpected = sorted(target_values - {0, target_semantic_id})
    if unexpected:
        raise PixelVisibilityError(
            "target-only mask contains foreign semantic ids: "
            + ", ".join(str(value) for value in unexpected)
        )
    modal_target = modal_mask == target_semantic_id
    target_footprint = target_only_mask == target_semantic_id
    if np.any(modal_target & ~target_footprint):
        raise PixelVisibilityError(
            "modal target pixels are not a subset of the target-only footprint; "
            "camera, pose, renderer or target geometry is inconsistent"
        )
    visible_pixels = int(np.count_nonzero(modal_target))
    target_pixels = int(np.count_nonzero(target_footprint))

    if target_pixels == 0:
        state = "out_of_view"
        visible_fraction: float | None = None
        occlusion_fraction: float | None = None
    else:
        visible_fraction = visible_pixels / target_pixels
        occlusion_fraction = 1.0 - visible_fraction
        if visible_pixels == target_pixels:
            state = "visible_clear"
        elif visible_pixels == 0:
            state = "fully_occluded"
        else:
            state = "visible_occluded"
    return {
        "frame_index": frame_index,
        "visible_pixels": visible_pixels,
        "target_pixels": target_pixels,
        "visible_fraction": visible_fraction,
        "occlusion_fraction": occlusion_fraction,
        "state": state,
    }


def compile_pixel_visibility_truth(
    *,
    normal_semantic_masks: Sequence[Any],
    target_only_semantic_masks_by_instance: Mapping[str, Sequence[Any]],
    semantic_ids_by_instance: Mapping[str, int],
    normal_context: Mapping[str, Any],
    target_only_contexts_by_instance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile exact in-view visibility ratios from paired semantic passes."""

    if (
        isinstance(normal_semantic_masks, (str, bytes))
        or not isinstance(normal_semantic_masks, Sequence)
        or not normal_semantic_masks
    ):
        raise PixelVisibilityError("normal_semantic_masks may not be empty")
    frame_count = len(normal_semantic_masks)
    normal = _normalize_context(
        normal_context,
        expected_pass_kind="modal_scene",
        frame_count=frame_count,
    )
    resolution = tuple(normal["resolution_hw"])
    normal_masks = [
        _normalize_mask(mask, resolution_hw=resolution, owner=f"normal[{index}]")
        for index, mask in enumerate(normal_semantic_masks)
    ]

    instance_ids = set(semantic_ids_by_instance)
    if not instance_ids or set(target_only_semantic_masks_by_instance) != instance_ids:
        raise PixelVisibilityError(
            "target-only mask instances must exactly match semantic_ids_by_instance"
        )
    if set(target_only_contexts_by_instance) != instance_ids:
        raise PixelVisibilityError(
            "target-only contexts must exactly match semantic_ids_by_instance"
        )
    semantic_ids = list(semantic_ids_by_instance.values())
    if any(
        isinstance(semantic_id, bool)
        or not isinstance(semantic_id, int)
        or semantic_id <= 0
        for semantic_id in semantic_ids
    ) or len(set(semantic_ids)) != len(semantic_ids):
        raise PixelVisibilityError(
            "target semantic ids must be unique positive integers"
        )

    per_instance: dict[str, Any] = {}
    for instance_id in sorted(instance_ids):
        target_masks_value = target_only_semantic_masks_by_instance[instance_id]
        if (
            isinstance(target_masks_value, (str, bytes))
            or not isinstance(target_masks_value, Sequence)
            or len(target_masks_value) != frame_count
        ):
            raise PixelVisibilityError(
                f"{instance_id}: target-only masks must align with normal frames"
            )
        target_context = _normalize_context(
            target_only_contexts_by_instance[instance_id],
            expected_pass_kind="target_only",
            frame_count=frame_count,
            target_instance_id=instance_id,
        )
        for field in _COMMON_CONTEXT_FIELDS:
            if target_context[field] != normal[field]:
                raise PixelVisibilityError(
                    f"{instance_id}: target-only {field} differs from the normal pass"
                )
        target_masks = [
            _normalize_mask(
                mask,
                resolution_hw=resolution,
                owner=f"{instance_id}.target_only[{index}]",
            )
            for index, mask in enumerate(target_masks_value)
        ]
        semantic_id = semantic_ids_by_instance[instance_id]
        frames = [
            _frame_truth(
                modal_mask=modal_mask,
                target_only_mask=target_mask,
                target_semantic_id=semantic_id,
                frame_index=frame_index,
            )
            for modal_mask, target_mask, frame_index in zip(
                normal_masks,
                target_masks,
                normal["frame_indices"],
            )
        ]
        state_counts = {
            state: sum(frame["state"] == state for frame in frames)
            for state in PIXEL_VISIBILITY_STATES
        }
        per_instance[instance_id] = {
            "semantic_id": semantic_id,
            "frames": frames,
            "state_counts": state_counts,
        }

    return {
        "schema": PIXEL_VISIBILITY_SCHEMA,
        "status": "computed_modal_target_only_v1",
        "authority": PIXEL_VISIBILITY_AUTHORITY,
        "renderer_backend": normal["renderer_backend"],
        "rgb_renderer_backend": normal["rgb_renderer_backend"],
        "camera_contract_id": normal["camera_contract_id"],
        "semantic_id_namespace": normal["semantic_id_namespace"],
        "resolution_hw": normal["resolution_hw"],
        "frame_indices": normal["frame_indices"],
        "camera_pose_ids": normal["camera_pose_ids"],
        "fraction_policy": "visible_pixels_divided_by_in_view_target_only_pixels_v1",
        "out_of_view_fraction_policy": "null_no_in_view_target_denominator",
        "per_instance": per_instance,
    }


def compile_depth_pixel_visibility_truth(
    *,
    normal_depth_m_frames: Sequence[Any],
    target_only_depth_m_frames_by_instance: Mapping[str, Sequence[Any]],
    semantic_ids_by_instance: Mapping[str, int],
    normal_context: Mapping[str, Any],
    target_only_contexts_by_instance: Mapping[str, Mapping[str, Any]],
    target_only_background_depth_m: float,
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compile native visibility from normal and show-only metric depth.

    A target-only depth frame uses a finite far-depth sentinel outside the
    target.  Pixels below that sentinel form the target's in-frustum
    footprint.  A target is modal-visible where the normal-pass depth agrees
    with its target-only depth within the declared absolute-plus-relative
    tolerance.  Every pass must share renderer, RGB backend, camera contract,
    frame indices and exact camera pose hashes.
    """

    if (
        isinstance(normal_depth_m_frames, (str, bytes))
        or not isinstance(normal_depth_m_frames, Sequence)
        or not normal_depth_m_frames
    ):
        raise PixelVisibilityError("normal_depth_m_frames may not be empty")
    frame_count = len(normal_depth_m_frames)
    normal = _normalize_context(
        normal_context,
        expected_pass_kind="modal_scene",
        frame_count=frame_count,
    )
    resolution = tuple(normal["resolution_hw"])
    normal_depths = _normalize_depth_frames(
        normal_depth_m_frames,
        frame_count=frame_count,
        resolution_hw=resolution,
        owner="normal_depth_m_frames",
    )
    background_depth_m = _finite_non_negative(
        target_only_background_depth_m,
        owner="target_only_background_depth_m",
    )
    if background_depth_m <= 0.0:
        raise PixelVisibilityError(
            "target_only_background_depth_m must be positive"
        )
    absolute_tolerance = _finite_non_negative(
        absolute_tolerance_m, owner="absolute_tolerance_m"
    )
    relative_tolerance_value = _finite_non_negative(
        relative_tolerance, owner="relative_tolerance"
    )
    if absolute_tolerance == 0.0 and relative_tolerance_value == 0.0:
        raise PixelVisibilityError("depth tolerance may not be identically zero")

    instance_ids = set(semantic_ids_by_instance)
    if (
        not instance_ids
        or set(target_only_depth_m_frames_by_instance) != instance_ids
    ):
        raise PixelVisibilityError(
            "target-only depth instances must exactly match semantic_ids_by_instance"
        )
    if set(target_only_contexts_by_instance) != instance_ids:
        raise PixelVisibilityError(
            "target-only contexts must exactly match semantic_ids_by_instance"
        )

    target_depths_by_instance: dict[str, list[np.ndarray]] = {}
    target_footprints_by_instance: dict[str, list[np.ndarray]] = {}
    residuals_by_instance: dict[str, list[np.ndarray]] = {}
    visible_candidates_by_instance: dict[str, list[np.ndarray]] = {}
    for instance_id in sorted(instance_ids):
        target_context = _normalize_context(
            target_only_contexts_by_instance[instance_id],
            expected_pass_kind="target_only",
            frame_count=frame_count,
            target_instance_id=instance_id,
        )
        for field in _COMMON_CONTEXT_FIELDS:
            if target_context[field] != normal[field]:
                raise PixelVisibilityError(
                    f"{instance_id}: target-only {field} differs from the normal pass"
                )
        target_depths = _normalize_depth_frames(
            target_only_depth_m_frames_by_instance[instance_id],
            frame_count=frame_count,
            resolution_hw=resolution,
            owner=f"{instance_id}.target_only_depth_m_frames",
        )
        target_depths_by_instance[instance_id] = target_depths
        target_footprints: list[np.ndarray] = []
        residuals: list[np.ndarray] = []
        visible_candidates: list[np.ndarray] = []
        for normal_depth, target_depth in zip(normal_depths, target_depths):
            footprint = target_depth < background_depth_m
            residual = np.abs(normal_depth - target_depth)
            tolerance = (
                absolute_tolerance
                + relative_tolerance_value * target_depth
            )
            visible = footprint & (residual <= tolerance)
            if np.any(visible & ~footprint):
                raise PixelVisibilityError(
                    f"{instance_id}: depth-visible pixels exceed target footprint"
                )
            target_footprints.append(np.ascontiguousarray(footprint))
            residuals.append(np.ascontiguousarray(residual))
            visible_candidates.append(np.ascontiguousarray(visible))
        target_footprints_by_instance[instance_id] = target_footprints
        residuals_by_instance[instance_id] = residuals
        visible_candidates_by_instance[instance_id] = visible_candidates

    normal_semantic_masks: list[np.ndarray] = []
    target_semantic_masks_by_instance: dict[str, list[np.ndarray]] = {
        instance_id: [] for instance_id in instance_ids
    }
    for frame_offset in range(frame_count):
        modal_mask = np.zeros(resolution, dtype=np.int32)
        best_residual = np.full(resolution, np.inf, dtype=np.float32)
        for instance_id in sorted(instance_ids):
            semantic_id = semantic_ids_by_instance[instance_id]
            footprint = target_footprints_by_instance[instance_id][frame_offset]
            target_semantic_masks_by_instance[instance_id].append(
                np.where(footprint, semantic_id, 0).astype(np.int32)
            )
            residual = residuals_by_instance[instance_id][frame_offset]
            visible = visible_candidates_by_instance[instance_id][frame_offset]
            wins = visible & (residual < best_residual)
            modal_mask[wins] = semantic_id
            best_residual[wins] = residual[wins]
        normal_semantic_masks.append(modal_mask)

    truth = compile_pixel_visibility_truth(
        normal_semantic_masks=normal_semantic_masks,
        target_only_semantic_masks_by_instance=target_semantic_masks_by_instance,
        semantic_ids_by_instance=semantic_ids_by_instance,
        normal_context=normal_context,
        target_only_contexts_by_instance=target_only_contexts_by_instance,
    )
    truth["authority"] = PIXEL_VISIBILITY_DEPTH_AUTHORITY
    truth["depth_comparison"] = {
        "units": "meters",
        "normal_pass": "full_scene_metric_depth",
        "target_only_pass": "show_only_target_metric_depth",
        "target_footprint_policy": "target_depth_below_background_sentinel_v1",
        "visibility_policy": (
            "absolute_normal_minus_target_depth_lte_absolute_plus_relative_v1"
        ),
        "target_only_background_depth_m": background_depth_m,
        "absolute_tolerance_m": absolute_tolerance,
        "relative_tolerance": relative_tolerance_value,
        "overlap_resolution": "minimum_depth_residual_then_instance_id_v1",
    }
    return truth


def bind_pixel_visibility_truth(
    value: Any,
    *,
    expected_instance_ids: Sequence[str],
    expected_frame_count: int,
    expected_resolution_hw: Sequence[int],
    expected_camera_pose_ids: Sequence[str] | None,
) -> dict[str, Any]:
    """Validate the cross-artifact binding needed by a QA fact table."""

    if not isinstance(value, Mapping) or value.get("schema") != PIXEL_VISIBILITY_SCHEMA:
        raise PixelVisibilityError("pixel truth has an unexpected schema")
    if value.get("status") != "computed_modal_target_only_v1":
        raise PixelVisibilityError("pixel truth is not computed")
    authority = value.get("authority")
    if authority not in PIXEL_VISIBILITY_AUTHORITIES:
        raise PixelVisibilityError("pixel truth has an unexpected authority")
    if authority == PIXEL_VISIBILITY_DEPTH_AUTHORITY:
        depth_comparison = value.get("depth_comparison")
        if not isinstance(depth_comparison, Mapping):
            raise PixelVisibilityError(
                "metric-depth pixel truth lacks its comparison contract"
            )
        if (
            depth_comparison.get("units") != "meters"
            or depth_comparison.get("normal_pass")
            != "full_scene_metric_depth"
            or depth_comparison.get("target_only_pass")
            != "show_only_target_metric_depth"
            or depth_comparison.get("target_footprint_policy")
            != "target_depth_below_background_sentinel_v1"
            or depth_comparison.get("visibility_policy")
            != (
                "absolute_normal_minus_target_depth_lte_absolute_plus_relative_v1"
            )
            or depth_comparison.get("overlap_resolution")
            != "minimum_depth_residual_then_instance_id_v1"
        ):
            raise PixelVisibilityError(
                "metric-depth pixel truth has an unsupported comparison policy"
            )
        background_depth = _finite_non_negative(
            depth_comparison.get("target_only_background_depth_m"),
            owner="target_only_background_depth_m",
        )
        absolute_tolerance = _finite_non_negative(
            depth_comparison.get("absolute_tolerance_m"),
            owner="absolute_tolerance_m",
        )
        relative_tolerance = _finite_non_negative(
            depth_comparison.get("relative_tolerance"),
            owner="relative_tolerance",
        )
        if background_depth <= 0.0 or (
            absolute_tolerance == 0.0 and relative_tolerance == 0.0
        ):
            raise PixelVisibilityError(
                "metric-depth pixel truth has invalid sentinel or tolerance"
            )
    if (
        value.get("fraction_policy")
        != "visible_pixels_divided_by_in_view_target_only_pixels_v1"
        or value.get("out_of_view_fraction_policy")
        != "null_no_in_view_target_denominator"
    ):
        raise PixelVisibilityError("pixel truth fraction policy is unsupported")
    if value.get("renderer_backend") != value.get("rgb_renderer_backend"):
        raise PixelVisibilityError(
            "pixel truth semantic and RGB renderer backends differ"
        )
    if value.get("resolution_hw") != list(expected_resolution_hw):
        raise PixelVisibilityError("pixel truth resolution differs from the camera")
    frame_indices = value.get("frame_indices")
    camera_pose_ids = value.get("camera_pose_ids")
    if (
        not isinstance(frame_indices, Sequence)
        or len(frame_indices) != expected_frame_count
        or list(frame_indices) != list(range(expected_frame_count))
        or not isinstance(camera_pose_ids, Sequence)
        or len(camera_pose_ids) != expected_frame_count
    ):
        raise PixelVisibilityError("pixel truth frame tracks have the wrong length")
    if expected_camera_pose_ids is not None and list(camera_pose_ids) != list(
        expected_camera_pose_ids
    ):
        raise PixelVisibilityError(
            "pixel truth camera poses differ from SensorRigTrajectory"
        )
    per_instance = value.get("per_instance")
    if not isinstance(per_instance, Mapping) or set(per_instance) != set(
        expected_instance_ids
    ):
        raise PixelVisibilityError(
            "pixel truth instances differ from the episode source slots"
        )
    semantic_ids: list[int] = []
    for instance_id, entry in per_instance.items():
        frames = entry.get("frames") if isinstance(entry, Mapping) else None
        semantic_id = entry.get("semantic_id") if isinstance(entry, Mapping) else None
        if (
            isinstance(semantic_id, bool)
            or not isinstance(semantic_id, int)
            or semantic_id <= 0
        ):
            raise PixelVisibilityError(
                f"{instance_id}: pixel truth semantic id is invalid"
            )
        semantic_ids.append(semantic_id)
        if not isinstance(frames, Sequence) or len(frames) != expected_frame_count:
            raise PixelVisibilityError(
                f"{instance_id}: pixel truth has the wrong frame count"
            )
        if [frame.get("frame_index") for frame in frames] != list(frame_indices):
            raise PixelVisibilityError(
                f"{instance_id}: pixel truth frame indices are inconsistent"
            )
        observed_states: list[str] = []
        for frame in frames:
            visible = frame.get("visible_pixels")
            target = frame.get("target_pixels")
            visible_fraction = frame.get("visible_fraction")
            occlusion_fraction = frame.get("occlusion_fraction")
            if (
                isinstance(visible, bool)
                or not isinstance(visible, int)
                or visible < 0
                or isinstance(target, bool)
                or not isinstance(target, int)
                or target < visible
            ):
                raise PixelVisibilityError(
                    f"{instance_id}: pixel counts are inconsistent"
                )
            if target == 0:
                expected_state = "out_of_view"
                fractions_valid = (
                    visible == 0
                    and visible_fraction is None
                    and occlusion_fraction is None
                )
            else:
                expected_visible_fraction = visible / target
                expected_occlusion_fraction = 1.0 - expected_visible_fraction
                expected_state = (
                    "visible_clear"
                    if visible == target
                    else "fully_occluded"
                    if visible == 0
                    else "visible_occluded"
                )
                fractions_valid = (
                    isinstance(visible_fraction, (int, float))
                    and not isinstance(visible_fraction, bool)
                    and isinstance(occlusion_fraction, (int, float))
                    and not isinstance(occlusion_fraction, bool)
                    and math.isclose(
                        float(visible_fraction),
                        expected_visible_fraction,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    and math.isclose(
                        float(occlusion_fraction),
                        expected_occlusion_fraction,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
            if not fractions_valid or frame.get("state") != expected_state:
                raise PixelVisibilityError(
                    f"{instance_id}: pixel fractions or state disagree with counts"
                )
            observed_states.append(expected_state)
        expected_state_counts = {
            state: observed_states.count(state) for state in PIXEL_VISIBILITY_STATES
        }
        if entry.get("state_counts") != expected_state_counts:
            raise PixelVisibilityError(
                f"{instance_id}: state_counts disagree with frame states"
            )
    if len(set(semantic_ids)) != len(semantic_ids):
        raise PixelVisibilityError("pixel truth semantic ids must be unique")
    return deepcopy(dict(value))


__all__ = [
    "PIXEL_VISIBILITY_AUTHORITY",
    "PIXEL_VISIBILITY_AUTHORITIES",
    "PIXEL_VISIBILITY_DEPTH_AUTHORITY",
    "PIXEL_VISIBILITY_SCHEMA",
    "PIXEL_VISIBILITY_STATES",
    "PixelVisibilityError",
    "bind_pixel_visibility_truth",
    "compile_depth_pixel_visibility_truth",
    "compile_pixel_visibility_truth",
]
