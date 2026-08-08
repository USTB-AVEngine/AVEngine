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
    if value.get("authority") != PIXEL_VISIBILITY_AUTHORITY:
        raise PixelVisibilityError("pixel truth has an unexpected authority")
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
    "PIXEL_VISIBILITY_SCHEMA",
    "PIXEL_VISIBILITY_STATES",
    "PixelVisibilityError",
    "bind_pixel_visibility_truth",
    "compile_pixel_visibility_truth",
]
