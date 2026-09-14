"""Question evidence derived from retained native masks, never from plan intent."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import numpy as np

IN_FOV_DEFINITION = (
    "target_pixels > 0 from the native target-only footprint, including fully_occluded; "
    "this is not visible_pixels > 0 and does not mean the instance is unoccluded or fully inside the frame"
)
REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON = (
    "registered_appearance_value_classifier_not_implemented"
)
PLACEHOLDER_NONHUMAN_MINIMUM_COLOR_PIXELS = 512
PLACEHOLDER_NONHUMAN_DOMINANCE_RATIO = 1.25
PLACEHOLDER_NONHUMAN_COLOR_COMPONENT_FRACTIONS = {
    "tricolor_white": 0.08,
    "tricolor_dark": 0.12,
    "tricolor_warm_brown": 0.08,
    "black_white_white": 0.12,
    "black_white_dark": 0.25,
    "red_white_white": 0.12,
    "red_white_warm_brown": 0.12,
    "white_tan_white": 0.12,
    "white_tan_warm_brown": 0.12,
    "yellow_coat": 0.12,
    "blue_gray_coat": 0.30,
    "dark": 0.55,
    "warm_brown": 0.12,
}


def _mask_frame_row(
    frame_index: int,
    frame_indices: Sequence[int],
    row_count: int,
) -> int:
    """Map a contract frame identity to a dense mask row without filling gaps."""
    declared = tuple(int(value) for value in frame_indices)
    if row_count == len(declared):
        try:
            return declared.index(int(frame_index))
        except ValueError as error:
            raise ValueError(f"mask has no declared frame {frame_index}") from error
    if 0 <= int(frame_index) < row_count:
        return int(frame_index)
    raise ValueError(f"mask has no frame row for explicit frame {frame_index}")


def _modal_array(data: Any) -> np.ndarray:
    """Read the long and short modal-mask aliases as one contract field."""
    if "depth_derived_modal_semantic" in data:
        modal = np.asarray(data["depth_derived_modal_semantic"])
        if "modal" in data and not np.array_equal(np.asarray(data["modal"]), modal):
            raise ValueError("modal semantic aliases disagree")
    elif "modal" in data:
        modal = np.asarray(data["modal"])
    else:
        raise ValueError("native masks lack modal semantic IDs")
    if modal.ndim != 3 or not np.issubdtype(modal.dtype, np.integer):
        raise ValueError("native modal masks must be integer [frame,height,width]")
    return modal


def derive_actor_occluders(
    masks_path: Path, pixel_truth: Mapping[str, Any], *,
    minimum_covered_pixels: int = 100, minimum_explained_fraction: float = 0.9,
) -> dict[str, Any]:
    """Identify only occlusions explained by another captured source instance.

    The function consumes the renderer-neutral pixel truth and mask contract.
    Sparse truth frames remain sparse; a missing frame is never synthesized from
    an adjacent mask row. Unidentified static geometry remains unresolved.
    """
    if pixel_truth.get("status") not in {"pass", "computed_modal_target_only_v1"}:
        raise ValueError("actor occluders require completed native pixel truth")
    frame_indices = pixel_truth.get("frame_indices")
    instances = pixel_truth.get("per_instance", {})
    if not isinstance(instances, Mapping) or not instances:
        raise ValueError("actor occluders require per_instance records")
    if not isinstance(frame_indices, list) or not frame_indices:
        # Older chain-one fixtures omitted the top-level list but carried an
        # explicit frame_index in every per-instance record. Reuse those
        # identities; never synthesize a dense range for a sparse probe.
        observed = {
            frame.get("frame_index")
            for record in instances.values()
            if isinstance(record, Mapping) and isinstance(record.get("frames"), list)
            for frame in record["frames"]
            if isinstance(frame, Mapping) and isinstance(frame.get("frame_index"), int)
        }
        if not observed:
            raise ValueError("actor occluders require explicit pixel truth frame indices")
        frame_indices = sorted(observed)
    semantic: dict[str, int] = {}
    for key, record in instances.items():
        if not isinstance(key, str) or not isinstance(record, Mapping):
            raise ValueError("pixel truth instance records are malformed")
        value = record.get("semantic_id")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"pixel truth semantic ID is invalid for {key}")
        semantic[key] = int(value)
    by_id = {value: key for key, value in semantic.items()}
    if len(semantic) != len(by_id):
        raise ValueError("native semantic IDs must be distinct")
    records = []
    rejected = Counter()
    with np.load(Path(masks_path), allow_pickle=False) as data:
        modal = _modal_array(data)
        for target, own_id in semantic.items():
            name = f"target_only_{target}"
            if name not in data:
                raise ValueError(f"missing or mismatched target-only masks: {target}")
            # NPZ indexing decompresses the array on each access.
            target_masks = np.asarray(data[name])
            if target_masks.shape != modal.shape:
                raise ValueError(f"missing or mismatched target-only masks: {target}")
            target_frames = instances[target].get("frames")
            if not isinstance(target_frames, list):
                raise ValueError(f"pixel truth has no frames for {target}")
            for frame in target_frames:
                if not isinstance(frame, Mapping):
                    raise ValueError(f"pixel truth frame record is malformed for {target}")
                f = frame.get("frame_index")
                if isinstance(f, bool) or not isinstance(f, int) or f not in frame_indices:
                    raise ValueError(f"pixel truth frame identity is unavailable for {target}")
                if frame.get("state") not in {"visible_occluded", "fully_occluded"}:
                    continue
                row = _mask_frame_row(f, frame_indices, modal.shape[0])
                occluded = (target_masks[row] > 0) & (modal[row] != own_id)
                total = int(occluded.sum())
                if total < minimum_covered_pixels:
                    rejected["too_few_occluded_pixels"] += 1
                    continue
                ids, counts = np.unique(modal[row][occluded], return_counts=True)
                known = [
                    (int(count), by_id[int(value)])
                    for value, count in zip(ids, counts)
                    if int(value) in by_id and int(value) != own_id
                ]
                if not known:
                    rejected["no_identified_foreground_actor"] += 1
                    continue
                known.sort(key=lambda value: (-value[0], value[1]))
                count, foreground = known[0]
                if count < minimum_covered_pixels or count / total < minimum_explained_fraction:
                    rejected["occlusion_not_explained_by_one_actor"] += 1
                    continue
                records.append({
                    "frame_index": f,
                    "target_instance_id": target,
                    "occluder_instance_ids": [foreground],
                    "occluded_target_pixels": total,
                    "foreground_actor_pixels": count,
                    "explained_fraction": count / total,
                })
    return {
        "status": "pass",
        "authority": "intersection_of_native_depth_modal_and_target_only_masks",
        "masks_path": str(Path(masks_path).resolve()),
        "frame_records": records,
        "unresolved_counts": dict(rejected),
        "minimum_covered_pixels": minimum_covered_pixels,
        "minimum_explained_fraction": minimum_explained_fraction,
        "claim_boundary": "identified source-instance occluders only; static-object identities are not inferred",
    }


def upper_body_window(
    mask_shape: Sequence[int], target_bbox: Sequence[int],
) -> tuple[int, int, int, int] | None:
    """The crop a clothed torso occupies inside a target-only bounding box.

    The garment a ``top_color`` registers sits on the upper body, so a review of
    it inspects the middle of the box rather than the whole silhouette, which
    also carries hair, skin, trousers and shoes. The fractions are the ones this
    review has always used; they are geometric, not per-asset.
    """
    x0, y0, x1, y1 = (int(value) for value in target_bbox)
    if x1 <= x0 or y1 <= y0:
        return None
    height, width = int(mask_shape[0]), int(mask_shape[1])
    xa = max(0, int(x0 + 0.1 * (x1 - x0)))
    xb = min(width, int(x1 - 0.1 * (x1 - x0)))
    ya = max(0, int(y0 + 0.18 * (y1 - y0)))
    yb = min(height, int(y0 + 0.58 * (y1 - y0)))
    if xb <= xa or yb <= ya:
        return None
    return xa, ya, xb, yb


def inspect_coarse_top_color(
    rgb: np.ndarray, visible_mask: np.ndarray, target_bbox: list[int], *,
    minimum_color_pixels: int = 512,
) -> dict[str, Any]:
    """Name the dominant colour family of a visible upper-body crop.

    This is the bounded diagnostic behind a registered shirt colour: it reports
    which family the torso pixels actually fall into, measured against the
    scene's own neutral reference so that a warm room does not rename a white
    shirt. It makes no accessory or clothing-pattern claim, and a crop with too
    few pixels or no dominant family stays unverified.
    """
    from avengine.rooms.appearance_color import (
        colour_family_counts,
        estimate_scene_neutral,
        white_balanced,
    )

    image = np.asarray(rgb)
    mask = np.asarray(visible_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("RGB and native instance mask dimensions differ")
    window = upper_body_window(mask.shape, target_bbox)
    if window is None:
        return {"status": "not_observable", "reason": "empty_target_footprint"}
    xa, ya, xb, yb = window
    torso = np.zeros(mask.shape, dtype=bool)
    torso[ya:yb, xa:xb] = True
    selected = torso & mask
    neutral = estimate_scene_neutral(image, exclude_mask=mask)
    pixels = white_balanced(image[selected], neutral)
    counts, _medians = colour_family_counts(pixels, neutral["reference_lightness"])
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    winner, count = ranked[0] if ranked else ("", 0)
    second = ranked[1][1] if len(ranked) > 1 else 0
    confident = count >= minimum_color_pixels and count >= 1.6 * max(1, second)
    return {
        "status": "pass" if confident else "not_observable",
        "observed_color": winner if confident else None,
        "color_pixels": count,
        "upper_body_visible_pixels": int(selected.sum()),
        "candidate_color_counts": counts,
        "scene_neutral": neutral,
        "crop_xyxy": [xa, ya, xb, yb],
        "minimum_color_pixels": minimum_color_pixels,
        "claim_boundary": "coarse RGB color diagnostic, not formal attribute certification",
    }


def bbox_touches_frame_edge(
    bbox: Sequence[int] | None,
    resolution_hw: Sequence[int],
) -> bool:
    """Return whether a target-only xyxy bbox (exclusive max) touches an image edge."""
    if bbox is None or len(bbox) != 4:
        return False
    if len(resolution_hw) != 2:
        raise ValueError("resolution_hw must be [height, width]")
    height, width = int(resolution_hw[0]), int(resolution_hw[1])
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if height <= 0 or width <= 0 or x1 <= x0 or y1 <= y0:
        return False
    return x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height


def summarize_pixel_visibility_semantics(
    frames: Sequence[Mapping[str, Any]],
    *,
    resolution_hw: Sequence[int],
    window_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Count in-FOV, visible, and edge-truncated frames from native pixel truth."""
    if window_frames is not None:
        if len(window_frames) != 2:
            raise ValueError("window_frames must be [start, end)")
        start, end = int(window_frames[0]), int(window_frames[1])
        selected = [
            frame for frame in frames
            if isinstance(frame, Mapping)
            and isinstance(frame.get("frame_index"), int)
            and not isinstance(frame.get("frame_index"), bool)
            and start <= int(frame["frame_index"]) < end
        ]
    else:
        selected = [frame for frame in frames if isinstance(frame, Mapping)]
    in_fov = 0
    visible = 0
    edge = 0
    missing = 0
    for frame in selected:
        target = frame.get("target_pixels")
        vis = frame.get("visible_pixels")
        if isinstance(target, bool) or not isinstance(target, int):
            missing += 1
            continue
        if target > 0:
            in_fov += 1
        if (not isinstance(vis, bool)) and isinstance(vis, int) and vis > 0:
            visible += 1
        if bbox_touches_frame_edge(frame.get("target_bbox_xyxy_px"), resolution_hw):
            edge += 1
    return {
        "in_fov_frame_count": in_fov,
        "in_fov_definition": IN_FOV_DEFINITION,
        "visible_pixel_frames": visible,
        "bbox_touches_frame_edge_frames": edge,
        "missing_pixel_frames": missing,
        "frame_count": len(selected),
        "window_frames": [int(window_frames[0]), int(window_frames[1])] if window_frames is not None else None,
        "resolution_hw": [int(resolution_hw[0]), int(resolution_hw[1])],
        "calibration": "placeholder; native pixel-truth tallies, not human answerability",
    }


def annotate_pixel_visibility_semantics(truth: Mapping[str, Any]) -> dict[str, Any]:
    """Copy pixel truth and add in_fov / visible / edge-touch fields without redefining state."""
    annotated = deepcopy(dict(truth))
    resolution = annotated.get("resolution_hw")
    if not isinstance(resolution, Sequence) or isinstance(resolution, (str, bytes)) or len(resolution) != 2:
        raise ValueError("pixel truth requires resolution_hw [height, width]")
    resolution_hw = [int(resolution[0]), int(resolution[1])]
    per_instance = annotated.get("per_instance")
    if not isinstance(per_instance, Mapping):
        raise ValueError("pixel truth per_instance is required")
    instances: dict[str, Any] = {}
    for instance_id, instance in per_instance.items():
        if not isinstance(instance, Mapping):
            instances[str(instance_id)] = instance
            continue
        record = deepcopy(dict(instance))
        frames = record.get("frames")
        frame_rows: list[Any] = []
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, Mapping):
                    frame_rows.append(frame)
                    continue
                row = deepcopy(dict(frame))
                target = row.get("target_pixels")
                row["in_fov"] = (
                    (not isinstance(target, bool))
                    and isinstance(target, int)
                    and target > 0
                )
                row["bbox_touches_frame_edge"] = bbox_touches_frame_edge(
                    row.get("target_bbox_xyxy_px"), resolution_hw
                )
                frame_rows.append(row)
        record["frames"] = frame_rows
        summary = summarize_pixel_visibility_semantics(frame_rows, resolution_hw=resolution_hw)
        record["in_fov_frame_count"] = summary["in_fov_frame_count"]
        record["visible_pixel_frames"] = summary["visible_pixel_frames"]
        record["bbox_touches_frame_edge_frames"] = summary["bbox_touches_frame_edge_frames"]
        record["in_fov_definition"] = IN_FOV_DEFINITION
        instances[str(instance_id)] = record
    annotated["per_instance"] = instances
    annotated["resolution_hw"] = resolution_hw
    annotated["in_fov_definition"] = IN_FOV_DEFINITION
    annotated["visibility_semantics_authority"] = "qa_evidence.annotate_pixel_visibility_semantics"
    return annotated


def annotate_achieved_conditions_visibility(
    achieved: Mapping[str, Any],
    pixel_truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy achieved_conditions and add visible/edge counts next to existing in_fov_frame_count."""
    result = deepcopy(dict(achieved))
    resolution = pixel_truth.get("resolution_hw") or [0, 0]
    instances = pixel_truth.get("per_instance") if isinstance(pixel_truth.get("per_instance"), Mapping) else {}
    measurements = result.get("anchor_event_measurements")
    if not isinstance(measurements, list):
        result["in_fov_definition"] = IN_FOV_DEFINITION
        return result
    updated: list[Any] = []
    for row in measurements:
        if not isinstance(row, Mapping):
            updated.append(row)
            continue
        item = deepcopy(dict(row))
        actor_id = str(item.get("actor_id"))
        instance = instances.get(actor_id)
        frames = instance.get("frames") if isinstance(instance, Mapping) else []
        window = item.get("window_frames")
        summary = summarize_pixel_visibility_semantics(
            frames if isinstance(frames, list) else [],
            resolution_hw=resolution if isinstance(resolution, Sequence) else [0, 0],
            window_frames=window if isinstance(window, list) and len(window) == 2 else None,
        )
        item["visible_pixel_frames"] = summary["visible_pixel_frames"]
        item["bbox_touches_frame_edge_frames"] = summary["bbox_touches_frame_edge_frames"]
        item["in_fov_definition"] = IN_FOV_DEFINITION
        updated.append(item)
    result["anchor_event_measurements"] = updated
    result["in_fov_definition"] = IN_FOV_DEFINITION
    return result


def nonhuman_appearance_placeholder_thresholds(
    *,
    minimum_color_pixels: int = PLACEHOLDER_NONHUMAN_MINIMUM_COLOR_PIXELS,
    dominance_ratio: float = PLACEHOLDER_NONHUMAN_DOMINANCE_RATIO,
    color_component_fractions: Mapping[str, float] | None = None,
    value_minimum_shares: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Named calibration for coarse colour evidence, shared by every entity kind.

    ``color_component_fractions`` is kept for the older multi-tone coat records
    and stays in the reuse key; the component shares a registered value actually
    has to meet now live with its colour-family predicate. ``value_minimum_shares``
    overrides that per-value share for a caller that wants a stricter reading.
    """
    fractions = dict(PLACEHOLDER_NONHUMAN_COLOR_COMPONENT_FRACTIONS)
    if color_component_fractions:
        fractions.update({str(key): float(value) for key, value in color_component_fractions.items()})
    from avengine.rooms.appearance_color import COLOUR_MODEL

    return {
        "label": "placeholder",
        "calibration": "placeholder_nonhuman_appearance_v1",
        "color_model": COLOUR_MODEL,
        "minimum_color_pixels": int(minimum_color_pixels),
        "dominance_ratio": float(dominance_ratio),
        "color_component_fractions": fractions,
        "value_minimum_shares": dict(value_minimum_shares or {}),
        "one_rule_for_every_entity_kind": {
            "minimum_color_pixels": int(minimum_color_pixels),
            "dominance_ratio": float(dominance_ratio),
            "function": "inspect_registered_appearance",
            "note": (
                "humans, animals and devices now run the same illumination-relative "
                "decision; the separate absolute-HSV human palette is gone"
            ),
        },
        "claim_boundary": (
            "coarse colour-family evidence measured against the scene's own neutral "
            "reference at human order of magnitude (512 supporting pixels); it names "
            "a family, not a fine-grained appearance, and is not formal certification"
        ),
    }


def _appearance_spec(
    record: Mapping[str, Any],
    *,
    asset_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the registered appearance field without parsing asset labels."""
    owners: list[tuple[str, Mapping[str, Any]]] = []
    for owner_key in ("registered_appearance", "realized_attributes", "appearance", "attributes"):
        value = record.get(owner_key)
        if isinstance(value, Mapping):
            owners.append((owner_key, value))
    asset_id = record.get("asset_id") or record.get("entity_asset_id")
    if isinstance(asset_registry, Mapping) and isinstance(asset_id, str):
        registered = asset_registry.get(asset_id)
        if isinstance(registered, Mapping):
            attrs = registered.get("realized_attributes")
            if isinstance(attrs, Mapping):
                owners.insert(0, ("source_asset_registry.realized_attributes", attrs))
    entity_class = str(record.get("entity_class") or record.get("actor_class") or "").casefold()
    species = record.get("species_id")
    if not entity_class and isinstance(species, str):
        entity_class = species.casefold()
    if "rigid" in entity_class or "device" in entity_class or "speaker" in entity_class or "object" in entity_class:
        wanted = (("finish", "finish"), ("surface_finish", "surface_finish"), ("body_color", "body_color"))
        kind = "device"
    elif "animal" in entity_class or any(token in entity_class for token in ("dog", "cat", "beagle", "quadruped")) or (isinstance(species, str) and species.casefold() not in {"human", "person"}):
        wanted = (("coat_profile", "coat_profile.value"), ("coat_value", "coat_value"), ("color", "color"))
        kind = "animal"
    else:
        wanted = (("top_color", "top_color"), ("shirt_color", "shirt_color"), ("color", "color"))
        kind = "human"
    found_fields: list[dict[str, str]] = []
    chosen: dict[str, str] | None = None
    for source, owner in owners:
        for key, field in wanted:
            value = owner.get(key)
            if key == "coat_profile" and isinstance(value, Mapping):
                value = value.get("value")
            if isinstance(value, str) and value.strip():
                item = {"field": field, "value": value.strip(), "source": source}
                found_fields.append(item)
                if chosen is None:
                    chosen = item
    missing_reason = (
        "neither finish nor body_color is registered"
        if kind == "device"
        else "registered appearance field is unavailable"
    )
    return {
        "field": chosen["field"] if chosen else None,
        "value": chosen["value"] if chosen else None,
        "kind": kind,
        "source": chosen["source"] if chosen else None,
        "appearance_field_used": chosen["field"] if chosen else None,
        "searched_fields": [field for _, field in wanted],
        "available_registered_fields": found_fields,
        "missing_reason": None if chosen else missing_reason,
    }


def _rgb_frame(
    capture_root: Path,
    frame_index: int,
    frame_indices: Sequence[int],
    rgb_array: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    path = capture_root / "frames" / f"frame_{int(frame_index):04d}.png"
    if path.is_file():
        import cv2
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"native RGB frame unavailable: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), str(path.resolve())
    if rgb_array is None:
        raise ValueError(f"native RGB frame unavailable for explicit frame {frame_index}")
    if rgb_array.ndim != 4 or rgb_array.shape[-1] != 3:
        raise ValueError("rgb.npy must have shape [frame,height,width,3]")
    row = _mask_frame_row(frame_index, frame_indices, rgb_array.shape[0])
    return np.asarray(rgb_array[row]), str((capture_root / "rgb.npy").resolve())


def inspect_registered_appearance(
    rgb: np.ndarray,
    visible_mask: np.ndarray,
    expected_value: str,
    *,
    entity_kind: str,
    minimum_color_pixels: int = PLACEHOLDER_NONHUMAN_MINIMUM_COLOR_PIXELS,
    dominance_ratio: float = PLACEHOLDER_NONHUMAN_DOMINANCE_RATIO,
    color_component_fractions: Mapping[str, float] | None = None,
    value_minimum_shares: Mapping[str, float] | None = None,
    target_bbox: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare a registered appearance value against actual masked RGB pixels.

    One rule serves every entity kind. The frame's own neutral reference is
    measured first, from the least chromatic pixels that do not belong to the
    inspected instance; the target's pixels are white balanced against it and
    sorted into colour families whose lightness bands are expressed relative to
    that same reference. The registered value is accepted only when the families
    it names carry a real share of the target and outweigh the families it could
    be confused with. Nothing here is keyed to a room, a map or an asset, so a
    warm interior and a baked scan are judged by the same numbers.
    """
    from avengine.rooms.appearance_color import (
        colour_family_counts,
        estimate_scene_neutral,
        evaluate_registered_value,
        white_balanced,
    )

    image = np.asarray(rgb)
    mask = np.asarray(visible_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("RGB and native instance mask dimensions differ")
    thresholds = nonhuman_appearance_placeholder_thresholds(
        minimum_color_pixels=minimum_color_pixels,
        dominance_ratio=dominance_ratio,
        color_component_fractions=color_component_fractions,
        value_minimum_shares=value_minimum_shares,
    )
    expected = str(expected_value).strip().casefold()
    kind = str(entity_kind).strip().casefold()
    inspected = mask
    crop: list[int] | None = None
    geometry = "whole_visible_instance_mask"
    if kind == "human" and target_bbox is not None and len(target_bbox) == 4:
        window = upper_body_window(mask.shape, target_bbox)
        if window is None:
            return _appearance_not_observable(
                expected_value, thresholds, reason="empty_target_footprint",
                geometry="upper_body_crop_of_the_target_bounding_box",
            )
        xa, ya, xb, yb = window
        torso = np.zeros(mask.shape, dtype=bool)
        torso[ya:yb, xa:xb] = True
        inspected = mask & torso
        crop = [xa, ya, xb, yb]
        geometry = "upper_body_crop_of_the_target_bounding_box"
    inspected_pixels = int(inspected.sum())
    if inspected_pixels == 0:
        return _appearance_not_observable(
            expected_value, thresholds,
            reason="no visible native RGB pixels in the target mask",
            geometry=geometry, crop=crop,
        )
    neutral = estimate_scene_neutral(image, exclude_mask=mask)
    balanced = white_balanced(image[inspected], neutral)
    counts, medians = colour_family_counts(balanced, neutral["reference_lightness"])
    share_override = thresholds["value_minimum_shares"].get(expected)
    decision = evaluate_registered_value(
        counts, medians, expected_value, kind,
        minimum_support_pixels=int(thresholds["minimum_color_pixels"]),
        dominance_ratio=float(thresholds["dominance_ratio"]),
        minimum_share=float(share_override) if share_override is not None else None,
    )
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    dominant = ranked[0][0] if ranked and ranked[0][1] > 0 else None
    # A family only names what was seen once it clears the same support floor the
    # registered value has to clear; below that the target was simply not read.
    named = dominant if ranked and ranked[0][1] >= int(thresholds["minimum_color_pixels"]) else None
    accepted = bool(decision["accepted"])
    record: dict[str, Any] = {
        "status": "pass" if accepted else "not_observable",
        # On a refusal this reports the family the pixels actually showed, which
        # is what the older palette diagnostic reported and what a reviewer needs
        # in order to see whether the registration or the render is at fault.
        "observed_value": expected if accepted else named,
        "observed_family": dominant,
        "visible_pixels": inspected_pixels,
        "candidate_counts": counts,
        "component_fractions": {
            name: round(count / max(1, inspected_pixels), 4) for name, count in counts.items()
        },
        "family_relative_lightness": dict(medians),
        "expected_value": expected_value,
        "entity_kind": kind,
        "inspected_geometry": geometry,
        "minimum_color_pixels": int(thresholds["minimum_color_pixels"]),
        "dominance_ratio": float(thresholds["dominance_ratio"]),
        "scene_neutral": neutral,
        "decision": decision,
        "appearance_thresholds": thresholds,
        "placeholder": True,
        "calibration": "placeholder_coarse_color_only",
        "color_model": thresholds["color_model"],
        "claim_boundary": thresholds["claim_boundary"],
    }
    if crop is not None:
        record["crop_xyxy"] = crop
    if decision["unsupported"]:
        record["reason"] = REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON
        record["gap_category"] = "interface_not_implemented"
    elif not accepted:
        record["reason"] = "; ".join(decision["rejections"])
    return record


def _appearance_not_observable(
    expected_value: str,
    thresholds: Mapping[str, Any],
    *,
    reason: str,
    geometry: str,
    crop: list[int] | None = None,
) -> dict[str, Any]:
    """A geometric refusal: the target has no pixels this review could read."""
    record: dict[str, Any] = {
        "status": "not_observable",
        "observed_value": None,
        "observed_family": None,
        "visible_pixels": 0,
        "candidate_counts": {},
        "component_fractions": {},
        "family_relative_lightness": {},
        "expected_value": expected_value,
        "inspected_geometry": geometry,
        "minimum_color_pixels": int(thresholds["minimum_color_pixels"]),
        "appearance_thresholds": dict(thresholds),
        "placeholder": True,
        "calibration": "placeholder_coarse_color_only",
        "color_model": thresholds["color_model"],
        "reason": reason,
        "gap_category": "target_geometry",
        "claim_boundary": thresholds["claim_boundary"],
    }
    if crop is not None:
        record["crop_xyxy"] = crop
    return record


def classifier_gap_fields(checks: Sequence[Any]) -> dict[str, Any]:
    """Actor-level reason when a registered appearance value has no classifier."""
    for check in checks:
        if (
            isinstance(check, Mapping)
            and check.get("reason") == REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON
        ):
            return {
                "reason": REGISTERED_APPEARANCE_CLASSIFIER_GAP_REASON,
                "gap_category": "interface_not_implemented",
            }
    return {}


def appearance_review_frame_selection(
    truth: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    frame_stride: int = 15,
) -> list[int]:
    """Resolve which explicit truth frames an appearance review inspects.

    The stride walk, the retained last frame and the audio-event neighbourhood
    stay exactly as the review has always selected them. Exposing the list lets
    a caller record the frames actually reviewed instead of re-deriving the
    rule, and lets shared visual evidence compare two selections directly.
    """
    frame_indices = truth.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError("pixel visibility truth requires explicit frame indices")
    selected_frames = list(frame_indices[::max(1, int(frame_stride))])
    selected_frames.append(int(frame_indices[-1]))
    selected_frames = sorted(set(int(value) for value in selected_frames))
    clock = plan.get("clock", {})
    fps = float(clock.get("frame_rate_hz", 15.0)) if isinstance(clock, Mapping) else 15.0
    sr = int(clock.get("sample_rate_hz", 16000)) if isinstance(clock, Mapping) else 16000
    for event in plan.get("audio_events", []) if isinstance(plan.get("audio_events"), list) else []:
        start = event.get("start_sample") if isinstance(event, Mapping) else None
        if isinstance(start, (int, float)):
            frame = int(float(start) * fps / sr)
            selected_frames.extend(i for i in (frame, frame + 1) if i in frame_indices)
    return sorted(set(selected_frames))


def build_pixel_appearance_review(
    capture_root: Path,
    plan: Mapping[str, Any], *,
    frame_stride: int = 15,
    asset_registry: Mapping[str, Mapping[str, Any]] | None = None,
    minimum_color_pixels: int = PLACEHOLDER_NONHUMAN_MINIMUM_COLOR_PIXELS,
    dominance_ratio: float = PLACEHOLDER_NONHUMAN_DOMINANCE_RATIO,
    color_component_fractions: Mapping[str, float] | None = None,
    value_minimum_shares: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build renderer-neutral appearance evidence from masked native RGB."""
    capture_root = Path(capture_root).resolve()
    thresholds = nonhuman_appearance_placeholder_thresholds(
        minimum_color_pixels=minimum_color_pixels,
        dominance_ratio=dominance_ratio,
        color_component_fractions=color_component_fractions,
        value_minimum_shares=value_minimum_shares,
    )
    truth = json.loads((capture_root / "pixel_visibility_truth.json").read_text(encoding="utf-8"))
    if not isinstance(truth, Mapping):
        raise ValueError("pixel visibility truth must be an object")
    declarations: dict[str, Mapping[str, Any]] = {}
    visual_plan = plan.get("visual_plan") if isinstance(plan.get("visual_plan"), Mapping) else plan
    values = visual_plan.get("actors") if isinstance(visual_plan, Mapping) else None
    if isinstance(values, Mapping):
        declarations = {str(key): value for key, value in values.items() if isinstance(value, Mapping)}
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        declarations = {
            str(row["actor_id"]): row
            for row in values
            if isinstance(row, Mapping) and isinstance(row.get("actor_id"), str)
        }
    frame_indices = truth.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError("pixel visibility truth requires explicit frame indices")
    selected_frames = appearance_review_frame_selection(
        truth, plan, frame_stride=frame_stride
    )
    rgb_array = None
    rgb_path = capture_root / "rgb.npy"
    if rgb_path.is_file():
        rgb_array = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    mask_path = capture_root / "native_pixel_masks_depth_authority_v1.npz"
    records: dict[str, Any] = {}
    pixel_semantics: dict[str, Any] = {}
    resolution = truth.get("resolution_hw")
    with np.load(mask_path, allow_pickle=False) as data:
        modal = _modal_array(data)
        if not isinstance(resolution, Sequence) or isinstance(resolution, (str, bytes)) or len(resolution) != 2:
            resolution = [int(modal.shape[1]), int(modal.shape[2])]
        resolution_hw = [int(resolution[0]), int(resolution[1])]
        # Resolve every actor declaration first, then walk the reviewed frames
        # once. The previous actor-major walk re-read the same native RGB frame
        # for every actor in it; each frame is now read once and every actor
        # that needs it is inspected against that one decoded image. The masked
        # pixel comparison per (actor, frame) is unchanged.
        pending: dict[str, dict[str, Any]] = {}
        frame_requests: dict[int, list[tuple[str, int, Mapping[str, Any]]]] = {}
        for actor_id, instance in truth.get("per_instance", {}).items():
            if not isinstance(instance, Mapping):
                continue
            instance_frames = instance.get("frames", [])
            pixel_semantics[str(actor_id)] = summarize_pixel_visibility_semantics(
                instance_frames if isinstance(instance_frames, list) else [],
                resolution_hw=resolution_hw,
            )
            declaration = declarations.get(str(actor_id), {})
            spec = _appearance_spec(declaration, asset_registry=asset_registry)
            if spec["value"] is None:
                records[str(actor_id)] = {
                    "status": "not_observable",
                    "value": None,
                    "attribute_value": None,
                    "attribute_field": spec["field"],
                    "appearance_field_used": spec["appearance_field_used"],
                    "appearance_source": spec["source"],
                    "entity_kind": spec["kind"],
                    "searched_fields": spec["searched_fields"],
                    "available_registered_fields": spec["available_registered_fields"],
                    "frame_refs": [],
                    "checks": [],
                    "reason": spec["missing_reason"],
                    "reviewer": "native_RGB_masked_registered_appearance_v1",
                    "pixel_visibility_semantics": pixel_semantics[str(actor_id)],
                    "appearance_thresholds": thresholds,
                    "placeholder": True,
                    "calibration": thresholds["calibration"],
                    "claim_boundary": thresholds["claim_boundary"],
                }
                continue
            semantic_id = int(instance["semantic_id"])
            slots: list[dict[str, Any] | None] = []
            for frame in instance_frames if isinstance(instance_frames, list) else []:
                if not isinstance(frame, Mapping):
                    continue
                f = int(frame.get("frame_index", -1))
                if f not in selected_frames or frame.get("state") not in {"visible_clear", "visible_occluded"}:
                    continue
                frame_requests.setdefault(f, []).append((str(actor_id), len(slots), frame))
                slots.append(None)
            pending[str(actor_id)] = {
                "spec": spec,
                "semantic_id": semantic_id,
                "instance": instance,
                "slots": slots,
            }
        for f in sorted(frame_requests):
            row = _mask_frame_row(f, frame_indices, modal.shape[0])
            rgb, source = _rgb_frame(capture_root, f, frame_indices, rgb_array)
            if rgb.shape[:2] != modal.shape[1:]:
                raise ValueError("native RGB and modal mask resolutions differ")
            modal_row = modal[row]
            for actor_id, position, frame in frame_requests[f]:
                spec = pending[actor_id]["spec"]
                check = inspect_registered_appearance(
                    rgb,
                    modal_row == pending[actor_id]["semantic_id"],
                    spec["value"],
                    entity_kind=spec["kind"],
                    minimum_color_pixels=int(thresholds["minimum_color_pixels"]),
                    dominance_ratio=float(thresholds["dominance_ratio"]),
                    color_component_fractions=thresholds["color_component_fractions"],
                    value_minimum_shares=thresholds.get("value_minimum_shares"),
                    target_bbox=frame.get("target_bbox_xyxy_px"),
                )
                check.update(
                    frame_index=f,
                    frame_path=source,
                    appearance_field=spec["field"],
                    appearance_field_used=spec["appearance_field_used"],
                    appearance_source=spec["source"],
                    entity_kind=spec["kind"],
                )
                pending[actor_id]["slots"][position] = check
        for actor_id, work in pending.items():
            spec = work["spec"]
            checks = [check for check in work["slots"] if check is not None]
            accepted = [
                int(check["frame_index"]) for check in checks
                if check["status"] == "pass"
                and check.get("observed_value") == spec["value"].casefold()
            ]
            records[str(actor_id)] = {
                "status": "reviewed" if accepted else "not_observable",
                "value": spec["value"],
                "attribute_value": spec["value"],
                "attribute_field": spec["field"],
                "appearance_field_used": spec["appearance_field_used"],
                "appearance_source": spec["source"],
                "entity_kind": spec["kind"],
                "searched_fields": spec["searched_fields"],
                "available_registered_fields": spec["available_registered_fields"],
                "frame_refs": sorted(set(accepted)),
                "checks": checks,
                "reviewer": "native_RGB_masked_registered_appearance_v1",
                "pixel_visibility_semantics": pixel_semantics[str(actor_id)],
                "appearance_thresholds": thresholds,
                "placeholder": True,
                "calibration": thresholds["calibration"],
                "claim_boundary": "automatic coarse appearance evidence; native RGB frames remain available for human review",
                **(classifier_gap_fields(checks) if not accepted else {}),
            }
    return {
        "schema": "avengine_qa_appearance_review_v2",
        "status": "research_only",
        "actors": records,
        "authority": "actual_RGB_and_native_depth_instance_masks",
        "formal_certification": False,
        "in_fov_definition": IN_FOV_DEFINITION,
        "pixel_visibility_semantics": pixel_semantics,
        "appearance_thresholds": thresholds,
        "placeholder": True,
        "calibration": thresholds["calibration"],
        "claim_boundary": "registered appearance values are compared against actual masked RGB; asset labels are never treated as pixel evidence",
    }


def audit_native_structural_clearance(
    readbacks: Mapping[str, Any], layout: Mapping[str, Any],
) -> dict[str, Any]:
    """Conservatively test actual body envelopes against existing wall pieces."""
    structural = [x for x in layout["objects"]
                  if x["semantic_class"] in {"wall", "doorframe", "window_frame", "glazing"}]
    overlaps = []
    checked = 0
    for actor_id, records in readbacks["bounds"].items():
        for frame, record in enumerate(records):
            checked += 1
            lo, hi = np.asarray(record["minimum_cm"]) / 100, np.asarray(record["maximum_cm"]) / 100
            low = np.array([lo[0], -hi[1], lo[2]])
            high = np.array([hi[0], -lo[1], hi[2]])
            for obstacle in structural:
                a, b = np.asarray(obstacle["bounds_xyz_m"])
                intersection = np.minimum(high, b) - np.maximum(low, a)
                if np.all(intersection > 0.005):
                    overlaps.append({
                        "actor_id": actor_id, "frame_index": frame,
                        "object_id": obstacle["object_id"], "category": obstacle["semantic_class"],
                        "overlap_aabb_m": intersection.tolist(),
                    })
    return {
        "status": "pass" if not overlaps else "needs_geometry_review",
        "method": "native_visual_AABB_vs_registered_structural_bounds",
        "checked_actor_frames": checked, "structural_object_count": len(structural),
        "overlaps": overlaps,
        "claim_boundary": "sampled structural-envelope test; positive AABB intersections need mesh/visual review; furniture interaction is not certified",
    }


def review_imported_pose_clearance(
    report: Mapping[str, Any], readbacks: Mapping[str, Any],
    layout: Mapping[str, Any], plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve conservative human bounds with the existing source skinning replay.

    The exact imported GLB, native root and verified animation phase are reused.
    This is a source-pose check, distinct from a native skinned-vertex readback.
    """
    import json
    import math
    from avengine.assets.glb import load_glb
    from avengine.assets.skinning import compile_skinning, action_time_bounds, sample_action_vertices

    result = dict(report)
    if not report["overlaps"]:
        return result
    actors = {a["actor_id"]: a for a in plan["visual_plan"]["actors"]}
    objects = {o["object_id"]: o for o in layout["objects"]}
    compiled = {}
    pose_cache = {}
    resolved, unresolved = [], []
    for overlap in report["overlaps"]:
        actor_id, frame = overlap["actor_id"], int(overlap["frame_index"])
        actor = actors[actor_id]
        delta = actor.get("ue_component_frame_delta", {})
        reference = actor.get("exact_runtime_binding", {}).get("import_manifest_ref", {})
        if (actor.get("body_plan_id") != "biped_human" or not reference.get("path")
            or any(abs(float(x)) > 1e-8 for x in delta.get("rotation_deg", [0, 0, 0]))
            or any(abs(float(x)) > 1e-8 for x in delta.get("translation_cm", [0, 0, 0]))):
            unresolved.append({**overlap, "reason": "source_pose_basis_not_supported"})
            continue
        manifest = json.loads(Path(reference["path"]).read_text())
        if manifest["content"]["skeletal_mesh"] != actor["skeletal_mesh_path"]:
            raise ValueError("source pose mesh differs from native imported mesh")
        source = manifest["source_glb"]
        if source not in compiled:
            compiled[source] = compile_skinning(load_glb(source))
        skin = compiled[source]
        state = next(x for x in plan["visual_plan"]["frames"][frame]["actor_states"]
                     if x["actor_id"] == actor_id)
        actual = readbacks["actors"][actor_id][frame]
        animation = readbacks["animations"][actor_id][frame]
        action = animation["animation_path"].rsplit(".", 1)[-1]
        start, end = action_time_bounds(skin, action)
        phase = float(state["action_phase"])
        if abs(phase * (end - start) - animation["observed_position_seconds"]) > 1 / 30:
            unresolved.append({**overlap, "reason": "source_and_native_animation_phase_differ"})
            continue
        key = (source, action, phase, tuple(actual["location_cm"]), tuple(actual["rotation_deg"]))
        if key not in pose_cache:
            vertices = sample_action_vertices(skin, action, start + phase * (end - start))
            local = vertices[:, [0, 2, 1]] * float(actor.get("actor_scale", 1))
            yaw = math.radians(actual["rotation_deg"][2])
            c, s = math.cos(yaw), math.sin(yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            ue = local @ rotation.T + np.asarray(actual["location_cm"]) / 100
            native = readbacks["bounds"][actor_id][frame]
            if (np.any(ue.min(axis=0) < np.asarray(native["minimum_cm"]) / 100 - 0.002)
                or np.any(ue.max(axis=0) > np.asarray(native["maximum_cm"]) / 100 + 0.002)):
                unresolved.append({**overlap, "reason": "replayed_pose_not_within_native_envelope"})
                continue
            world = ue * np.array([1, -1, 1])
            pose_cache[key] = (world.min(axis=0), world.max(axis=0))
        low, high = pose_cache[key]
        wall_low, wall_high = np.asarray(objects[overlap["object_id"]]["bounds_xyz_m"])
        intersection = np.minimum(high, wall_high) - np.maximum(low, wall_low)
        if np.any(intersection < -0.005):
            resolved.append({
                **overlap, "source_glb": source, "source_import_manifest": reference["path"],
                "source_action": action, "source_action_time_seconds": start + phase * (end - start),
                "separating_axis_clearance_m": float(-intersection.min()),
                "source_pose_bounds_m": [low.tolist(), high.tolist()],
            })
        else:
            unresolved.append({**overlap, "reason": "source_pose_bounds_still_intersect_structure"})
    result.update({
        "status": "pass" if not unresolved else "needs_geometry_review",
        "source_pose_review": {
            "resolved_count": len(resolved), "unresolved_count": len(unresolved),
            "unique_pose_count": len(pose_cache), "resolved": resolved, "unresolved": unresolved,
            "basis": "imported unit-scale human GLB [x,y,z] to UE [x,z,y], actual native yaw/location",
            "claim_boundary": "source skinning at native-verified phases resolves conservative envelopes; not native vertex or physical-contact certification",
        },
    })
    return result


# --------------------------------------------------------------------------
# Shared visual evidence for audio-track members that reuse one native capture
# --------------------------------------------------------------------------
#
# A binding group renders one visual episode and binds several audio variants
# to it (`materialize_audio_variant` links each member's `capture` at the same
# physical directory). Appearance review, actor occluders and the visibility
# annotation read only that capture, the visual actor declarations and their
# registered attributes, so every member recomputed identical products from
# identical pixels. This pack computes them once per (capture, binding,
# parameter, evidence-version) identity and lets later members reuse them.
#
# Reuse never trusts the stored answer on its own: the retained capture inputs
# are re-stated, and a bounded sample of the recorded appearance and occluder
# observations is recomputed from the actual RGB, depth-authority modal masks
# and target-only masks before the pack is accepted.

SHARED_VISUAL_EVIDENCE_SCHEMA = "avengine_qa_shared_visual_evidence_v1"
SHARED_VISUAL_EVIDENCE_PRODUCER = "qa_evidence.build_shared_visual_evidence"
SHARED_VISUAL_EVIDENCE_CLAIM = "build.claim"
SHARED_VISUAL_EVIDENCE_MANIFEST = "manifest.json"
SHARED_VISUAL_EVIDENCE_DEFAULT_WAIT_SECONDS = 600.0
SHARED_VISUAL_EVIDENCE_VERIFICATION_FRAMES = 2

_CAPTURE_PIXEL_INPUTS = (
    "pixel_visibility_truth.json",
    "native_pixel_masks_depth_authority_v1.npz",
    "rgb.npy",
    "ue_visual_only.mp4",
    "visual.mp4",
    "metric_depth_native.npz",
)


class SharedVisualEvidenceError(RuntimeError):
    """A shared visual evidence pack could not be built, read or trusted."""


def visual_input_identity(path: Path) -> dict[str, Any] | None:
    """Describe one retained visual input by its ordinary filesystem identity.

    Size and modification time are the identity a cache needs: they change
    whenever the capture is re-rendered or repaired. This is deliberately not a
    content digest and is not an integrity contract. Its resolution is the
    host filesystem's timestamp granularity, measured at about one millisecond
    on this server, so a same-size rewrite inside one tick is not visible here.
    That is why reuse never rests on this record alone: a pack is accepted only
    after recorded observations are recomputed from the actual pixels and
    masks. A stale or ambiguous identity costs a rebuild, never a wrong answer.
    """
    value = Path(path)
    if value.is_dir():
        entries = sorted(entry for entry in value.iterdir() if entry.is_file())
        if not entries:
            return None
        stats = [entry.stat() for entry in entries]
        return {
            "path": str(value.resolve()),
            "kind": "directory",
            "entry_count": len(entries),
            "total_size_bytes": sum(stat.st_size for stat in stats),
            "latest_mtime_ns": max(stat.st_mtime_ns for stat in stats),
        }
    if not value.is_file():
        return None
    stat = value.stat()
    return {
        "path": str(value.resolve()),
        "kind": "file",
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def capture_visual_input_identities(capture_root: Path) -> dict[str, Any]:
    """Identify every retained capture input the visual evidence actually reads."""
    root = Path(capture_root).resolve()
    identities: dict[str, Any] = {}
    for name in _CAPTURE_PIXEL_INPUTS:
        identity = visual_input_identity(root / name)
        if identity is not None:
            identities[name] = identity
    frames = visual_input_identity(root / "frames")
    if frames is not None:
        identities["frames"] = frames
    if "pixel_visibility_truth.json" not in identities:
        raise SharedVisualEvidenceError(
            f"capture has no pixel visibility truth for shared visual evidence: {root}"
        )
    if "native_pixel_masks_depth_authority_v1.npz" not in identities:
        raise SharedVisualEvidenceError(
            f"capture has no depth-authority masks for shared visual evidence: {root}"
        )
    return identities


def shared_visual_evidence_key(
    capture_root: Path,
    plan: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    asset_registry: Mapping[str, Mapping[str, Any]] | None = None,
    frame_stride: int = 15,
    thresholds: Mapping[str, Any] | None = None,
    occluder_minimum_covered_pixels: int = 100,
    occluder_minimum_explained_fraction: float = 0.9,
) -> dict[str, Any]:
    """Describe exactly what a shared visual evidence pack was derived from.

    Two members may share a pack only when this whole record matches: the same
    retained capture pixels, the same visual actor/asset bindings and resolved
    appearance fields, the same reviewed frame selection, the same thresholds
    and the same evidence schema versions. Audio program, voice bindings and
    the sound pool are deliberately absent; they never reach these products.
    """
    root = Path(capture_root).resolve()
    resolved_thresholds = dict(thresholds) if thresholds is not None else nonhuman_appearance_placeholder_thresholds()
    declarations: dict[str, Mapping[str, Any]] = {}
    visual_plan = plan.get("visual_plan") if isinstance(plan.get("visual_plan"), Mapping) else plan
    values = visual_plan.get("actors") if isinstance(visual_plan, Mapping) else None
    if isinstance(values, Mapping):
        declarations = {str(key): value for key, value in values.items() if isinstance(value, Mapping)}
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        declarations = {
            str(row["actor_id"]): row
            for row in values
            if isinstance(row, Mapping) and isinstance(row.get("actor_id"), str)
        }
    bindings: list[dict[str, Any]] = []
    for actor_id in sorted(set(declarations) | {str(key) for key in truth.get("per_instance", {})}):
        declaration = declarations.get(actor_id, {})
        instance = truth.get("per_instance", {}).get(actor_id)
        spec = _appearance_spec(declaration, asset_registry=asset_registry)
        bindings.append({
            "actor_id": actor_id,
            "asset_id": declaration.get("asset_id") or declaration.get("entity_asset_id"),
            "entity_class": declaration.get("entity_class") or declaration.get("actor_class"),
            "species_id": declaration.get("species_id"),
            "semantic_id": (
                int(instance["semantic_id"])
                if isinstance(instance, Mapping) and isinstance(instance.get("semantic_id"), int)
                and not isinstance(instance.get("semantic_id"), bool)
                else None
            ),
            "appearance_field": spec["field"],
            "appearance_value": spec["value"],
            "appearance_source": spec["source"],
            "entity_kind": spec["kind"],
        })
    return {
        "capture_root": str(root),
        "capture_inputs": capture_visual_input_identities(root),
        "asset_bindings": bindings,
        "frame_indices": [int(value) for value in truth.get("frame_indices", [])],
        "reviewed_frames": appearance_review_frame_selection(truth, plan, frame_stride=frame_stride),
        "resolution_hw": (
            [int(truth["resolution_hw"][0]), int(truth["resolution_hw"][1])]
            if isinstance(truth.get("resolution_hw"), Sequence)
            and not isinstance(truth.get("resolution_hw"), (str, bytes))
            and len(truth["resolution_hw"]) == 2
            else None
        ),
        "parameters": {
            "frame_stride": int(frame_stride),
            "appearance_thresholds": resolved_thresholds,
            "occluder_minimum_covered_pixels": int(occluder_minimum_covered_pixels),
            "occluder_minimum_explained_fraction": float(occluder_minimum_explained_fraction),
        },
        "evidence_versions": {
            "shared_pack": SHARED_VISUAL_EVIDENCE_SCHEMA,
            "appearance_review": "avengine_qa_appearance_review_v2",
            "pixel_truth_status": truth.get("status"),
            "visibility_semantics_authority": "qa_evidence.annotate_pixel_visibility_semantics",
            "occluder_authority": "intersection_of_native_depth_modal_and_target_only_masks",
        },
    }


def build_shared_visual_evidence(
    capture_root: Path,
    plan: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    asset_registry: Mapping[str, Mapping[str, Any]] | None = None,
    frame_stride: int = 15,
    thresholds: Mapping[str, Any] | None = None,
    occluder_minimum_covered_pixels: int = 100,
    occluder_minimum_explained_fraction: float = 0.9,
) -> dict[str, Any]:
    """Compute the audio-independent visual evidence for one native capture."""
    root = Path(capture_root).resolve()
    resolved_thresholds = dict(thresholds) if thresholds is not None else nonhuman_appearance_placeholder_thresholds()
    key = shared_visual_evidence_key(
        root, plan, truth,
        asset_registry=asset_registry,
        frame_stride=frame_stride,
        thresholds=resolved_thresholds,
        occluder_minimum_covered_pixels=occluder_minimum_covered_pixels,
        occluder_minimum_explained_fraction=occluder_minimum_explained_fraction,
    )
    timings: dict[str, float] = {}
    started = time.monotonic()
    annotated = annotate_pixel_visibility_semantics(truth)
    timings["annotate_pixel_visibility_semantics_s"] = time.monotonic() - started

    started = time.monotonic()
    review = build_pixel_appearance_review(
        root, plan,
        frame_stride=frame_stride,
        asset_registry=asset_registry,
        minimum_color_pixels=int(resolved_thresholds["minimum_color_pixels"]),
        dominance_ratio=float(resolved_thresholds["dominance_ratio"]),
        color_component_fractions=resolved_thresholds["color_component_fractions"],
        value_minimum_shares=resolved_thresholds.get("value_minimum_shares"),
    )
    timings["build_pixel_appearance_review_s"] = time.monotonic() - started

    started = time.monotonic()
    occluders = derive_actor_occluders(
        root / "native_pixel_masks_depth_authority_v1.npz",
        annotated,
        minimum_covered_pixels=occluder_minimum_covered_pixels,
        minimum_explained_fraction=occluder_minimum_explained_fraction,
    )
    timings["derive_actor_occluders_s"] = time.monotonic() - started

    return {
        "schema": SHARED_VISUAL_EVIDENCE_SCHEMA,
        "producer": SHARED_VISUAL_EVIDENCE_PRODUCER,
        "key": key,
        "annotated_pixel_visibility_truth": annotated,
        "appearance_review": review,
        "actor_occluders": occluders,
        "stage_timings_s": timings,
        "build_total_s": sum(timings.values()),
        "claim_boundary": (
            "audio-independent visual evidence for one native capture; the "
            "audio program, voice bindings and per-member facts are not derived here"
        ),
    }


def _verification_checks(review: Mapping[str, Any], limit: int) -> list[tuple[str, Mapping[str, Any]]]:
    """Pick a bounded, deterministic sample of recorded appearance observations."""
    selected: list[tuple[str, Mapping[str, Any]]] = []
    actors = review.get("actors")
    if not isinstance(actors, Mapping):
        return selected
    for actor_id in sorted(actors):
        record = actors[actor_id]
        if not isinstance(record, Mapping):
            continue
        checks = [check for check in record.get("checks", []) if isinstance(check, Mapping)]
        if not checks:
            continue
        # First and last reviewed frames bracket the episode; both are recorded
        # observations, so a re-read compares like with like.
        candidates = [checks[0]] if limit <= 1 else [checks[0], checks[-1]]
        for check in candidates[:max(1, int(limit))]:
            selected.append((str(actor_id), check))
    return selected


def verify_shared_visual_evidence(
    pack: Mapping[str, Any],
    capture_root: Path,
    *,
    verification_frames: int = SHARED_VISUAL_EVIDENCE_VERIFICATION_FRAMES,
) -> dict[str, Any]:
    """Re-observe real pixels before a stored pack is reused.

    Filesystem identity alone would only prove that nothing was rewritten. This
    re-opens the actual RGB, the depth-authority modal masks and the target-only
    masks and recomputes a bounded sample of the recorded appearance and
    occluder observations. A registered label is never accepted in place of the
    observation it claims to summarise.
    """
    root = Path(capture_root).resolve()
    key = pack.get("key")
    if not isinstance(key, Mapping):
        return {"status": "rejected", "reason": "pack has no key record"}
    recorded = key.get("capture_inputs")
    if not isinstance(recorded, Mapping):
        return {"status": "rejected", "reason": "pack key has no capture input identities"}
    try:
        current = capture_visual_input_identities(root)
    except SharedVisualEvidenceError as error:
        return {"status": "rejected", "reason": str(error)}
    if current != recorded:
        drifted = sorted(
            name for name in set(current) | set(recorded)
            if current.get(name) != recorded.get(name)
        )
        return {
            "status": "rejected",
            "reason": "retained capture inputs changed since the pack was built",
            "changed_inputs": drifted,
        }

    review = pack.get("appearance_review")
    truth = pack.get("annotated_pixel_visibility_truth")
    if not isinstance(review, Mapping) or not isinstance(truth, Mapping):
        return {"status": "rejected", "reason": "pack lacks appearance review or annotated truth"}
    thresholds = key.get("parameters", {}).get("appearance_thresholds")
    if not isinstance(thresholds, Mapping):
        return {"status": "rejected", "reason": "pack key lacks appearance thresholds"}
    frame_indices = truth.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        return {"status": "rejected", "reason": "pack truth lacks explicit frame indices"}

    samples = _verification_checks(review, verification_frames)
    masks_path = root / "native_pixel_masks_depth_authority_v1.npz"
    rgb_array = None
    rgb_path = root / "rgb.npy"
    if rgb_path.is_file():
        rgb_array = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    reobserved: list[dict[str, Any]] = []
    occluder_reobserved: list[dict[str, Any]] = []
    try:
        with np.load(masks_path, allow_pickle=False) as data:
            modal = _modal_array(data)
            for actor_id, check in samples:
                frame_index = check.get("frame_index")
                if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                    return {"status": "rejected", "reason": f"recorded check has no frame identity: {actor_id}"}
                instance = truth.get("per_instance", {}).get(actor_id)
                if not isinstance(instance, Mapping) or not isinstance(instance.get("semantic_id"), int):
                    return {"status": "rejected", "reason": f"pack truth has no semantic ID for {actor_id}"}
                row = _mask_frame_row(frame_index, frame_indices, modal.shape[0])
                rgb, _source = _rgb_frame(root, frame_index, frame_indices, rgb_array)
                if rgb.shape[:2] != modal.shape[1:]:
                    return {"status": "rejected", "reason": "native RGB and modal mask resolutions differ"}
                frame_record = next(
                    (
                        row_value for row_value in instance.get("frames", [])
                        if isinstance(row_value, Mapping) and row_value.get("frame_index") == frame_index
                    ),
                    None,
                )
                observed = inspect_registered_appearance(
                    rgb,
                    modal[row] == int(instance["semantic_id"]),
                    check.get("expected_value"),
                    entity_kind=str(check.get("entity_kind") or "human"),
                    minimum_color_pixels=int(thresholds["minimum_color_pixels"]),
                    dominance_ratio=float(thresholds["dominance_ratio"]),
                    color_component_fractions=thresholds["color_component_fractions"],
                    value_minimum_shares=thresholds.get("value_minimum_shares"),
                    target_bbox=frame_record.get("target_bbox_xyxy_px") if isinstance(frame_record, Mapping) else None,
                )
                for field in ("status", "observed_value", "visible_pixels"):
                    if observed.get(field) != check.get(field):
                        return {
                            "status": "rejected",
                            "reason": "re-observed pixels disagree with the recorded appearance check",
                            "actor_id": actor_id,
                            "frame_index": frame_index,
                            "field": field,
                            "recorded": check.get(field),
                            "reobserved": observed.get(field),
                        }
                reobserved.append({
                    "actor_id": actor_id, "frame_index": frame_index,
                    "status": observed.get("status"),
                    "observed_value": observed.get("observed_value"),
                    "visible_pixels": observed.get("visible_pixels"),
                })

            occluders = pack.get("actor_occluders")
            records = occluders.get("frame_records") if isinstance(occluders, Mapping) else None
            if isinstance(records, list) and records:
                record = records[0]
                target = str(record.get("target_instance_id"))
                name = f"target_only_{target}"
                instance = truth.get("per_instance", {}).get(target)
                if name not in data or not isinstance(instance, Mapping):
                    return {"status": "rejected", "reason": f"pack occluder target is unavailable: {target}"}
                target_masks = np.asarray(data[name])
                if target_masks.shape != modal.shape:
                    return {"status": "rejected", "reason": f"target-only mask shape differs: {target}"}
                row = _mask_frame_row(int(record["frame_index"]), frame_indices, modal.shape[0])
                own_id = int(instance["semantic_id"])
                occluded = (target_masks[row] > 0) & (modal[row] != own_id)
                total = int(occluded.sum())
                if total != int(record.get("occluded_target_pixels", -1)):
                    return {
                        "status": "rejected",
                        "reason": "re-observed masks disagree with the recorded occluded pixel count",
                        "frame_index": int(record["frame_index"]),
                        "recorded": record.get("occluded_target_pixels"),
                        "reobserved": total,
                    }
                occluder_reobserved.append({
                    "frame_index": int(record["frame_index"]),
                    "target_instance_id": target,
                    "occluded_target_pixels": total,
                })
    except (OSError, ValueError) as error:
        return {"status": "rejected", "reason": f"pack inputs could not be re-observed: {error}"}

    return {
        "status": "pass",
        "capture_inputs_unchanged": sorted(current),
        "reobserved_appearance_checks": reobserved,
        "reobserved_occluder_records": occluder_reobserved,
        "method": "restat_retained_capture_inputs_then_recompute_sampled_pixel_and_mask_observations",
        "claim_boundary": (
            "bounded re-observation of recorded checks against actual RGB, modal "
            "and target-only masks; it is a reuse guard, not a fresh full review"
        ),
    }


def _pack_products(pack: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "annotated_pixel_visibility_truth.json": pack["annotated_pixel_visibility_truth"],
        "appearance_review.json": pack["appearance_review"],
        "actor_occluders.json": pack["actor_occluders"],
    }


def write_shared_visual_evidence(pack: Mapping[str, Any], pack_root: Path) -> Path:
    """Publish a pack atomically so a reader never sees a partial directory."""
    root = Path(pack_root)
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging_{os.getpid()}_{time.time_ns()}"
    staging.mkdir()
    try:
        for name, value in _pack_products(pack).items():
            (staging / name).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        manifest = {
            "schema": pack["schema"],
            "producer": pack["producer"],
            "key": pack["key"],
            "stage_timings_s": pack.get("stage_timings_s", {}),
            "build_total_s": pack.get("build_total_s"),
            "products": sorted(_pack_products(pack)),
            "written_unix_ns": time.time_ns(),
            "claim_boundary": pack.get("claim_boundary"),
        }
        (staging / SHARED_VISUAL_EVIDENCE_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        final = root / f"pack_{time.time_ns()}_{os.getpid()}"
        os.rename(staging, final)
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_shared_visual_evidence(pack_dir: Path) -> dict[str, Any] | None:
    """Read one published pack directory, or None when it is not complete."""
    root = Path(pack_dir)
    manifest_path = root / SHARED_VISUAL_EVIDENCE_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SHARED_VISUAL_EVIDENCE_SCHEMA:
        return None
    products: dict[str, Any] = {}
    for name in ("annotated_pixel_visibility_truth.json", "appearance_review.json", "actor_occluders.json"):
        path = root / name
        if not path.is_file():
            return None
        try:
            products[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return {
        "schema": manifest["schema"],
        "producer": manifest.get("producer"),
        "key": manifest.get("key"),
        "annotated_pixel_visibility_truth": products["annotated_pixel_visibility_truth.json"],
        "appearance_review": products["appearance_review.json"],
        "actor_occluders": products["actor_occluders.json"],
        "stage_timings_s": manifest.get("stage_timings_s", {}),
        "build_total_s": manifest.get("build_total_s"),
        "pack_dir": str(root.resolve()),
        "claim_boundary": manifest.get("claim_boundary"),
    }


def find_shared_visual_evidence(pack_root: Path, key: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the published pack whose key matches exactly, if one exists."""
    root = Path(pack_root)
    if not root.is_dir():
        return None
    for entry in sorted(root.glob("pack_*")):
        if not entry.is_dir():
            continue
        pack = read_shared_visual_evidence(entry)
        if pack is not None and pack.get("key") == dict(key):
            return pack
    return None


def shared_visual_pack_root(shared_root: Path, capture_root: Path) -> Path:
    """Give one capture a readable directory under the shared root.

    The complete resolved capture path is represented as directory components.
    The old last-three-components token made sibling captures such as
    v0_capture/.../capture and v1_capture/.../capture collide. A full path
    identity keeps different captures in different pack/master roots, while
    members whose capture symlinks resolve to the same directory still reuse
    one root. URL-encoding is only path-name encoding; it is not a content hash
    or a new integrity contract.
    """
    resolved = Path(capture_root).expanduser().resolve()
    path_parts = [
        quote(part, safe="")
        for part in resolved.parts
        if part not in ("/", "")
    ]
    token_parts = ["absolute" if resolved.is_absolute() else "relative", *path_parts]
    if len(token_parts) == 1:
        token_parts.append("capture")
    return (
        Path(shared_root).expanduser().resolve()
        / "capture_by_path"
        / Path(*token_parts)
    )


def _claim_shared_build(pack_root: Path, *, stale_seconds: float) -> bool:
    """Try to become the process that builds this pack."""
    claim = Path(pack_root) / SHARED_VISUAL_EVIDENCE_CLAIM
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            age = time.time() - claim.stat().st_mtime
        except OSError:
            return False
        if age <= stale_seconds:
            return False
        # The holder is gone or wedged well past a normal build. Take the claim
        # over; a duplicate build only wastes work, it cannot corrupt a pack,
        # because publication is an atomic rename of a complete directory.
        try:
            claim.unlink()
            descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except (FileExistsError, OSError):
            return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "unix_ns": time.time_ns()}, stream)
    return True


def _release_shared_build(pack_root: Path) -> None:
    try:
        (Path(pack_root) / SHARED_VISUAL_EVIDENCE_CLAIM).unlink()
    except OSError:
        pass


def acquire_shared_visual_evidence(
    capture_root: Path,
    plan: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    shared_root: Path | None,
    asset_registry: Mapping[str, Mapping[str, Any]] | None = None,
    frame_stride: int = 15,
    thresholds: Mapping[str, Any] | None = None,
    occluder_minimum_covered_pixels: int = 100,
    occluder_minimum_explained_fraction: float = 0.9,
    verification_frames: int = SHARED_VISUAL_EVIDENCE_VERIFICATION_FRAMES,
    wait_seconds: float = SHARED_VISUAL_EVIDENCE_DEFAULT_WAIT_SECONDS,
    poll_seconds: float = 2.0,
) -> dict[str, Any]:
    """Reuse or build the audio-independent visual evidence for one capture.

    With no shared root this simply builds, which is what a single-episode
    finalization has always done. With a shared root the first member builds
    and publishes; later members reuse only after the key matches exactly and
    the bounded pixel/mask re-observation passes.
    """
    root = Path(capture_root).resolve()
    resolved_thresholds = dict(thresholds) if thresholds is not None else nonhuman_appearance_placeholder_thresholds()
    build_arguments = {
        "asset_registry": asset_registry,
        "frame_stride": frame_stride,
        "thresholds": resolved_thresholds,
        "occluder_minimum_covered_pixels": occluder_minimum_covered_pixels,
        "occluder_minimum_explained_fraction": occluder_minimum_explained_fraction,
    }
    if shared_root is None:
        started = time.monotonic()
        pack = build_shared_visual_evidence(root, plan, truth, **build_arguments)
        pack["reuse"] = {
            "status": "built",
            "shared": False,
            "reason": "no shared visual root was declared for this episode",
            "elapsed_s": time.monotonic() - started,
        }
        return pack

    key = shared_visual_evidence_key(
        root, plan, truth,
        asset_registry=asset_registry,
        frame_stride=frame_stride,
        thresholds=resolved_thresholds,
        occluder_minimum_covered_pixels=occluder_minimum_covered_pixels,
        occluder_minimum_explained_fraction=occluder_minimum_explained_fraction,
    )
    pack_root = shared_visual_pack_root(shared_root, root)
    rejected: list[dict[str, Any]] = []

    def try_reuse() -> dict[str, Any] | None:
        found = find_shared_visual_evidence(pack_root, key)
        if found is None:
            return None
        started = time.monotonic()
        verification = verify_shared_visual_evidence(
            found, root, verification_frames=verification_frames
        )
        if verification.get("status") != "pass":
            rejected.append({"pack_dir": found.get("pack_dir"), **verification})
            return None
        found["reuse"] = {
            "status": "reused",
            "shared": True,
            "pack_dir": found.get("pack_dir"),
            "pack_root": str(pack_root),
            "verification": verification,
            "elapsed_s": time.monotonic() - started,
            "rejected_packs": rejected,
        }
        found["stage_timings_s"] = {"verify_shared_visual_evidence_s": time.monotonic() - started}
        found["build_total_s"] = 0.0
        return found

    reused = try_reuse()
    if reused is not None:
        return reused

    try:
        pack_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        started = time.monotonic()
        pack = build_shared_visual_evidence(root, plan, truth, **build_arguments)
        pack["reuse"] = {
            "status": "built_unshared",
            "shared": False,
            "reason": f"shared visual root is not writable: {error}",
            "pack_root": str(pack_root),
            "elapsed_s": time.monotonic() - started,
        }
        return pack

    claimed = _claim_shared_build(pack_root, stale_seconds=max(1.0, float(wait_seconds)))
    if not claimed:
        # Another member is building the same pack. Wait for it rather than
        # repeating the work, and fall back to a local build if it never
        # arrives, so a stalled peer can never block this finalization.
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        waited = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(max(0.1, float(poll_seconds)))
            reused = try_reuse()
            if reused is not None:
                reused["reuse"]["status"] = "reused_after_wait"
                reused["reuse"]["waited_s"] = time.monotonic() - waited
                return reused
            if not (pack_root / SHARED_VISUAL_EVIDENCE_CLAIM).exists():
                break
        claimed = _claim_shared_build(pack_root, stale_seconds=max(1.0, float(wait_seconds)))
        if not claimed:
            started = time.monotonic()
            pack = build_shared_visual_evidence(root, plan, truth, **build_arguments)
            pack["reuse"] = {
                "status": "built_unshared",
                "shared": False,
                "reason": "a peer build was still in progress after the declared wait",
                "pack_root": str(pack_root),
                "waited_s": time.monotonic() - waited,
                "elapsed_s": time.monotonic() - started,
                "rejected_packs": rejected,
            }
            return pack

    try:
        started = time.monotonic()
        pack = build_shared_visual_evidence(root, plan, truth, **build_arguments)
        published = write_shared_visual_evidence(pack, pack_root)
        pack["reuse"] = {
            "status": "built_and_published",
            "shared": True,
            "pack_dir": str(published.resolve()),
            "pack_root": str(pack_root),
            "elapsed_s": time.monotonic() - started,
            "rejected_packs": rejected,
        }
        return pack
    finally:
        _release_shared_build(pack_root)
