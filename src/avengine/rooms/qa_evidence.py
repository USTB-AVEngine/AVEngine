"""Question evidence derived from retained native masks, never from plan intent."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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
            if name not in data or np.asarray(data[name]).shape != modal.shape:
                raise ValueError(f"missing or mismatched target-only masks: {target}")
            target_masks = np.asarray(data[name])
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


def inspect_coarse_top_color(
    rgb: np.ndarray, visible_mask: np.ndarray, target_bbox: list[int], *,
    minimum_color_pixels: int = 512,
) -> dict[str, Any]:
    """Check coarse registered shirt colors in actual native RGB.

    This bounded HSV diagnostic uses a visible upper-body crop. It makes no
    accessory, clothing-pattern or fine-attribute claim, and low-resolution
    or competing color evidence stays unverified.
    """
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3 or visible_mask.shape != rgb.shape[:2]:
        raise ValueError("RGB and native instance mask dimensions differ")
    x0, y0, x1, y1 = [int(x) for x in target_bbox]
    height, width = rgb.shape[:2]
    if x1 <= x0 or y1 <= y0:
        return {"status": "not_observable", "reason": "empty_target_footprint"}
    torso = np.zeros((height, width), dtype=bool)
    xa = max(0, int(x0 + 0.1 * (x1 - x0)))
    xb = min(width, int(x1 - 0.1 * (x1 - x0)))
    ya = max(0, int(y0 + 0.18 * (y1 - y0)))
    yb = min(height, int(y0 + 0.58 * (y1 - y0)))
    torso[ya:yb, xa:xb] = True
    selected = torso & np.asarray(visible_mask, dtype=bool)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
    hue, saturation, value = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
    color = selected & (saturation >= 0.18) & (value >= 0.16)
    counts = {
        "blue": int((color & (hue >= 190) & (hue < 270)).sum()),
        "green": int((color & (hue >= 70) & (hue < 175)).sum()),
        "yellow": int((color & (hue >= 38) & (hue < 70)).sum()),
        "burgundy": int((color & ((hue < 16) | (hue >= 345)) & (saturation >= 0.28)).sum()),
        "pink": int((color & (hue >= 300) & (hue < 345)).sum()),
        "white": int((selected & (saturation < 0.1) & (value > 0.65)).sum()),
    }
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    winner, count = ranked[0]
    second = ranked[1][1]
    confident = count >= minimum_color_pixels and count >= 1.6 * max(1, second)
    return {
        "status": "pass" if confident else "not_observable",
        "observed_color": winner if confident else None,
        "color_pixels": count, "upper_body_visible_pixels": int(selected.sum()),
        "candidate_color_counts": counts, "crop_xyxy": [xa, ya, xb, yb],
        "minimum_color_pixels": minimum_color_pixels,
        "claim_boundary": "coarse RGB color diagnostic, not formal attribute certification",
    }


def _appearance_spec(
    record: Mapping[str, Any],
    *,
    asset_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str] | None:
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
        wanted = (("finish", "finish"), ("surface_finish", "surface_finish"))
        kind = "device"
    elif "animal" in entity_class or any(token in entity_class for token in ("dog", "cat", "beagle", "quadruped")) or (isinstance(species, str) and species.casefold() not in {"human", "person"}):
        wanted = (("coat_profile", "coat_profile.value"), ("coat_value", "coat_value"), ("color", "color"))
        kind = "animal"
    else:
        wanted = (("top_color", "top_color"), ("shirt_color", "shirt_color"), ("color", "color"))
        kind = "human"
    for source, owner in owners:
        for key, field in wanted:
            value = owner.get(key)
            if key == "coat_profile" and isinstance(value, Mapping):
                value = value.get("value")
            if isinstance(value, str) and value.strip():
                return {
                    "field": field,
                    "value": value.strip(),
                    "kind": kind,
                    "source": source,
                }
    return None


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
    minimum_color_pixels: int = 8,
    target_bbox: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare registered appearance values against actual masked RGB pixels."""
    import cv2
    image = np.asarray(rgb)
    mask = np.asarray(visible_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("RGB and native instance mask dimensions differ")
    expected = str(expected_value).strip().casefold()
    if (entity_kind == "human" and expected in
            {"blue", "green", "yellow", "burgundy", "pink", "white"}
            and target_bbox is not None and len(target_bbox) == 4):
        # Preserve the established human shirt palette and its 512-pixel /
        # 1.6 dominance rule. Skin and animal/device colors are not competing
        # shirt values simply because an upper-body mask includes bare arms.
        observed = inspect_coarse_top_color(image, mask, list(target_bbox))
        matched = observed["status"] == "pass" and observed.get("observed_color") == expected
        return {
            "status": "pass" if matched else "not_observable",
            "observed_value": observed.get("observed_color"),
            "visible_pixels": observed.get("upper_body_visible_pixels", 0),
            "candidate_counts": observed.get("candidate_color_counts", {}),
            "expected_value": expected_value,
            "minimum_color_pixels": observed.get("minimum_color_pixels", 512),
            "calibration": "existing_coarse_human_rule_not_formal_certification",
            "crop_xyxy": observed.get("crop_xyxy"),
            "claim_boundary": observed["claim_boundary"],
        }
    if target_bbox is not None and len(target_bbox) == 4 and entity_kind == "human":
        x0, y0, x1, y1 = (int(value) for value in target_bbox)
        torso = np.zeros(mask.shape, dtype=bool)
        xa = max(0, int(x0 + 0.1 * (x1 - x0)))
        xb = min(mask.shape[1], int(x1 - 0.1 * (x1 - x0)))
        ya = max(0, int(y0 + 0.18 * (y1 - y0)))
        yb = min(mask.shape[0], int(y0 + 0.58 * (y1 - y0)))
        if xb > xa and yb > ya:
            torso[ya:yb, xa:xb] = True
            mask = mask & torso
    pixels = image[mask]
    expected = str(expected_value).strip().casefold()
    if pixels.size == 0:
        return {
            "status": "not_observable",
            "observed_value": None,
            "visible_pixels": 0,
            "candidate_counts": {},
            "expected_value": expected_value,
            "reason": "no visible native RGB pixels in the target mask",
        }
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
    hue = hsv[:, 0] * 2.0
    saturation = hsv[:, 1] / 255.0
    value = hsv[:, 2] / 255.0
    counts = {
        "blue": int(((hue >= 190) & (hue < 270) & (saturation >= 0.18) & (value >= 0.16)).sum()),
        "green": int(((hue >= 70) & (hue < 175) & (saturation >= 0.18) & (value >= 0.16)).sum()),
        "yellow": int(((hue >= 38) & (hue < 70) & (saturation >= 0.18) & (value >= 0.16)).sum()),
        "burgundy": int((((hue < 16) | (hue >= 345)) & (saturation >= 0.28) & (value >= 0.12)).sum()),
        "pink": int(((hue >= 300) & (hue < 345) & (saturation >= 0.18) & (value >= 0.16)).sum()),
        "white": int(((saturation < 0.2) & (value > 0.6)).sum()),
        "dark": int((value < 0.35).sum()),
        "warm_brown": int(((hue >= 5) & (hue < 45) & (saturation >= 0.12) & (value < 0.85)).sum()),
    }
    coarse = {
        "standard_black_white": "black_white",
        "standard_red_white": "red_white",
        "standard_red": "red",
        "standard_yellow": "yellow_coat",
        "standard_blue": "blue_gray_coat",
    }.get(expected, expected) if entity_kind == "animal" else expected
    # These are coat-profile semantics, not names parsed out of asset IDs.
    # In the registered British Shorthair profile, blue denotes gray-blue fur.
    counts["blue_gray_coat"] = int(((saturation <= 0.28) & (value >= 0.15) & (value <= 0.85)).sum())
    counts["yellow_coat"] = int(((hue >= 30) & (hue <= 75) & (saturation >= 0.05) & (value >= 0.25)).sum())
    total = max(1, len(pixels))
    unsupported = False
    if coarse in counts and coarse not in {"dark", "warm_brown", "yellow_coat", "blue_gray_coat"}:
        expected_count = counts[expected]
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        second = next((count for name, count in ranked if name != expected), 0)
        accepted = expected_count >= minimum_color_pixels and expected_count >= 1.25 * max(1, second)
        observed = expected if accepted else (ranked[0][0] if ranked[0][1] >= minimum_color_pixels else None)
    elif coarse in {"standard_tricolor", "light_tricolor", "dark_tricolor"}:
        white_fraction = counts["white"] / total
        dark_fraction = counts["dark"] / total
        warm_fraction = counts["warm_brown"] / total
        accepted = (
            counts["white"] >= max(2, minimum_color_pixels // 4)
            and counts["dark"] >= max(4, minimum_color_pixels // 2)
            and counts["warm_brown"] >= max(2, minimum_color_pixels // 4)
            and white_fraction > 0.0
            and dark_fraction > 0.0
            and warm_fraction > 0.0
        )
        observed = expected if accepted else None
    elif coarse == "black_white":
        accepted = (
            counts["white"] >= max(2, minimum_color_pixels // 4)
            and counts["dark"] >= max(4, minimum_color_pixels // 2)
        )
        observed = expected if accepted else None
    elif coarse == "red_white":
        accepted = (counts["white"] >= max(2, minimum_color_pixels // 4)
                    and counts["warm_brown"] >= max(2, minimum_color_pixels // 4))
        observed = expected if accepted else None
    elif coarse in {"yellow_coat", "blue_gray_coat"}:
        minimum_fraction = 0.30 if coarse == "blue_gray_coat" else 0.12
        accepted = counts[coarse] >= minimum_color_pixels and counts[coarse] / total >= minimum_fraction
        observed = expected if accepted else None
    elif coarse in {"black_ash", "black", "charcoal", "dark"}:
        accepted = counts["dark"] >= minimum_color_pixels and counts["dark"] / total >= 0.55
        observed = expected if accepted else None
    elif coarse in {"walnut_veneer", "walnut", "ruddy", "standard_ruddy", "brown", "red"}:
        accepted = counts["warm_brown"] >= minimum_color_pixels and counts["warm_brown"] / total >= 0.12
        observed = expected if accepted else None
    else:
        accepted = False
        observed = None
        unsupported = True
    return {
        "status": "pass" if accepted else "not_observable",
        "observed_value": observed,
        "visible_pixels": int(len(pixels)),
        "candidate_counts": counts,
        "expected_value": expected_value,
        "minimum_color_pixels": minimum_color_pixels,
        "coarse_color_predicate": coarse,
        **({"reason": "registered_appearance_value_classifier_not_implemented",
            "gap_category": "interface_not_implemented"} if unsupported else {}),
        "calibration": "placeholder_coarse_color_only",
        "claim_boundary": "coarse native masked color comparison; does not certify texture, wood grain or fine phenotype",
    }


def build_pixel_appearance_review(
    capture_root: Path,
    plan: Mapping[str, Any], *,
    frame_stride: int = 15,
    asset_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build renderer-neutral appearance evidence from masked native RGB."""
    capture_root = Path(capture_root).resolve()
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
    selected_frames = sorted(set(selected_frames))
    rgb_array = None
    rgb_path = capture_root / "rgb.npy"
    if rgb_path.is_file():
        rgb_array = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    mask_path = capture_root / "native_pixel_masks_depth_authority_v1.npz"
    records: dict[str, Any] = {}
    with np.load(mask_path, allow_pickle=False) as data:
        modal = _modal_array(data)
        for actor_id, instance in truth.get("per_instance", {}).items():
            if not isinstance(instance, Mapping):
                continue
            declaration = declarations.get(str(actor_id), {})
            spec = _appearance_spec(declaration, asset_registry=asset_registry)
            if spec is None:
                records[str(actor_id)] = {
                    "status": "not_observable",
                    "value": None,
                    "frame_refs": [],
                    "checks": [],
                    "reason": "registered appearance field is unavailable",
                    "reviewer": "native_RGB_masked_registered_appearance_v1",
                }
                continue
            semantic_id = int(instance["semantic_id"])
            checks: list[dict[str, Any]] = []
            accepted: list[int] = []
            for frame in instance.get("frames", []):
                if not isinstance(frame, Mapping):
                    continue
                f = int(frame.get("frame_index", -1))
                if f not in selected_frames or frame.get("state") not in {"visible_clear", "visible_occluded"}:
                    continue
                row = _mask_frame_row(f, frame_indices, modal.shape[0])
                rgb, source = _rgb_frame(capture_root, f, frame_indices, rgb_array)
                if rgb.shape[:2] != modal.shape[1:]:
                    raise ValueError("native RGB and modal mask resolutions differ")
                check = inspect_registered_appearance(
                    rgb,
                    modal[row] == semantic_id,
                    spec["value"],
                    entity_kind=spec["kind"],
                    target_bbox=frame.get("target_bbox_xyxy_px"),
                )
                check.update(
                    frame_index=f,
                    frame_path=source,
                    appearance_field=spec["field"],
                    appearance_source=spec["source"],
                    entity_kind=spec["kind"],
                )
                checks.append(check)
                if check["status"] == "pass" and check.get("observed_value") == spec["value"]:
                    accepted.append(f)
            records[str(actor_id)] = {
                "status": "reviewed" if accepted else "not_observable",
                "value": spec["value"],
                "attribute_value": spec["value"],
                "attribute_field": spec["field"],
                "appearance_source": spec["source"],
                "entity_kind": spec["kind"],
                "frame_refs": sorted(set(accepted)),
                "checks": checks,
                "reviewer": "native_RGB_masked_registered_appearance_v1",
                "claim_boundary": "automatic coarse appearance evidence; native RGB frames remain available for human review",
            }
    return {
        "schema": "avengine_qa_appearance_review_v2",
        "status": "research_only",
        "actors": records,
        "authority": "actual_RGB_and_native_depth_instance_masks",
        "formal_certification": False,
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
