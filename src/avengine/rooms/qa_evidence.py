"""Question evidence derived from retained native masks, never from plan intent."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def derive_actor_occluders(
    masks_path: Path, pixel_truth: Mapping[str, Any], *,
    minimum_covered_pixels: int = 100, minimum_explained_fraction: float = 0.9,
) -> dict[str, Any]:
    """Identify only occlusions explained by one other captured source instance.

    Depth equality determines the visible modal source ID. Its intersection
    with an occluded target-only footprint then identifies the foreground
    actor. Unlabeled static geometry remains unknown and never receives a
    guessed furniture name.
    """
    if pixel_truth.get("status") not in {"pass", "computed_modal_target_only_v1"}:
        raise ValueError("actor occluders require completed native pixel truth")
    instances = pixel_truth.get("per_instance", {})
    semantic = {str(key): int(record["semantic_id"]) for key, record in instances.items()}
    by_id = {value: key for key, value in semantic.items()}
    if len(semantic) != len(by_id):
        raise ValueError("native semantic IDs must be distinct")
    records = []
    rejected = Counter()
    with np.load(masks_path, allow_pickle=False) as data:
        modal = data["depth_derived_modal_semantic"]
        if modal.ndim != 3:
            raise ValueError("native modal masks must be [frame,height,width]")
        for target, own_id in semantic.items():
            target_masks = data[f"target_only_{target}"]
            if target_masks.shape != modal.shape:
                raise ValueError("target and modal mask shapes differ")
            for frame in instances[target]["frames"]:
                f = int(frame["frame_index"])
                if frame.get("state") not in {"visible_occluded", "fully_occluded"}:
                    continue
                occluded = (target_masks[f] > 0) & (modal[f] != own_id)
                total = int(occluded.sum())
                if total < minimum_covered_pixels:
                    rejected["too_few_occluded_pixels"] += 1
                    continue
                ids, counts = np.unique(modal[f][occluded], return_counts=True)
                known = [(int(count), by_id[int(value)]) for value, count in zip(ids, counts)
                         if int(value) in by_id and int(value) != own_id]
                if not known:
                    rejected["no_identified_foreground_actor"] += 1
                    continue
                known.sort(reverse=True)
                count, foreground = known[0]
                if count < minimum_covered_pixels or count / total < minimum_explained_fraction:
                    rejected["occlusion_not_explained_by_one_actor"] += 1
                    continue
                records.append({
                    "frame_index": f, "target_instance_id": target,
                    "occluder_instance_ids": [foreground],
                    "occluded_target_pixels": total, "foreground_actor_pixels": count,
                    "explained_fraction": count / total,
                })
    return {
        "status": "pass", "authority": "intersection_of_native_depth_modal_and_target_only_masks",
        "masks_path": str(masks_path.resolve()), "frame_records": records,
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


def build_pixel_appearance_review(
    capture_root: Path, plan: Mapping[str, Any], *,
    frame_stride: int = 15,
) -> dict[str, Any]:
    """Save reviewable per-frame coarse-color evidence for the unified miner."""
    import cv2
    import json

    truth = json.loads((capture_root / "pixel_visibility_truth.json").read_text())
    declarations = {x["actor_id"]: x for x in plan["visual_plan"]["actors"]}
    frames = truth["frame_indices"]
    selected_frames = set(frames[::frame_stride]) | {frames[-1]}
    clock = plan["clock"]
    fps, sr = float(clock["frame_rate_hz"]), int(clock["sample_rate_hz"])
    for event in plan.get("audio_events", []):
        f = int(event["start_sample"] * fps / sr)
        selected_frames.update(i for i in (f, f + 1) if i in frames)
    records = {}
    with np.load(capture_root / "native_pixel_masks_depth_authority_v1.npz", allow_pickle=False) as data:
        modal = data["depth_derived_modal_semantic"]
        for actor_id, instance in truth["per_instance"].items():
            declaration = declarations[actor_id]
            attrs = declaration.get("realized_attributes", {})
            expected = attrs.get("top_color", attrs.get("coat_value"))
            if expected not in {"blue", "green", "yellow", "burgundy", "pink", "white"}:
                continue
            checks, accepted = [], []
            for frame in instance["frames"]:
                f = int(frame["frame_index"])
                if f not in selected_frames or frame["state"] not in {"visible_clear", "visible_occluded"}:
                    continue
                path = capture_root / "frames" / f"frame_{f:04d}.png"
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise ValueError(f"native RGB frame unavailable: {path}")
                check = inspect_coarse_top_color(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    modal[f] == instance["semantic_id"], frame["target_bbox_xyxy_px"])
                check.update(frame_index=f, frame_path=str(path.resolve()))
                checks.append(check)
                if check["status"] == "pass" and check["observed_color"] == expected:
                    accepted.append(f)
            records[actor_id] = {
                "status": "pass" if accepted else "not_observable",
                "value": expected, "frame_refs": accepted, "checks": checks,
                "reviewer": "native_RGB_upper_body_HSV_diagnostic",
            }
    return {
        "status": "research_only", "actors": records,
        "authority": "actual_RGB_and_native_depth_instance_masks",
        "formal_certification": False,
        "claim_boundary": "automatic coarse-color evidence; retain frames for visual review",
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
