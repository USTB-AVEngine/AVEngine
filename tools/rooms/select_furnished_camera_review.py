"""Choose a furnished-room camera from actual SPEAR per-person visibility masks.

This consumes the existing native camera review and emits a fixed-camera
research plan. It ranks the least visible person first so a large foreground
actor cannot hide a missing participant in an average visibility score.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
from avengine.rooms.furniture_layout import clock_config


def rank_segments(segments, visible_counts, target_counts):
    actor_ids = list(visible_counts)
    if not actor_ids or set(actor_ids) != set(target_counts):
        raise ValueError("visible and target actor closure differs")
    frame_counts = {len(v) for v in visible_counts.values()} | {len(v) for v in target_counts.values()}
    if len(frame_counts) != 1:
        raise ValueError("native pixel frame counts differ")
    frame_count = frame_counts.pop()
    rows = []
    covered = set()
    for segment in segments:
        start, end = int(segment["frame_start"]), int(segment["frame_end"])
        if not 0 <= start <= end < frame_count:
            raise ValueError("camera segment exceeds native frame count")
        frames = set(range(start, end + 1))
        if frames & covered:
            raise ValueError("camera segments overlap")
        covered.update(frames)
        # Use the latter half of each hold after the camera jump has settled.
        begin = start + (end - start + 1) // 2
        per_actor = {}
        for actor_id in actor_ids:
            visible = np.asarray(visible_counts[actor_id][begin:end + 1], dtype=float)
            target = np.asarray(target_counts[actor_id][begin:end + 1], dtype=float)
            if (visible < 0).any() or (target < visible).any():
                raise ValueError("native pixel counts are inconsistent")
            ratios = np.divide(visible, target, out=np.zeros_like(visible), where=target > 0)
            per_actor[actor_id] = {
                "median_visible_pixels": float(np.median(visible)),
                "median_visible_fraction": float(np.median(ratios)),
            }
        row = {
            "candidate_id": segment["candidate_id"],
            "frame_start": start,
            "frame_end": end,
            "representative_frame": end,
            "evaluated_frames": list(range(begin, end + 1)),
            "per_actor": per_actor,
            "minimum_visible_pixels": min(x["median_visible_pixels"] for x in per_actor.values()),
            "minimum_visible_fraction": min(x["median_visible_fraction"] for x in per_actor.values()),
            "total_visible_pixels": sum(x["median_visible_pixels"] for x in per_actor.values()),
        }
        rows.append(row)
    if covered != set(range(frame_count)):
        raise ValueError("camera segments do not cover the native frames")
    rows.sort(key=lambda r: (-r["minimum_visible_pixels"], -r["minimum_visible_fraction"],
                             -r["total_visible_pixels"], r["candidate_id"]))
    if not rows or rows[0]["minimum_visible_pixels"] <= 0:
        raise ValueError("no reviewed camera shows every actor")
    return rows


def count_native_masks(visible, target, *, semantic_id, frame_count):
    if visible.shape != target.shape or visible.ndim != 3 or visible.dtype != np.bool_:
        raise ValueError("native visibility masks have invalid shapes or types")
    if visible.shape[0] != frame_count:
        raise ValueError("native masks differ from the declared clock")
    if not np.issubdtype(target.dtype, np.integer):
        raise ValueError("native target masks must carry semantic IDs")
    if np.any((target != 0) & (target != semantic_id)):
        raise ValueError("native target mask contains another actor ID")
    if np.any(visible & (target == 0)):
        raise ValueError("visible pixels are outside the target footprint")
    return np.count_nonzero(visible, axis=(1, 2)), np.count_nonzero(target, axis=(1, 2))


def select_camera(*, episode_root: Path, capture_root: Path, output: Path, frame_count: int):
    episode_root, capture_root, output = (p.expanduser().resolve() for p in (episode_root, capture_root, output))
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    episode = load(episode_root / "episode_plan.json")
    plan = episode["visual_plan"]
    captured = load(capture_root / "visual_plan.json")
    if captured != plan:
        raise ValueError("native capture did not execute this camera review plan")
    receipt = load(capture_root / "research_receipt.json")
    native = receipt.get("native_pixel", {})
    if native.get("status") != "pass":
        raise ValueError("completed native pixel capture is required")
    actor_ids = [a["actor_id"] for a in plan["actors"]]
    if set(native["semantic_ids_by_actor"]) != set(actor_ids):
        raise ValueError("native actor closure differs from the review plan")
    visible_counts, target_counts = {}, {}
    with np.load(capture_root / "native_pixel_masks_depth_authority_v1.npz", allow_pickle=False) as masks:
        for actor_id in actor_ids:
            visible = masks[f"modal_visible_{actor_id}"]
            target = masks[f"target_only_{actor_id}"]
            visible_counts[actor_id], target_counts[actor_id] = count_native_masks(
                visible, target, semantic_id=native["semantic_ids_by_actor"][actor_id],
                frame_count=len(plan["frames"]),
            )
    segments = plan["camera_review"]["segments"]
    ranking = rank_segments(segments, visible_counts, target_counts)
    winner = ranking[0]
    selected_segment = next(s for s in segments if s["candidate_id"] == winner["candidate_id"])
    camera = deepcopy(selected_segment["camera"])
    clock = clock_config(frame_count=frame_count,
                         frame_rate_hz=episode["clock"]["frame_rate_hz"],
                         sample_rate_hz=episode["clock"]["sample_rate_hz"])
    result = deepcopy(episode)
    result["kind"] = "furnished_residential_native_selected_plan"
    result["clock"] = clock
    result["status"] = "research_candidate"
    result["planning_boundary"] = {
        **result.get("planning_boundary", {}),
        "camera_review_only": False, "native_selection_pending": False,
        "native_selection_source": "SPEAR per-person depth visibility",
        "visual_quality_review_pending": True,
    }
    visual = result["visual_plan"]
    visual["camera"] = camera
    visual["clock"] = clock
    visual.pop("camera_review", None)
    visual["camera_selection"] = {
        "selection_mode": "native_pixel_review",
        "selected_candidate_id": winner["candidate_id"],
        "capture_root": str(capture_root),
        "ranking_rule": "maximize minimum per-person visible pixels, then visibility fraction",
        "metrics": winner,
        "visual_quality_review_pending": True,
    }
    visual["render"] = {**visual.get("render", {}), "frame_count": frame_count,
                        "fps_num": clock["frame_rate_hz"], "fps_den": 1,
                        "ticks_per_frame": clock["ticks_per_frame"]}
    base_states = plan["frames"][0]["actor_states"]
    visual["frames"] = []
    for index in range(frame_count):
        states = deepcopy(base_states)
        for state in states:
            state["frame_index"] = index
        visual["frames"].append({
            "frame_index": index, "pts_ticks": index * clock["ticks_per_frame"],
            "actor_states": states, "camera_state": {**camera, "frame_index": index},
        })
    exposure = receipt.get("capture_exposure_readback", {})
    if exposure.get("status") == "pass":
        visual["camera"]["exposure_bias_ev"] = exposure["bias_ev"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "native_camera_selected",
        "episode_root": str(episode_root), "capture_root": str(capture_root),
        "selected_candidate_id": winner["candidate_id"], "ranking": ranking,
        "representative_image": str(capture_root / "frames" / f"frame_{winner['representative_frame']:04d}.png"),
        "qualification": False, "visual_quality_review_pending": True,
    }
    for name, value in (("episode_plan.json", result), ("camera_selection.json", report)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-count", default=240, type=int)
    args = parser.parse_args()
    report = select_camera(**vars(args))
    print(json.dumps({key: report[key] for key in ("status", "selected_candidate_id", "representative_image")}, indent=2))


if __name__ == "__main__":
    main()
