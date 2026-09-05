#!/usr/bin/env python3
"""Build a short multi-view camera review plan for one furnished room.

The input episode supplies fixed actor declarations/states. AVEngine regenerates
only a dense, target-independent room camera pool, ranks it with the existing
geometry/FOV/GLB-ray scorer, and holds each selected view for a few frames so
the existing SPEAR runner can capture native pixels for human selection.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tools/rooms"))

from avengine.rooms.furniture_layout import (  # noqa: E402
    clock_config,
    generate_camera_candidates,
    load_room_layout,
    score_camera_candidates,
)
from plan_furnished_residential_episode import (  # noqa: E402
    _camera_for_runtime,
    _load_json,
    _load_static_triangle_geometry,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_episode(path: Path) -> dict[str, Any]:
    value = _load_json(path, owner="episode plan")
    if not isinstance(value, Mapping):
        raise ValueError("episode plan must be an object")
    return dict(value)


def _target_bounds_from_episode(episode: Mapping[str, Any]) -> list[dict[str, list[float]]]:
    layout = episode.get("seat_layout")
    placements = layout.get("actor_placements") if isinstance(layout, Mapping) else None
    if not isinstance(placements, list):
        raise ValueError("camera review requires episode seat_layout.actor_placements")
    bounds: list[dict[str, list[float]]] = []
    for placement in placements:
        reference = placement.get("seat_reference") if isinstance(placement, Mapping) else None
        if not isinstance(reference, Mapping):
            continue
        position = reference.get("position_authoring_m")
        if not isinstance(position, Sequence) or len(position) != 3:
            continue
        x, y, z = [float(value) for value in position]
        bounds.append(
            {
                "minimum_m": [x - 0.35, y - 0.35, max(0.0, z - 0.15)],
                "maximum_m": [x + 0.35, y + 0.35, z + 1.20],
            }
        )
    if not bounds:
        raise ValueError("camera review has no actor seat target bounds")
    return bounds


def _obstacle_bounds(layout: Mapping[str, Any]) -> list[Any]:
    return [
        item["bounds_xyz_m"]
        for item in layout.get("objects", [])
        if str(item.get("navigation_role") or "ground_blocker")
        not in {"walkable_surface", "walkable_floor_covering", "elevated_object"}
    ]


def _select_review_candidates(
    scored: Mapping[str, Any],
    *,
    candidate_count: int,
) -> list[dict[str, Any]]:
    raw = scored.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("scored camera pool has no candidates")
    ranked = sorted(
        (deepcopy(dict(item)) for item in raw if isinstance(item, Mapping)),
        key=lambda item: (
            -float(item.get("review_score", item.get("post_join_score", 0.0))),
            str(item.get("candidate_id")),
        ),
    )
    selected: list[dict[str, Any]] = []
    used_points: set[str] = set()
    for item in ranked:
        point = str(item.get("geometry_point_id"))
        if point in used_points:
            continue
        selected.append(item)
        used_points.add(point)
        if len(selected) >= candidate_count:
            return selected
    for item in ranked:
        if item["candidate_id"] in {value["candidate_id"] for value in selected}:
            continue
        selected.append(item)
        if len(selected) >= candidate_count:
            return selected
    if len(selected) < candidate_count:
        raise ValueError(
            f"camera pool has only {len(selected)} usable review candidates; requested {candidate_count}"
        )
    return selected


def build_camera_review_plan(
    episode: Mapping[str, Any],
    layout: Mapping[str, Any],
    *,
    candidate_count: int = 8,
    hold_frames: int = 6,
    grid_step_m: float = 0.75,
    camera_height_m: float = 1.55,
) -> dict[str, Any]:
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    if isinstance(hold_frames, bool) or not isinstance(hold_frames, int) or not 5 <= hold_frames <= 10:
        raise ValueError("hold_frames must be between 5 and 10")
    raw_clock = episode.get("clock")
    if not isinstance(raw_clock, Mapping):
        raise ValueError("input episode clock is missing")
    frame_rate = float(raw_clock["frame_rate_hz"])
    sample_rate = int(raw_clock["sample_rate_hz"])
    target_bounds = _target_bounds_from_episode(episode)
    actor_positions = [
        [
            (item["minimum_m"][axis] + item["maximum_m"][axis]) * 0.5
            for axis in range(3)
        ]
        for item in target_bounds
    ]
    pool = generate_camera_candidates(
        layout,
        grid_step_m=grid_step_m,
        camera_height_m=camera_height_m,
    )
    triangle_geometry = _load_static_triangle_geometry(layout)
    scored = score_camera_candidates(
        pool,
        actor_positions_m=actor_positions,
        target_bounds_m=target_bounds,
        obstacle_bounds_m=_obstacle_bounds(layout),
        room_bounds_xy_m=layout["geometry"].get("bounds_xy_m"),
        triangle_vertices_m=triangle_geometry["vertices"] if triangle_geometry else None,
        triangle_indices=triangle_geometry["triangles"] if triangle_geometry else None,
    )
    for candidate in scored["candidates"]:
        full = int(candidate.get("fully_framed_target_count", 0))
        total = int(candidate.get("target_count", len(target_bounds)))
        margin = float(candidate.get("target_frame_margin", -1.0))
        mesh_occlusion = int(candidate.get("target_mesh_occluded_count", 0))
        # Keep all-target framing as a strong preference, then prefer useful
        # on-screen occupancy and low inter-object/mesh obstruction. Native
        # pixel visibility remains the final review authority.
        occupancy = max(-1.0, min(1.0, 1.0 - margin))
        candidate["review_score"] = (
            float(candidate.get("post_join_score", 0.0))
            + (30.0 if full == total else -30.0 * (total - full))
            + 4.0 * occupancy
            - 14.0 * mesh_occlusion
        )
        candidate["review_visibility_status"] = (
            "geometry_candidate" if full == total and mesh_occlusion == 0 else "geometry_warning"
        )
    selected = _select_review_candidates(scored, candidate_count=candidate_count)
    selected = [
        _camera_for_runtime(candidate)
        for candidate in selected
    ]
    review_frame_count = candidate_count * hold_frames
    clock = clock_config(
        frame_count=review_frame_count,
        frame_rate_hz=frame_rate,
        sample_rate_hz=sample_rate,
    )
    visual_plan = deepcopy(dict(episode["visual_plan"]))
    base_actors = visual_plan.get("actors")
    base_frames = visual_plan.get("frames")
    if not isinstance(base_actors, list) or not isinstance(base_frames, list) or not base_frames:
        raise ValueError("input visual plan lacks actors or frames")
    base_states = base_frames[0].get("actor_states")
    if not isinstance(base_states, list) or not base_states:
        raise ValueError("camera review requires a non-empty fixed actor state")
    frames: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for candidate_index, camera in enumerate(selected):
        start = candidate_index * hold_frames
        end = start + hold_frames - 1
        segments.append(
            {
                "candidate_id": camera["candidate_id"],
                "frame_start": start,
                "frame_end": end,
                "camera": camera,
                "review_score": camera.get("review_score"),
                "review_visibility_status": camera.get("review_visibility_status"),
            }
        )
        for frame_index in range(start, end + 1):
            pts_ticks = frame_index * clock["ticks_per_frame"]
            states = []
            for state in base_states:
                item = deepcopy(dict(state))
                item["frame_index"] = frame_index
                item["action_time_ticks"] = pts_ticks
                states.append(item)
            camera_state = deepcopy(camera)
            camera_state["frame_index"] = frame_index
            frames.append(
                {
                    "frame_index": frame_index,
                    "pts_ticks": pts_ticks,
                    "actor_states": states,
                    "camera_state": camera_state,
                }
            )
    visual_plan["camera"] = deepcopy(selected[0])
    visual_plan["camera_candidates"] = deepcopy(selected)
    visual_plan["camera_review"] = {
        "candidate_count": candidate_count,
        "hold_frames": hold_frames,
        "grid_step_m": grid_step_m,
        "segments": segments,
        "native_selection_pending": True,
        "static_triangle_geometry_used": triangle_geometry is not None,
    }
    visual_plan["render"] = {
        **dict(visual_plan.get("render", {})),
        "frame_count": review_frame_count,
        "fps_num": clock["frame_rate_hz"],
        "fps_den": 1,
        "ticks_per_frame": clock["ticks_per_frame"],
    }
    visual_plan["frames"] = frames
    result = deepcopy(dict(episode))
    result["kind"] = "furnished_residential_camera_review_plan"
    result["status"] = "research_candidate"
    result["clock"] = clock
    result["visual_plan"] = visual_plan
    result["camera_candidates"] = {
        "generation": deepcopy(pool["generation"]),
        "review_selected_candidates": deepcopy(selected),
        "review_segments": segments,
    }
    result["planning_boundary"] = {
        "camera_review_only": True,
        "research_only": True,
        "target_independent_pool": True,
        "native_selection_pending": True,
        "actor_state_authority": "input_episode_plan",
        "action_engine": "not_created",
        "review_cameras_used": False,
    }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--room", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=6)
    parser.add_argument("--grid-step-m", type=float, default=0.75)
    parser.add_argument("--camera-height-m", type=float, default=1.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace existing output: {output}")
    episode = _load_episode(args.episode_root.expanduser().resolve() / "episode_plan.json")
    room_path = args.room
    if room_path is None:
        room_path = Path(episode["room_layout"]["manifest_path"])
    layout = load_room_layout(room_path, asset_root=args.asset_root)
    result = build_camera_review_plan(
        episode,
        layout,
        candidate_count=args.candidate_count,
        hold_frames=args.hold_frames,
        grid_step_m=args.grid_step_m,
        camera_height_m=args.camera_height_m,
    )
    output.mkdir(parents=True)
    _write_json(output / "episode_plan.json", result)
    _write_json(output / "camera_review.json", result["camera_candidates"])
    print(json.dumps({"output": str(output), "candidate_count": args.candidate_count, "hold_frames": args.hold_frames, "frame_count": result["clock"]["frame_count"], "static_triangle_geometry_used": result["visual_plan"]["camera_review"]["static_triangle_geometry_used"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
