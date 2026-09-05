#!/usr/bin/env python3
"""Plan a static furnished residential episode without starting UE/SPEAR.

The room manifest supplies static geometry and seat references.  AVEngine then
creates a target-independent camera candidate pool and a bounded seat layout.
Pose-agent bindings may be joined later; this tool never drops an actor root on
the seat surface and never invents a UE map path.
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

from avengine.rooms.furniture_layout import (  # noqa: E402
    DEFAULT_ACTOR_COUNT,
    DEFAULT_SEAT_COUNT,
    FurnitureLayoutError,
    authoring_to_habitat,
    build_seat_placements,
    clock_config,
    generate_camera_candidates,
    habitat_to_ue_cm,
    load_room_layout,
    score_camera_candidates,
)


def _load_json(path: Path, *, owner: str) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FurnitureLayoutError(f"cannot read {owner} {path}: {exc}") from exc
    return value


def _pose_bindings(path: str | Path | None) -> Any:
    if path is None:
        return None
    value = _load_json(Path(path).expanduser().resolve(), owner="pose bindings")
    if not isinstance(value, (Mapping, list)):
        raise FurnitureLayoutError("pose bindings JSON must be an object or list")
    return value


def _camera_for_runtime(candidate: Mapping[str, Any]) -> dict[str, Any]:
    camera = deepcopy(dict(candidate))
    camera["ue_position_cm"] = list(candidate["position_ue_cm"])
    camera["ue_roll_deg"] = float(candidate.get("roll_deg", 0.0))
    return camera


def _actor_record(placement: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actor_id": placement["actor_id"],
        "asset_id": placement.get("asset_id"),
        "template_id": placement.get("template_id"),
        "body_plan_id": placement.get("body_plan_id"),
        "blueprint_class_path": placement.get("blueprint_class_path"),
        "ue_animation": placement.get("ue_animation"),
        "seat_affordance_id": placement["seat_affordance_id"],
        "placement_status": placement["placement_status"],
        "reference_is_not_actor_root": True,
    }


def _actor_state(
    placement: Mapping[str, Any],
    *,
    frame_index: int,
    pts_ticks: int,
) -> dict[str, Any]:
    root_habitat = placement.get("root_position_habitat_m")
    rotation = placement.get("rotation_xyzw")
    root_transform = None
    translation_ue_cm = None
    actor_yaw_ue_deg = None
    if root_habitat is not None:
        root_transform = {
            "translation_m": list(root_habitat),
            "rotation_xyzw": list(rotation or [0.0, 0.0, 0.0, 1.0]),
            "scale": [1.0, 1.0, 1.0],
        }
        translation_ue_cm = habitat_to_ue_cm(root_habitat)
        actor_yaw_ue_deg = float(placement["seat_reference"]["facing_yaw_deg"])
    return {
        "actor_id": placement["actor_id"],
        "root_transform": root_transform,
        "translation_m": list(root_habitat) if root_habitat is not None else None,
        "translation_ue_cm": translation_ue_cm,
        "rotation_xyzw": list(rotation) if rotation is not None else None,
        "actor_yaw_ue_deg": actor_yaw_ue_deg,
        "action_id": "seated_idle",
        "action_phase": 0.0,
        "action_time_ticks": pts_ticks,
        "ue_animation": placement.get("ue_animation"),
        "contacts": {
            "seat_affordance_id": placement["seat_affordance_id"],
            "reference_is_not_actor_root": True,
        },
        "placement_status": placement["placement_status"],
        "frame_index": frame_index,
    }


def build_episode_plan(
    layout: Mapping[str, Any],
    *,
    seat_count: int = DEFAULT_SEAT_COUNT,
    actor_count: int = DEFAULT_ACTOR_COUNT,
    pose_bindings: Any = None,
    frame_count: int = 75,
    frame_rate_hz: float = 15.0,
    sample_rate_hz: int = 16_000,
    grid_step_m: float = 2.0,
    camera_height_m: float = 1.55,
    map_path: str | None = None,
    scene_id: str | None = None,
) -> dict[str, Any]:
    """Build one path-free plan from a normalized room layout."""

    clock = clock_config(
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        sample_rate_hz=sample_rate_hz,
    )
    camera_set = generate_camera_candidates(
        layout,
        grid_step_m=grid_step_m,
        camera_height_m=camera_height_m,
    )
    if not camera_set["candidates"]:
        raise FurnitureLayoutError("camera candidate generation returned no candidates")
    seat_layout = build_seat_placements(
        layout,
        seat_count=seat_count,
        actor_count=actor_count,
        pose_bindings=pose_bindings,
    )
    selected_camera = _camera_for_runtime(camera_set["candidates"][0])
    placements = seat_layout["actor_placements"]
    actors = [_actor_record(item) for item in placements]
    frames: list[dict[str, Any]] = []
    for frame_index in range(clock["frame_count"]):
        pts_ticks = frame_index * clock["ticks_per_frame"]
        camera_state = deepcopy(selected_camera)
        camera_state["frame_index"] = frame_index
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": pts_ticks,
                "actor_states": [
                    _actor_state(item, frame_index=frame_index, pts_ticks=pts_ticks)
                    for item in placements
                ],
                "camera_state": camera_state,
            }
        )

    effective_map_path = map_path if map_path is not None else layout.get("map_path")
    effective_scene_id = scene_id or str(layout.get("scene_id") or layout["room_id"])
    scene = {
        "scene_id": effective_scene_id,
        "room_id": layout["room_id"],
        "map_path": effective_map_path if isinstance(effective_map_path, str) else None,
        "map_path_status": "declared" if isinstance(effective_map_path, str) else "not_declared",
        "claim_boundary": "map path is an input declaration; no UE stage was launched",
    }
    visual_plan = {
        "backend_role": "production_visual",
        "authority": {
            "room_identity_and_layout": "normalized_room_metadata",
            "camera_candidate_generation": "authoring_geometry_grid",
            "actor_state": "pose_binding_or_pending",
            "backend_may_replan": False,
        },
        "room": {
            "room_id": layout["room_id"],
            "scene_id": effective_scene_id,
            "authoring_geometry_status": layout["geometry"]["authoring_geometry_status"],
            "native_validation_status": "not_run",
        },
        "camera": selected_camera,
        "camera_candidates": deepcopy(camera_set["candidates"]),
        "camera_generation": deepcopy(camera_set["generation"]),
        "actors": actors,
        "render": {
            "frame_count": clock["frame_count"],
            "fps_num": clock["frame_rate_hz"],
            "fps_den": 1,
            "ticks_per_frame": clock["ticks_per_frame"],
        },
        "frames": frames,
        "claim_boundary": (
            "camera candidates are geometry-only authoring candidates; target LOS, pose and native UE readback are pending"
        ),
    }
    return {
        "kind": "furnished_residential_episode_plan",
        "status": "research_candidate",
        "scene": scene,
        "clock": clock,
        "visual_lighting": deepcopy(layout.get("visual_lighting", {})),
        "visual_plan": visual_plan,
        "camera_candidates": deepcopy(camera_set),
        "seat_layout": seat_layout,
        "room_layout": {
            "room_id": layout["room_id"],
            "manifest_path": layout.get("manifest_path"),
            "geometry": deepcopy(layout["geometry"]),
            "resources": deepcopy(layout.get("resources", {})),
            "authoring_geometry_status": "candidate",
            "native_validation_status": "not_run",
        },
        "planning_boundary": {
            "target_independent_candidates": True,
            "target_scoring": "post_actor_question_join",
            "target_los": "not_evaluated",
            "native_spear_ue": "not_run",
            "action_engine": "not_created",
            "review_cameras_used": False,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", "--room-manifest", dest="room", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-bindings", type=Path)
    parser.add_argument("--map-path")
    parser.add_argument("--scene-id")
    parser.add_argument("--seat-count", type=int, default=DEFAULT_SEAT_COUNT)
    parser.add_argument("--actor-count", type=int, default=DEFAULT_ACTOR_COUNT)
    parser.add_argument("--frame-count", type=int, choices=(75, 150), default=75)
    parser.add_argument("--frame-rate-hz", type=float, default=15.0)
    parser.add_argument("--sample-rate-hz", type=int, default=16_000)
    parser.add_argument("--grid-step-m", type=float, default=2.0)
    parser.add_argument("--camera-height-m", type=float, default=1.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace existing output: {output}")
    layout = load_room_layout(args.room, asset_root=args.asset_root)
    bindings = _pose_bindings(args.pose_bindings)
    plan = build_episode_plan(
        layout,
        seat_count=args.seat_count,
        actor_count=args.actor_count,
        pose_bindings=bindings,
        frame_count=args.frame_count,
        frame_rate_hz=args.frame_rate_hz,
        sample_rate_hz=args.sample_rate_hz,
        grid_step_m=args.grid_step_m,
        camera_height_m=args.camera_height_m,
        map_path=args.map_path,
        scene_id=args.scene_id,
    )
    output.mkdir(parents=True)
    _write_json(output / "room_layout.json", layout)
    _write_json(output / "camera_candidates.json", plan["camera_candidates"])
    _write_json(output / "seat_layout.json", plan["seat_layout"])
    _write_json(output / "episode_plan.json", plan)
    print(json.dumps({"output": str(output), "room_id": layout["room_id"], "camera_candidates": len(plan["camera_candidates"]["candidates"]), "seats": plan["seat_layout"]["selected_seat_ids"], "native_validation_status": "not_run"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
