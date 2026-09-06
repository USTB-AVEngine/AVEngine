"""Question-driven research Episodes over existing room geometry and runtimes.

Room capabilities are planning possibilities. Only native readbacks, pixel
evidence and rendered PCM establish which conditions a particular Episode met.
The planner never adds room furniture and never selects by room name.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.camera_pose import yaw_rotation_xyzw
from avengine.rooms.furniture_layout import (
    FurnitureLayoutError, _mesh_ray_occluded, authoring_to_habitat,
    build_seat_placements, clock_config, generate_camera_candidates,
    habitat_to_ue_cm, load_room_layout,
)
from avengine.rooms.furnished_episode import (
    _actor_record, _actor_state, _load_static_triangle_geometry,
)
from avengine.routes.raster_pathfinder import (
    RasterPathfinder, RasterShortestPath, _erode_binary,
    build_polygon_raster_obstacle_map,
)
from avengine.routes.trajectory import resample_polyline_by_arc_length
from avengine.runtime_profiles import (
    build_asset_emitter_binding, load_source_asset_runtime_registry,
    resolve_source_asset_runtime_profile,
)


class QAPlanningError(ValueError):
    """A request cannot be realized using the supplied existing resources."""


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _positive(value: Any, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise QAPlanningError(f"{owner} must be positive and finite")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise QAPlanningError(f"{owner} must be positive and finite")
    return value


def build_room_navigation(
    layout: Mapping[str, Any], *, resolution_m: float = 0.08,
    clearance_m: float = 0.38, body_height_m: float = 1.9,
    floor_height_m: float | None = None,
) -> tuple[RasterPathfinder, dict[str, Any]]:
    """Reuse the raster A* adapter on retained floors and conservative blockers.

    The dominant floor height excludes lower exterior ground slabs. A bounding
    rectangle alone is not walkable evidence: cells also need an existing
    floor footprint. Multi-level or arbitrary scan navigation requires its
    native navmesh adapter and is not silently approximated by this function.
    """
    resolution_m = _positive(resolution_m, "resolution_m")
    clearance_m = _positive(clearance_m, "clearance_m")
    floors = [x for x in layout.get("objects", [])
              if str(x.get("semantic_class", "")).lower() == "floor"]
    if not floors:
        raise QAPlanningError("no registered floor geometry; need a retained native navigation resource")
    if floor_height_m is None:
        levels = Counter(round(float(x["bounds_xyz_m"][1][2]), 3) for x in floors)
        floor_height_m = max(levels, key=lambda level: (levels[level], level))
    floors = [x for x in floors
              if abs(float(x["bounds_xyz_m"][1][2]) - floor_height_m) <= 0.025]
    if not floors:
        raise QAPlanningError("the requested floor level has no registered floor")
    xmin, ymin, xmax, ymax = layout["geometry"]["bounds_xy_m"]
    polygon = [[xmin, -ymax], [xmax, -ymax], [xmax, -ymin], [xmin, -ymin]]
    blockers = []
    for item in layout.get("objects", []):
        low, high = item["bounds_xyz_m"]
        if str(item.get("semantic_class", "")).lower() in {"floor", "rug", "carpet", "ceiling"}:
            continue
        if item.get("navigation_role") in {"walkable_surface", "walkable_floor_covering"}:
            continue
        if high[2] <= floor_height_m + 0.04 or low[2] >= floor_height_m + body_height_m:
            continue
        blockers.append({
            "object_id": item["object_id"],
            "footprint_xz_m": [[low[0], -high[1]], [high[0], -high[1]],
                               [high[0], -low[1]], [low[0], -low[1]]],
            "blocks_source_center": True,
        })
    pf, _ = build_polygon_raster_obstacle_map(
        polygon_xz_m=polygon, rigid_obstacles=blockers,
        floor_height_m=floor_height_m, meters_per_pixel=resolution_m,
        minimum_clearance_m=0.0,
    )
    bounds = pf.get_bounds()
    binary = pf.get_topdown_view(resolution_m, floor_height_m).astype(bool)
    x = bounds[0, 0] + (np.arange(binary.shape[1]) + 0.5) * resolution_m
    z = bounds[0, 2] + (np.arange(binary.shape[0]) + 0.5) * resolution_m
    xx, zz = np.meshgrid(x, z)
    supported = np.zeros_like(binary)
    for floor in floors:
        low, high = floor["bounds_xyz_m"]
        supported |= ((xx >= low[0]) & (xx <= high[0])
                      & (zz >= -high[1]) & (zz <= -low[1]))
    binary = _erode_binary(binary & supported, clearance_m, resolution_m)
    if not binary.any():
        raise QAPlanningError("no free floor cells at the requested body clearance")
    pf = RasterPathfinder(binary, bounds_m=bounds, floor_height_m=floor_height_m)
    return pf, {
        "authority": "existing_floor_footprints_and_object_bounds_with_raster_astar",
        "source_manifest": layout.get("manifest_path"),
        "floor_object_ids": [x["object_id"] for x in floors],
        "blocking_object_ids": [x["object_id"] for x in blockers],
        "floor_height_m": floor_height_m, "resolution_m": resolution_m,
        "clearance_m": clearance_m, "body_height_m": body_height_m,
        "free_area_m2": int(binary.sum()) * resolution_m ** 2,
        "bounds_habitat_m": bounds.tolist(),
        "geometry_status": "planning_candidate",
        "native_collision_status": "not_run",
        "claim_boundary": "conservative metadata footprints; native capture and path review remain separate",
    }


def navigation_points(pf: RasterPathfinder, nav: Mapping[str, Any]) -> np.ndarray:
    binary = pf.get_topdown_view(nav["resolution_m"], nav["floor_height_m"])
    pixels = np.argwhere(binary)
    bounds = pf.get_bounds()
    return np.column_stack((
        bounds[0, 0] + (pixels[:, 1] + 0.5) * nav["resolution_m"],
        np.full(len(pixels), nav["floor_height_m"]),
        bounds[0, 2] + (pixels[:, 0] + 0.5) * nav["resolution_m"],
    ))


def source_declaration(registry: Mapping[str, Any], asset_id: str, actor_id: str) -> dict[str, Any]:
    record = resolve_source_asset_runtime_profile(registry, asset_id)
    backend = deepcopy(record.get("runtime_backends", {}).get("spear_unreal"))
    if not isinstance(backend, dict) or record.get("entity_class") == "rigid_object":
        raise QAPlanningError(f"{asset_id} has no articulated SPEAR runtime")
    mesh = backend.get("skeletal_mesh_path")
    import_ref = backend.get("ue_import_manifest_ref", {})
    if not mesh and import_ref.get("path"):
        mesh = read_json(import_ref["path"]).get("content", {}).get("skeletal_mesh")
    if not mesh:
        raise QAPlanningError(f"{asset_id} lacks an exact imported mesh reference")
    timeline = record["timeline"]
    idle = timeline["idle_action_id"]
    walk = timeline["walking_action_id"]
    animations = backend.get("animation_paths_by_action_id") or {
        idle: backend.get("idle_animation"), walk: backend.get("walking_animation"),
    }
    if not animations.get(idle) or not animations.get(walk):
        raise QAPlanningError(f"{asset_id} lacks retained idle/walk animations")
    emitter = build_asset_emitter_binding(registry, source_slot_id=actor_id, asset_id=asset_id)
    offset = emitter["emitter_offset_m"]
    attributes = deepcopy(record["realized_attributes"])
    if attributes.get("top_color"):
        attributes["coat_value"] = attributes["top_color"]
    elif isinstance(attributes.get("coat_profile"), Mapping):
        attributes["coat_value"] = attributes["coat_profile"]["value"]
    result = {
        **backend, "actor_id": actor_id, "asset_id": asset_id,
        "asset_revision": record["revision"], "template_id": timeline["template_id"],
        "body_plan_id": timeline["body_plan_id"], "identity": deepcopy(record["identity"]),
        "realized_attributes": attributes, "display_label": record["display_label"],
        "skeletal_mesh_path": mesh, "animation_paths_by_action_id": animations,
        "idle_animation": animations[idle], "walking_animation": animations[walk],
        "idle_action_id": idle, "walking_action_id": walk,
        "walk_phase_period_frames": timeline["walk_phase_period_frames"],
        "emitter_local_ue_cm": [offset[0] * 100, offset[2] * 100, offset[1] * 100],
        "emitter_binding": emitter,
        "exact_runtime_binding": {
            "source": "source_asset_runtime_registry_and_native_import",
            "asset_id": asset_id, "revision": record["revision"],
            "import_manifest_ref": deepcopy(import_ref),
            "asset_bound_lineage": deepcopy(record.get("asset_bound_lineage")),
            "status": "declared_pending_native_readback",
        },
    }
    return result


def room_capabilities(
    layout: Mapping[str, Any], nav: Mapping[str, Any] | None,
    source_pool: Sequence[Mapping[str, Any]], sound_pool: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    speech = [x for x in sound_pool if x.get("transcript")]
    classes = {x.get("sound_class") for x in sound_pool if x.get("sound_class")}
    return {
        "room_id": layout["room_id"], "backend": layout.get("backend_route"),
        "status": "potential_only", "native_episode_conditions": "not_run",
        "potential": {
            "min_entities": len(source_pool), "events": bool(sound_pool),
            "appearance": any(x.get("realized_attributes") for x in source_pool),
            "speech_content": bool(speech), "motion": bool(nav),
            "after_sound": bool(sound_pool), "pixel_visibility": bool(layout.get("resources", {}).get("visual_geometry")),
            "occlusion": bool(layout.get("objects")), "entry": bool(nav),
            "distinct_sound_classes": len(classes),
            "seated": len(layout.get("seats", [])),
        },
        "evidence_refs": {
            "room_manifest": layout.get("manifest_path"),
            "navigation": deepcopy(nav),
            "source_assets": [x["asset_id"] for x in source_pool],
            "sounds": [x["sound_asset_id"] for x in sound_pool],
        },
    }


def match_question_conditions(qa_ids: Sequence[str], capabilities: Mapping[str, Any]) -> dict[str, Any]:
    from avengine.qa.unified_catalog import get_requirements

    possible, gaps = [], {}
    facts = capabilities["potential"]
    for qa_id in qa_ids:
        req = get_requirements(qa_id)
        req = req.get("potential_requirements", req)
        missing = []
        for key, required in req.items():
            if key not in facts or required in (None, False, 0):
                continue
            actual = facts[key]
            if isinstance(required, bool):
                if not actual:
                    missing.append(key)
            elif isinstance(required, Mapping):
                if required and not actual:
                    missing.append(key)
            elif isinstance(required, (int, float)) and not isinstance(actual, bool):
                if actual < required:
                    missing.append(key)
        if missing:
            gaps[qa_id] = missing
        else:
            possible.append(qa_id)
    return {
        "candidate_qa_ids": possible, "unsatisfied_potential_conditions": gaps,
        "status": "candidate" if possible else "unsupported",
        "episode_validity": "not_run",
        "claim_boundary": "resources permit planning; no native pixel, sound or answer validation implied",
    }


def _path(pf: RasterPathfinder, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    query = RasterShortestPath(requested_start=start, requested_end=end)
    if not pf.find_path(query) or query.geodesic_distance < 0.6:
        return None
    points = np.asarray(query.points)
    # Remove grid stair-steps only when the same retained navigation accepts
    # the whole shortcut; this keeps natural headings without a second solver.
    retained = [points[0]]
    index = 0
    while index < len(points) - 1:
        candidate = len(points) - 1
        while candidate > index + 1:
            count = max(2, int(np.ceil(np.linalg.norm(points[candidate] - points[index])
                                      / (pf.meters_per_pixel * 0.25))) + 1)
            segment = np.linspace(points[index], points[candidate], count)
            if all(pf.is_navigable(point) for point in segment):
                break
            candidate -= 1
        retained.append(points[candidate])
        index = candidate
    return np.asarray(retained)


def sample_activity_routes(
    pf: RasterPathfinder, nav: Mapping[str, Any], *, actor_count: int,
    frame_count: int, fps: int, rng: np.random.Generator, activity: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    points = navigation_points(pf, nav)
    if len(points) > 1600:
        points = points[rng.choice(len(points), 1600, replace=False)]
    patterns = ["hold_walk_hold", "walk_hold_walk", "stand", "hold_walk_hold"]
    rng.shuffle(patterns)
    for attempt in range(96):
        hub = points[int(rng.integers(len(points)))]
        near = points[np.linalg.norm(points - hub, axis=1) < 3.1]
        if len(near) < actor_count * 3:
            continue
        starts = [hub]
        for point in near[rng.permutation(len(near))]:
            if min(np.linalg.norm(point - p) for p in starts) >= 0.95:
                starts.append(point)
                if len(starts) == actor_count:
                    break
        if len(starts) != actor_count:
            continue
        routes, records = {}, {}
        failed = False
        for index, start in enumerate(starts):
            actor_id = f"source{index + 1}"
            pattern = "stand" if activity == "standing" else patterns[index % len(patterns)]
            route = np.repeat(start[None, :], frame_count, axis=0)
            polyline = None
            if pattern != "stand":
                ends = near[(np.linalg.norm(near - start, axis=1) >= 1.5)
                            & (np.linalg.norm(near - start, axis=1) < 4.5)]
                for end in ends[rng.permutation(len(ends))[:48]]:
                    candidate = _path(pf, start, end)
                    if candidate is None:
                        continue
                    length = np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum()
                    if length < min(5.0, frame_count / fps * 0.55):
                        polyline = candidate
                        break
                if polyline is None:
                    failed = True
                    break
                speed = float(rng.uniform(0.5, 0.8))
                moving_frames = max(2, min(frame_count - 4,
                    int(math.ceil(length / speed * fps)) + 1))
                if pattern == "hold_walk_hold":
                    start_frame = int(rng.integers(1, max(2, frame_count - moving_frames)))
                    end_frame = start_frame + moving_frames
                    route[start_frame:end_frame] = resample_polyline_by_arc_length(polyline, moving_frames)
                    route[end_frame:] = route[end_frame - 1]
                    intervals = [[start_frame, end_frame]]
                else:
                    # Movement is interrupted by a real stationary interval.
                    pause = int(rng.integers(max(2, fps), max(3, fps * 3)))
                    moving_frames = min(moving_frames, frame_count - pause - 2)
                    moving = resample_polyline_by_arc_length(polyline, moving_frames)
                    split = moving_frames // 2
                    route[:split] = moving[:split]
                    route[split:split + pause] = moving[split - 1]
                    route[split + pause: moving_frames + pause] = moving[split:]
                    route[moving_frames + pause:] = moving[-1]
                    intervals = [[0, split], [split + pause, moving_frames + pause]]
            else:
                intervals = []
            if not all(pf.is_navigable(p) for p in route):
                failed = True
                break
            if any(np.linalg.norm(route - r, axis=1).min() < 0.78 for r in routes.values()):
                failed = True
                break
            routes[actor_id] = route
            records[actor_id] = {
                "activity": pattern, "moving_frame_intervals": intervals,
                "polyline_habitat_m": polyline.tolist() if polyline is not None else [start.tolist()],
                "path_length_m": float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
                "speed_max_mps": float(np.max(np.linalg.norm(np.diff(route, axis=0), axis=1)) * fps),
                "path_solver": "avengine.routes.raster_pathfinder.RasterPathfinder",
                "all_sampled_centers_navigable": True,
            }
        if not failed:
            return routes, {
                "sampling_attempt": attempt, "actors": records,
                "minimum_actor_separation_m": 0.78,
                "native_body_collision_status": "not_run",
            }
    raise QAPlanningError("bounded route sampling found no separated paths; try another room/seed/activity")


def _look_camera(camera: Mapping[str, Any], target: Sequence[float], yaw_offset_deg: float = 0) -> dict[str, Any]:
    result = deepcopy(dict(camera))
    origin = np.asarray(result["position_authoring_m"], dtype=float)
    delta = np.asarray(target) - origin
    yaw = math.degrees(math.atan2(delta[1], delta[0])) + yaw_offset_deg
    pitch = math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1])))
    yr, pr = math.radians(yaw), math.radians(pitch)
    forward = [math.cos(pr) * math.cos(yr), math.cos(pr) * math.sin(yr), math.sin(pr)]
    result.update({
        "yaw_deg": yaw, "pitch_deg": pitch, "yaw_blender_deg": yaw,
        "pitch_blender_deg": pitch, "ue_yaw_deg": -yaw, "ue_pitch_deg": pitch,
        "forward_blender": forward, "forward_ue": [forward[0], -forward[1], forward[2]],
        "ue_roll_deg": 0.0,
    })
    return result


def _projected_visible(camera: Mapping[str, Any], point: np.ndarray) -> bool:
    forward = np.asarray(camera["forward_blender"], dtype=float)
    right = np.array([-forward[1], forward[0], 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    delta = point - camera["position_authoring_m"]
    depth = float(np.dot(delta, forward))
    tangent = math.tan(math.radians(camera["horizontal_fov_deg"]) / 2)
    return bool(depth > 0.1 and abs(np.dot(delta, right)) < depth * tangent * 0.93
                and abs(np.dot(delta, up)) < depth * tangent / (16 / 9) * 0.9)


def select_question_camera(
    layout: Mapping[str, Any], pf: RasterPathfinder, routes: Mapping[str, np.ndarray],
    actors: Sequence[Mapping[str, Any]], *, rng: np.random.Generator,
    camera_motion: str, qa_ids: Sequence[str], camera_fov_deg: float = 85.0,
    sampling_policy: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    conditioned = sampling_policy == "conditioned_static_v2"
    if sampling_policy not in (None, "conditioned_static_v2"):
        raise QAPlanningError(f"unknown sampling_policy: {sampling_policy}")
    if conditioned and camera_motion != "static":
        raise QAPlanningError("conditioned_static_v2 requires a static camera")
    floor = float(next(iter(routes.values()))[0, 1])
    pool = generate_camera_candidates(
        layout, grid_step_m=0.55, camera_height_m=floor + 1.55,
        yaw_candidates_deg=(0.0,), pitch_candidates_deg=(0.0,),
        clearance_m=0.4, horizontal_fov_deg=camera_fov_deg,
    )
    candidates = [x for x in pool["candidates"] if pf.is_navigable(
        [x["position_habitat_m"][0], floor, x["position_habitat_m"][2]])]
    centers = []
    for actor in actors:
        route = routes[actor["actor_id"]]
        height = float(actor.get("emitter_local_ue_cm", [0, 0, 150])[2]) / 100
        centers.append([route[0, 0], -route[0, 2], floor + max(0.35, height - 0.25)])
    target = np.mean(centers, axis=0)
    candidates = [_look_camera(x, target) for x in candidates]
    # Candidate points have already passed the existing room clearance logic.
    if not conditioned:
        candidates.sort(key=lambda c: -sum(
            3 * int(_projected_visible(c, np.asarray(p))) - max(0, np.linalg.norm(
                np.asarray(p) - c["position_authoring_m"]) - 4.5) for p in centers))
    geometry = _load_static_triangle_geometry(layout)
    if geometry is None:
        raise QAPlanningError("visual triangle geometry unavailable for camera/path condition sampling")
    frame_count = len(next(iter(routes.values())))
    sample_frames = sorted(set([0, frame_count // 5, frame_count // 2,
                               frame_count * 4 // 5, frame_count - 1]))
    ranked = []
    from avengine.qa.unified_catalog import get_requirements
    focus_late_angle = any(get_requirements(q)["question_family"] == "post_sound_azimuth"
                           for q in qa_ids) and camera_motion == "follow_group"
    wants_transitions = bool(set(qa_ids) & {"QA-07", "QA-09", "QA-11", "QA-20", "QA-24"})
    def group_target(frame: int) -> np.ndarray:
        return np.mean([
            [routes[a["actor_id"]][frame, 0], -routes[a["actor_id"]][frame, 2],
             floor + max(0.35, float(a["emitter_local_ue_cm"][2]) / 100 - 0.25)]
            for a in actors], axis=0)
    for candidate in (candidates if conditioned else candidates[:40]):
        vis = {}
        total = 0.0
        for actor in actors:
            route = routes[actor["actor_id"]]
            h = max(0.35, float(actor["emitter_local_ue_cm"][2]) / 100 - 0.25)
            states = []
            for f in sample_frames:
                point = np.array([route[f, 0], -route[f, 2], floor + h])
                frame_camera = (_look_camera(candidate, group_target(f))
                                if camera_motion == "follow_group" else candidate)
                if not _projected_visible(frame_camera, point):
                    state = "out_of_view"
                elif _mesh_ray_occluded(candidate["position_authoring_m"], point,
                                        geometry["vertices"], geometry["triangles"]):
                    state = "geometry_occluded"
                else:
                    state = "geometry_visible"
                states.append(state)
            visible = states.count("geometry_visible")
            total += 8 * int(visible > 0) + visible
            if visible:
                distance = np.linalg.norm(np.asarray(candidate["position_authoring_m"]) - centers[len(vis)])
                total += max(0, 4.5 - distance)
            if wants_transitions and len(set(states)) > 1:
                total += 4
            vis[actor["actor_id"]] = states
        # Prevent a rig being placed on a source's planned walking path.
        rig = np.asarray(candidate["position_habitat_m"]); rig[1] = floor
        if any(np.linalg.norm(route - rig, axis=1).min() < 0.8 for route in routes.values()):
            continue
        late_angles = []
        late_camera = _look_camera(candidate, group_target(frame_count - 1))
        origin = np.asarray(candidate["position_authoring_m"])
        fyaw = math.radians(late_camera["yaw_deg"])
        forward = np.array([math.cos(fyaw), math.sin(fyaw)])
        right = np.array([math.sin(fyaw), -math.cos(fyaw)])
        for actor in actors:
            p = routes[actor["actor_id"]][-1]
            direction = np.array([p[0], -p[2]]) - origin[:2]
            late_angles.append(math.degrees(math.atan2(
                np.dot(direction, right), np.dot(direction, forward))))
        separation = max((abs((a - b + 180) % 360 - 180) for a in late_angles for b in late_angles),
                         default=0.0)
        if focus_late_angle:
            both_late_visible = all(states[-1] == "geometry_visible" for states in vis.values())
            both_initial_visible = all(states[0] == "geometry_visible" for states in vis.values())
            total += (100 if both_late_visible and both_initial_visible and separation > 64 else -100)
            total += min(separation, 90) * 0.2
        candidate["planned_late_azimuth_separation_deg"] = separation
        if conditioned:
            if any("geometry_visible" in values for values in vis.values()):
                ranked.append((0.0, 0.0, candidate, vis))
        else:
            ranked.append((total, float(rng.random()), candidate, vis))
    if not ranked:
        raise QAPlanningError("no camera with route clearance")
    if conditioned:
        _, _, camera, states = ranked[int(rng.integers(len(ranked)))]
    else:
        ranked.sort(key=lambda x: (-x[0], x[1]))
        _, _, camera, states = ranked[0]
    if not any("geometry_visible" in x for x in states.values()):
        raise QAPlanningError("no camera can observe any target through existing geometry")
    camera_frames = []
    sweep = float(rng.uniform(16, 28)) * (-1 if rng.random() < 0.5 else 1)
    for f in range(frame_count):
        offset = sweep * math.sin(math.pi * f / max(1, frame_count - 1)) if camera_motion == "pan" else 0.0
        aim = group_target(f) if camera_motion == "follow_group" else target
        state = _look_camera(camera, aim, offset)
        state["frame_index"] = f
        camera_frames.append(state)
    return camera_frames[0], camera_frames, {
        "generation": pool["generation"], "candidate_count": len(candidates),
        "selected_candidate_id": camera["candidate_id"],
        "selection": "question_target_torso_geometry_and_path_clearance",
        "sample_frames": sample_frames, "planning_visibility": states,
        "camera_motion": camera_motion,
        "planned_late_azimuth_separation_deg": camera["planned_late_azimuth_separation_deg"],
        "native_observability": "not_run",
        **({"selection": "uniform_over_legal", "checked_candidate_count": len(candidates),
            "legal_candidate_ids": [item[2]["candidate_id"] for item in ranked]}
           if conditioned else {}),
        "claim_boundary": "torso ray/FOV estimates select candidates; only native RGB and pixel evidence validate questions",
    }


def schedule_audio(
    actors: Sequence[Mapping[str, Any]], sounds: Sequence[Mapping[str, Any]], *,
    clock: Mapping[str, Any], rng: np.random.Generator, mode: str = "sequential",
    silent_actor_count: int = 0, sampling_policy: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import soundfile as sf

    sr, duration = int(clock["sample_rate_hz"]), float(clock["duration_seconds"])
    if sampling_policy not in (None, "conditioned_static_v2"):
        raise QAPlanningError(f"unknown sampling_policy: {sampling_policy}")
    if mode not in {"sequential", "overlap", "repeat"}:
        raise QAPlanningError("audio mode must be sequential, overlap or repeat")
    if silent_actor_count < 0 or silent_actor_count >= len(actors):
        raise QAPlanningError("silent_actor_count must retain at least one speaking entity")
    selected_actors = list(actors)
    rng.shuffle(selected_actors)
    selected_actors = selected_actors[:len(actors) - silent_actor_count]
    available = list(sounds)
    rng.shuffle(available)
    if len(available) < len(selected_actors):
        raise QAPlanningError("not enough independent sound bindings")
    bindings, events = [], []
    for actor, sound in zip(selected_actors, available):
        binding = {**deepcopy(sound), "actor_id": actor["actor_id"]}
        info = sf.info(str(Path(binding["path"]).expanduser().resolve()))
        if info.samplerate != sr or info.channels != 1:
            raise QAPlanningError("sound pool must contain prepared mono audio at the episode sample rate")
        binding["sample_count"] = info.frames
        binding["duration_seconds"] = info.frames / sr
        binding["sound_class"] = binding.get("sound_class") or ("speech" if binding.get("transcript") else None)
        bindings.append(binding)
    reserve = min(3.0, duration * 0.2)
    total = sum(x["duration_seconds"] for x in bindings)
    if mode == "sequential":
        slack = duration - reserve - total
        if slack < 0.3:
            raise QAPlanningError("complete sound clips plus a silent query tail do not fit")
        gaps = rng.dirichlet(np.ones(len(bindings) + 1)) * slack
        cursor = float(gaps[0])
        starts = []
        for index, binding in enumerate(bindings):
            starts.append(cursor)
            cursor += binding["duration_seconds"] + float(gaps[index + 1])
    else:
        latest = duration - reserve - max(x["duration_seconds"] for x in bindings)
        if latest < 0.4:
            raise QAPlanningError("sound clips do not fit with a silent tail")
        starts = sorted(float(x) for x in rng.uniform(0.2, min(latest, max(0.6, duration * 0.25)), len(bindings)))
    for index, (binding, start) in enumerate(zip(bindings, starts)):
        begin = int(round(start * sr))
        events.append({
            **deepcopy(binding), "event_id": f"event_{index + 1:03d}",
            "start_sample": begin, "end_sample": begin + binding["sample_count"],
            "start_tick": begin * 48000 // sr,
            "end_tick": (begin + binding["sample_count"]) * 48000 // sr,
            "linear_gain": float(binding.get("linear_gain", 0.15)),
            "event_unit": "independent_source_playback_onset",
        })
    if mode == "repeat":
        if sampling_policy == "conditioned_static_v2":
            begin = max(x["end_sample"] for x in events) + int(sr * rng.uniform(0.25, 0.65))
            legal = [e for e in events if begin + e["sample_count"] < (duration - reserve) * sr]
            first = legal[int(rng.integers(len(legal)))] if legal else None
        else:
            first = min(events, key=lambda e: e["sample_count"])
            begin = max(x["end_sample"] for x in events) + int(sr * rng.uniform(0.25, 0.65))
        if first is not None and begin + first["sample_count"] < (duration - reserve) * sr:
            event = deepcopy(first)
            event.update(event_id=f"event_{len(events) + 1:03d}", start_sample=begin,
                         end_sample=begin + first["sample_count"], start_tick=begin * 3,
                         end_tick=(begin + first["sample_count"]) * 3)
            events.append(event)
    return sorted(events, key=lambda x: x["start_sample"]), bindings


def build_qa_episode_plan(
    *, room: Mapping[str, Any], request: Mapping[str, Any],
    source_registry: Mapping[str, Any], sounds: Sequence[Mapping[str, Any]],
    condition_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], RasterPathfinder]:
    if request.get("sampling_policy") == "conditioned_static_v2":
        from avengine.capture.qa_plan_adapters import load_planning_resources
        from avengine.rooms.conditioned_sampler import build_conditioned_plan
        space, mesh, layout = load_planning_resources(room, request)
        clock = clock_config(frame_count=int(request.get("frame_count", 240)),
                             frame_rate_hz=float(request.get("frame_rate_hz", 15)),
                             sample_rate_hz=int(request.get("sample_rate_hz", 16000)))
        effective_request = deepcopy(dict(request))
        camera_config = effective_request.setdefault("camera", {})
        camera_config.setdefault("resolution_hw", layout.get("capture_resolution_hw", [720, 1280]))
        plan = build_conditioned_plan(room=room, request=effective_request, source_registry=source_registry,
                sounds=sounds, space=space, mesh=mesh, clock=clock, condition_profile=condition_profile,
                region=request.get("planning_region_m"))
        plan["visual_lighting"] = deepcopy(layout.get("visual_lighting", {}))
        return plan, layout, space.pathfinder
    seed = int(request.get("seed", 0))
    sampling_policy = request.get("sampling_policy")
    if sampling_policy not in (None, "conditioned_static_v2"):
        raise QAPlanningError(f"unknown sampling_policy: {sampling_policy}")
    if sampling_policy == "conditioned_static_v2":
        camera_request = request.get("camera", {})
        motion = camera_request.get("motion", request.get("camera_motion", "static"))
        if motion != "static":
            raise QAPlanningError("conditioned_static_v2 requires a static camera")
    rng = np.random.default_rng(seed)
    qa_ids = list(request.get("qa_ids", [f"QA-{i:02d}" for i in range(1, 25)]))
    from avengine.qa.unified_catalog import get_requirements
    activity = str(request.get("activity", "auto"))
    if activity == "auto":
        requirements = [get_requirements(q) for q in qa_ids]
        requirements = [r.get("potential_requirements", r) for r in requirements]
        activity = ("walking" if any(
            r.get("motion") or r.get("entry") or r.get("after_sound")
            for r in requirements) else "standing")
    if activity not in {"walking", "standing", "seated"}:
        raise QAPlanningError("activity must be auto, walking, standing or seated")
    if request.get("camera_motion", "static") not in {"static", "pan", "follow_group"}:
        raise QAPlanningError("camera_motion must be static, pan or follow_group")
    layout = load_room_layout(room["manifest"], asset_root=room.get("asset_root"), require_seats=False)
    route = str(room.get("backend", layout.get("backend_route")))
    if route not in {"spear_unreal", "spear_usd"}:
        raise QAPlanningError(f"unsupported furnished-room route {route}; keep its native room-family executor")
    if not str(room.get("map_path", "")).startswith("/Game/"):
        raise QAPlanningError("the selected room needs its existing UE map reference")
    clock = clock_config(frame_count=int(request.get("frame_count", 240)),
                         frame_rate_hz=float(request.get("frame_rate_hz", 15)),
                         sample_rate_hz=int(request.get("sample_rate_hz", 16000)))
    pf, nav = build_room_navigation(
        layout, clearance_m=float(request.get("body_clearance_m", 0.38)),
        floor_height_m=room.get("floor_height_m"))
    selected_assets = list(request["source_asset_ids"])
    if len(selected_assets) < 2 or len(selected_assets) > 4 or len(set(selected_assets)) != len(selected_assets):
        raise QAPlanningError("select two to four distinct registered source assets")
    rng.shuffle(selected_assets)
    actors = ([source_declaration(source_registry, asset, f"source{i + 1}")
               for i, asset in enumerate(selected_assets)] if activity != "seated" else [])
    frames = []
    if activity == "seated":
        if not room.get("pose_bindings"):
            raise QAPlanningError("seated activity needs the existing calibrated pose bindings")
        bindings = read_json(room["pose_bindings"])
        values = bindings.get("assets", bindings) if isinstance(bindings, Mapping) else bindings
        by_asset = {str(value["asset_id"]): value for value in values}
        if any(asset not in by_asset for asset in selected_assets):
            raise QAPlanningError("selected seated assets are absent from the retained pose bindings")
        # Existing pose/seat joining owns the root reference and anatomical yaw.
        # A pool of calibrated poses does not fix those poses to old chair IDs.
        values = [deepcopy(by_asset[asset]) for asset in selected_assets]
        attributes = room.get("source_attributes", {})
        for i, value in enumerate(values):
            value["actor_id"] = f"source{i + 1}"
            value.pop("seat_affordance_id", None)
            value.pop("seat_id", None)
            value["realized_attributes"] = deepcopy(
                attributes.get(value["asset_id"], value.get("realized_attributes", {})))
        seats = build_seat_placements(layout, seat_count=len(values),
                                     actor_count=len(values), pose_bindings={"assets": values})
        placements = seats["actor_placements"]
        actors = [_actor_record(x) for x in placements]
        for actor, binding in zip(actors, values):
            actor["realized_attributes"] = deepcopy(binding.get("realized_attributes", {}))
        routes = {p["actor_id"]: np.repeat(np.asarray(p["root_position_habitat_m"])[None, :],
                                         clock["frame_count"], axis=0) for p in placements}
        route_record = {"actors": {x["actor_id"]: {"activity": "seated"} for x in actors},
                        "seat_layout": seats}
    else:
        routes, route_record = sample_activity_routes(
            pf, nav, actor_count=len(actors), frame_count=clock["frame_count"],
            fps=int(clock["frame_rate_hz"]), rng=rng, activity=activity)
    capability = room_capabilities(layout, nav, actors, sounds)
    matching = match_question_conditions(qa_ids, capability)
    if matching["status"] == "unsupported":
        raise QAPlanningError(f"room has no requested potential: {matching}")
    camera, cameras, camera_record = select_question_camera(
        layout, pf, routes, actors, rng=rng,
        camera_motion=motion if sampling_policy else str(request.get("camera_motion", "static")), qa_ids=qa_ids,
        camera_fov_deg=float(request.get("camera", {}).get("fov_deg", request.get("camera_fov_deg", 85.0))
                             if sampling_policy else request.get("camera_fov_deg", 85.0)),
        sampling_policy=sampling_policy)
    if room.get("exposure_bias_ev") is not None:
        for c in cameras:
            c["exposure_bias_ev"] = float(room["exposure_bias_ev"])
        camera["exposure_bias_ev"] = float(room["exposure_bias_ev"])
    headings = {}
    phases = {x["actor_id"]: float(rng.random()) for x in actors}
    hub = np.mean([r[0] for r in routes.values()], axis=0)
    for actor in actors:
        point = routes[actor["actor_id"]][0]
        delta = hub - point
        headings[actor["actor_id"]] = math.degrees(math.atan2(delta[2], delta[0]))
    for f in range(clock["frame_count"]):
        states = []
        for actor in actors:
            aid = actor["actor_id"]
            if activity == "seated":
                placement = next(p for p in placements if p["actor_id"] == aid)
                state = _actor_state(placement, frame_index=f, pts_ticks=f * clock["ticks_per_frame"])
            else:
                path = routes[aid]
                delta = path[min(f + 1, len(path) - 1)] - path[f]
                moving = float(np.linalg.norm(delta)) > 1e-5
                if moving:
                    headings[aid] = math.degrees(math.atan2(delta[2], delta[0]))
                yaw = (headings[aid] - float(actor["ue_anatomical_forward_yaw_deg"]) + 180) % 360 - 180
                action = actor["walking_action_id"] if moving else actor["idle_action_id"]
                phases[aid] = (phases[aid] + (1 / actor["walk_phase_period_frames"] if moving else 0)) % 1
                point = path[f].tolist()
                rotation = yaw_rotation_xyzw(-yaw)
                state = {
                    "actor_id": aid, "translation_m": point,
                    "translation_ue_cm": habitat_to_ue_cm(point),
                    "rotation_xyzw": list(rotation), "actor_yaw_ue_deg": yaw,
                    "root_transform": {"translation_m": point, "rotation_xyzw": list(rotation), "scale": [1, 1, 1]},
                    "action_id": action, "action_phase": phases[aid],
                    "action_time_ticks": f * clock["ticks_per_frame"],
                    "ue_animation": actor["animation_paths_by_action_id"][action],
                    "moving": moving, "frame_index": f,
                }
            states.append(state)
        frames.append({"frame_index": f, "pts_ticks": f * clock["ticks_per_frame"],
                       "actor_states": states, "camera_state": cameras[f]})
    events, bindings = schedule_audio(
        actors, sounds, clock=clock, rng=rng,
        mode=str(request.get("audio_mode", "sequential")),
        silent_actor_count=int(request.get("silent_actor_count", 0)), sampling_policy=sampling_policy)
    plan = {
        "kind": "avengine_question_driven_episode", "status": "research_candidate",
        "renderer_backend": "spear_unreal_native",
        "episode_id": str(request["episode_id"]), "seed": seed, "clock": clock,
        "scene": {"scene_id": layout["scene_id"], "room_id": layout["room_id"],
                  "map_path": room["map_path"], "backend": route},
        "request": deepcopy(dict(request)), "room_capabilities": capability,
        "question_condition_match": matching, "activity_plan": route_record,
        "camera_condition_sampling": camera_record, "audio_events": events,
        "voice_bindings": bindings,
        "visual_lighting": deepcopy(layout.get("visual_lighting", {})),
        "visual_plan": {
            "backend_role": "production_visual", "camera": camera, "actors": actors,
            "frames": frames, "render": {"frame_count": clock["frame_count"],
                                        "fps_num": clock["frame_rate_hz"], "fps_den": 1,
                                        "ticks_per_frame": clock["ticks_per_frame"]},
            "authority": {"actor_state": "avengine_question_condition_planner",
                          "camera_listener": "avengine_question_condition_planner",
                          "backend_may_replan": False},
        },
        "resources": deepcopy(dict(room)),
        "evidence_status": {"native_visual": "not_run", "native_audio": "not_run",
                            "qa_validity": "not_run", "model_evaluation": "not_run"},
        "qualification_claim": False, "formal_dataset_registration_authorized": False,
    }
    return plan, layout, pf
