"""Plan a static furnished residential episode without starting UE/SPEAR.

The room manifest supplies static geometry and seat references.  AVEngine then
creates a target-independent camera candidate pool and a bounded seat layout.
Pose-agent bindings may be joined later; this tool never drops an actor root on
the seat surface and never invents a UE map path.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.rooms.furniture_layout import (  # noqa: E402
    DEFAULT_ACTOR_COUNT,
    DEFAULT_SEAT_COUNT,
    FurnitureLayoutError,
    authoring_to_habitat,
    build_seat_placements,
    camera_obstacle_bounds,
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


def _load_static_triangle_geometry(layout: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read an available visual GLB for CPU authored-ray scoring."""

    resources = layout.get("resources")
    visual = resources.get("visual_geometry") if isinstance(resources, Mapping) else None
    path = visual.get("resolved") if isinstance(visual, Mapping) else None
    if not isinstance(path, str) or not path:
        return None
    try:
        import numpy as np
        from avengine.acoustics.gltf import extract_triangle_scene

        scene = extract_triangle_scene(path)
        vertices = np.asarray(scene.vertices, dtype=np.float64)
        # Room GLBs are exported Y-up from Blender: GLB (X,Y,Z) maps to
        # authoring Blender (X,-Z,Y).
        authoring_vertices = np.column_stack(
            (vertices[:, 0], -vertices[:, 2], vertices[:, 1])
        )
        return {
            "vertices": authoring_vertices,
            "triangles": np.asarray(scene.triangles, dtype=np.int64),
            "source": path,
        }
    except (ImportError, OSError, ValueError, RuntimeError):
        return None


def _pose_bindings(
    path: str | Path | None,
    request_path: str | Path | None = None,
) -> Any:
    if path is None and request_path is None:
        return None
    value = (
        _load_json(Path(path).expanduser().resolve(), owner="pose bindings")
        if path is not None
        else {"assets": []}
    )
    if not isinstance(value, (Mapping, list)):
        raise FurnitureLayoutError("pose bindings JSON must be an object or list")
    if request_path is None:
        return value
    request = _load_json(
        Path(request_path).expanduser().resolve(), owner="pose binding request"
    )
    if not isinstance(request, Mapping) or not isinstance(request.get("assets"), list):
        raise FurnitureLayoutError("pose binding request must contain an assets list")
    request_by_asset = {
        str(item.get("asset_id")): item
        for item in request["assets"]
        if isinstance(item, Mapping) and item.get("asset_id")
    }
    if isinstance(value, Mapping) and isinstance(value.get("assets"), list):
        merged_assets = []
        for raw in value["assets"]:
            if not isinstance(raw, Mapping):
                raise FurnitureLayoutError("pose binding assets must be objects")
            asset_id = str(raw.get("asset_id"))
            merged = dict(request_by_asset.get(asset_id, {}))
            merged.update(raw)
            request_ref = request_by_asset.get(asset_id, {}).get("seat_reference")
            manifest_ref = raw.get("seat_reference")
            if isinstance(request_ref, Mapping) or isinstance(manifest_ref, Mapping):
                seat_ref = dict(request_ref or {})
                seat_ref.update(manifest_ref or {})
                merged["seat_reference"] = seat_ref
            merged_assets.append(merged)
        return {**dict(value), "assets": merged_assets}
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
        "skeletal_mesh_path": placement.get("skeletal_mesh_path"),
        "animation_paths_by_action_id": deepcopy(placement.get("animation_paths_by_action_id")),
        "idle_animation": placement.get("ue_animation"),
        "walking_animation": placement.get("ue_animation"),
        "ue_component_frame_delta": deepcopy(placement.get("ue_component_frame_delta")),
        "exact_runtime_binding": deepcopy(placement.get("exact_runtime_binding")),
        "actor_scale": placement.get("actor_scale", 1.0),
        "emitter_local_ue_cm": deepcopy(placement.get("emitter_local_ue_cm")),
        "ue_anatomical_forward_yaw_deg": placement.get("ue_anatomical_forward_yaw_deg"),
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
    # camera_pose itself imports rooms.contracts. Import at call time so
    # importing that public module does not recurse through rooms.__init__.
    from avengine.camera_pose import yaw_rotation_xyzw

    root_habitat = placement.get("root_position_habitat_m")
    rotation = placement.get("rotation_xyzw")
    root_transform = None
    translation_ue_cm = None
    actor_yaw_ue_deg = None
    actor_yaw_blender_deg = placement.get("pose_actor_yaw_blender_deg")
    if root_habitat is not None:
        seat_yaw = float(placement["seat_reference"]["facing_yaw_deg"])
        anatomical_yaw = placement.get("ue_anatomical_forward_yaw_deg")
        if anatomical_yaw is None:
            actor_yaw_ue_deg = seat_yaw
        else:
            # Seat theta is chair-to-table in authoring XY.  The imported
            # asset's anatomical UE forward yaw is applied; reference actor
            # yaw from the pose request is deliberately ignored.
            actor_yaw_ue_deg = (
                -seat_yaw - float(anatomical_yaw) + 180.0
            ) % 360.0 - 180.0
        if actor_yaw_blender_deg is None and actor_yaw_ue_deg is not None:
            # Authoring Blender +Z yaw maps to Habitat +Y yaw under the
            # room's (X,Z,-Y) basis; the UE actor yaw is its mirrored value.
            actor_yaw_blender_deg = -float(actor_yaw_ue_deg)
        if rotation is None and actor_yaw_blender_deg is not None:
            rotation = yaw_rotation_xyzw(float(actor_yaw_blender_deg))
        root_transform = {
            "translation_m": list(root_habitat),
            "rotation_xyzw": list(rotation or [0.0, 0.0, 0.0, 1.0]),
            "scale": [1.0, 1.0, 1.0],
        }
        translation_ue_cm = habitat_to_ue_cm(root_habitat)
    return {
        "actor_id": placement["actor_id"],
        "root_transform": root_transform,
        "translation_m": list(root_habitat) if root_habitat is not None else None,
        "translation_ue_cm": translation_ue_cm,
        "rotation_xyzw": list(rotation) if rotation is not None else None,
        "actor_yaw_ue_deg": actor_yaw_ue_deg,
        "actor_yaw_blender_deg": actor_yaw_blender_deg,
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

def _overview_target_bounds(layout: Mapping[str, Any]) -> list[dict[str, list[float]]]:
    """Build compact living/dining targets for a room-wide overview score."""

    assemblies = layout.get("furniture_assemblies")
    assembly_targets: list[Mapping[str, Any]] = []
    if isinstance(assemblies, list):
        for item in assemblies:
            if not isinstance(item, Mapping):
                continue
            text = f"{item.get('kind', '')} {item.get('object_id', '')}".lower()
            center = item.get("center_xy_m")
            if isinstance(center, Sequence) and len(center) == 2 and any(
                token in text for token in ("dining", "living", "sofa", "table")
            ):
                assembly_targets.append(item)
    targets: list[dict[str, list[float]]] = []
    if assembly_targets:
        for item in assembly_targets:
            x, y = [float(value) for value in item["center_xy_m"]]
            targets.append(
                {
                    "minimum_m": [x - 0.30, y - 0.30, 0.35],
                    "maximum_m": [x + 0.30, y + 0.30, 1.70],
                }
            )
        return targets

    preferred: list[Mapping[str, Any]] = []
    fallback: list[Mapping[str, Any]] = []
    for item in layout.get("objects", []):
        if str(item.get("navigation_role") or "ground_blocker") in {
            "walkable_surface", "walkable_floor_covering", "elevated_object"
        }:
            continue
        center = item.get("center_authoring_m")
        if not isinstance(center, Sequence) or len(center) != 3:
            continue
        text = f"{item.get('semantic_class', '')} {item.get('object_id', '')}".lower()
        (preferred if any(token in text for token in ("dining", "chair", "sofa", "cushion", "coffee", "table")) else fallback).append(item)
    source_items = preferred if len(preferred) >= 2 else fallback + preferred
    for item in source_items:
        x, y, z = [float(value) for value in item["center_authoring_m"]]
        targets.append(
            {
                "minimum_m": [x - 0.16, y - 0.16, max(0.25, min(z, 1.05) - 0.25)],
                "maximum_m": [x + 0.16, y + 0.16, min(1.75, max(z, 0.95) + 0.25)],
            }
        )
    if not targets:
        bounds = layout["geometry"]["bounds_xy_m"]
        x = (bounds[0] + bounds[2]) * 0.5
        y = (bounds[1] + bounds[3]) * 0.5
        targets.append({"minimum_m": [x - 0.16, y - 0.16, 0.25], "maximum_m": [x + 0.16, y + 0.16, 1.75]})
    return targets

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
    overview_only: bool = False,
) -> dict[str, Any]:
    """Build one path-free plan from a normalized room layout."""

    clock = clock_config(
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        sample_rate_hz=sample_rate_hz,
    )
    camera_pool = generate_camera_candidates(
        layout,
        grid_step_m=grid_step_m,
        camera_height_m=camera_height_m,
    )
    if not camera_pool["candidates"]:
        raise FurnitureLayoutError("camera candidate generation returned no candidates")
    if overview_only:
        seat_layout = {
            "requested_seat_count": 0,
            "available_seat_count": len(layout.get("seats", [])),
            "selected_seat_ids": [],
            "selected_seats": [],
            "actor_placements": [],
            "placement_policy": "overview_no_actor_placement",
            "authoring_geometry_status": "candidate",
            "native_validation_status": "not_run",
            "claim_boundary": "overview excludes actors and seat placement",
        }
    else:
        seat_layout = build_seat_placements(
            layout,
            seat_count=seat_count,
            actor_count=actor_count,
            pose_bindings=pose_bindings,
        )
    placements = seat_layout["actor_placements"]
    bound_actor_positions = [
        item["root_position_authoring_m"]
        for item in placements
        if item.get("root_position_authoring_m") is not None
    ]
    target_bounds: list[dict[str, list[float]]] = []
    for placement in placements:
        seat_position = placement["seat_reference"]["position_authoring_m"]
        target_bounds.append(
            {
                "minimum_m": [
                    seat_position[0] - 0.35,
                    seat_position[1] - 0.35,
                    max(0.0, seat_position[2] - 0.15),
                ],
                "maximum_m": [
                    seat_position[0] + 0.35,
                    seat_position[1] + 0.35,
                    seat_position[2] + 1.20,
                ],
            }
        )
    obstacle_bounds = camera_obstacle_bounds(layout, camera_height_m=camera_height_m)
    scoring_target_bounds = (
        target_bounds
        if bound_actor_positions
        else (_overview_target_bounds(layout) if overview_only else None)
    )
    triangle_geometry = _load_static_triangle_geometry(layout) if (bound_actor_positions or overview_only) else None
    scored_camera_pool = score_camera_candidates(
        camera_pool,
        actor_positions_m=bound_actor_positions or None,
        target_bounds_m=scoring_target_bounds,
        obstacle_bounds_m=obstacle_bounds if (bound_actor_positions or overview_only) else None,
        room_bounds_xy_m=layout["geometry"].get("bounds_xy_m") if (bound_actor_positions or overview_only) else None,
        triangle_vertices_m=triangle_geometry["vertices"] if triangle_geometry else None,
        triangle_indices=triangle_geometry["triangles"] if triangle_geometry else None,
    )
    selected_source = (
        scored_camera_pool["candidates"][0]
        if bound_actor_positions or overview_only
        else camera_pool["candidates"][0]
    )
    selected_camera = _camera_for_runtime(selected_source)
    actors = [_actor_record(item) for item in placements]
    camera_selection = {
        "selection_mode": (
            "overview_geometry_only"
            if overview_only
            else (
                "post_actor_question_scoring"
                if bound_actor_positions
                else "geometry_first_pending_actor_join"
            )
        ),
        "actor_count": len(placements),
        "actor_positions_used": len(bound_actor_positions),
        "coverage_target": "all_bound_actor_seat_references",
        "selected_candidate_id": selected_source["candidate_id"],
        "target_los_status": "not_evaluated",
        "native_validation_status": "not_run",
        "clearance_basis": "authoring_geometry_grid_clearance",
        "target_count": selected_source.get("target_count", 0),
        "fully_framed_target_count": selected_source.get("fully_framed_target_count", 0),
        "target_frame_margin": selected_source.get("target_frame_margin"),
        "target_occluded_count": selected_source.get("target_occluded_count", 0),
        "target_geometry_visibility": selected_source.get("target_geometry_visibility", "not_evaluated"),
        "geometry_clearance_m": selected_source.get("geometry_clearance_m"),
        "static_triangle_geometry_used": triangle_geometry is not None,
        "static_triangle_geometry_source": triangle_geometry.get("source") if triangle_geometry else None,
    }
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
            "actor_state": "overview_none" if overview_only else "pose_binding_or_pending",
            "backend_may_replan": False,
        },
        "room": {
            "room_id": layout["room_id"],
            "scene_id": effective_scene_id,
            "authoring_geometry_status": layout["geometry"]["authoring_geometry_status"],
            "native_validation_status": "not_run",
        },
        "camera": selected_camera,
        "camera_candidates": deepcopy(
            scored_camera_pool["candidates"]
            if (bound_actor_positions or overview_only)
            else camera_pool["candidates"]
        ),
        "camera_generation": deepcopy(camera_pool["generation"]),
        "camera_selection": camera_selection,
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
        "camera_candidates": {
            **deepcopy(camera_pool),
            "post_join_selection": camera_selection,
            "post_join_scored_candidates": deepcopy(scored_camera_pool["candidates"]),
        },
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
            "overview_only": overview_only,
            "target_independent_candidates": True,
            "target_scoring": (
                "post_geometry_overview" if overview_only else "post_actor_question_join"
            ),
            "target_los": "not_evaluated",
            "native_spear_ue": "not_run",
            "action_engine": "not_created",
            "review_cameras_used": False,
        },
    }


def reuse_camera_from_plan(
    plan: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    """Reuse one already selected production camera without reusing actor state."""

    source = _load_json(source_path, owner="camera source plan")
    if not isinstance(source, Mapping):
        raise FurnitureLayoutError("camera source plan must be an object")
    source_visual = source.get("visual_plan")
    if not isinstance(source_visual, Mapping):
        raise FurnitureLayoutError("camera source plan has no visual_plan")
    source_camera = source_visual.get("camera")
    if not isinstance(source_camera, Mapping):
        raise FurnitureLayoutError("camera source plan has no visual_plan.camera")
    camera = deepcopy(dict(source_camera))
    for field in ("position_authoring_m", "position_habitat_m", "position_ue_cm"):
        value = camera.get(field)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in value
            )
        ):
            raise FurnitureLayoutError(f"camera source plan has invalid {field}")
    source_frames = source_visual.get("frames")
    if source_frames is not None:
        if not isinstance(source_frames, list) or not source_frames:
            raise FurnitureLayoutError("camera source plan frames must be a non-empty list")
        for frame in source_frames:
            state = frame.get("camera_state") if isinstance(frame, Mapping) else None
            if not isinstance(state, Mapping):
                raise FurnitureLayoutError("camera source plan frame has no camera_state")
            fixed_state = {key: value for key, value in state.items() if key != "frame_index"}
            if fixed_state != dict(source_camera):
                raise FurnitureLayoutError(
                    "camera source plan is dynamic; only fixed cameras can be reused"
                )
    plan["visual_plan"]["camera"] = camera
    for frame in plan["visual_plan"]["frames"]:
        frame["camera_state"] = {**deepcopy(camera), "frame_index": frame["frame_index"]}
    source_reuse = source.get("camera_reuse")
    plan["camera_reuse"] = {
        "source_plan_path": str(source_path),
        "source_candidate_id": camera.get("candidate_id"),
        "camera_pose_only": True,
        "source_camera_reuse": deepcopy(source_reuse) if isinstance(source_reuse, Mapping) else None,
        "claim_boundary": "camera pose reused; room metadata and actor states come from this fresh plan",
    }
    selection = {
        **plan["visual_plan"].get("camera_selection", {}),
        "selected_candidate_id": camera.get("candidate_id"),
        "source_plan_path": str(source_path),
    }
    if not plan.get("planning_boundary", {}).get("overview_only"):
        selection["selection_mode"] = "reused_existing_native_selected_camera"
    plan["visual_plan"]["camera_selection"] = selection
    plan["planning_boundary"]["camera_reused_from_existing_plan"] = True
    plan["planning_boundary"]["camera_source_plan"] = str(source_path)
    return plan


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def plan_furnished_residential_episode(
    *,
    room: Path,
    output: Path,
    activity: str,
    pose_bindings: Path | None = None,
    pose_request: Path | None = None,
    asset_root: Path | None = None,
    map_path: str | None = None,
    scene_id: str | None = None,
    seat_count: int = DEFAULT_SEAT_COUNT,
    actor_count: int = DEFAULT_ACTOR_COUNT,
    frame_count: int = 75,
    frame_rate_hz: float = 15.0,
    sample_rate_hz: int = 16_000,
    grid_step_m: float = 2.0,
    camera_height_m: float = 1.55,
    camera_source_plan: Path | None = None,
    overview_only: bool = False,
) -> dict[str, Any]:
    """Create a fresh static-seated plan from room semantics and pose bindings."""

    if activity != "seated":
        raise FurnitureLayoutError(
            "furnished residential planning currently supports activity='seated' only"
        )
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    layout = load_room_layout(room, asset_root=asset_root)
    bindings = _pose_bindings(pose_bindings, pose_request)
    plan = build_episode_plan(
        layout,
        seat_count=seat_count,
        actor_count=actor_count,
        pose_bindings=bindings,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        sample_rate_hz=sample_rate_hz,
        grid_step_m=grid_step_m,
        camera_height_m=camera_height_m,
        map_path=map_path,
        scene_id=scene_id,
        overview_only=overview_only,
    )
    if camera_source_plan is not None:
        plan = reuse_camera_from_plan(
            plan, camera_source_plan.expanduser().resolve()
        )
    output.mkdir(parents=True)
    _write_json(output / "room_layout.json", layout)
    _write_json(output / "camera_candidates.json", plan["camera_candidates"])
    _write_json(output / "seat_layout.json", plan["seat_layout"])
    _write_json(output / "episode_plan.json", plan)
    return plan
