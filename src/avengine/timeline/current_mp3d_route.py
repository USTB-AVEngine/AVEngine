"""Author a current MP3D two-Beagle research route from external inputs.

This route author is deliberately separate from the retained M2 Blender request
and from formal M2/M5 evidence.  It reads the qualified Beagle package and its
75-state request as action/timing source data, then creates one new M2-compatible
primary request for the current MP3D sample room.  M5 current-visual uses that
primary request to instantiate two copies of the same Beagle asset with its
existing fixed instance offsets.  The plain accompanying explanation records
both resulting skin-root paths without adding a schema, digest contract,
baseline, or formal gate.
"""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from avengine.backends.rlr.sdk import ExternalRlrSdkError, require_outside_git_checkout
from avengine.camera_pose import CameraPoseError, apply_camera_listener_pose
from avengine.contracts.json_io import sha256_file, write_json
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m1.contracts import (
    ValidatedM1Inputs,
    load_and_validate_inputs as load_m1_inputs,
    validate_loaded_scene_asset_graph,
    validate_scene_asset_graph,
)
from avengine.m1.habitat_capture import prepare_installed_habitat_runtime
from avengine.assets.contracts import (
    ValidatedM2Inputs,
    compute_applied_state_hash,
    load_and_validate_inputs as load_m2_inputs,
    validate_capture_request,
)
from avengine.assets.habitat_capture import (
    _apply_root_with_habitat,
    compile_frame_applications,
    load_runtime_asset_bundle,
    quaternion_xyzw_to_matrix,
)
from avengine.assets.timeline import (
    FRAME_COUNT,
    IDLE_LEAD_FRAME_COUNT,
    IDLE_TAIL_FRAME_COUNT,
    WALK_FRAME_COUNT,
)
from avengine.timeline.current_visual import (
    CurrentVisualError,
    CURRENT_ACTOR_IDS,
    CURRENT_ACTOR_OFFSETS_M,
    CURRENT_SEMANTIC_IDS,
    _current_mp3d_room_error,
    _instantiate_semantic_actor,
    _make_current_configuration,
    _require_no_actor_semantic_collision,
    _resolve_external_scene,
)
from avengine.capture.mp3d_capture import MP3DCaptureError, _pathfinder_path_record
from avengine.routes.geometry import (
    M6XGeometryError,
    build_runtime_obstacle_map,
    evaluate_source_center_gate,
)
from avengine.routes.room_feasibility import RoomFeasibilityCompiler, RoomFeasibilityError
from avengine.routes.trajectory import M6XTrajectoryError, resample_polyline_by_arc_length


EXTERNAL_REVIEW_ROOT = Path("/data/avengine_external/review")
CURRENT_MP3D_ROOM_MANIFEST_RELATIVE = Path(
    "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
)
CURRENT_MP3D_M1_REQUEST_RELATIVE = Path(
    "examples/m1/requests/habitat_mp3d_example.json"
)
_CURRENT_VISUAL_OFFSETS = np.asarray(CURRENT_ACTOR_OFFSETS_M, dtype=np.float64)
_MAXIMUM_PATH_ATTEMPTS = 4096
_MAXIMUM_ENDPOINT_CANDIDATES = 512


class CurrentMP3DRouteError(RuntimeError):
    """The explicit current-MP3D research route cannot be authored safely."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require_external_path(
    value: str | Path,
    *,
    owner: str,
    directory: bool,
) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raise CurrentMP3DRouteError(f"{owner} must be an absolute path")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise CurrentMP3DRouteError(
            f"{owner} cannot be resolved: {raw}: {exc}"
        ) from exc
    if raw != resolved:
        raise CurrentMP3DRouteError(
            f"{owner} must use its canonical path without a symlink hop: {raw}"
        )
    if directory:
        if not resolved.is_dir():
            raise CurrentMP3DRouteError(f"{owner} is not a directory: {resolved}")
    elif not resolved.is_file() or resolved.is_symlink():
        raise CurrentMP3DRouteError(f"{owner} is not a regular file: {resolved}")
    try:
        return require_outside_git_checkout(resolved, owner=owner)
    except (ExternalRlrSdkError, OSError, RuntimeError) as exc:
        raise CurrentMP3DRouteError(str(exc)) from exc


def _fresh_external_output(path: str | Path) -> Path:
    root = EXTERNAL_REVIEW_ROOT.resolve(strict=True)
    output = Path(path)
    if not output.is_absolute():
        raise CurrentMP3DRouteError("--output must be an absolute path")
    if output.parent != root:
        raise CurrentMP3DRouteError(
            f"--output must be an immediate fresh child of {root}: {output}"
        )
    if os.path.lexists(output):
        raise CurrentMP3DRouteError(f"refusing to replace route output: {output}")
    return output


def _current_room_inputs() -> ValidatedM1Inputs:
    root = _repository_root()
    room = root / CURRENT_MP3D_ROOM_MANIFEST_RELATIVE
    request = root / CURRENT_MP3D_M1_REQUEST_RELATIVE
    try:
        inputs = load_m1_inputs(room, request)
    except (OSError, ValueError) as exc:
        raise CurrentMP3DRouteError(
            f"current MP3D room inputs are invalid: {exc}"
        ) from exc
    reason = _current_mp3d_room_error(inputs)
    if reason is not None:
        raise CurrentMP3DRouteError(reason)
    return inputs


def _source_action_distance(request: Mapping[str, Any]) -> float:
    states = request.get("states")
    if not isinstance(states, list) or len(states) != FRAME_COUNT:
        raise CurrentMP3DRouteError("source M2 request must contain exactly 75 states")
    start = IDLE_LEAD_FRAME_COUNT
    end = start + WALK_FRAME_COUNT
    try:
        points = np.asarray(
            [state["root_transform"]["translation_m"] for state in states[start:end]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CurrentMP3DRouteError("source M2 walk root path is invalid") from exc
    if points.shape != (WALK_FRAME_COUNT, 3) or not np.all(np.isfinite(points)):
        raise CurrentMP3DRouteError("source M2 walk root path is invalid")
    distance = float(np.linalg.norm(np.diff(points[:, (0, 2)], axis=0), axis=1).sum())
    if not math.isfinite(distance) or distance <= 1.0e-6:
        raise CurrentMP3DRouteError(
            "source M2 walk path has no positive horizontal distance"
        )
    return distance


def _timing_projection(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    states = request["states"]
    return [
        {
            "frame_index": int(state["frame_index"]),
            "pts_ticks": int(state["pts_ticks"]),
            "action_id": str(state["action_id"]),
            "action_time_ticks": int(state["action_time_ticks"]),
        }
        for state in states
    ]


def _camera_frustum_predicate(
    camera_request: Mapping[str, Any],
) -> Callable[[np.ndarray], bool]:
    """Return a conservative geometric prefilter for fixed M1 view0.

    This controls route selection only. The current-visual renderer retains
    semantic-pixel readback as its final runtime visibility authority.
    """

    rig = camera_request["primary_camera_rig"]
    calibration = rig["shared_calibration"]
    origin = np.asarray(rig["world_from_rig"]["translation_m"], dtype=np.float64)
    rotation = quaternion_xyzw_to_matrix(rig["world_from_rig"]["rotation_xyzw"])
    forward = rotation @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    right = rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    up = rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    height, width = calibration["resolution_hw"]
    half_horizontal = math.radians(float(calibration["hfov_degrees"]) * 0.5)
    half_vertical = math.atan(math.tan(half_horizontal) * float(height) / float(width))
    near = max(float(calibration["near_m"]), 2.5)
    far = min(float(calibration["far_m"]), 12.0)
    margin_m = 0.15

    def visible(point: np.ndarray) -> bool:
        value = np.asarray(point, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            return False
        delta = value - origin
        depth = float(np.dot(delta, forward))
        if not near <= depth <= far:
            return False
        horizontal = abs(float(np.dot(delta, right)))
        vertical = abs(float(np.dot(delta, up)))
        return horizontal + margin_m <= depth * math.tan(
            half_horizontal
        ) and vertical + margin_m <= depth * math.tan(half_vertical)

    return visible


def _path_with_m2_timing(walk_path: np.ndarray) -> np.ndarray:
    path = np.asarray(walk_path, dtype=np.float64)
    if path.shape != (WALK_FRAME_COUNT, 3) or not np.all(np.isfinite(path)):
        raise CurrentMP3DRouteError("native walk path does not have 45 finite points")
    result = np.concatenate(
        (
            np.repeat(path[:1], IDLE_LEAD_FRAME_COUNT, axis=0),
            path,
            np.repeat(path[-1:], IDLE_TAIL_FRAME_COUNT, axis=0),
        ),
        axis=0,
    )
    if result.shape != (FRAME_COUNT, 3):
        raise AssertionError("M2 timing expansion changed the 75-frame contract")
    return np.ascontiguousarray(result)


def _point_has_offset_clearance(
    pathfinder: Any,
    point: np.ndarray,
    *,
    maximum_snap_error_m: float,
    maximum_y_delta_m: float,
    minimum_navmesh_clearance_m: float,
    visible_from_current_camera: Callable[[np.ndarray], bool],
) -> bool:
    for offset in np.concatenate(
        (np.zeros((1, 3), dtype=np.float64), _CURRENT_VISUAL_OFFSETS),
        axis=0,
    ):
        candidate = np.asarray(point + offset, dtype=np.float64)
        if not visible_from_current_camera(candidate):
            return False
        if not bool(pathfinder.is_navigable(candidate, maximum_y_delta_m)):
            return False
        snapped = np.asarray(pathfinder.snap_point(candidate), dtype=np.float64)
        if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
            return False
        if float(np.linalg.norm(snapped - candidate)) > maximum_snap_error_m:
            return False
        clearance = float(pathfinder.distance_to_closest_obstacle(snapped, 10.0))
        if not math.isfinite(clearance) or clearance < minimum_navmesh_clearance_m:
            return False
    return True


def _select_base_skin_path(
    *,
    habitat_sim: Any,
    pathfinder: Any,
    region: Any,
    seed: int,
    target_distance_m: float,
    distance_tolerance_m: float,
    maximum_snap_error_m: float,
    maximum_y_delta_m: float,
    minimum_navmesh_clearance_m: float,
    visible_from_current_camera: Callable[[np.ndarray], bool],
) -> np.ndarray:
    points = np.asarray(region.sample_points_m(), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise CurrentMP3DRouteError(
            "current MP3D feasibility region has too few samples"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    eligible = [
        int(index)
        for index in order
        if _point_has_offset_clearance(
            pathfinder,
            points[int(index)],
            maximum_snap_error_m=maximum_snap_error_m,
            maximum_y_delta_m=maximum_y_delta_m,
            minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
            visible_from_current_camera=visible_from_current_camera,
        )
    ]
    eligible = eligible[:_MAXIMUM_ENDPOINT_CANDIDATES]
    if len(eligible) < 2:
        raise CurrentMP3DRouteError(
            "current MP3D sample has fewer than two offset-safe native pathfinder endpoints"
        )
    minimum_distance = target_distance_m - distance_tolerance_m
    maximum_distance = target_distance_m + distance_tolerance_m
    attempts = 0
    for start_index in eligible:
        start = points[start_index]
        for end_index in eligible:
            if start_index == end_index:
                continue
            end = points[end_index]
            chord = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
            if chord > maximum_distance or chord < minimum_distance * 0.5:
                continue
            query = habitat_sim.ShortestPath()
            query.requested_start = np.asarray(
                pathfinder.snap_point(start), dtype=np.float64
            )
            query.requested_end = np.asarray(
                pathfinder.snap_point(end), dtype=np.float64
            )
            attempts += 1
            if attempts > _MAXIMUM_PATH_ATTEMPTS:
                break
            if not bool(pathfinder.find_path(query)):
                continue
            distance = float(query.geodesic_distance)
            if (
                not math.isfinite(distance)
                or not minimum_distance <= distance <= maximum_distance
            ):
                continue
            try:
                walk = resample_polyline_by_arc_length(
                    np.asarray(query.points, dtype=np.float64),
                    WALK_FRAME_COUNT,
                    owner="native current MP3D ShortestPath",
                )
            except M6XTrajectoryError:
                continue
            base = _path_with_m2_timing(walk)
            if all(
                _point_has_offset_clearance(
                    pathfinder,
                    point,
                    maximum_snap_error_m=maximum_snap_error_m,
                    maximum_y_delta_m=maximum_y_delta_m,
                    minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
                    visible_from_current_camera=visible_from_current_camera,
                )
                for point in base
            ):
                return base
        if attempts > _MAXIMUM_PATH_ATTEMPTS:
            break
    raise CurrentMP3DRouteError(
        "native PathFinder found no deterministic offset-safe two-Beagle route "
        f"within {target_distance_m:.3f}±{distance_tolerance_m:.3f} m"
    )


def _pathfinder_record(
    pathfinder: Any,
    path: np.ndarray,
    *,
    owner: str,
    maximum_snap_error_m: float,
    maximum_y_delta_m: float,
    maximum_step_endpoint_error_m: float,
) -> dict[str, Any]:
    try:
        record = _pathfinder_path_record(
            pathfinder,
            path,
            owner=owner,
            maximum_snap_error_m=maximum_snap_error_m,
            maximum_y_delta_m=maximum_y_delta_m,
            maximum_step_endpoint_error_m=maximum_step_endpoint_error_m,
            expected_frame_count=FRAME_COUNT,
            include_trajectory_sha256=False,
        )
    except MP3DCaptureError as exc:
        raise CurrentMP3DRouteError(str(exc)) from exc
    return dict(record)


def _actor_root_path(
    source_frames: Sequence[Any], base_skin_path: np.ndarray
) -> np.ndarray:
    # The navmesh floor path is the world-contact support plane, and the
    # world-contact rig anchors the actor root on that plane; the skin root
    # rides above it through actor_from_skin_root. Subtracting the per-frame
    # skin-to-actor lift here sank the whole actor below the floor.
    roots: list[np.ndarray] = []
    for index, frame in enumerate(source_frames):
        skin = np.asarray(frame.world_from_skin_root, dtype=np.float64)
        actor = np.asarray(frame.world_from_actor, dtype=np.float64)
        if skin.shape != (4, 4) or actor.shape != (4, 4):
            raise CurrentMP3DRouteError("source M2 frame matrix is invalid")
        roots.append(np.asarray(base_skin_path[index], dtype=np.float64))
    result = np.asarray(roots, dtype=np.float64)
    if result.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(result)):
        raise CurrentMP3DRouteError("derived M2 actor-root path is invalid")
    return result


def _new_primary_request(
    *,
    source_inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    source_frames: Sequence[Any],
    base_skin_path: np.ndarray,
) -> dict[str, Any]:
    request = deepcopy(source_inputs.request)
    request["request_id"] = (
        f"{source_inputs.request['request_id']}_current_mp3d_two_beagle_research"
    )
    request["room_id"] = room_inputs.room["room_id"]
    request["seed"] = room_inputs.request["seed"]
    root_path = _actor_root_path(source_frames, base_skin_path)
    manifest_sha256 = sha256_file(source_inputs.asset_path)
    for index, state in enumerate(request["states"]):
        state["root_transform"]["translation_m"] = root_path[index].tolist()
        state["applied_state_hash"] = compute_applied_state_hash(
            source_inputs.asset,
            state,
            asset_manifest_sha256=manifest_sha256,
        )
    errors = validate_capture_request(
        request,
        asset=source_inputs.asset,
        asset_manifest_sha256=manifest_sha256,
    )
    if errors:
        raise CurrentMP3DRouteError(
            "generated M2-compatible request is invalid: " + "; ".join(errors)
        )
    return request


def _current_visual_paths(base_skin_path: np.ndarray) -> dict[str, np.ndarray]:
    return {
        actor_id: np.ascontiguousarray(base_skin_path + _CURRENT_VISUAL_OFFSETS[index])
        for index, actor_id in enumerate(CURRENT_ACTOR_IDS)
    }


def _camera_yaw_toward(
    camera_position_m: np.ndarray, target_position_m: np.ndarray
) -> float:
    delta = np.asarray(target_position_m, dtype=np.float64) - np.asarray(
        camera_position_m, dtype=np.float64
    )
    horizontal = delta[[0, 2]]
    if (
        not np.all(np.isfinite(horizontal))
        or float(np.linalg.norm(horizontal)) <= 1.0e-6
    ):
        raise CurrentMP3DRouteError("research camera cannot share the route target XZ")
    return math.degrees(math.atan2(-float(horizontal[0]), -float(horizontal[1])))


def _camera_floor_navigation_record(
    pathfinder: Any,
    floor_point: np.ndarray,
    *,
    required_island_id: int,
    maximum_snap_error_m: float,
    maximum_y_delta_m: float,
    minimum_navmesh_clearance_m: float,
) -> dict[str, Any] | None:
    """Directly prove one selected research camera floor point on the live navmesh."""

    requested = np.asarray(floor_point, dtype=np.float64)
    if requested.shape != (3,) or not np.all(np.isfinite(requested)):
        raise CurrentMP3DRouteError(
            "research camera floor candidate is not finite vec3"
        )
    if not bool(pathfinder.is_navigable(requested, maximum_y_delta_m)):
        return None
    snapped = np.asarray(pathfinder.snap_point(requested), dtype=np.float64)
    if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
        raise CurrentMP3DRouteError("PathFinder returned invalid research camera snap")
    snap_error = float(np.linalg.norm(snapped - requested))
    if snap_error > maximum_snap_error_m:
        return None
    clearance = float(pathfinder.distance_to_closest_obstacle(snapped, 10.0))
    if not math.isfinite(clearance) or clearance < minimum_navmesh_clearance_m:
        return None
    island_id = int(pathfinder.get_island(snapped))
    if island_id != required_island_id:
        return None
    return {
        "requested_floor_point_m": requested.tolist(),
        "snapped_floor_point_m": snapped.tolist(),
        "navigable": True,
        "snap_error_m": snap_error,
        "maximum_snap_error_m": maximum_snap_error_m,
        "navmesh_clearance_m": clearance,
        "minimum_navmesh_clearance_m": minimum_navmesh_clearance_m,
        "island_id": island_id,
    }


def _research_camera_candidates(
    *,
    room_inputs: ValidatedM1Inputs,
    region: Any,
    pathfinder: Any,
    visual_paths: Mapping[str, np.ndarray],
    required_island_id: int,
    seed: int,
    maximum_snap_error_m: float,
    maximum_y_delta_m: float,
    minimum_navmesh_clearance_m: float,
    camera_selection: str = "framing",
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    """Return deterministic static M1 requests that frame both visual actors.

    ``framing`` prefers a comfortable framing distance. ``lateral_sweep``
    prefers the camera whose view maximizes the actors' azimuth sweep, so a
    walking route crosses the field of view and stays audible as motion.
    """

    if camera_selection not in ("framing", "lateral_sweep"):
        raise CurrentMP3DRouteError(
            f"unknown camera_selection: {camera_selection!r}"
        )

    route_points = np.concatenate(
        [
            np.asarray(visual_paths[actor_id], dtype=np.float64)
            for actor_id in CURRENT_ACTOR_IDS
        ],
        axis=0,
    )
    target = np.mean(route_points, axis=0)
    target[1] += 0.35  # conservative Beagle body-height projection probe
    navigation = room_inputs.room["navigation"]
    camera_height = float(navigation["agent_height_m"])
    candidates: list[
        tuple[tuple[float, float, float], dict[str, Any], dict[str, Any]]
    ] = []
    points = np.asarray(region.sample_points_m(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index in rng.permutation(len(points)):
        floor_point = np.asarray(points[int(index)], dtype=np.float64)
        floor_navigation = _camera_floor_navigation_record(
            pathfinder,
            floor_point,
            required_island_id=required_island_id,
            maximum_snap_error_m=maximum_snap_error_m,
            maximum_y_delta_m=maximum_y_delta_m,
            minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
        )
        if floor_navigation is None:
            continue
        camera_position = np.asarray(
            floor_navigation["snapped_floor_point_m"], dtype=np.float64
        )
        camera_position[1] += camera_height
        try:
            yaw = _camera_yaw_toward(camera_position, target)
            request = apply_camera_listener_pose(
                room_inputs.request,
                request_id=(
                    f"{room_inputs.request['request_id']}_"
                    f"current_mp3d_two_beagle_research_{seed}"
                ),
                position_m=camera_position.tolist(),
                yaw_deg=yaw,
            )
        except (CameraPoseError, KeyError, TypeError, ValueError) as exc:
            raise CurrentMP3DRouteError(
                f"could not author research M1 camera rig: {exc}"
            ) from exc
        visible = _camera_frustum_predicate(request)
        if not all(
            visible(np.asarray(point, dtype=np.float64) + np.asarray([0.0, 0.35, 0.0]))
            for point in route_points
        ):
            continue
        distance = float(np.linalg.norm(camera_position[[0, 2]] - target[[0, 2]]))
        yaw_radians = math.radians(yaw)
        cos_yaw, sin_yaw = math.cos(yaw_radians), math.sin(yaw_radians)
        lateral_sweep_degrees = 0.0
        for actor_id in CURRENT_ACTOR_IDS:
            relative = (
                np.asarray(visual_paths[actor_id], dtype=np.float64)
                - camera_position
            )
            local_x = cos_yaw * relative[:, 0] - sin_yaw * relative[:, 2]
            local_z = sin_yaw * relative[:, 0] + cos_yaw * relative[:, 2]
            azimuth = np.degrees(
                np.unwrap(np.arctan2(local_x, -local_z))
            )
            lateral_sweep_degrees = max(
                lateral_sweep_degrees, float(azimuth.max() - azimuth.min())
            )
        if camera_selection == "lateral_sweep":
            score = (
                -round(lateral_sweep_degrees, 1),
                abs(distance - 4.5),
                float(camera_position[0]),
                float(camera_position[2]),
            )
        else:
            score = (
                abs(distance - 4.5),
                float(camera_position[0]),
                float(camera_position[2]),
            )
        candidates.append(
            (
                score,
                request,
                {
                    "camera_floor_navigation": floor_navigation,
                    "camera_position_m": camera_position.tolist(),
                    "camera_yaw_degrees": yaw,
                    "camera_island_id": required_island_id,
                    "analytic_frustum_probe_height_m": 0.35,
                    "route_target_m": target.tolist(),
                    "camera_selection": camera_selection,
                    "lateral_sweep_degrees": round(lateral_sweep_degrees, 2),
                },
            )
        )
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise CurrentMP3DRouteError(
            "native PathFinder found no same-island static M1 camera candidate that "
            "frames both two-Beagle paths"
        )
    return tuple((request, record) for _score, request, record in candidates[:64])


def _camera_agent_state(
    camera_request: Mapping[str, Any], habitat_sim: Any, qt: Any
) -> Any:
    rig = camera_request["primary_camera_rig"]
    transform = rig["world_from_rig"]
    position = np.asarray(transform["translation_m"], dtype=np.float64)
    qx, qy, qz, qw = normalized_quaternion_xyzw(transform["rotation_xyzw"])
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = qt.quaternion(qw, qx, qy, qz)
    return state


def _select_semantically_visible_camera(
    simulator: Any,
    *,
    candidates: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    bundle: Any,
    frames: Sequence[Any],
    habitat_sim: Any,
    qt: Any,
    mn: Any,
    modality_to_uuid: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use one native scene/actor setup to select an all-frame visible camera."""

    if not candidates:
        raise CurrentMP3DRouteError("semantic camera selection has no candidates")
    semantic_uuid = modality_to_uuid["semantic"]
    semantic_sensor = simulator.sensors[semantic_uuid]
    initial_world_time = float(simulator.get_world_time())
    agent = simulator.initialize_agent(
        0,
        _camera_agent_state(candidates[0][0], habitat_sim, qt),
    )

    # Every no-actor candidate preflight occurs before template registration, so
    # scene pixels cannot be mistaken for actor semantic IDs.
    no_actor_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for request, record in candidates:
        state = _camera_agent_state(request, habitat_sim, qt)
        agent.set_state(state, reset_sensors=False, infer_sensor_states=True)
        observation = simulator.render_sensors([semantic_sensor])
        image = np.asarray(observation[semantic_uuid])
        try:
            _require_no_actor_semantic_collision(image)
        except CurrentVisualError:
            pass
        else:
            no_actor_candidates.append((request, record))
        if float(simulator.get_world_time()) != initial_world_time:
            raise CurrentMP3DRouteError(
                "semantic camera preflight advanced Habitat world time"
            )
    if not no_actor_candidates:
        raise CurrentMP3DRouteError(
            "every analytically framed research camera sees a pre-existing actor semantic ID"
        )

    manager = simulator.metadata_mediator.ao_template_manager
    config_path = bundle.paths_by_role["habitat_ao_config"]
    loaded_ids = manager.load_configs(str(config_path))
    handle_prefix = config_path.stem.removesuffix(".ao_config")
    base_handles = manager.get_template_handles(handle_prefix)
    if len(loaded_ids) != 1 or len(base_handles) != 1:
        raise CurrentMP3DRouteError(
            "semantic preflight expected one source AO template"
        )
    actors: list[Any] = []
    bindings: list[Any] = []
    for actor_index, semantic_id in enumerate(CURRENT_SEMANTIC_IDS):
        actor, binding = _instantiate_semantic_actor(
            simulator,
            bundle=bundle,
            habitat_sim=habitat_sim,
            base_handle=base_handles[0],
            semantic_id=semantic_id,
            actor_index=actor_index,
        )
        actors.append(actor)
        bindings.append(binding)

    for candidate_index, (request, record) in enumerate(no_actor_candidates):
        state = _camera_agent_state(request, habitat_sim, qt)
        counts: list[tuple[int, int]] = []
        visible = True
        for frame in frames:
            agent.set_state(state, reset_sensors=False, infer_sensor_states=True)
            for actor_index, (actor, binding) in enumerate(
                zip(actors, bindings, strict=True)
            ):
                skin_root = np.asarray(
                    frame.world_from_skin_root, dtype=np.float64
                ).copy()
                skin_root[:3, 3] += _CURRENT_VISUAL_OFFSETS[actor_index]
                joints = np.asarray(
                    binding.map_pose(frame.joint_rotations_xyzw), dtype=np.float64
                )
                _apply_root_with_habitat(actor, skin_root, qt=qt, mn=mn)
                actor.joint_positions = joints.copy()
            observation = simulator.render_sensors([semantic_sensor])
            image = np.asarray(observation[semantic_uuid])
            frame_counts = tuple(
                int(np.count_nonzero(image == semantic_id))
                for semantic_id in CURRENT_SEMANTIC_IDS
            )
            if any(value == 0 for value in frame_counts):
                visible = False
                break
            counts.append(frame_counts)
            if float(simulator.get_world_time()) != initial_world_time:
                raise CurrentMP3DRouteError(
                    f"semantic route preflight advanced world time at frame {frame.frame_index}"
                )
        if visible and len(counts) == FRAME_COUNT:
            return request, {
                **record,
                "candidate_index": candidate_index,
                "semantic_visibility_preflight": {
                    "frame_count": FRAME_COUNT,
                    "semantic_ids": list(CURRENT_SEMANTIC_IDS),
                    "minimum_visibility_pixels": [
                        min(item[index] for item in counts)
                        for index in range(len(CURRENT_SEMANTIC_IDS))
                    ],
                    "observation_calls_per_frame": 1,
                    "physics_steps": 0,
                    "world_time_advanced": False,
                },
            }
    raise CurrentMP3DRouteError(
        "no analytically framed same-island M1 camera candidate passed native two-Beagle "
        "semantic visibility for all 75 frames"
    )


def _pair_separation(
    paths: Mapping[str, np.ndarray], minimum_required_m: float
) -> dict[str, Any]:
    actor0, actor1 = (
        np.asarray(paths[key], dtype=np.float64) for key in CURRENT_ACTOR_IDS
    )
    separations = np.linalg.norm(actor0[:, (0, 2)] - actor1[:, (0, 2)], axis=1)
    frame = int(np.argmin(separations))
    minimum = float(separations[frame])
    if minimum < minimum_required_m:
        raise CurrentMP3DRouteError(
            "two current-visual Beagle centers violate required separation: "
            f"{minimum:.6f} < {minimum_required_m:.6f} m"
        )
    return {
        "center_only": True,
        "full_body_collision_claim": False,
        "minimum_required_m": minimum_required_m,
        "minimum_observed_m": minimum,
        "minimum_frame_index": frame,
        "per_frame_m": separations.tolist(),
    }


def author_current_mp3d_two_beagle_route(
    *,
    source_animal_manifest_path: str | Path,
    source_m2_request_path: str | Path,
    runtime_prefix: str | Path,
    mp3d_root: str | Path,
    magnum_python_site: str | Path,
    output_directory: str | Path,
    seed: int = 20_260_820,
    camera_selection: str = "framing",
    distance_tolerance_m: float = 0.15,
    minimum_center_separation_m: float = 0.75,
) -> dict[str, Any]:
    """Write one fresh current-MP3D two-Beagle research route.

    The output primary request remains a normal, independently validated M2 v1
    request.  The second Beagle exists only in the plain research explanation
    because M2 v1 deliberately models one articulated asset per request; M5
    current-visual owns its two-instance realization.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CurrentMP3DRouteError("seed must be an integer")
    if (
        not isinstance(distance_tolerance_m, (int, float))
        or isinstance(distance_tolerance_m, bool)
        or float(distance_tolerance_m) <= 0.0
    ):
        raise CurrentMP3DRouteError(
            "distance_tolerance_m must be a positive finite number"
        )
    if (
        not isinstance(minimum_center_separation_m, (int, float))
        or isinstance(minimum_center_separation_m, bool)
        or float(minimum_center_separation_m) <= 0.0
    ):
        raise CurrentMP3DRouteError(
            "minimum_center_separation_m must be a positive finite number"
        )
    tolerance = float(distance_tolerance_m)
    separation = float(minimum_center_separation_m)
    if not math.isfinite(tolerance) or not math.isfinite(separation):
        raise CurrentMP3DRouteError("route thresholds must be finite")

    asset_path = _require_external_path(
        source_animal_manifest_path,
        owner="--source-animal-manifest",
        directory=False,
    )
    request_path = _require_external_path(
        source_m2_request_path,
        owner="--source-m2-request",
        directory=False,
    )
    prefix = _require_external_path(
        runtime_prefix, owner="--runtime-prefix", directory=True
    )
    data_root = _require_external_path(mp3d_root, owner="--mp3d-root", directory=True)
    magnum_site = _require_external_path(
        magnum_python_site,
        owner="--magnum-python-site",
        directory=True,
    )
    output = _fresh_external_output(output_directory)
    try:
        source_inputs = load_m2_inputs(asset_path, request_path)
    except (OSError, ValueError) as exc:
        raise CurrentMP3DRouteError(
            f"source Beagle M2 inputs are invalid: {exc}"
        ) from exc
    room_inputs = _current_room_inputs()
    source_distance = _source_action_distance(source_inputs.request)
    source_timing = _timing_projection(source_inputs.request)
    try:
        static_errors = validate_scene_asset_graph(
            room_inputs, None, mp3d_root=data_root
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CurrentMP3DRouteError(
            f"current MP3D static scene validation failed: {exc}"
        ) from exc
    if static_errors:
        raise CurrentMP3DRouteError(
            "current MP3D static scene validation failed: " + "; ".join(static_errors)
        )
    try:
        bundle = load_runtime_asset_bundle(source_inputs)
        source_frames = compile_frame_applications(source_inputs, bundle)
        runtime = prepare_installed_habitat_runtime(
            runtime_prefix=prefix,
            mp3d_root=data_root,
            magnum_python_site=magnum_site,
            rlr_sdk_root=os.environ.get("AVENGINE_RLR_SDK_ROOT"),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CurrentMP3DRouteError(str(exc)) from exc
    if len(source_frames) != FRAME_COUNT:
        raise CurrentMP3DRouteError("source M2 application sequence is not 75 frames")
    if not bool(getattr(runtime.habitat_sim, "built_with_bullet", False)):
        raise CurrentMP3DRouteError(
            "current MP3D route author requires Bullet-enabled Habitat"
        )

    scene = _resolve_external_scene(room_inputs, runtime)
    configuration, modality_to_uuid = _make_current_configuration(
        room_inputs=room_inputs,
        installed_runtime=runtime,
        scene=scene,
        include_audio_sensor=False,
    )
    qa_views = room_inputs.request.get("qa_views")
    if not isinstance(qa_views, list) or len(qa_views) != 1:
        raise CurrentMP3DRouteError(
            "current MP3D M1 request must have one topdown QA view"
        )
    floor_height = float(qa_views[0]["height_m"])
    maximum_snap_error_m = 0.03
    maximum_y_delta_m = 0.25
    maximum_step_endpoint_error_m = 0.03
    minimum_navmesh_clearance_m = 0.10

    habitat_sim = runtime.habitat_sim
    mn = runtime.magnum
    qt = runtime.quaternion
    try:
        with habitat_sim.Simulator(configuration) as simulator:
            navmesh_loaded = bool(
                simulator.pathfinder.load_nav_mesh(str(scene["navmesh"]))
            )
            if not navmesh_loaded or not bool(simulator.pathfinder.is_loaded):
                raise CurrentMP3DRouteError(
                    "Habitat could not load current MP3D navmesh"
                )
            loaded_errors, _ = validate_loaded_scene_asset_graph(
                room_inputs,
                None,
                simulator,
                declared_navmesh_loaded=navmesh_loaded,
                mp3d_root=data_root,
            )
            if loaded_errors:
                raise CurrentMP3DRouteError(
                    "loaded current MP3D scene validation failed: "
                    + "; ".join(loaded_errors)
                )
            obstacle_map = build_runtime_obstacle_map(
                simulator.pathfinder,
                simulator.get_rigid_object_manager(),
                mn,
                floor_height_m=floor_height,
                meters_per_pixel=0.05,
            )
            region = RoomFeasibilityCompiler(obstacle_map).compile(
                source_center_height_m=floor_height,
                minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
                minimum_rigid_clearance_m=0.0,
                sample_spacing_m=0.15,
            )
            base_skin_path = _select_base_skin_path(
                habitat_sim=habitat_sim,
                pathfinder=simulator.pathfinder,
                region=region,
                seed=seed,
                target_distance_m=source_distance,
                distance_tolerance_m=tolerance,
                maximum_snap_error_m=maximum_snap_error_m,
                maximum_y_delta_m=maximum_y_delta_m,
                minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
                visible_from_current_camera=lambda _point: True,
            )
            visual_paths = _current_visual_paths(base_skin_path)
            path_records = {
                "m2_primary_skin_root": _pathfinder_record(
                    simulator.pathfinder,
                    base_skin_path,
                    owner="M2 primary skin-root route",
                    maximum_snap_error_m=maximum_snap_error_m,
                    maximum_y_delta_m=maximum_y_delta_m,
                    maximum_step_endpoint_error_m=maximum_step_endpoint_error_m,
                ),
                **{
                    actor_id: _pathfinder_record(
                        simulator.pathfinder,
                        path,
                        owner=f"current visual {actor_id}",
                        maximum_snap_error_m=maximum_snap_error_m,
                        maximum_y_delta_m=maximum_y_delta_m,
                        maximum_step_endpoint_error_m=maximum_step_endpoint_error_m,
                    )
                    for actor_id, path in visual_paths.items()
                },
            }
            islands = {record["island_id"] for record in path_records.values()}
            if len(islands) != 1:
                raise CurrentMP3DRouteError(
                    f"current two-Beagle paths do not share one navmesh island: {sorted(islands)}"
                )
            source_gate = evaluate_source_center_gate(
                simulator.pathfinder,
                obstacle_map,
                {
                    "m2_primary_skin_root": base_skin_path,
                    **visual_paths,
                },
                maximum_floor_snap_xz_m=maximum_snap_error_m,
                maximum_floor_y_delta_m=maximum_y_delta_m,
                minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
                minimum_rigid_clearance_m=0.0,
            )
            if source_gate["status"] != "pass":
                raise CurrentMP3DRouteError(
                    "native M6x source-center feasibility rejected route"
                )
            pair = _pair_separation(visual_paths, separation)
            request = _new_primary_request(
                source_inputs=source_inputs,
                room_inputs=room_inputs,
                source_frames=source_frames,
                base_skin_path=base_skin_path,
            )
            generated_inputs = ValidatedM2Inputs(
                asset_path=source_inputs.asset_path,
                request_path=Path("<in-memory-current-mp3d-request>"),
                asset=source_inputs.asset,
                request=request,
            )
            generated_frames = compile_frame_applications(generated_inputs, bundle)
            generated_actor = np.asarray(
                [
                    np.asarray(frame.world_from_actor, dtype=np.float64)[:3, 3]
                    for frame in generated_frames
                ],
                dtype=np.float64,
            )
            if not np.allclose(generated_actor, base_skin_path, rtol=0.0, atol=2.0e-6):
                raise CurrentMP3DRouteError(
                    "generated M2 request no longer realizes the selected "
                    "support-plane actor-root path"
                )
            camera_candidates = _research_camera_candidates(
                camera_selection=camera_selection,
                room_inputs=room_inputs,
                region=region,
                pathfinder=simulator.pathfinder,
                visual_paths=visual_paths,
                required_island_id=next(iter(islands)),
                seed=seed,
                maximum_snap_error_m=maximum_snap_error_m,
                maximum_y_delta_m=maximum_y_delta_m,
                minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
            )
            research_m1_request, camera_selection = _select_semantically_visible_camera(
                simulator,
                candidates=camera_candidates,
                bundle=bundle,
                frames=generated_frames,
                habitat_sim=habitat_sim,
                qt=qt,
                mn=mn,
                modality_to_uuid=modality_to_uuid,
            )
            obstacle_summary = obstacle_map.summary()
            region_summary = region.summary()
    except (
        CurrentMP3DRouteError,
        M6XGeometryError,
        RoomFeasibilityError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        if isinstance(exc, CurrentMP3DRouteError):
            raise
        raise CurrentMP3DRouteError(str(exc)) from exc

    output.mkdir()
    primary_path = output / "primary_m2_request.json"
    research_m1_path = output / "research_m1_request.json"
    explanation_path = output / "two_beagle_route_explanation.json"
    readme_path = output / "README.md"
    write_json(primary_path, request)
    write_json(research_m1_path, research_m1_request)
    try:
        generated_on_disk = load_m2_inputs(asset_path, primary_path)
    except (OSError, ValueError) as exc:
        raise CurrentMP3DRouteError(
            f"written primary M2 request failed ordinary validation: {exc}"
        ) from exc
    if generated_on_disk.request != request:
        raise CurrentMP3DRouteError("written primary M2 request differs after readback")
    try:
        written_m1 = load_m1_inputs(room_inputs.room_path, research_m1_path)
    except (OSError, ValueError) as exc:
        raise CurrentMP3DRouteError(
            f"written research M1 request failed ordinary validation: {exc}"
        ) from exc
    if written_m1.request != research_m1_request:
        raise CurrentMP3DRouteError(
            "written research M1 request differs after readback"
        )

    explanation: dict[str, Any] = {
        "research_only": True,
        "episode_counted": False,
        "formal_release_status": "not_run",
        "claim_boundary": (
            "current MP3D two-Beagle route authoring only; center-path navigation "
            "and separation do not claim body-volume collision, audio, RLR, a formal "
            "episode, or pre/post equivalence"
        ),
        "output": {
            "primary_m2_request": primary_path.name,
            "research_m1_request": research_m1_path.name,
        },
        "inputs": {
            "source_animal_manifest": str(asset_path),
            "source_m2_request": str(request_path),
            "current_room_manifest": str(room_inputs.room_path),
            "base_current_m1_request": str(room_inputs.request_path),
            "runtime_prefix": str(prefix),
            "mp3d_root": str(data_root),
            "magnum_python_site": str(magnum_site),
        },
        "provenance": {
            "legacy_blender_request_mutated": False,
            "source_request_room_id": source_inputs.request["room_id"],
            "new_request_room_id": request["room_id"],
            "source_action_timing_preserved": True,
            "source_action_timing": source_timing,
            "source_walk_horizontal_distance_m": source_distance,
            "selected_geodesic_distance_band_m": [
                source_distance - tolerance,
                source_distance + tolerance,
            ],
            "existing_m1_camera_rig_mutated": False,
            "new_research_m1_camera_request": research_m1_path.name,
            "m2_v1_single_actor_boundary": (
                "M2 v1 remains one articulated asset per request; M5 current-visual "
                "owns the second same-asset instance."
            ),
        },
        "current_visual_realization": {
            "actor_ids": list(CURRENT_ACTOR_IDS),
            "same_asset_for_both_instances": True,
            "instance_offsets_m": _CURRENT_VISUAL_OFFSETS.tolist(),
            "conservative_camera_frustum_prefilter": True,
            "renderer_semantic_visibility_is_final_authority": True,
            "primary_m2_skin_root_path_m": base_skin_path.tolist(),
            "research_m1_camera": camera_selection,
            "actor_skin_root_paths_m": {
                actor_id: path.tolist() for actor_id, path in visual_paths.items()
            },
        },
        "native_pathfinder": {
            "all_paths_share_single_island": True,
            "island_id": next(iter(islands)),
            "maximum_snap_error_m": maximum_snap_error_m,
            "maximum_y_delta_m": maximum_y_delta_m,
            "maximum_step_endpoint_error_m": maximum_step_endpoint_error_m,
            "paths": path_records,
        },
        "m6x_source_center_feasibility": source_gate,
        "m6x_runtime_obstacle_map": obstacle_summary,
        "m6x_feasible_region": region_summary,
        "two_beagle_center_separation": pair,
        "no_new_hash_or_gate": (
            "No new persistent hash, contract, baseline, or release gate was added; "
            "M2's existing required pose/applied-state fields were recomputed only "
            "because this is a new ordinary M2-compatible request."
        ),
    }
    write_json(explanation_path, explanation)
    readme_path.write_text(
        "# Current MP3D two-Beagle research route\n\n"
        "This directory is a fresh research-only route-authoring result. "
        "`primary_m2_request.json` is a new M2-compatible request and "
        "`research_m1_request.json` is a new static camera/listener request for the "
        "current MP3D sample; neither mutates or relabels an existing Blender input. "
        "`two_beagle_route_explanation.json` records the two same-asset paths, native "
        "PathFinder/frustum checks, and all-frame semantic preflight. No formal episode, "
        "audio/RLR, body-volume collision, or equivalence claim is made.\n",
        encoding="utf-8",
    )
    return {
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "output_directory": str(output),
        "primary_m2_request": str(primary_path),
        "research_m1_request": str(research_m1_path),
        "explanation": str(explanation_path),
        "frame_count": FRAME_COUNT,
        "actor_ids": list(CURRENT_ACTOR_IDS),
    }


__all__ = ["CurrentMP3DRouteError", "author_current_mp3d_two_beagle_route"]
