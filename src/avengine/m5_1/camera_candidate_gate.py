"""Pure CPU admission gates for static camera/listener candidates.

The evaluator is scene agnostic.  A caller supplies a loaded PathFinder and a
ray-query callback from the same runtime scene.  Failed candidates retain
diagnostic evidence but deliberately do not expose a ``room_gate`` that could
be consumed by the framing solver.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Real
from typing import Any

import numpy as np

CAMERA_CANDIDATE_GATE_SCHEMA = "avengine_camera_candidate_gate_v1"
RUNTIME_PROVENANCE = "habitat_cpu_runtime"


class CameraCandidateGateError(ValueError):
    """A camera candidate or runtime query contract is invalid."""


def _finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CameraCandidateGateError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CameraCandidateGateError(f"{owner} must be a finite number")
    return 0.0 if result == 0.0 else result


def _vec3(value: Any, *, owner: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise CameraCandidateGateError(f"{owner} must contain three numbers")
    try:
        components = list(value)
    except TypeError as error:
        raise CameraCandidateGateError(f"{owner} must contain three numbers") from error
    if len(components) != 3:
        raise CameraCandidateGateError(f"{owner} must contain three numbers")
    return np.asarray(
        [
            _finite(item, owner=f"{owner}[{index}]")
            for index, item in enumerate(components)
        ],
        dtype=np.float64,
    )


def _nonnegative(value: Any, *, owner: str, positive: bool = False) -> float:
    result = _finite(value, owner=owner)
    if result < 0.0 or (positive and result <= 0.0):
        comparator = "positive" if positive else "nonnegative"
        raise CameraCandidateGateError(f"{owner} must be {comparator}")
    return result


def _identifier(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraCandidateGateError(f"{owner} must be a non-empty string")
    return value.strip()


def _status(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"status": "pass" if passed else "fail", **evidence}


def _runtime_bounds(value: Any, *, owner: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a Magnum Range3D or a two-vector bounds tuple."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise CameraCandidateGateError(f"{owner} must contain two corners")
        minimum_value, maximum_value = value
    else:
        minimum_value = getattr(value, "min", None)
        maximum_value = getattr(value, "max", None)
        if callable(minimum_value):
            minimum_value = minimum_value()
        if callable(maximum_value):
            maximum_value = maximum_value()
    minimum = _vec3(minimum_value, owner=f"{owner}.minimum")
    maximum = _vec3(maximum_value, owner=f"{owner}.maximum")
    if np.any(minimum >= maximum):
        raise CameraCandidateGateError(f"{owner} minimum must be smaller than maximum")
    return minimum, maximum


class HabitatRuntimeCameraProvider:
    """Bind camera admission queries to one loaded Habitat simulator.

    Every capability claim below is read from the simulator.  The caller only
    declares the scene identity it expects; it cannot supply navigation,
    physics, bounds, or raycast booleans.
    """

    def __init__(
        self,
        simulator: Any,
        habitat_sim: Any,
        mn: Any,
        declared_scene_id: Any,
        *,
        provider_id: Any = "habitat-runtime-camera-provider",
    ) -> None:
        self._simulator = simulator
        self._habitat_sim = habitat_sim
        self._mn = mn
        if isinstance(declared_scene_id, Mapping):
            expected_configured_scene = _identifier(
                declared_scene_id.get("configured_scene_id"),
                owner="declared configured_scene_id",
            )
            expected_loaded_scene = _identifier(
                declared_scene_id.get("loaded_scene_id"),
                owner="declared loaded_scene_id",
            )
            expected_dataset = _identifier(
                declared_scene_id.get("active_dataset"),
                owner="declared active_dataset",
            )
            expected_stage_surface = _identifier(
                declared_scene_id.get("stage_surface"),
                owner="declared stage_surface",
            )
        else:
            expected_scene = _identifier(declared_scene_id, owner="declared_scene_id")
            expected_configured_scene = expected_scene
            expected_loaded_scene = expected_scene
            expected_dataset = None
            expected_stage_surface = None
        owner_id = _identifier(provider_id, owner="provider_id")

        pathfinder = getattr(simulator, "pathfinder", None)
        if pathfinder is None or getattr(pathfinder, "is_loaded", None) is not True:
            raise CameraCandidateGateError("Habitat runtime pathfinder is not loaded")
        self.pathfinder = pathfinder

        config = getattr(simulator, "config", None)
        sim_cfg = getattr(config, "sim_cfg", None)
        if getattr(sim_cfg, "enable_physics", None) is not True:
            raise CameraCandidateGateError("Habitat runtime physics is not enabled")
        configured_scene = _identifier(
            getattr(sim_cfg, "scene_id", None), owner="configured Habitat scene_id"
        )
        loaded_scene = _identifier(
            getattr(simulator, "curr_scene_name", None),
            owner="loaded Habitat scene_id",
        )
        if (
            configured_scene != expected_configured_scene
            or loaded_scene != expected_loaded_scene
        ):
            raise CameraCandidateGateError(
                "configured or loaded Habitat scene identity differs from declaration"
            )

        cast_ray = getattr(simulator, "cast_ray", None)
        if not callable(cast_ray):
            raise CameraCandidateGateError("Habitat runtime cast_ray is unavailable")
        self._cast_ray = cast_ray

        library_query = getattr(simulator, "get_physics_simulation_library", None)
        if not callable(library_query):
            raise CameraCandidateGateError(
                "Habitat runtime physics-library readback is unavailable"
            )
        physics_library = _identifier(
            str(library_query()), owner="Habitat physics library"
        )
        library_key = physics_library.casefold().replace("-", "_").replace(" ", "_")
        if library_key in {"none", "no_physics", "nophysics"}:
            raise CameraCandidateGateError("Habitat runtime has no physics library")

        active_dataset = _identifier(
            str(getattr(simulator, "active_dataset", "")),
            owner="loaded Habitat dataset",
        )
        stage_query = getattr(simulator, "get_stage_initialization_template", None)
        if not callable(stage_query):
            raise CameraCandidateGateError(
                "Habitat runtime stage-template readback is unavailable"
            )
        stage = stage_query()
        stage_surface = _identifier(
            str(getattr(stage, "render_asset_fullpath", "")),
            owner="loaded Habitat stage surface",
        )
        if expected_dataset is not None and active_dataset != expected_dataset:
            raise CameraCandidateGateError(
                "loaded Habitat dataset differs from declaration"
            )
        if (
            expected_stage_surface is not None
            and stage_surface != expected_stage_surface
        ):
            raise CameraCandidateGateError(
                "loaded Habitat stage surface differs from declaration"
            )

        scene_bounds = getattr(simulator, "scene_aabb", None)
        if scene_bounds is None:
            raise CameraCandidateGateError("Habitat runtime scene_aabb is unavailable")
        minimum, maximum = _runtime_bounds(scene_bounds, owner="scene_aabb")
        self.room_bounds_m = {
            "minimum_m": minimum.tolist(),
            "maximum_m": maximum.tolist(),
        }
        self.runtime_context = {
            "provider_id": owner_id,
            "scene_id": loaded_scene,
            "configured_scene_id": configured_scene,
            "active_dataset": active_dataset,
            "stage_surface": stage_surface,
            "physics_library": physics_library,
            "pathfinder_loaded": True,
            "physics_enabled": True,
            "raycast_enabled": True,
            "room_bounds_source": "loaded_scene_aabb",
        }

    def line_of_sight_nearest_hit(
        self, origin: list[float], target: list[float]
    ) -> dict[str, Any]:
        origin_vector = _vec3(origin, owner="ray origin")
        target_vector = _vec3(target, owner="ray target")
        delta = target_vector - origin_vector
        target_distance = float(np.linalg.norm(delta))
        if target_distance <= 1.0e-9:
            raise CameraCandidateGateError("ray origin and target must differ")
        direction = delta / target_distance
        ray = self._habitat_sim.geo.Ray(
            self._mn.Vector3(origin_vector), self._mn.Vector3(direction)
        )
        result = self._cast_ray(ray, max_distance=target_distance, buffer_distance=0.0)
        has_hits = getattr(result, "has_hits", None)
        if not callable(has_hits):
            raise CameraCandidateGateError("Habitat cast_ray returned invalid results")

        nearest_distance: float | None = None
        nearest_hit: Any = None
        hits = list(getattr(result, "hits", ()))
        if bool(has_hits()) != bool(hits):
            raise CameraCandidateGateError("Habitat ray hit status and hit list differ")
        for hit in hits:
            distance = _nonnegative(
                getattr(hit, "ray_distance", None), owner="Habitat ray hit distance"
            )
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_hit = hit

        context = self.runtime_context
        evidence: dict[str, Any] = {
            "provider_id": context["provider_id"],
            "scene_id": context["scene_id"],
            "physics_enabled": True,
            "raycast_enabled": True,
            "endpoint_policy": "full_ray_buffer_zero_no_endpoint_tolerance",
            "ray_origin_m": origin_vector.tolist(),
            "ray_direction_unit": direction.tolist(),
            "ray_max_distance_m": target_distance,
            "ray_buffer_distance_m": 0.0,
            "nearest_hit_distance_m": nearest_distance,
        }
        if nearest_hit is not None:
            evidence["nearest_hit"] = {
                "object_id": int(getattr(nearest_hit, "object_id")),
                "point_m": _vec3(
                    getattr(nearest_hit, "point", None), owner="Habitat hit point"
                ).tolist(),
                "normal": _vec3(
                    getattr(nearest_hit, "normal", None), owner="Habitat hit normal"
                ).tolist(),
            }
        return evidence


def _room_bounds(value: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, Mapping):
        raise CameraCandidateGateError("room_bounds_m must be an object")
    minimum = _vec3(value.get("minimum_m"), owner="room_bounds_m.minimum_m")
    maximum = _vec3(value.get("maximum_m"), owner="room_bounds_m.maximum_m")
    if np.any(minimum >= maximum):
        raise CameraCandidateGateError(
            "room_bounds_m minimum_m must be smaller than maximum_m"
        )
    return minimum, maximum


def _runtime_provider(value: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    pathfinder = getattr(value, "pathfinder", None)
    line_of_sight_query = getattr(value, "line_of_sight_nearest_hit", None)
    room_bounds_m = getattr(value, "room_bounds_m", None)
    context_value = getattr(value, "runtime_context", None)
    if pathfinder is None or not callable(line_of_sight_query):
        raise CameraCandidateGateError(
            "runtime_provider must expose pathfinder and line_of_sight_nearest_hit"
        )
    if getattr(pathfinder, "is_loaded", None) is not True:
        raise CameraCandidateGateError(
            "runtime_provider.pathfinder.is_loaded must be explicitly true"
        )
    value = context_value
    if not isinstance(value, Mapping):
        raise CameraCandidateGateError(
            "runtime_provider.runtime_context must be an object"
        )
    provider_id = _identifier(value.get("provider_id"), owner="runtime provider_id")
    scene_id = _identifier(value.get("scene_id"), owner="runtime scene_id")
    required = {
        "pathfinder_loaded": True,
        "physics_enabled": True,
        "raycast_enabled": True,
    }
    for field_name, expected in required.items():
        if value.get(field_name) is not expected:
            raise CameraCandidateGateError(
                f"runtime_context.{field_name} must be explicitly true"
            )
    if value.get("room_bounds_source") not in {
        "loaded_scene_aabb",
        "loaded_scene_pathfinder_bounds",
    }:
        raise CameraCandidateGateError(
            "runtime_context.room_bounds_source must identify loaded runtime bounds"
        )
    context = {
        "provider_id": provider_id,
        "scene_id": scene_id,
        **required,
        "room_bounds_source": value["room_bounds_source"],
    }
    if room_bounds_m is None:
        raise CameraCandidateGateError(
            "runtime_provider must expose loaded-scene room_bounds_m"
        )
    return pathfinder, line_of_sight_query, room_bounds_m, context


def _actor_inputs(
    actor_floor_paths_m: Any,
    actor_visibility_anchors_m: Any,
) -> tuple[
    dict[str, list[np.ndarray]],
    dict[str, dict[str, list[np.ndarray]]],
    int,
]:
    if not isinstance(actor_floor_paths_m, Mapping) or len(actor_floor_paths_m) < 2:
        raise CameraCandidateGateError(
            "actor_floor_paths_m must contain at least two actors"
        )
    if not isinstance(actor_visibility_anchors_m, Mapping):
        raise CameraCandidateGateError("actor_visibility_anchors_m must be an object")

    floors: dict[str, list[np.ndarray]] = {}
    anchors: dict[str, dict[str, list[np.ndarray]]] = {}
    frame_count: int | None = None
    for actor_id_value in sorted(
        actor_floor_paths_m, key=lambda item: str(item).encode("utf-8")
    ):
        actor_id = _identifier(actor_id_value, owner="actor ID")
        floor_value = actor_floor_paths_m[actor_id_value]
        if isinstance(floor_value, (str, bytes)) or not isinstance(
            floor_value, Sequence
        ):
            raise CameraCandidateGateError(
                f"actor {actor_id!r} floor path must be a non-empty sequence"
            )
        floor_path = [
            _vec3(point, owner=f"actor {actor_id!r} floor frame {index}")
            for index, point in enumerate(floor_value)
        ]
        if not floor_path:
            raise CameraCandidateGateError(
                f"actor {actor_id!r} floor path must be non-empty"
            )
        if frame_count is None:
            frame_count = len(floor_path)
        elif len(floor_path) != frame_count:
            raise CameraCandidateGateError("actor floor paths must share frame count")
        floors[actor_id] = floor_path

        actor_anchors_value = actor_visibility_anchors_m.get(actor_id_value)
        if not isinstance(actor_anchors_value, Mapping) or len(actor_anchors_value) < 2:
            raise CameraCandidateGateError(
                f"actor {actor_id!r} requires at least two visibility anchors"
            )
        actor_anchors: dict[str, list[np.ndarray]] = {}
        for anchor_id_value in sorted(
            actor_anchors_value, key=lambda item: str(item).encode("utf-8")
        ):
            anchor_id = _identifier(anchor_id_value, owner="visibility anchor ID")
            points_value = actor_anchors_value[anchor_id_value]
            if isinstance(points_value, (str, bytes)) or not isinstance(
                points_value, Sequence
            ):
                raise CameraCandidateGateError(
                    f"actor {actor_id!r} anchor {anchor_id!r} must be a sequence"
                )
            points = [
                _vec3(
                    point,
                    owner=(f"actor {actor_id!r} anchor {anchor_id!r} frame {index}"),
                )
                for index, point in enumerate(points_value)
            ]
            if len(points) != frame_count:
                raise CameraCandidateGateError(
                    f"actor {actor_id!r} anchor {anchor_id!r} frame count differs"
                )
            actor_anchors[anchor_id] = points
        anchors[actor_id] = actor_anchors
    assert frame_count is not None
    if set(anchors) != set(floors):
        raise CameraCandidateGateError(
            "visibility anchor actors must exactly match floor-path actors"
        )
    return floors, anchors, frame_count


def _runtime_floor_record(
    pathfinder: Any,
    point: np.ndarray,
    *,
    maximum_y_delta_m: float,
    maximum_snap_error_m: float,
    minimum_clearance_m: float,
    maximum_clearance_query_m: float,
) -> dict[str, Any]:
    navigable = bool(pathfinder.is_navigable(point, maximum_y_delta_m))
    snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float64)
    if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
        raise CameraCandidateGateError("PathFinder returned an invalid snap point")
    snap_error = float(np.linalg.norm(snapped - point))
    clearance = float(
        pathfinder.distance_to_closest_obstacle(snapped, maximum_clearance_query_m)
    )
    if not math.isfinite(clearance) or clearance < 0.0:
        raise CameraCandidateGateError("PathFinder returned an invalid clearance")
    island = int(pathfinder.get_island(snapped))
    if island < 0:
        raise CameraCandidateGateError("PathFinder returned an invalid island")
    passed = (
        navigable
        and snap_error <= maximum_snap_error_m
        and clearance >= minimum_clearance_m
    )
    return _status(
        passed,
        query_m=point.tolist(),
        snapped_m=snapped.tolist(),
        navigable=navigable,
        snap_error_m=snap_error,
        maximum_snap_error_m=maximum_snap_error_m,
        clearance_m=clearance,
        minimum_clearance_m=minimum_clearance_m,
        island=island,
    )


def _same_island_record(
    pathfinder: Any,
    floors: Mapping[str, Sequence[np.ndarray]],
    *,
    listener_island: int,
    maximum_y_delta_m: float,
    maximum_snap_error_m: float,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    actor_islands: dict[str, list[int]] = {}
    for actor_id in sorted(floors, key=lambda item: item.encode("utf-8")):
        islands: list[int] = []
        for frame_index, point in enumerate(floors[actor_id]):
            navigable = bool(pathfinder.is_navigable(point, maximum_y_delta_m))
            snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float64)
            if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
                raise CameraCandidateGateError(
                    f"PathFinder returned an invalid actor snap for {actor_id!r}"
                )
            snap_error = float(np.linalg.norm(snapped - point))
            island = int(pathfinder.get_island(snapped))
            islands.append(island)
            if (
                not navigable
                or snap_error > maximum_snap_error_m
                or island != listener_island
            ):
                failures.append(
                    {
                        "actor_id": actor_id,
                        "frame_index": frame_index,
                        "navigable": navigable,
                        "snap_error_m": snap_error,
                        "island": island,
                    }
                )
        actor_islands[actor_id] = sorted(set(islands))
    return _status(
        not failures,
        listener_island=listener_island,
        actor_islands=actor_islands,
        failed_frame_count=len(failures),
        first_failure=failures[0] if failures else None,
    )


def _line_of_sight_record(
    *,
    origin: np.ndarray,
    anchors: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
    line_of_sight_query: Any,
    runtime_context: Mapping[str, Any],
    tolerance_m: float,
) -> tuple[dict[str, dict[str, Any]], bool]:
    records: dict[str, dict[str, Any]] = {}
    all_pass = True
    for actor_id in sorted(anchors, key=lambda item: item.encode("utf-8")):
        failures: list[dict[str, Any]] = []
        query_count = 0
        minimum_margin = math.inf
        for anchor_id in sorted(
            anchors[actor_id], key=lambda item: item.encode("utf-8")
        ):
            for frame_index, target in enumerate(anchors[actor_id][anchor_id]):
                distance = float(np.linalg.norm(target - origin))
                if distance <= 1.0e-9:
                    raise CameraCandidateGateError(
                        "camera coincides with a visibility anchor"
                    )
                query = line_of_sight_query(origin.tolist(), target.tolist())
                if not isinstance(query, Mapping):
                    raise CameraCandidateGateError(
                        "line-of-sight provider must return structured runtime evidence"
                    )
                required_query = {
                    "provider_id": runtime_context["provider_id"],
                    "scene_id": runtime_context["scene_id"],
                    "physics_enabled": True,
                    "raycast_enabled": True,
                    "endpoint_policy": "full_ray_buffer_zero_no_endpoint_tolerance",
                }
                for field_name, expected in required_query.items():
                    if query.get(field_name) != expected:
                        raise CameraCandidateGateError(
                            f"line-of-sight evidence {field_name} differs from runtime provider"
                        )
                nearest_value = query.get("nearest_hit_distance_m")
                if nearest_value is None:
                    nearest = math.inf
                else:
                    nearest = _nonnegative(
                        nearest_value, owner="line-of-sight nearest hit distance"
                    )
                margin = nearest - distance
                minimum_margin = min(minimum_margin, margin)
                query_count += 1
                if nearest + tolerance_m < distance:
                    failures.append(
                        {
                            "frame_index": frame_index,
                            "anchor_id": anchor_id,
                            "target_m": target.tolist(),
                            "target_distance_m": distance,
                            "nearest_hit_m": nearest,
                        }
                    )
        passed = not failures
        all_pass = all_pass and passed
        records[actor_id] = _status(
            passed,
            anchor_ids=sorted(anchors[actor_id], key=lambda item: item.encode("utf-8")),
            query_count=query_count,
            passed_query_count=query_count - len(failures),
            tolerance_m=tolerance_m,
            minimum_hit_margin_m=(
                None if math.isinf(minimum_margin) else minimum_margin
            ),
            first_failure=failures[0] if failures else None,
        )
    return records, all_pass


def evaluate_camera_candidates(
    *,
    runtime_provider: Any,
    candidates: Any,
    actor_floor_paths_m: Any,
    actor_visibility_anchors_m: Any,
    floor_height_m: Any,
    evaluation_id: Any,
    maximum_y_delta_m: Any = 0.25,
    maximum_snap_error_m: Any = 0.05,
    minimum_clearance_m: Any = 0.25,
    maximum_clearance_query_m: Any = 10.0,
    line_of_sight_tolerance_m: Any = 0.03,
) -> list[dict[str, Any]]:
    """Evaluate candidates and return deterministic pass/fail evidence.

    Only a candidate that passes every runtime hard gate receives a
    ``room_gate``.  That object can be forwarded directly to
    :func:`avengine.camera_framing.solve_static_camera_candidates`.
    """

    owner_id = _identifier(evaluation_id, owner="evaluation_id")
    pathfinder, line_of_sight_query, room_bounds_m, runtime = _runtime_provider(
        runtime_provider
    )
    floor_height = _finite(floor_height_m, owner="floor_height_m")
    max_y = _nonnegative(maximum_y_delta_m, owner="maximum_y_delta_m")
    max_snap = _nonnegative(maximum_snap_error_m, owner="maximum_snap_error_m")
    min_clearance = _nonnegative(minimum_clearance_m, owner="minimum_clearance_m")
    max_clearance_query = _nonnegative(
        maximum_clearance_query_m,
        owner="maximum_clearance_query_m",
        positive=True,
    )
    los_tolerance = _nonnegative(
        line_of_sight_tolerance_m, owner="line_of_sight_tolerance_m"
    )
    room_minimum, room_maximum = _room_bounds(room_bounds_m)
    floors, anchors, frame_count = _actor_inputs(
        actor_floor_paths_m, actor_visibility_anchors_m
    )

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise CameraCandidateGateError("candidates must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(candidates):
        if not isinstance(value, Mapping):
            raise CameraCandidateGateError(f"candidate {index} must be an object")
        candidate_id = _identifier(value.get("candidate_id"), owner="candidate_id")
        if candidate_id in seen_ids:
            raise CameraCandidateGateError("candidate IDs must be unique")
        seen_ids.add(candidate_id)
        position = _vec3(
            value.get("position_m"), owner=f"candidate {candidate_id!r}.position_m"
        )
        yaw = _finite(value.get("yaw_deg"), owner=f"candidate {candidate_id!r}.yaw_deg")
        priority = _finite(
            value.get("priority", 0.0), owner=f"candidate {candidate_id!r}.priority"
        )
        normalized.append(
            {
                "candidate_id": candidate_id,
                "position_m": position,
                "yaw_deg": yaw,
                "priority": priority,
            }
        )
    if not normalized:
        raise CameraCandidateGateError("candidates must be a non-empty sequence")
    normalized.sort(
        key=lambda value: (
            value["priority"],
            value["candidate_id"].encode("utf-8"),
        )
    )

    results: list[dict[str, Any]] = []
    for candidate in normalized:
        position = candidate["position_m"]
        floor_point = np.asarray(
            [position[0], floor_height, position[2]], dtype=np.float64
        )
        listener = _runtime_floor_record(
            pathfinder,
            floor_point,
            maximum_y_delta_m=max_y,
            maximum_snap_error_m=max_snap,
            minimum_clearance_m=min_clearance,
            maximum_clearance_query_m=max_clearance_query,
        )
        room_pass = bool(
            np.all(position >= room_minimum) and np.all(position <= room_maximum)
        )
        room = _status(
            room_pass,
            position_m=position.tolist(),
            minimum_m=room_minimum.tolist(),
            maximum_m=room_maximum.tolist(),
        )
        island = _same_island_record(
            pathfinder,
            floors,
            listener_island=int(listener["island"]),
            maximum_y_delta_m=max_y,
            maximum_snap_error_m=max_snap,
        )
        los_by_actor, los_pass = _line_of_sight_record(
            origin=position,
            anchors=anchors,
            line_of_sight_query=line_of_sight_query,
            runtime_context=runtime,
            tolerance_m=los_tolerance,
        )
        hard_gates: dict[str, Any] = {
            "listener_navmesh": listener,
            "same_navmesh_island": island,
            "room_bounds": room,
            **{
                f"line_of_sight_{actor_id}": record
                for actor_id, record in los_by_actor.items()
            },
        }
        passed = (
            listener["status"] == "pass"
            and island["status"] == "pass"
            and room["status"] == "pass"
            and los_pass
        )
        result: dict[str, Any] = {
            "schema": CAMERA_CANDIDATE_GATE_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "priority": candidate["priority"],
            "position_m": position.tolist(),
            "yaw_deg": candidate["yaw_deg"],
            "status": "pass" if passed else "fail",
            "frame_count": frame_count,
            "evidence": {
                "provenance": RUNTIME_PROVENANCE,
                "runtime_context": deepcopy(runtime),
                "hard_gates": deepcopy(hard_gates),
                "claim_boundary": {
                    "listener_navmesh_runtime_validated": True,
                    "line_of_sight_runtime_queries_complete": True,
                    "live_ue_full_body_clearance_validated": False,
                    "native_pixel_validation_status": "pending",
                    "qualification_claim": False,
                    "formal_dataset_count": 0,
                },
            },
            "room_gate": None,
        }
        if passed:
            result["room_gate"] = {
                "status": "pass",
                "authority_id": f"{owner_id}/{candidate['candidate_id']}",
                "provenance": RUNTIME_PROVENANCE,
                "native_habitat_validation_status": "pass",
                "line_of_sight_validation_status": "pass",
                "full_body_clearance_status": "pending_live_ue",
                "hard_gates": deepcopy(hard_gates),
            }
        results.append(result)
    return results
