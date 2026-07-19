"""ReplicaCAD ``apt_0`` human/Beagle review capture.

This module is the small Habitat-native adapter between ReplicaCAD's scene
instance layout and the already validated M5.1 articulated capture core.  It
does not treat the scene instance's furniture as M1 source-marker objects and
it does not use the MP3D-specific scene-graph validator.  Instead, it binds the
selected ReplicaCAD dataset, scene instance, stage, lighting setup, object
counts, and explicit navmesh before delegating the per-frame articulated work
to :func:`avengine.m5_1.mixed_capture.capture_human_beagle_paths`.

The resulting gate is deliberately a research-review gate.  It proves the
selected visual scene, actor-root routes, camera/listener placement, and
visibility checks.  It makes no acoustic-package or room-qualification claim.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import (
    _make_configuration,
    _resolved_assets,
    _resolved_scene,
    discover_runtime_root,
)
from avengine.m5_1.legacy_route import FRAME_COUNT, FRAME_RATE_HZ
from avengine.m5_1.mixed_capture import (
    MixedCaptureResult,
    capture_human_beagle_paths,
)
from avengine.m5_1.mp3d_capture import (
    MP3DCaptureError,
    MP3DRoutePaths,
    _assert_route_geometry,
    _pathfinder_path_record,
    _semantic_visibility_record,
    write_mp3d_contact_sheet,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


REPLICACAD_ROUTE_SCHEMA = "avengine_m5_1_replicacad_center_route_v1"
REPLICACAD_GATE_SCHEMA = "avengine_m5_1_replicacad_mixed_visual_gate_v1"
REPLICACAD_SCENE_ID = "apt_0"
CONTACT_SHEET_NAME = "replicacad_mixed_contact_sheet.png"
GATE_EVIDENCE_NAME = "replicacad_gate_evidence.json"


class ReplicaCADCaptureError(RuntimeError):
    """The ReplicaCAD closure, placement, or mixed capture failed."""


@dataclass(frozen=True)
class ReplicaCADNavigationEvidence:
    """Real ReplicaCAD PathFinder and selected-scene readback."""

    paths: MP3DRoutePaths
    record: Mapping[str, Any]


@dataclass(frozen=True)
class ReplicaCADCaptureResult:
    """M5.1 mixed capture plus the ReplicaCAD-specific review gate."""

    capture: MixedCaptureResult
    gate_evidence_path: Path
    gate_evidence: Mapping[str, Any]
    contact_sheet_path: Path


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplicaCADCaptureError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReplicaCADCaptureError(f"{owner} must be a finite number")
    return result


def _finite_point(value: Any, *, owner: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplicaCADCaptureError(
            f"{owner} must be a finite length-3 point"
        ) from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ReplicaCADCaptureError(f"{owner} must be a finite length-3 point")
    return point


def _positive_number(value: Any, *, owner: str) -> float:
    result = _finite_number(value, owner=owner)
    if result <= 0.0:
        raise ReplicaCADCaptureError(f"{owner} must be positive")
    return result


def load_replicacad_route_manifest(path: str | Path) -> dict[str, Any]:
    """Load the bounded 18-second ReplicaCAD route contract."""

    try:
        route = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReplicaCADCaptureError(str(exc)) from exc
    if route.get("schema") != REPLICACAD_ROUTE_SCHEMA:
        raise ReplicaCADCaptureError(
            f"ReplicaCAD route schema must be {REPLICACAD_ROUTE_SCHEMA!r}"
        )
    for key in ("route_id", "room_id", "request_id"):
        if not isinstance(route.get(key), str) or not route[key]:
            raise ReplicaCADCaptureError(f"ReplicaCAD route {key} must be non-empty")
    if route.get("frame_count") != FRAME_COUNT:
        raise ReplicaCADCaptureError(
            f"ReplicaCAD route frame_count must be {FRAME_COUNT}"
        )
    if route.get("frame_rate_hz") != FRAME_RATE_HZ:
        raise ReplicaCADCaptureError(
            f"ReplicaCAD route frame_rate_hz must be {FRAME_RATE_HZ}"
        )
    if route.get("path_generation") != "linear_endpoint_interpolation_v1":
        raise ReplicaCADCaptureError("unsupported ReplicaCAD path_generation")
    if route.get("center_navigation_semantics") != "actor_root_center_only":
        raise ReplicaCADCaptureError(
            "ReplicaCAD route must declare actor_root_center_only navigation"
        )

    routes = route.get("routes")
    if not isinstance(routes, dict) or set(routes) != {"human0", "dog0"}:
        raise ReplicaCADCaptureError(
            "ReplicaCAD routes must contain exactly human0 and dog0"
        )
    for actor_id in ("human0", "dog0"):
        actor_route = routes[actor_id]
        if not isinstance(actor_route, dict):
            raise ReplicaCADCaptureError(f"{actor_id} route must be an object")
        _finite_point(actor_route.get("start_m"), owner=f"{actor_id}.start_m")
        _finite_point(actor_route.get("end_m"), owner=f"{actor_id}.end_m")

    navigation = route.get("pathfinder_gate")
    if not isinstance(navigation, dict):
        raise ReplicaCADCaptureError("ReplicaCAD route requires pathfinder_gate")
    for key in (
        "require_declared_navmesh",
        "require_every_frame_navigable",
        "require_one_shared_island",
        "require_segment_no_sliding",
    ):
        if navigation.get(key) is not True:
            raise ReplicaCADCaptureError(f"pathfinder_gate.{key} must be true")
    for key in (
        "maximum_snap_error_m",
        "maximum_y_delta_m",
        "maximum_step_endpoint_error_m",
    ):
        _positive_number(navigation.get(key), owner=f"pathfinder_gate.{key}")

    minimum_separation = _finite_number(
        route.get("minimum_center_separation_m"),
        owner="minimum_center_separation_m",
    )
    if minimum_separation < 0.3:
        raise ReplicaCADCaptureError("minimum_center_separation_m cannot be below 0.3")
    movement = route.get("movement_gate")
    if not isinstance(movement, dict):
        raise ReplicaCADCaptureError("ReplicaCAD route requires movement_gate")
    for key in ("minimum_path_length_m", "minimum_endpoint_displacement_m"):
        _positive_number(movement.get(key), owner=f"movement_gate.{key}")

    placement = route.get("placement_gate")
    if not isinstance(placement, dict):
        raise ReplicaCADCaptureError("ReplicaCAD route requires placement_gate")
    _positive_number(
        placement.get("minimum_navmesh_clearance_m"),
        owner="placement_gate.minimum_navmesh_clearance_m",
    )
    _positive_number(
        placement.get("maximum_camera_floor_snap_error_m"),
        owner="placement_gate.maximum_camera_floor_snap_error_m",
    )
    if placement.get("require_rigid_object_center_clearance") is not True:
        raise ReplicaCADCaptureError(
            "placement_gate.require_rigid_object_center_clearance must be true"
        )
    _positive_number(
        placement.get("minimum_rigid_object_center_clearance_m"),
        owner="placement_gate.minimum_rigid_object_center_clearance_m",
    )
    if placement.get("require_camera_floor_navigable") is not True:
        raise ReplicaCADCaptureError(
            "placement_gate.require_camera_floor_navigable must be true"
        )
    if placement.get("require_camera_actor_line_of_sight") is not True:
        raise ReplicaCADCaptureError(
            "placement_gate.require_camera_actor_line_of_sight must be true"
        )
    offsets = placement.get("visibility_anchor_height_offsets_m")
    if not isinstance(offsets, dict) or set(offsets) != {"human0", "dog0"}:
        raise ReplicaCADCaptureError(
            "visibility anchor offsets must contain exactly human0 and dog0"
        )
    for actor_id in ("human0", "dog0"):
        values = offsets[actor_id]
        if not isinstance(values, list) or len(values) < 2:
            raise ReplicaCADCaptureError(
                f"{actor_id} requires at least two visibility height offsets"
            )
        for index, value in enumerate(values):
            _positive_number(
                value,
                owner=(
                    "placement_gate.visibility_anchor_height_offsets_m."
                    f"{actor_id}[{index}]"
                ),
            )

    semantic_ids = route.get("semantic_ids")
    if not isinstance(semantic_ids, dict) or set(semantic_ids) != {
        "human0",
        "dog0",
    }:
        raise ReplicaCADCaptureError(
            "semantic_ids must contain exactly human0 and dog0"
        )
    values: list[int] = []
    for actor_id in ("human0", "dog0"):
        value = semantic_ids[actor_id]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReplicaCADCaptureError(
                f"semantic_ids.{actor_id} must be a nonnegative integer"
            )
        values.append(value)
    if len(set(values)) != 2:
        raise ReplicaCADCaptureError("human and dog semantic IDs must differ")

    visibility = route.get("visibility_gate")
    if not isinstance(visibility, dict):
        raise ReplicaCADCaptureError("ReplicaCAD route requires visibility_gate")
    minimum_frames = visibility.get("minimum_visible_frames_per_actor")
    if (
        isinstance(minimum_frames, bool)
        or not isinstance(minimum_frames, int)
        or not 1 <= minimum_frames <= FRAME_COUNT
    ):
        raise ReplicaCADCaptureError(
            "minimum_visible_frames_per_actor must lie within the capture"
        )
    minimum_pixels = visibility.get("minimum_visible_pixels_per_frame")
    if (
        isinstance(minimum_pixels, bool)
        or not isinstance(minimum_pixels, int)
        or minimum_pixels < 1
    ):
        raise ReplicaCADCaptureError(
            "minimum_visible_pixels_per_frame must be a positive integer"
        )
    return route


def derive_replicacad_route_paths(
    route: Mapping[str, Any],
) -> MP3DRoutePaths:
    """Expand the two endpoint pairs to exact 270-frame center paths."""

    try:
        routes = route["routes"]
        human_start = _finite_point(routes["human0"]["start_m"], owner="human0.start_m")
        human_end = _finite_point(routes["human0"]["end_m"], owner="human0.end_m")
        dog_start = _finite_point(routes["dog0"]["start_m"], owner="dog0.start_m")
        dog_end = _finite_point(routes["dog0"]["end_m"], owner="dog0.end_m")
    except (KeyError, TypeError) as exc:
        raise ReplicaCADCaptureError("ReplicaCAD route lacks endpoint data") from exc
    return MP3DRoutePaths(
        human=np.ascontiguousarray(
            np.linspace(human_start, human_end, FRAME_COUNT), dtype=np.float64
        ),
        beagle=np.ascontiguousarray(
            np.linspace(dog_start, dog_end, FRAME_COUNT), dtype=np.float64
        ),
    )


def _external_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _replicacad_file_record(path: Path, *, root: Path) -> dict[str, Any]:
    """Return a portable external-dataset locator plus byte identity."""

    resolved = path.resolve()
    dataset_root = root.resolve()
    try:
        relative = resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ReplicaCADCaptureError(
            f"ReplicaCAD artifact escapes AVENGINE_REPLICACAD_ROOT: {resolved}"
        ) from exc
    return {
        "root_id": "AVENGINE_REPLICACAD_ROOT",
        "relative_path": relative.as_posix(),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


@contextmanager
def _replicacad_root_environment(root: Path) -> Iterator[None]:
    key = "AVENGINE_REPLICACAD_ROOT"
    previous = os.environ.get(key)
    os.environ[key] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _resolved_role_paths(room_inputs: Any, runtime: Path) -> dict[str, Path]:
    records = _resolved_assets(room_inputs, runtime)
    roles: dict[str, Path] = {}
    for record in records:
        role = str(record["role"])
        if role in roles:
            raise ReplicaCADCaptureError(f"duplicate ReplicaCAD asset role: {role}")
        if not record["exists"]:
            raise ReplicaCADCaptureError(
                f"ReplicaCAD asset is missing: {record['declared_path']}"
            )
        roles[role] = Path(record["resolved_path"]).resolve()
    return roles


def _assert_selected_closure(
    *,
    room_inputs: Any,
    runtime: Path,
    root: Path,
) -> dict[str, Path]:
    if room_inputs.room.get("room_id") != "replicacad_apt_0":
        raise ReplicaCADCaptureError(
            "ReplicaCAD review room_id must be replicacad_apt_0"
        )
    scene = room_inputs.room.get("scene", {})
    if (
        scene.get("scene_id_kind") != "handle"
        or scene.get("scene_id") != REPLICACAD_SCENE_ID
    ):
        raise ReplicaCADCaptureError("ReplicaCAD review must select handle apt_0")
    roles = _resolved_role_paths(room_inputs, runtime)
    expected = {
        "scene_dataset_config": root / "replicaCAD.scene_dataset_config.json",
        "scene_instance": root / "configs/scenes/apt_0.scene_instance.json",
        "stage_config": root / "configs/stages/frl_apartment_stage.stage_config.json",
        "render_surface_mesh": root / "stages/frl_apartment_stage.glb",
        "navmesh": root / "navmeshes/apt_0.navmesh",
        "lighting_config": (
            root / "configs/lighting/frl_apartment_stage.lighting_config.json"
        ),
    }
    for role, path in expected.items():
        if roles.get(role) != path.resolve():
            raise ReplicaCADCaptureError(
                f"ReplicaCAD role {role} does not resolve to selected apt_0 closure"
            )
    resolved_scene = _resolved_scene(room_inputs, runtime)
    if (
        Path(resolved_scene["dataset_config"]).resolve()
        != expected["scene_dataset_config"].resolve()
    ):
        raise ReplicaCADCaptureError("configured ReplicaCAD dataset differs")
    if resolved_scene["scene_id"] != REPLICACAD_SCENE_ID:
        raise ReplicaCADCaptureError("configured ReplicaCAD scene handle differs")
    if Path(resolved_scene["navmesh"]).resolve() != expected["navmesh"].resolve():
        raise ReplicaCADCaptureError("configured ReplicaCAD navmesh differs")
    return {role: path.resolve() for role, path in expected.items()}


def _camera_floor_record(
    pathfinder: Any,
    *,
    camera_position_m: Sequence[float],
    route_floor_y_m: float,
    maximum_snap_error_m: float,
    minimum_clearance_m: float,
) -> dict[str, Any]:
    camera = _finite_point(camera_position_m, owner="camera position")
    floor_point = np.asarray([camera[0], route_floor_y_m, camera[2]], dtype=np.float64)
    navigable = bool(pathfinder.is_navigable(floor_point, maximum_snap_error_m))
    snapped = np.asarray(pathfinder.snap_point(floor_point), dtype=np.float64)
    if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
        raise ReplicaCADCaptureError("PathFinder returned invalid camera floor snap")
    snap_error = float(np.linalg.norm(snapped - floor_point))
    clearance = float(pathfinder.distance_to_closest_obstacle(snapped, 10.0))
    passed = (
        navigable
        and snap_error <= maximum_snap_error_m
        and math.isfinite(clearance)
        and clearance >= minimum_clearance_m
    )
    if not passed:
        raise ReplicaCADCaptureError(
            "ReplicaCAD camera/listener floor placement failed navmesh clearance"
        )
    return {
        "status": "pass",
        "camera_position_m": camera.tolist(),
        "floor_query_m": floor_point.tolist(),
        "snapped_floor_m": snapped.tolist(),
        "navigable": navigable,
        "snap_error_m": snap_error,
        "maximum_snap_error_m": maximum_snap_error_m,
        "clearance_m": clearance,
        "minimum_clearance_m": minimum_clearance_m,
    }


def _path_clearance_record(
    pathfinder: Any,
    paths: MP3DRoutePaths,
    *,
    minimum_clearance_m: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for actor_id, path in (("human0", paths.human), ("dog0", paths.beagle)):
        values = np.asarray(
            [
                float(pathfinder.distance_to_closest_obstacle(point, 10.0))
                for point in path
            ],
            dtype=np.float64,
        )
        if (
            values.shape != (FRAME_COUNT,)
            or not np.all(np.isfinite(values))
            or np.any(values < minimum_clearance_m)
        ):
            raise ReplicaCADCaptureError(
                f"{actor_id} route violates ReplicaCAD navmesh clearance"
            )
        result[actor_id] = {
            "status": "pass",
            "query_count": FRAME_COUNT,
            "minimum_clearance_m": float(np.min(values)),
            "maximum_clearance_m": float(np.max(values)),
            "required_minimum_clearance_m": minimum_clearance_m,
            "failed_frame_indices": np.flatnonzero(values < minimum_clearance_m)
            .astype(int)
            .tolist(),
            "clearance_sequence_sha256": canonical_json_sha256(values.tolist()),
        }
    return result


def _rigid_object_center_clearance_record(
    object_manager: Any,
    mn: Any,
    paths: MP3DRoutePaths,
    *,
    minimum_clearance_m: float,
) -> dict[str, Any]:
    """Measure actor-root clearance from every furnished-object collision OBB.

    The returned record deliberately retains failed frame indices instead of
    throwing at the first intersection.  The native preflight consumes the
    complete record and fails closed before any visual capture is attempted.
    """

    threshold = _positive_number(
        minimum_clearance_m, owner="minimum rigid-object center clearance"
    )

    objects = tuple(
        sorted(
            object_manager.get_objects_by_handle_substring().values(),
            key=lambda value: str(value.handle).encode("utf-8"),
        )
    )
    if not objects:
        raise ReplicaCADCaptureError(
            "ReplicaCAD furnished-scene collision gate found no rigid objects"
        )
    prepared: list[tuple[str, Any, Any, np.ndarray, np.ndarray]] = []
    for value in objects:
        bounds = value.collision_shape_aabb
        lower = np.asarray(bounds.min, dtype=np.float64)
        upper = np.asarray(bounds.max, dtype=np.float64)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(upper < lower)
        ):
            raise ReplicaCADCaptureError(
                f"rigid object has an invalid collision AABB: {value.handle}"
            )
        prepared.append(
            (
                str(value.handle),
                value.transformation,
                value.transformation.inverted(),
                lower,
                upper,
            )
        )

    actor_records: dict[str, Any] = {}
    for actor_id, points in (("human0", paths.human), ("dog0", paths.beagle)):
        minimum_observed = math.inf
        failed_frames: set[int] = set()
        first_failure: dict[str, Any] | None = None
        frame_minima: list[float] = []
        for frame_index, point in enumerate(points):
            frame_minimum = math.inf
            for handle, forward, inverse, lower, upper in prepared:
                local = np.asarray(
                    inverse.transform_point(mn.Vector3(point)), dtype=np.float64
                )
                nearest_local = np.minimum(np.maximum(local, lower), upper)
                nearest_world = np.asarray(
                    forward.transform_point(mn.Vector3(nearest_local)),
                    dtype=np.float64,
                )
                clearance = float(
                    np.linalg.norm(np.asarray(point, dtype=np.float64) - nearest_world)
                )
                frame_minimum = min(frame_minimum, clearance)
                if clearance < threshold:
                    failed_frames.add(frame_index)
                    if first_failure is None:
                        first_failure = {
                            "frame_index": frame_index,
                            "object_handle": handle,
                            "world_root_m": np.asarray(point).tolist(),
                            "local_root_m": local.tolist(),
                            "nearest_world_obb_m": nearest_world.tolist(),
                            "clearance_m": clearance,
                        }
            frame_minima.append(frame_minimum)
            minimum_observed = min(minimum_observed, frame_minimum)
        failed_frame_indices = sorted(failed_frames)
        actor_records[actor_id] = {
            "status": "fail" if failed_frame_indices else "pass",
            "frame_count": len(points),
            "rigid_object_count": len(prepared),
            "query_count": len(points) * len(prepared),
            "required_minimum_clearance_m": threshold,
            "minimum_observed_clearance_m": minimum_observed,
            "failed_frame_indices": failed_frame_indices,
            "first_failure": first_failure,
            "frame_minimum_clearance_sha256": canonical_json_sha256(frame_minima),
        }
    failed_actors = {
        actor_id: record["failed_frame_indices"]
        for actor_id, record in actor_records.items()
        if record["status"] != "pass"
    }
    return {
        "status": "fail" if failed_actors else "pass",
        "semantics": "actor_root_point_vs_rigid_collision_oriented_aabb_v1",
        "full_body_collision_claim": False,
        "rigid_object_count": len(prepared),
        "required_minimum_clearance_m": threshold,
        "failed_actor_frame_indices": failed_actors,
        "actors": actor_records,
    }


def _line_of_sight_record(
    simulator: Any,
    habitat_sim: Any,
    mn: Any,
    *,
    camera_position_m: Sequence[float],
    paths: MP3DRoutePaths,
    offsets_m: Mapping[str, Sequence[float]],
    tolerance_m: float = 0.03,
) -> dict[str, Any]:
    camera = _finite_point(camera_position_m, owner="camera position")
    result: dict[str, Any] = {}
    for actor_id, path in (("human0", paths.human), ("dog0", paths.beagle)):
        heights = tuple(float(value) for value in offsets_m[actor_id])
        passed_count = 0
        minimum_margin = math.inf
        nearest_failure: dict[str, Any] | None = None
        for frame_index, root in enumerate(path):
            for offset in heights:
                target = np.asarray(root, dtype=np.float64).copy()
                target[1] += offset
                delta = target - camera
                distance = float(np.linalg.norm(delta))
                if distance <= 1.0e-9:
                    raise ReplicaCADCaptureError(
                        "camera coincides with visibility anchor"
                    )
                ray = habitat_sim.geo.Ray(
                    mn.Vector3(camera), mn.Vector3(delta / distance)
                )
                cast = simulator.cast_ray(ray, buffer_distance=0.0)
                nearest = (
                    float(cast.hits[0].ray_distance) if cast.has_hits() else math.inf
                )
                margin = nearest - distance
                minimum_margin = min(minimum_margin, margin)
                passed = nearest + tolerance_m >= distance
                if passed:
                    passed_count += 1
                elif nearest_failure is None:
                    nearest_failure = {
                        "frame_index": frame_index,
                        "height_offset_m": offset,
                        "target_m": target.tolist(),
                        "target_distance_m": distance,
                        "nearest_hit_m": nearest,
                    }
        query_count = FRAME_COUNT * len(heights)
        if passed_count != query_count:
            raise ReplicaCADCaptureError(
                f"{actor_id} camera line-of-sight failed: {nearest_failure}"
            )
        result[actor_id] = {
            "status": "pass",
            "frame_count": FRAME_COUNT,
            "height_offsets_m": list(heights),
            "query_count": query_count,
            "passed_query_count": passed_count,
            "tolerance_m": tolerance_m,
            "minimum_hit_margin_m": (
                None if math.isinf(minimum_margin) else minimum_margin
            ),
        }
    return result


def validate_replicacad_paths_and_placement(
    *,
    route: Mapping[str, Any],
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    replicacad_root: str | Path,
    runtime_root: str | Path | None = None,
) -> ReplicaCADNavigationEvidence:
    """Load real ``apt_0`` and validate its selected closure and placements."""

    root = Path(replicacad_root).resolve()
    if not root.is_dir():
        raise ReplicaCADCaptureError(f"ReplicaCAD root is missing: {root}")
    runtime = discover_runtime_root(runtime_root)
    paths = derive_replicacad_route_paths(route)
    try:
        geometry = _assert_route_geometry(route, paths)
    except MP3DCaptureError as exc:
        raise ReplicaCADCaptureError(str(exc)) from exc
    with _replicacad_root_environment(root):
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
        if room_inputs.room["room_id"] != route["room_id"]:
            raise ReplicaCADCaptureError("route room_id differs from room manifest")
        if room_inputs.request["request_id"] != route["request_id"]:
            raise ReplicaCADCaptureError("route request_id differs from M1 request")
        closure = _assert_selected_closure(
            room_inputs=room_inputs, runtime=runtime, root=root
        )

        # The pinned audio-enabled build imports numpy-quaternion first.
        import quaternion as qt

        import habitat_sim
        import magnum as mn

        configuration, modality_to_uuid, _listener_uuid, configured_scene = (
            _make_configuration(
                room_inputs,
                runtime,
                Path(m1_request_path).resolve().parent
                / ".replicacad_preflight_not_retained",
            )
        )
        if configured_scene["scene_id"] != REPLICACAD_SCENE_ID:
            raise ReplicaCADCaptureError("Habitat configuration changed scene handle")
        with habitat_sim.Simulator(configuration) as simulator:
            loaded = bool(simulator.pathfinder.load_nav_mesh(str(closure["navmesh"])))
            if not loaded or not simulator.pathfinder.is_loaded:
                raise ReplicaCADCaptureError(
                    "PathFinder failed to load the declared apt_0 navmesh"
                )
            stage = simulator.get_stage_initialization_template()
            stage_surface = (
                Path(stage.render_asset_fullpath).resolve()
                if stage is not None
                else None
            )
            current_scene = str(simulator.curr_scene_name)
            if current_scene != REPLICACAD_SCENE_ID:
                raise ReplicaCADCaptureError(
                    f"Habitat current scene is {current_scene!r}, not apt_0"
                )
            if stage_surface != closure["render_surface_mesh"]:
                raise ReplicaCADCaptureError(
                    "Habitat loaded a different ReplicaCAD stage surface"
                )

            instance = load_json(closure["scene_instance"])
            lighting = load_json(closure["lighting_config"])
            expected_rigid = len(instance.get("object_instances", []))
            expected_articulated = len(instance.get("articulated_object_instances", []))
            actual_rigid = len(
                simulator.get_rigid_object_manager().get_objects_by_handle_substring()
            )
            actual_articulated = len(
                simulator.get_articulated_object_manager().get_objects_by_handle_substring()
            )
            expected_lights = len(lighting.get("lights", {}))
            actual_lights = len(simulator.get_current_light_setup())
            if actual_rigid != expected_rigid:
                raise ReplicaCADCaptureError(
                    f"ReplicaCAD rigid-object count {actual_rigid} != {expected_rigid}"
                )
            if actual_articulated != expected_articulated:
                raise ReplicaCADCaptureError(
                    "ReplicaCAD articulated-object count differs from scene instance"
                )
            if actual_lights != expected_lights:
                raise ReplicaCADCaptureError(
                    f"ReplicaCAD light count {actual_lights} != {expected_lights}"
                )

            rig = room_inputs.request["primary_camera_rig"]
            camera_state = habitat_sim.AgentState()
            camera_state.position = np.asarray(
                rig["world_from_rig"]["translation_m"], dtype=np.float64
            )
            x, y, z, w = normalized_quaternion_xyzw(
                rig["world_from_rig"]["rotation_xyzw"]
            )
            camera_state.rotation = qt.quaternion(w, x, y, z)
            simulator.initialize_agent(0, camera_state)

            semantic_uuid = modality_to_uuid["semantic"]
            baseline = simulator.render_sensors([simulator.sensors[semantic_uuid]])
            semantic = np.asarray(baseline[semantic_uuid])
            expected_shape = tuple(rig["shared_calibration"]["resolution_hw"])
            if semantic.shape != expected_shape:
                raise ReplicaCADCaptureError(
                    "ReplicaCAD baseline semantic shape differs from request"
                )
            semantic_ids = {
                actor_id: int(route["semantic_ids"][actor_id])
                for actor_id in ("human0", "dog0")
            }
            baseline_counts = {
                actor_id: int(np.count_nonzero(semantic == semantic_id))
                for actor_id, semantic_id in semantic_ids.items()
            }
            if any(baseline_counts.values()):
                raise ReplicaCADCaptureError(
                    "ReplicaCAD room baseline collides with actor semantic IDs"
                )

            gate = route["pathfinder_gate"]
            path_records = {
                "human0": _pathfinder_path_record(
                    simulator.pathfinder,
                    paths.human,
                    owner="human0",
                    maximum_snap_error_m=float(gate["maximum_snap_error_m"]),
                    maximum_y_delta_m=float(gate["maximum_y_delta_m"]),
                    maximum_step_endpoint_error_m=float(
                        gate["maximum_step_endpoint_error_m"]
                    ),
                ),
                "dog0": _pathfinder_path_record(
                    simulator.pathfinder,
                    paths.beagle,
                    owner="dog0",
                    maximum_snap_error_m=float(gate["maximum_snap_error_m"]),
                    maximum_y_delta_m=float(gate["maximum_y_delta_m"]),
                    maximum_step_endpoint_error_m=float(
                        gate["maximum_step_endpoint_error_m"]
                    ),
                ),
            }
            if path_records["human0"]["island_id"] != path_records["dog0"]["island_id"]:
                raise ReplicaCADCaptureError(
                    "human and dog routes occupy different navmesh islands"
                )
            placement = route["placement_gate"]
            minimum_clearance = float(placement["minimum_navmesh_clearance_m"])
            clearance = _path_clearance_record(
                simulator.pathfinder,
                paths,
                minimum_clearance_m=minimum_clearance,
            )
            rigid_object_clearance = _rigid_object_center_clearance_record(
                simulator.get_rigid_object_manager(),
                mn,
                paths,
                minimum_clearance_m=float(
                    placement["minimum_rigid_object_center_clearance_m"]
                ),
            )
            if rigid_object_clearance["status"] != "pass":
                raise ReplicaCADCaptureError(
                    "actor root route violates furnished rigid-object center "
                    "clearance: failed_actor_frame_indices="
                    f"{rigid_object_clearance['failed_actor_frame_indices']}"
                )
            camera_floor = _camera_floor_record(
                simulator.pathfinder,
                camera_position_m=rig["world_from_rig"]["translation_m"],
                route_floor_y_m=float(paths.human[0, 1]),
                maximum_snap_error_m=float(
                    placement["maximum_camera_floor_snap_error_m"]
                ),
                minimum_clearance_m=minimum_clearance,
            )
            line_of_sight = _line_of_sight_record(
                simulator,
                habitat_sim,
                mn,
                camera_position_m=rig["world_from_rig"]["translation_m"],
                paths=paths,
                offsets_m=placement["visibility_anchor_height_offsets_m"],
            )
            bounds = simulator.pathfinder.get_bounds()
            record: dict[str, Any] = {
                "status": "pass",
                "implementation": "habitat_sim.Simulator+PathFinder",
                "selected_scene": {
                    "scene_id": REPLICACAD_SCENE_ID,
                    "dataset_config": _replicacad_file_record(
                        closure["scene_dataset_config"], root=root
                    ),
                    "scene_instance": _replicacad_file_record(
                        closure["scene_instance"], root=root
                    ),
                    "stage_config": _replicacad_file_record(
                        closure["stage_config"], root=root
                    ),
                    "stage_surface": _replicacad_file_record(
                        closure["render_surface_mesh"], root=root
                    ),
                    "lighting_config": _replicacad_file_record(
                        closure["lighting_config"], root=root
                    ),
                    "rigid_object_count": actual_rigid,
                    "articulated_object_count": actual_articulated,
                    "light_count": actual_lights,
                    "scene_instance_counts_match": True,
                    "stage_surface_matches": True,
                },
                "declared_navmesh_loaded": True,
                "declared_navmesh": _replicacad_file_record(
                    closure["navmesh"], root=root
                ),
                "navigable_area_m2": float(simulator.pathfinder.navigable_area),
                "island_count": int(simulator.pathfinder.num_islands),
                "shared_island_id": path_records["human0"]["island_id"],
                "bounds_m": [
                    [float(component) for component in bounds[0]],
                    [float(component) for component in bounds[1]],
                ],
                "center_navigation_semantics": "actor_root_center_only",
                "routes": path_records,
                "geometry": geometry,
                "clearance": clearance,
                "rigid_object_center_clearance": rigid_object_clearance,
                "camera_listener_floor": camera_floor,
                "camera_actor_line_of_sight": line_of_sight,
                "semantic_baseline": {
                    "sensor_uuid": semantic_uuid,
                    "shape": list(semantic.shape),
                    "actor_semantic_ids": semantic_ids,
                    "actor_id_pixel_counts": baseline_counts,
                    "no_actor_id_collision": True,
                },
            }
    return ReplicaCADNavigationEvidence(paths=paths, record=record)


def _root_relative_file_record(
    path: str | Path, *, root: str | Path, root_id: str
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    root_path = Path(root).resolve()
    try:
        relative = resolved.relative_to(root_path)
    except ValueError as exc:
        raise ReplicaCADCaptureError(
            f"{resolved} escapes portable root {root_id}"
        ) from exc
    return {
        "root_id": root_id,
        "relative_path": relative.as_posix(),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _repository_file_record(path: str | Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    return _root_relative_file_record(
        path, root=repository, root_id="AVENGINE_REPOSITORY_ROOT"
    )


def _portable_path_string(value: str, roots: Sequence[tuple[str, Path]]) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    resolved = path.resolve()
    for root_id, root in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        suffix = relative.as_posix()
        return f"${{{root_id}}}/{suffix}" if suffix != "." else f"${{{root_id}}}"
    raise ReplicaCADCaptureError(
        f"capture evidence contains an undeclared absolute path: {resolved}"
    )


def _portableize_evidence_paths(value: Any, roots: Sequence[tuple[str, Path]]) -> Any:
    if isinstance(value, dict):
        return {
            key: _portableize_evidence_paths(item, roots) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portableize_evidence_paths(item, roots) for item in value]
    if isinstance(value, tuple):
        return [_portableize_evidence_paths(item, roots) for item in value]
    if isinstance(value, str):
        return _portable_path_string(value, roots)
    return value


_HUMAN_DERIVED_FILENAMES = {
    "promoted_glb": "promoted_source.glb",
    "visual_glb": "visual.glb",
    "rebase_report": "rebase_report.json",
    "actions_npz": "walking_actions.npz",
    "habitat_urdf": "human.urdf",
    "habitat_ao_config": "human.ao_config.json",
    "habitat_joint_mapping": "joint_mapping.json",
}
_PORTABLE_ROOT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _authenticated_absolute_file_record(
    record: Any,
    roots: Sequence[tuple[str, Path]],
    *,
    owner: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise ReplicaCADCaptureError(f"{owner} must be a file record")
    raw = record.get("path")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ReplicaCADCaptureError(
            f"{owner} must bind the freshly generated absolute file"
        )
    resolved = Path(raw).resolve()
    if not any(_contains_portable_root(resolved, root) for _root_id, root in roots):
        raise ReplicaCADCaptureError(f"{owner} escapes declared portable roots")
    if not resolved.is_file():
        raise ReplicaCADCaptureError(f"{owner} file is missing")
    if record.get("byte_size") != resolved.stat().st_size:
        raise ReplicaCADCaptureError(f"{owner} byte size differs before portability")
    if record.get("sha256") != sha256_file(resolved):
        raise ReplicaCADCaptureError(f"{owner} SHA-256 differs before portability")
    return resolved


def _contains_portable_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _portable_bound_file_record(
    path: Path, roots: Sequence[tuple[str, Path]], *, owner: str
) -> dict[str, Any]:
    resolved = path.resolve()
    for root_id, root in roots:
        if _contains_portable_root(resolved, root):
            relative = resolved.relative_to(root.resolve())
            return {
                "root_id": root_id,
                "relative_path": relative.as_posix(),
                "byte_size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
    raise ReplicaCADCaptureError(f"{owner} escapes declared portable roots")


def _portableize_generated_human_runtime_documents(
    capture_root: Path,
    roots: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    """Rebind generated human JSON without preserving staging identities."""

    root = capture_root.resolve()
    human = root / "runtime/human"
    manifest_path = human / "human_runtime_manifest.json"
    rebase_path = human / "rebase_report.json"
    try:
        manifest = load_json(manifest_path)
        rebase = load_json(rebase_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReplicaCADCaptureError(
            f"generated human runtime JSON cannot be loaded: {exc}"
        ) from exc
    if manifest.get("schema") != "avengine_m5_1_rocketbox_human_runtime_v1":
        raise ReplicaCADCaptureError("generated human runtime manifest schema differs")
    manifest_content = dict(manifest)
    declared_manifest_content = manifest_content.pop("manifest_content_sha256", None)
    if declared_manifest_content != canonical_json_sha256(manifest_content):
        raise ReplicaCADCaptureError(
            "generated human runtime manifest content hash differs before portability"
        )
    old_manifest_file_sha256 = sha256_file(manifest_path)

    source_path = _authenticated_absolute_file_record(
        manifest.get("source"), roots, owner="human runtime source"
    )
    derived = manifest.get("derived")
    if not isinstance(derived, Mapping) or set(derived) != set(
        _HUMAN_DERIVED_FILENAMES
    ):
        raise ReplicaCADCaptureError("human runtime derived file set differs")
    derived_paths: dict[str, Path] = {}
    for role, filename in _HUMAN_DERIVED_FILENAMES.items():
        path = _authenticated_absolute_file_record(
            derived.get(role), roots, owner=f"human runtime derived.{role}"
        )
        expected = (human / filename).resolve()
        if path != expected:
            raise ReplicaCADCaptureError(
                f"human runtime derived.{role} does not bind {filename}"
            )
        derived_paths[role] = path

    if rebase.get("schema") != "avengine_m2_skin_root_rebase_local_tr_v2":
        raise ReplicaCADCaptureError("human rebase report schema differs")
    rebase_source = _authenticated_absolute_file_record(
        rebase.get("source"), roots, owner="human rebase source"
    )
    rebase_output = _authenticated_absolute_file_record(
        rebase.get("output"), roots, owner="human rebase output"
    )
    if (
        rebase_source != derived_paths["promoted_glb"]
        or rebase_output != derived_paths["visual_glb"]
    ):
        raise ReplicaCADCaptureError(
            "human rebase source/output differs from runtime derived files"
        )

    portable_rebase = deepcopy(rebase)
    portable_rebase["source"] = _portable_bound_file_record(
        rebase_source, roots, owner="human rebase source"
    )
    portable_rebase["output"] = _portable_bound_file_record(
        rebase_output, roots, owner="human rebase output"
    )
    write_json(rebase_path, portable_rebase)
    if load_json(rebase_path) != portable_rebase:
        raise ReplicaCADCaptureError("portable human rebase report differs on readback")

    portable_manifest = deepcopy(manifest)
    portable_manifest.pop("manifest_content_sha256", None)
    portable_source = _portable_bound_file_record(
        source_path, roots, owner="human runtime source"
    )
    if portable_source["root_id"] != "AVENGINE_LEGACY_ROOT":
        raise ReplicaCADCaptureError(
            "Rocketbox source must bind the declared AVENGINE_LEGACY_ROOT"
        )
    portable_manifest["source"] = portable_source
    portable_manifest["derived"] = {
        role: _portable_bound_file_record(
            path, roots, owner=f"human runtime derived.{role}"
        )
        for role, path in derived_paths.items()
    }
    portable_manifest["manifest_content_sha256"] = canonical_json_sha256(
        portable_manifest
    )
    write_json(manifest_path, portable_manifest)
    if load_json(manifest_path) != portable_manifest:
        raise ReplicaCADCaptureError(
            "portable human runtime manifest differs on readback"
        )
    manifest_check = dict(portable_manifest)
    declared = manifest_check.pop("manifest_content_sha256")
    if declared != canonical_json_sha256(manifest_check):
        raise ReplicaCADCaptureError(
            "portable human runtime manifest content hash differs"
        )
    return {
        "old_manifest_file_sha256": old_manifest_file_sha256,
        "manifest_file": file_record(manifest_path, relative_to=root),
        "manifest_content_sha256": declared,
        "rebase_report": file_record(rebase_path, relative_to=root),
    }


def _bind_capture_identity_and_emitter_anchors(
    evidence: dict[str, Any], route: Mapping[str, Any]
) -> None:
    """Add explicit route identity and source-emitter array bindings."""

    expected_anchor_order = [
        "human0.head",
        "human0.mouth_emitter",
        "dog0.mouth_emitter",
    ]
    if evidence.get("anchor_order") != expected_anchor_order:
        raise ReplicaCADCaptureError(
            "mixed capture anchor_order differs from the explicit emitter binding"
        )
    artifacts = evidence.get("array_artifacts")
    anchor_artifact = (
        artifacts.get("anchor_positions_m") if isinstance(artifacts, Mapping) else None
    )
    if (
        not isinstance(anchor_artifact, Mapping)
        or anchor_artifact.get("shape") != [FRAME_COUNT, 3, 3]
        or anchor_artifact.get("readback_verified") is not True
    ):
        raise ReplicaCADCaptureError(
            "mixed capture anchor_positions_m artifact lacks exact readback"
        )
    actors = evidence.get("actors")
    if not isinstance(actors, list) or any(
        not isinstance(actor, dict) for actor in actors
    ):
        raise ReplicaCADCaptureError("mixed capture actors must be mutable records")
    actor_by_id = {actor.get("actor_id"): actor for actor in actors}
    if set(actor_by_id) != {"human0", "dog0"} or len(actors) != 2:
        raise ReplicaCADCaptureError(
            "mixed capture actors differ from the declared human/dog pair"
        )
    for actor_id, anchor_index in (("human0", 1), ("dog0", 2)):
        anchor_id = expected_anchor_order[anchor_index]
        actor = actor_by_id[actor_id]
        if not isinstance(actor.get("emitter_link"), str) or not actor["emitter_link"]:
            raise ReplicaCADCaptureError(f"{actor_id} lacks a declared emitter link")
        if not anchor_id.startswith(f"{actor_id}."):
            raise ReplicaCADCaptureError(
                f"{actor_id} emitter anchor does not belong to the actor"
            )
        actor["emitter_anchor_index"] = anchor_index
        actor["emitter_anchor_id"] = anchor_id

    evidence["room_id"] = route["room_id"]
    evidence["request_id"] = route["request_id"]
    evidence["route_id"] = route["route_id"]


def _rebind_portable_human_runtime_references(
    evidence: dict[str, Any], portability: Mapping[str, Any]
) -> None:
    """Update the two exact capture references after manifest portability."""

    old_sha256 = portability.get("old_manifest_file_sha256")
    manifest_file = portability.get("manifest_file")
    if not isinstance(old_sha256, str) or not isinstance(manifest_file, Mapping):
        raise ReplicaCADCaptureError("human runtime portability result is malformed")
    runtime = evidence.get("runtime")
    retained = (
        runtime.get("human_package_manifest") if isinstance(runtime, dict) else None
    )
    if (
        not isinstance(retained, dict)
        or retained.get("sha256") != old_sha256
        or retained.get("path") != manifest_file.get("path")
    ):
        raise ReplicaCADCaptureError(
            "capture runtime manifest reference differs before portable rebinding"
        )
    retained.clear()
    retained.update(manifest_file)

    heading = evidence.get("heading_alignment")
    actors = heading.get("actors") if isinstance(heading, Mapping) else None
    if not isinstance(actors, list):
        raise ReplicaCADCaptureError("capture heading actors are missing")
    human_records = [
        value
        for value in actors
        if isinstance(value, dict) and value.get("actor_id") == "human0"
    ]
    if len(human_records) != 1:
        raise ReplicaCADCaptureError("capture requires one human heading record")
    source = human_records[0].get("binding_source")
    if (
        not isinstance(source, dict)
        or source.get("kind") != "generated_human_runtime_manifest"
        or source.get("sha256") != old_sha256
    ):
        raise ReplicaCADCaptureError(
            "human heading source differs before portable rebinding"
        )
    source["path"] = "${AVENGINE_CAPTURE_ROOT}/" + str(manifest_file["path"])
    source["sha256"] = manifest_file["sha256"]

    def stale_reference_count(value: Any) -> int:
        if isinstance(value, Mapping):
            return sum(stale_reference_count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(stale_reference_count(item) for item in value)
        return int(value == old_sha256)

    if stale_reference_count(evidence):
        raise ReplicaCADCaptureError(
            "capture evidence retains a stale human runtime manifest SHA-256"
        )


def _portable_mixed_capture(
    capture: MixedCaptureResult,
    *,
    route: Mapping[str, Any],
    replicacad_root: Path,
    runtime_root: Path,
) -> MixedCaptureResult:
    repository = Path(__file__).resolve().parents[3]
    legacy = repository.parent / "AVEngine"
    roots: list[tuple[str, Path]] = [
        ("AVENGINE_CAPTURE_ROOT", capture.output_dir.resolve()),
        ("AVENGINE_REPOSITORY_ROOT", repository.resolve()),
        ("AVENGINE_HABITAT_RUNTIME_ROOT", runtime_root.resolve()),
        ("AVENGINE_REPLICACAD_ROOT", replicacad_root.resolve()),
    ]
    if legacy.is_dir():
        roots.insert(2, ("AVENGINE_LEGACY_ROOT", legacy.resolve()))
    human_portability = _portableize_generated_human_runtime_documents(
        capture.output_dir, roots
    )
    evidence = _portableize_evidence_paths(dict(capture.evidence), roots)
    evidence.pop("evidence_content_sha256", None)
    _rebind_portable_human_runtime_references(evidence, human_portability)
    _bind_capture_identity_and_emitter_anchors(evidence, route)
    evidence["path_roots"] = {
        root_id: {
            "locator_kind": "environment_or_bundle_root",
            "absolute_path_recorded": False,
        }
        for root_id, _root in roots
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    write_json(capture.output_dir / "evidence.json", evidence)
    return MixedCaptureResult(
        output_dir=capture.output_dir,
        rgb=capture.rgb,
        semantic=capture.semantic,
        actor_world_matrices=capture.actor_world_matrices,
        skin_root_world_matrices=capture.skin_root_world_matrices,
        anchor_positions_m=capture.anchor_positions_m,
        semantic_visibility_pixels=capture.semantic_visibility_pixels,
        records=capture.records,
        evidence=evidence,
    )


def _semantic_visibility_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items() if key != "per_frame_visible_pixels"
    }


def _verified_output_record_path(
    root: Path, record: Mapping[str, Any], *, owner: str
) -> Path:
    raw = record.get("path", record.get("relative_path"))
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ReplicaCADCaptureError(f"{owner} lacks a relative output path")
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReplicaCADCaptureError(f"{owner} escapes the capture package") from exc
    if not candidate.is_file():
        raise ReplicaCADCaptureError(f"{owner} is missing on readback")
    if record.get("byte_size") != candidate.stat().st_size:
        raise ReplicaCADCaptureError(f"{owner} byte size differs on readback")
    if record.get("sha256") != sha256_file(candidate):
        raise ReplicaCADCaptureError(f"{owner} SHA-256 differs on readback")
    return candidate


def _verified_json_document(
    path: Path,
    expected: Mapping[str, Any],
    *,
    owner: str,
    content_hash_key: str,
) -> dict[str, Any]:
    try:
        readback = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReplicaCADCaptureError(f"{owner} JSON readback failed: {exc}") from exc
    if readback != expected:
        raise ReplicaCADCaptureError(f"{owner} JSON differs on readback")
    declared = readback.get(content_hash_key)
    content = dict(readback)
    content.pop(content_hash_key, None)
    if declared != canonical_json_sha256(content):
        raise ReplicaCADCaptureError(f"{owner} content hash differs on readback")
    return readback


def _portable_capture_locator_path(
    root: Path, record: Mapping[str, Any], *, owner: str
) -> Path:
    if record.get("root_id") != "AVENGINE_CAPTURE_ROOT":
        raise ReplicaCADCaptureError(f"{owner} is not capture-root relative")
    raw = record.get("relative_path")
    if not isinstance(raw, str) or not raw:
        raise ReplicaCADCaptureError(f"{owner} lacks relative_path")
    return _verified_output_record_path(
        root,
        {**record, "path": raw},
        owner=owner,
    )


def _readback_portable_human_runtime(root: Path) -> None:
    human = root / "runtime/human"
    manifest = load_json(human / "human_runtime_manifest.json")
    content = dict(manifest)
    declared = content.pop("manifest_content_sha256", None)
    if declared != canonical_json_sha256(content):
        raise ReplicaCADCaptureError(
            "portable human runtime manifest content hash differs on final readback"
        )
    source = manifest.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("root_id") != "AVENGINE_LEGACY_ROOT"
        or not isinstance(source.get("relative_path"), str)
    ):
        raise ReplicaCADCaptureError(
            "portable human source does not bind AVENGINE_LEGACY_ROOT"
        )
    derived = manifest.get("derived")
    if not isinstance(derived, Mapping) or set(derived) != set(
        _HUMAN_DERIVED_FILENAMES
    ):
        raise ReplicaCADCaptureError(
            "portable human runtime derived file set differs on readback"
        )
    resolved: dict[str, Path] = {}
    for role, filename in _HUMAN_DERIVED_FILENAMES.items():
        record = derived.get(role)
        if not isinstance(record, Mapping):
            raise ReplicaCADCaptureError(
                f"portable human runtime derived.{role} is missing"
            )
        path = _portable_capture_locator_path(
            root, record, owner=f"portable human runtime derived.{role}"
        )
        if path != (human / filename).resolve():
            raise ReplicaCADCaptureError(
                f"portable human runtime derived.{role} path differs"
            )
        resolved[role] = path

    rebase = load_json(human / "rebase_report.json")
    for field, role in (("source", "promoted_glb"), ("output", "visual_glb")):
        record = rebase.get(field)
        if not isinstance(record, Mapping):
            raise ReplicaCADCaptureError(f"portable rebase {field} is missing")
        path = _portable_capture_locator_path(
            root, record, owner=f"portable human rebase {field}"
        )
        if path != resolved[role]:
            raise ReplicaCADCaptureError(
                f"portable human rebase {field} differs from manifest"
            )


def _assert_portable_json_bundle(
    root: Path, *, declared_root_ids: Sequence[str]
) -> None:
    """Reject stale staging paths and undeclared absolute JSON paths."""

    allowed = set(declared_root_ids)
    if "AVENGINE_CAPTURE_ROOT" not in allowed:
        raise ReplicaCADCaptureError("capture path_roots lacks AVENGINE_CAPTURE_ROOT")

    def inspect(value: Any, *, owner: str) -> None:
        if isinstance(value, Mapping):
            if "relative_path" in value:
                root_id = value.get("root_id")
                relative = value.get("relative_path")
                if root_id not in allowed:
                    raise ReplicaCADCaptureError(
                        f"{owner} uses undeclared portable root: {root_id!r}"
                    )
                if (
                    not isinstance(relative, str)
                    or not relative
                    or Path(relative).is_absolute()
                    or PureWindowsPath(relative).is_absolute()
                    or "\\" in relative
                    or ".." in Path(relative).parts
                ):
                    raise ReplicaCADCaptureError(f"{owner} has an unsafe relative_path")
            for key, item in value.items():
                inspect(item, owner=f"{owner}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                inspect(item, owner=f"{owner}[{index}]")
            return
        if not isinstance(value, str):
            return
        if ".staging-" in value or ".staging." in value:
            raise ReplicaCADCaptureError(f"{owner} retains a staging path")
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ReplicaCADCaptureError(f"{owner} retains an absolute path")
        for root_id in _PORTABLE_ROOT_REFERENCE.findall(value):
            if root_id not in allowed:
                raise ReplicaCADCaptureError(
                    f"{owner} references undeclared portable root {root_id}"
                )

    for path in sorted(root.rglob("*.json"), key=lambda item: item.as_posix()):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ReplicaCADCaptureError(
                f"bundle JSON readback failed for {path.relative_to(root)}: {exc}"
            ) from exc
        inspect(document, owner=path.relative_to(root).as_posix())


def _readback_replicacad_capture_result(result: ReplicaCADCaptureResult) -> None:
    """Re-open the complete staging package before immutable publication."""

    root = result.capture.output_dir.resolve()
    if not root.is_dir():
        raise ReplicaCADCaptureError("ReplicaCAD staging capture is missing")
    capture_evidence_path = root / "evidence.json"
    capture_evidence = _verified_json_document(
        capture_evidence_path,
        result.capture.evidence,
        owner="mixed capture evidence",
        content_hash_key="evidence_content_sha256",
    )
    arrays = capture_evidence.get("array_artifacts")
    if not isinstance(arrays, Mapping):
        raise ReplicaCADCaptureError("mixed capture evidence lacks array artifacts")
    retained_arrays = {
        "rgb": result.capture.rgb,
        "semantic": result.capture.semantic,
        "actor_world_matrices": result.capture.actor_world_matrices,
        "skin_root_world_matrices": result.capture.skin_root_world_matrices,
        "anchor_positions_m": result.capture.anchor_positions_m,
        "semantic_visibility_pixels": result.capture.semantic_visibility_pixels,
    }
    for name, expected in retained_arrays.items():
        record = arrays.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("readback_verified") is not True
        ):
            raise ReplicaCADCaptureError(f"{name} lacks a successful readback record")
        path = _verified_output_record_path(root, record, owner=f"{name} array")
        try:
            readback = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ReplicaCADCaptureError(
                f"{name} array readback failed: {exc}"
            ) from exc
        if (
            readback.shape != expected.shape
            or readback.dtype != expected.dtype
            or not np.array_equal(readback, expected)
        ):
            raise ReplicaCADCaptureError(f"{name} array differs on final readback")

    readback_record = capture_evidence.get("readback")
    frame_record = (
        readback_record.get("frame_records")
        if isinstance(readback_record, Mapping)
        else None
    )
    if not isinstance(frame_record, Mapping):
        raise ReplicaCADCaptureError("mixed capture frame readback record is missing")
    frame_path = _verified_output_record_path(
        root, frame_record, owner="frame readback"
    )
    try:
        retained_records = json.loads(frame_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReplicaCADCaptureError(f"frame readback JSON failed: {exc}") from exc
    expected_records = json.loads(
        json.dumps(list(result.capture.records), ensure_ascii=False, allow_nan=False)
    )
    if retained_records != expected_records:
        raise ReplicaCADCaptureError("frame records differ on final readback")

    gate = _verified_json_document(
        result.gate_evidence_path,
        result.gate_evidence,
        owner="ReplicaCAD gate evidence",
        content_hash_key="evidence_content_sha256",
    )
    if gate.get("gate_count") != 19 or gate.get("passed_gate_count") != 19:
        raise ReplicaCADCaptureError("ReplicaCAD gate readback is not 19/19")
    mixed_capture = gate.get("mixed_capture")
    capture_record = (
        mixed_capture.get("evidence") if isinstance(mixed_capture, Mapping) else None
    )
    if not isinstance(capture_record, Mapping):
        raise ReplicaCADCaptureError("ReplicaCAD gate lacks capture evidence binding")
    if capture_record.get("evidence_content_sha256") != capture_evidence.get(
        "evidence_content_sha256"
    ):
        raise ReplicaCADCaptureError(
            "ReplicaCAD gate capture content identity differs on readback"
        )
    _verified_output_record_path(
        root, capture_record, owner="gate-bound mixed capture evidence"
    )
    contact_sheet = gate.get("contact_sheet")
    contact_record = (
        contact_sheet.get("file") if isinstance(contact_sheet, Mapping) else None
    )
    if (
        not isinstance(contact_record, Mapping)
        or not isinstance(contact_sheet, Mapping)
        or contact_sheet.get("readback_verified") is not True
    ):
        raise ReplicaCADCaptureError("ReplicaCAD contact sheet lacks readback proof")
    contact_path = _verified_output_record_path(
        root, contact_record, owner="ReplicaCAD contact sheet"
    )
    if contact_path != result.contact_sheet_path.resolve():
        raise ReplicaCADCaptureError("ReplicaCAD contact sheet path differs")
    _readback_portable_human_runtime(root)
    path_roots = capture_evidence.get("path_roots")
    if not isinstance(path_roots, Mapping):
        raise ReplicaCADCaptureError("capture evidence lacks declared path_roots")
    _assert_portable_json_bundle(root, declared_root_ids=tuple(path_roots))


def _capture_replicacad_route_in_staging(
    *,
    route_manifest_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    human_runtime_glb_path: str | Path,
    beagle_animal_manifest_path: str | Path,
    beagle_m2_request_path: str | Path,
    output_dir: str | Path,
    replicacad_root: str | Path,
    runtime_root: str | Path | None = None,
    review_configuration_hook: Callable[..., Mapping[str, Any]] | None = None,
    review_scene_hook: Callable[..., Mapping[str, Any]] | None = None,
    review_scene_readback_hook: Callable[..., Mapping[str, Any]] | None = None,
) -> ReplicaCADCaptureResult:
    """Build one complete ``apt_0`` capture in an unpublished directory."""

    route_path = Path(route_manifest_path).resolve()
    route = load_replicacad_route_manifest(route_path)
    root = Path(replicacad_root).resolve()
    with _replicacad_root_environment(root):
        navigation = validate_replicacad_paths_and_placement(
            route=route,
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
            replicacad_root=root,
            runtime_root=runtime_root,
        )
        capture = capture_human_beagle_paths(
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
            human_runtime_glb_path=human_runtime_glb_path,
            beagle_animal_manifest_path=beagle_animal_manifest_path,
            beagle_m2_request_path=beagle_m2_request_path,
            human_root_path_m=navigation.paths.human,
            beagle_root_path_m=navigation.paths.beagle,
            output_dir=output_dir,
            runtime_root=runtime_root,
            route_provenance={
                "route_manifest": _repository_file_record(route_path),
                "route_id": route["route_id"],
                "path_generation": route["path_generation"],
                "path_consumption": (
                    "derived_once_from_manifest_endpoints_then_verbatim"
                ),
                "real_replicacad_preflight": navigation.record,
            },
            require_legacy_camera=False,
            human_semantic_id=int(route["semantic_ids"]["human0"]),
            beagle_semantic_id=int(route["semantic_ids"]["dog0"]),
            review_configuration_hook=review_configuration_hook,
            review_scene_hook=review_scene_hook,
            review_scene_readback_hook=review_scene_readback_hook,
        )
        capture = _portable_mixed_capture(
            capture,
            route=route,
            replicacad_root=root,
            runtime_root=discover_runtime_root(runtime_root),
        )

    semantic_visibility = _semantic_visibility_record(
        capture.semantic, route["semantic_ids"]
    )
    minimum_frames = int(route["visibility_gate"]["minimum_visible_frames_per_actor"])
    minimum_pixels = int(route["visibility_gate"]["minimum_visible_pixels_per_frame"])
    for actor_id in ("human0", "dog0"):
        record = semantic_visibility[actor_id]
        if record["visible_frame_count"] < minimum_frames:
            raise ReplicaCADCaptureError(
                f"{actor_id} visible-frame count is below ReplicaCAD gate"
            )
        if record["minimum_visible_pixels"] < minimum_pixels:
            raise ReplicaCADCaptureError(
                f"{actor_id} minimum visible pixels is below ReplicaCAD gate"
            )

    contact_sheet_path = capture.output_dir / CONTACT_SHEET_NAME
    contact_sheet = write_mp3d_contact_sheet(
        rgb=capture.rgb,
        semantic=capture.semantic,
        semantic_ids=route["semantic_ids"],
        output_path=contact_sheet_path,
    )
    gates = {
        "dataset_config_selected": True,
        "scene_instance_selected": True,
        "stage_surface_selected": True,
        "scene_instance_object_counts_match": True,
        "scene_lighting_count_matches": True,
        "declared_navmesh_loaded": True,
        "human_route_all_frames_navigable": True,
        "dog_route_all_frames_navigable": True,
        "routes_share_one_island": True,
        "routes_no_sliding": True,
        "actor_center_separation": True,
        "actor_route_clearance": True,
        "actor_rigid_object_center_clearance": (
            navigation.record.get("rigid_object_center_clearance", {}).get("status")
            == "pass"
        ),
        "camera_listener_floor_placement": True,
        "camera_actor_line_of_sight": True,
        "semantic_ids_absent_before_actor_creation": True,
        "human_semantic_visibility": True,
        "dog_semantic_visibility": True,
        "articulated_state_readback": capture.evidence.get("status") == "pass",
    }
    passed = sum(bool(value) for value in gates.values())
    evidence: dict[str, Any] = {
        "schema": REPLICACAD_GATE_SCHEMA,
        "status": "pass" if passed == len(gates) else "fail",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": (
            "ReplicaCAD apt_0 visual, actor-root navigation, placement, and "
            "visibility research review only; no full-body collision, acoustic "
            "package, material-truth, RLR, or room-admission claim"
        ),
        "room_id": route["room_id"],
        "request_id": route["request_id"],
        "route_id": route["route_id"],
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FRAME_RATE_HZ,
        "gate_count": len(gates),
        "passed_gate_count": passed,
        "gates": gates,
        "native_execution": {
            "habitat_sim": "pass",
            "rlr_audio_propagation": "not_run",
            "media_mux": "not_run",
        },
        "route_manifest": _repository_file_record(route_path),
        "room_manifest": _repository_file_record(room_manifest_path),
        "m1_request": _repository_file_record(m1_request_path),
        "replicacad_root_contract": {
            "root_id": "AVENGINE_REPLICACAD_ROOT",
            "locator_kind": "environment_root_relative",
            "redistribution": "external_required",
        },
        "pathfinder": navigation.record,
        "mixed_capture": {
            "evidence": {
                **_root_relative_file_record(
                    capture.output_dir / "evidence.json",
                    root=capture.output_dir,
                    root_id="AVENGINE_CAPTURE_ROOT",
                ),
                "evidence_content_sha256": capture.evidence["evidence_content_sha256"],
            },
            "semantic_visibility": {
                actor_id: _semantic_visibility_summary(record)
                for actor_id, record in semantic_visibility.items()
            },
        },
        "contact_sheet": contact_sheet,
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    gate_evidence_path = capture.output_dir / GATE_EVIDENCE_NAME
    write_json(gate_evidence_path, evidence)
    if evidence["status"] != "pass":
        raise ReplicaCADCaptureError("ReplicaCAD gate aggregation failed")
    return ReplicaCADCaptureResult(
        capture=capture,
        gate_evidence_path=gate_evidence_path,
        gate_evidence=evidence,
        contact_sheet_path=contact_sheet_path,
    )


def capture_replicacad_route(
    *,
    route_manifest_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    human_runtime_glb_path: str | Path,
    beagle_animal_manifest_path: str | Path,
    beagle_m2_request_path: str | Path,
    output_dir: str | Path,
    replicacad_root: str | Path,
    runtime_root: str | Path | None = None,
    review_configuration_hook: Callable[..., Mapping[str, Any]] | None = None,
    review_scene_hook: Callable[..., Mapping[str, Any]] | None = None,
    review_scene_readback_hook: Callable[..., Mapping[str, Any]] | None = None,
) -> ReplicaCADCaptureResult:
    """Build, read back, then atomically publish one immutable capture."""

    unresolved = Path(output_dir).expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    if os.path.lexists(unresolved):
        raise ReplicaCADCaptureError(
            f"refusing to replace existing capture output: {unresolved}"
        )
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    parent = unresolved.parent.resolve(strict=True)
    destination = parent / unresolved.name
    policy = WorkspacePathPolicy.from_roots([parent])
    try:
        destination = policy.resolve_output(
            destination, owner="ReplicaCAD capture package"
        )
    except (FileExistsError, ValueError) as exc:
        raise ReplicaCADCaptureError(str(exc)) from exc
    staging = policy.resolve_output(
        destination.with_name(f".{destination.name}.staging-{uuid4().hex}"),
        owner="ReplicaCAD capture staging directory",
    )
    try:
        staged = _capture_replicacad_route_in_staging(
            route_manifest_path=route_manifest_path,
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
            human_runtime_glb_path=human_runtime_glb_path,
            beagle_animal_manifest_path=beagle_animal_manifest_path,
            beagle_m2_request_path=beagle_m2_request_path,
            output_dir=staging,
            replicacad_root=replicacad_root,
            runtime_root=runtime_root,
            review_configuration_hook=review_configuration_hook,
            review_scene_hook=review_scene_hook,
            review_scene_readback_hook=review_scene_readback_hook,
        )
        if (
            staged.capture.output_dir.resolve() != staging
            or staged.gate_evidence_path.resolve() != staging / GATE_EVIDENCE_NAME
            or staged.contact_sheet_path.resolve() != staging / CONTACT_SHEET_NAME
        ):
            raise ReplicaCADCaptureError(
                "ReplicaCAD staging result escaped its publication directory"
            )
        _readback_replicacad_capture_result(staged)
        published = atomic_publish_directory(policy, staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    capture = MixedCaptureResult(
        output_dir=published,
        rgb=staged.capture.rgb,
        semantic=staged.capture.semantic,
        actor_world_matrices=staged.capture.actor_world_matrices,
        skin_root_world_matrices=staged.capture.skin_root_world_matrices,
        anchor_positions_m=staged.capture.anchor_positions_m,
        semantic_visibility_pixels=staged.capture.semantic_visibility_pixels,
        records=staged.capture.records,
        evidence=staged.capture.evidence,
    )
    return ReplicaCADCaptureResult(
        capture=capture,
        gate_evidence_path=published / GATE_EVIDENCE_NAME,
        gate_evidence=staged.gate_evidence,
        contact_sheet_path=published / CONTACT_SHEET_NAME,
    )


__all__ = [
    "CONTACT_SHEET_NAME",
    "GATE_EVIDENCE_NAME",
    "REPLICACAD_GATE_SCHEMA",
    "REPLICACAD_ROUTE_SCHEMA",
    "ReplicaCADCaptureError",
    "ReplicaCADCaptureResult",
    "ReplicaCADNavigationEvidence",
    "capture_replicacad_route",
    "derive_replicacad_route_paths",
    "load_replicacad_route_manifest",
    "validate_replicacad_paths_and_placement",
]
