"""Real-navmesh MP3D mixed human/Beagle visual canary.

The route manifest owns two deterministic center-line endpoints.  This module
expands each line to the formal 270 visual frames, loads the room's declared
navmesh with Habitat's real :class:`PathFinder`, and requires every frame
center to be navigable before delegating to :func:`capture_human_beagle_paths`.

Only actor centers are navigation-qualified.  The canary does not claim that
the complete articulated meshes have collision clearance, and it performs no
physics steps, acoustic rendering, or video muxing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
)
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m1.contracts import (
    load_and_validate_inputs as load_m1_inputs,
    validate_loaded_scene_asset_graph,
)
from avengine.m1.habitat_capture import (
    _make_configuration,
    _resolved_scene,
    discover_runtime_root,
)
from avengine.m5_1.legacy_route import FRAME_COUNT, FRAME_RATE_HZ
from avengine.m5_1.mixed_capture import (
    MixedCaptureResult,
    capture_human_beagle_paths,
)


MP3D_ROUTE_SCHEMA = "avengine_m5_1_mp3d_center_route_v1"
MP3D_GATE_SCHEMA = "avengine_m5_1_mp3d_mixed_visual_gate_v1"
CONTACT_SHEET_NAME = "mp3d_mixed_contact_sheet.png"
GATE_EVIDENCE_NAME = "mp3d_gate_evidence.json"
_CONTACT_FRAME_INDICES = tuple(
    int(value) for value in np.linspace(0, FRAME_COUNT - 1, 9).round()
)


class MP3DCaptureError(RuntimeError):
    """The MP3D route, navigation gate, or mixed capture failed."""


@dataclass(frozen=True)
class MP3DRoutePaths:
    """Two deterministic 270-frame actor-center trajectories."""

    human: np.ndarray
    beagle: np.ndarray


@dataclass(frozen=True)
class MP3DNavigationEvidence:
    """Readback from the real PathFinder loaded with the declared navmesh."""

    paths: MP3DRoutePaths
    record: Mapping[str, Any]


@dataclass(frozen=True)
class MP3DCaptureResult:
    """Generic mixed capture plus the MP3D-specific gate evidence."""

    capture: MixedCaptureResult
    gate_evidence_path: Path
    gate_evidence: Mapping[str, Any]
    contact_sheet_path: Path


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MP3DCaptureError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MP3DCaptureError(f"{owner} must be a finite number")
    return result


def _finite_point(value: Any, *, owner: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DCaptureError(f"{owner} must be a finite length-3 point") from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise MP3DCaptureError(f"{owner} must be a finite length-3 point")
    return point


def load_mp3d_route_manifest(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on the bounded MP3D route contract."""

    try:
        route = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise MP3DCaptureError(str(exc)) from exc
    if route.get("schema") != MP3D_ROUTE_SCHEMA:
        raise MP3DCaptureError(f"MP3D route schema must be {MP3D_ROUTE_SCHEMA!r}")
    for key in ("route_id", "room_id", "request_id"):
        if not isinstance(route.get(key), str) or not route[key]:
            raise MP3DCaptureError(f"MP3D route {key} must be a non-empty string")
    if route.get("frame_count") != FRAME_COUNT:
        raise MP3DCaptureError(f"MP3D route frame_count must be {FRAME_COUNT}")
    if route.get("frame_rate_hz") != FRAME_RATE_HZ:
        raise MP3DCaptureError(
            f"MP3D route frame_rate_hz must be {FRAME_RATE_HZ}"
        )
    if route.get("path_generation") != "linear_endpoint_interpolation_v1":
        raise MP3DCaptureError("unsupported MP3D path_generation")
    if route.get("center_navigation_semantics") != "actor_root_center_only":
        raise MP3DCaptureError(
            "MP3D route must declare actor_root_center_only navigation semantics"
        )

    routes = route.get("routes")
    if not isinstance(routes, dict) or set(routes) != {"human0", "dog0"}:
        raise MP3DCaptureError("MP3D routes must contain exactly human0 and dog0")
    for actor_id in ("human0", "dog0"):
        actor_route = routes[actor_id]
        if not isinstance(actor_route, dict):
            raise MP3DCaptureError(f"MP3D route {actor_id} must be an object")
        _finite_point(actor_route.get("start_m"), owner=f"{actor_id}.start_m")
        _finite_point(actor_route.get("end_m"), owner=f"{actor_id}.end_m")

    navigation = route.get("pathfinder_gate")
    if not isinstance(navigation, dict):
        raise MP3DCaptureError("MP3D route requires pathfinder_gate")
    if navigation.get("require_declared_navmesh") is not True:
        raise MP3DCaptureError("MP3D route must require the declared navmesh")
    if navigation.get("require_every_frame_navigable") is not True:
        raise MP3DCaptureError("MP3D route must require every frame navigable")
    if navigation.get("require_one_shared_island") is not True:
        raise MP3DCaptureError("MP3D route must require one shared navmesh island")
    if navigation.get("require_segment_no_sliding") is not True:
        raise MP3DCaptureError("MP3D route must require no-sliding route segments")
    maximum_snap_error = _finite_number(
        navigation.get("maximum_snap_error_m"),
        owner="pathfinder_gate.maximum_snap_error_m",
    )
    if maximum_snap_error <= 0.0:
        raise MP3DCaptureError("maximum_snap_error_m must be positive")
    maximum_y_delta = _finite_number(
        navigation.get("maximum_y_delta_m"),
        owner="pathfinder_gate.maximum_y_delta_m",
    )
    if maximum_y_delta <= 0.0:
        raise MP3DCaptureError("maximum_y_delta_m must be positive")
    maximum_step_error = _finite_number(
        navigation.get("maximum_step_endpoint_error_m"),
        owner="pathfinder_gate.maximum_step_endpoint_error_m",
    )
    if maximum_step_error <= 0.0:
        raise MP3DCaptureError("maximum_step_endpoint_error_m must be positive")

    separation = _finite_number(
        route.get("minimum_center_separation_m"),
        owner="minimum_center_separation_m",
    )
    if separation < 0.3:
        raise MP3DCaptureError("minimum_center_separation_m cannot be below 0.3")
    movement = route.get("movement_gate")
    if not isinstance(movement, dict):
        raise MP3DCaptureError("MP3D route requires movement_gate")
    for key in ("minimum_path_length_m", "minimum_endpoint_displacement_m"):
        if _finite_number(movement.get(key), owner=f"movement_gate.{key}") <= 0.0:
            raise MP3DCaptureError(f"movement_gate.{key} must be positive")

    semantic_ids = route.get("semantic_ids")
    if not isinstance(semantic_ids, dict) or set(semantic_ids) != {
        "human0",
        "dog0",
    }:
        raise MP3DCaptureError("semantic_ids must contain exactly human0 and dog0")
    retained_ids: list[int] = []
    for actor_id in ("human0", "dog0"):
        value = semantic_ids[actor_id]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MP3DCaptureError(f"semantic_ids.{actor_id} must be nonnegative int")
        retained_ids.append(value)
    if len(set(retained_ids)) != 2:
        raise MP3DCaptureError("human and dog semantic IDs must differ")
    visibility = route.get("visibility_gate")
    if not isinstance(visibility, dict):
        raise MP3DCaptureError("MP3D route requires visibility_gate")
    minimum_frames = visibility.get("minimum_visible_frames_per_actor")
    if (
        isinstance(minimum_frames, bool)
        or not isinstance(minimum_frames, int)
        or minimum_frames < 1
        or minimum_frames > FRAME_COUNT
    ):
        raise MP3DCaptureError(
            "minimum_visible_frames_per_actor must lie within the capture"
        )
    minimum_pixels = visibility.get("minimum_visible_pixels_per_frame")
    if (
        isinstance(minimum_pixels, bool)
        or not isinstance(minimum_pixels, int)
        or minimum_pixels < 1
    ):
        raise MP3DCaptureError(
            "minimum_visible_pixels_per_frame must be a positive integer"
        )
    return route


def derive_mp3d_route_paths(route: Mapping[str, Any]) -> MP3DRoutePaths:
    """Expand the two manifest endpoint pairs to exact 270-frame paths."""

    try:
        routes = route["routes"]
        human_start = _finite_point(routes["human0"]["start_m"], owner="human0.start_m")
        human_end = _finite_point(routes["human0"]["end_m"], owner="human0.end_m")
        dog_start = _finite_point(routes["dog0"]["start_m"], owner="dog0.start_m")
        dog_end = _finite_point(routes["dog0"]["end_m"], owner="dog0.end_m")
    except (KeyError, TypeError) as exc:
        raise MP3DCaptureError("MP3D route lacks endpoint data") from exc
    return MP3DRoutePaths(
        human=np.ascontiguousarray(
            np.linspace(human_start, human_end, FRAME_COUNT), dtype=np.float64
        ),
        beagle=np.ascontiguousarray(
            np.linspace(dog_start, dog_end, FRAME_COUNT), dtype=np.float64
        ),
    )


def _path_metrics(path: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return {
        "path_length_m": float(steps.sum()),
        "endpoint_displacement_m": float(np.linalg.norm(path[-1] - path[0])),
        "maximum_center_step_m": float(np.max(steps)),
    }


def _assert_route_geometry(
    route: Mapping[str, Any], paths: MP3DRoutePaths
) -> dict[str, Any]:
    separation = np.linalg.norm(paths.human - paths.beagle, axis=1)
    minimum_separation = float(np.min(separation))
    required_separation = float(route["minimum_center_separation_m"])
    if minimum_separation < required_separation:
        raise MP3DCaptureError(
            "MP3D actor centers violate minimum separation: "
            f"{minimum_separation} < {required_separation}"
        )
    required_length = float(route["movement_gate"]["minimum_path_length_m"])
    required_displacement = float(
        route["movement_gate"]["minimum_endpoint_displacement_m"]
    )
    movement: dict[str, dict[str, float]] = {}
    for actor_id, path in (("human0", paths.human), ("dog0", paths.beagle)):
        metrics = _path_metrics(path)
        if metrics["path_length_m"] < required_length:
            raise MP3DCaptureError(f"{actor_id} path is shorter than movement gate")
        if metrics["endpoint_displacement_m"] < required_displacement:
            raise MP3DCaptureError(
                f"{actor_id} endpoint displacement is below movement gate"
            )
        movement[actor_id] = metrics
    return {
        "minimum_center_separation_m": minimum_separation,
        "maximum_center_separation_m": float(np.max(separation)),
        "required_minimum_center_separation_m": required_separation,
        "movement": movement,
        "required_minimum_path_length_m": required_length,
        "required_minimum_endpoint_displacement_m": required_displacement,
    }


def _pathfinder_path_record(
    pathfinder: Any,
    path: np.ndarray,
    *,
    owner: str,
    maximum_snap_error_m: float,
    maximum_y_delta_m: float,
    maximum_step_endpoint_error_m: float,
) -> dict[str, Any]:
    navigable = np.asarray(
        [
            bool(
                pathfinder.is_navigable(
                    np.asarray(point, dtype=np.float64), maximum_y_delta_m
                )
            )
            for point in path
        ],
        dtype=np.bool_,
    )
    snapped = np.stack(
        [
            np.asarray(
                pathfinder.snap_point(np.asarray(point, dtype=np.float64)),
                dtype=np.float64,
            )
            for point in path
        ],
        axis=0,
    )
    if snapped.shape != path.shape or not np.all(np.isfinite(snapped)):
        raise MP3DCaptureError(f"PathFinder returned invalid snap points for {owner}")
    snap_errors = np.linalg.norm(snapped - path, axis=1)
    navigable_count = int(np.count_nonzero(navigable))
    maximum_snap_error = float(np.max(snap_errors))
    if navigable_count != FRAME_COUNT:
        failed = np.flatnonzero(~navigable).astype(int).tolist()
        raise MP3DCaptureError(
            f"{owner} has non-navigable center frames: {failed[:20]}"
        )
    if maximum_snap_error > maximum_snap_error_m:
        raise MP3DCaptureError(
            f"{owner} maximum navmesh snap error {maximum_snap_error} exceeds "
            f"{maximum_snap_error_m}"
        )
    islands = np.asarray(
        [int(pathfinder.get_island(point)) for point in snapped], dtype=np.int64
    )
    if islands.shape != (FRAME_COUNT,) or np.any(islands < 0):
        raise MP3DCaptureError(f"{owner} has an invalid navmesh island readback")
    unique_islands = np.unique(islands)
    if len(unique_islands) != 1:
        raise MP3DCaptureError(f"{owner} crosses navmesh islands")
    stepped = np.stack(
        [
            np.asarray(
                pathfinder.try_step_no_sliding(path[index], path[index + 1]),
                dtype=np.float64,
            )
            for index in range(FRAME_COUNT - 1)
        ],
        axis=0,
    )
    if stepped.shape != (FRAME_COUNT - 1, 3) or not np.all(np.isfinite(stepped)):
        raise MP3DCaptureError(f"PathFinder returned invalid no-sliding steps for {owner}")
    step_errors = np.linalg.norm(stepped - path[1:], axis=1)
    maximum_step_error = float(np.max(step_errors))
    passed_segment_count = int(
        np.count_nonzero(step_errors <= maximum_step_endpoint_error_m)
    )
    if passed_segment_count != FRAME_COUNT - 1:
        failed = np.flatnonzero(
            step_errors > maximum_step_endpoint_error_m
        ).astype(int)
        raise MP3DCaptureError(
            f"{owner} has no-sliding segment failures: {failed[:20].tolist()}"
        )
    return {
        "frame_count": FRAME_COUNT,
        "navigable_frame_count": navigable_count,
        "all_frames_navigable": True,
        "maximum_snap_error_m": maximum_snap_error,
        "required_maximum_snap_error_m": maximum_snap_error_m,
        "maximum_y_delta_m": maximum_y_delta_m,
        "island_id": int(unique_islands[0]),
        "unique_island_count": 1,
        "segment_count": FRAME_COUNT - 1,
        "no_sliding_passed_segment_count": passed_segment_count,
        "maximum_step_endpoint_error_m": maximum_step_error,
        "required_maximum_step_endpoint_error_m": (
            maximum_step_endpoint_error_m
        ),
        "trajectory_sha256": canonical_json_sha256(path.tolist()),
        "start_m": path[0].tolist(),
        "end_m": path[-1].tolist(),
    }


def _external_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_mp3d_paths_with_declared_navmesh(
    *,
    route: Mapping[str, Any],
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    runtime_root: str | Path | None = None,
) -> MP3DNavigationEvidence:
    """Load the declared MP3D navmesh and validate every actor center."""

    inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    if inputs.room["room_id"] != route["room_id"]:
        raise MP3DCaptureError("MP3D route room_id differs from room manifest")
    if inputs.request["request_id"] != route["request_id"]:
        raise MP3DCaptureError("MP3D route request_id differs from M1 request")
    runtime = discover_runtime_root(runtime_root)
    resolved_scene = _resolved_scene(inputs, runtime)
    raw_navmesh = resolved_scene.get("navmesh")
    if raw_navmesh is None:
        raise MP3DCaptureError("MP3D room does not resolve a declared navmesh")
    navmesh_path = Path(raw_navmesh).resolve()
    if not navmesh_path.is_file():
        raise MP3DCaptureError(f"declared MP3D navmesh is missing: {navmesh_path}")

    # The pinned editable Habitat build imports numpy-quaternion first.
    import quaternion as qt

    import habitat_sim

    paths = derive_mp3d_route_paths(route)
    geometry = _assert_route_geometry(route, paths)
    maximum_snap_error = float(route["pathfinder_gate"]["maximum_snap_error_m"])
    maximum_y_delta = float(route["pathfinder_gate"]["maximum_y_delta_m"])
    maximum_step_error = float(
        route["pathfinder_gate"]["maximum_step_endpoint_error_m"]
    )
    configuration, modality_to_uuid, _listener_uuid, configured_scene = (
        _make_configuration(
            inputs,
            runtime,
            Path(m1_request_path).resolve().parent / ".mp3d_preflight_not_retained",
        )
    )
    if Path(configured_scene["navmesh"]).resolve() != navmesh_path:
        raise MP3DCaptureError("M1 configuration changed the declared navmesh")
    with habitat_sim.Simulator(configuration) as simulator:
        loaded = bool(simulator.pathfinder.load_nav_mesh(str(navmesh_path)))
        if not loaded or not simulator.pathfinder.is_loaded:
            raise MP3DCaptureError(
                "PathFinder failed to load the declared MP3D navmesh"
            )
        graph_errors, loaded_graph = validate_loaded_scene_asset_graph(
            inputs,
            runtime,
            simulator,
            declared_navmesh_loaded=loaded,
        )
        if graph_errors:
            raise MP3DCaptureError(
                "MP3D loaded scene graph differs from declarations: "
                + "; ".join(graph_errors)
            )
        rig = inputs.request["primary_camera_rig"]
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
        baseline_observation = simulator.render_sensors(
            [simulator.sensors[semantic_uuid]]
        )
        baseline_semantic = np.asarray(baseline_observation[semantic_uuid])
        expected_shape = tuple(
            rig["shared_calibration"]["resolution_hw"]
        )
        if baseline_semantic.shape != expected_shape:
            raise MP3DCaptureError("MP3D baseline semantic shape differs from request")
        semantic_ids = {
            actor_id: int(route["semantic_ids"][actor_id])
            for actor_id in ("human0", "dog0")
        }
        baseline_counts = {
            actor_id: int(np.count_nonzero(baseline_semantic == semantic_id))
            for actor_id, semantic_id in semantic_ids.items()
        }
        if any(baseline_counts.values()):
            raise MP3DCaptureError(
                "MP3D actor semantic IDs collide with the no-actor room baseline"
            )

        pathfinder = simulator.pathfinder
        bounds = pathfinder.get_bounds()
        record = {
            "implementation": "habitat_sim.PathFinder",
            "declared_navmesh_loaded": True,
            "declared_navmesh": _external_file_record(navmesh_path),
            "loaded_scene_asset_graph": loaded_graph,
            "semantic_baseline": {
                "sensor_uuid": semantic_uuid,
                "shape": list(baseline_semantic.shape),
                "actor_semantic_ids": semantic_ids,
                "actor_id_pixel_counts": baseline_counts,
                "no_actor_id_collision": True,
            },
            "navigable_area_m2": float(pathfinder.navigable_area),
            "island_count": int(pathfinder.num_islands),
            "bounds_m": [
                [float(component) for component in bounds[0]],
                [float(component) for component in bounds[1]],
            ],
            "center_navigation_semantics": "actor_root_center_only",
            "routes": {
                "human0": _pathfinder_path_record(
                    pathfinder,
                    paths.human,
                    owner="human0",
                    maximum_snap_error_m=maximum_snap_error,
                    maximum_y_delta_m=maximum_y_delta,
                    maximum_step_endpoint_error_m=maximum_step_error,
                ),
                "dog0": _pathfinder_path_record(
                    pathfinder,
                    paths.beagle,
                    owner="dog0",
                    maximum_snap_error_m=maximum_snap_error,
                    maximum_y_delta_m=maximum_y_delta,
                    maximum_step_endpoint_error_m=maximum_step_error,
                ),
            },
            "geometry": geometry,
        }
    human_island = record["routes"]["human0"]["island_id"]
    dog_island = record["routes"]["dog0"]["island_id"]
    if human_island != dog_island:
        raise MP3DCaptureError("human and dog routes occupy different navmesh islands")
    record["shared_island_id"] = human_island
    return MP3DNavigationEvidence(paths=paths, record=record)


def _semantic_box(mask: np.ndarray, semantic_id: int) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask == semantic_id)
    if len(rows) == 0:
        return None
    return (
        int(np.min(columns)),
        int(np.min(rows)),
        int(np.max(columns)),
        int(np.max(rows)),
    )


def _semantic_visibility_record(
    semantic: np.ndarray, semantic_ids: Mapping[str, int]
) -> dict[str, Any]:
    array = np.asarray(semantic)
    if array.ndim != 3 or array.shape[0] != FRAME_COUNT:
        raise MP3DCaptureError("semantic capture must be [270,height,width]")
    result: dict[str, Any] = {}
    for actor_id in ("human0", "dog0"):
        semantic_id = int(semantic_ids[actor_id])
        masks = array == semantic_id
        counts = np.count_nonzero(masks, axis=(1, 2)).astype(np.int64)
        border = np.any(masks[:, 0, :], axis=1)
        border |= np.any(masks[:, -1, :], axis=1)
        border |= np.any(masks[:, :, 0], axis=1)
        border |= np.any(masks[:, :, -1], axis=1)
        result[actor_id] = {
            "semantic_id": semantic_id,
            "visible_frame_count": int(np.count_nonzero(counts > 0)),
            "minimum_visible_pixels": int(np.min(counts)),
            "minimum_visible_pixels_frame_index": int(np.argmin(counts)),
            "maximum_visible_pixels": int(np.max(counts)),
            "maximum_visible_pixels_frame_index": int(np.argmax(counts)),
            "border_touch_frame_count": int(np.count_nonzero(border)),
            "per_frame_visible_pixels": counts.tolist(),
            "per_frame_visible_pixels_sha256": canonical_json_sha256(counts.tolist()),
        }
    return result


def write_mp3d_contact_sheet(
    *,
    rgb: np.ndarray,
    semantic: np.ndarray,
    semantic_ids: Mapping[str, int],
    output_path: str | Path,
    frame_indices: Sequence[int] = _CONTACT_FRAME_INDICES,
) -> dict[str, Any]:
    """Write a nine-frame RGB sheet with semantic actor boxes and read it back."""

    rgb_array = np.asarray(rgb)
    semantic_array = np.asarray(semantic)
    if (
        rgb_array.ndim != 4
        or rgb_array.shape[0] != FRAME_COUNT
        or rgb_array.shape[-1] != 3
        or rgb_array.dtype != np.uint8
        or semantic_array.shape != rgb_array.shape[:3]
    ):
        raise MP3DCaptureError("contact-sheet arrays have incompatible shapes")
    indices = tuple(int(index) for index in frame_indices)
    if len(indices) != 9 or len(set(indices)) != 9:
        raise MP3DCaptureError("contact sheet requires nine distinct frames")
    if min(indices) < 0 or max(indices) >= FRAME_COUNT:
        raise MP3DCaptureError("contact-sheet frame index lies outside capture")
    height, width = rgb_array.shape[1:3]
    label_height = 24
    sheet = Image.new("RGB", (width * 3, (height + label_height) * 3), "black")
    draw = ImageDraw.Draw(sheet)
    styles = (
        ("human0", int(semantic_ids["human0"]), (0, 255, 255)),
        ("dog0", int(semantic_ids["dog0"]), (255, 220, 0)),
    )
    selected_visibility: list[dict[str, Any]] = []
    for tile_index, frame_index in enumerate(indices):
        column = tile_index % 3
        row = tile_index // 3
        origin_x = column * width
        origin_y = row * (height + label_height)
        sheet.paste(Image.fromarray(rgb_array[frame_index]), (origin_x, origin_y))
        visibility: dict[str, int] = {}
        for actor_id, semantic_id, color in styles:
            actor_mask = semantic_array[frame_index] == semantic_id
            pixels = int(np.count_nonzero(actor_mask))
            visibility[actor_id] = pixels
            box = _semantic_box(semantic_array[frame_index], semantic_id)
            if box is not None:
                x0, y0, x1, y1 = box
                draw.rectangle(
                    (origin_x + x0, origin_y + y0, origin_x + x1, origin_y + y1),
                    outline=color,
                    width=2,
                )
        draw.text(
            (origin_x + 5, origin_y + height + 5),
            f"frame {frame_index:03d}  H:{visibility['human0']} D:{visibility['dog0']}",
            fill=(255, 255, 255),
        )
        selected_visibility.append(
            {"frame_index": frame_index, "visible_pixels": visibility}
        )

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet_pixels = np.asarray(sheet).copy()
    sheet.save(destination, format="PNG", compress_level=6)
    with Image.open(destination) as readback:
        readback.load()
        readback_pixels = np.asarray(readback)
        if (
            readback.mode != "RGB"
            or readback.size != sheet.size
            or not np.array_equal(readback_pixels, sheet_pixels)
        ):
            raise MP3DCaptureError("MP3D contact sheet differs on readback")
    return {
        "file": file_record(destination, relative_to=destination.parent),
        "format": "PNG",
        "mode": "RGB",
        "size_wh": list(sheet.size),
        "selected_frames": selected_visibility,
        "semantic_box_colors_rgb": {
            "human0": [0, 255, 255],
            "dog0": [255, 220, 0],
        },
        "readback_verified": True,
    }


def _input_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def capture_mp3d_route(
    *,
    route_manifest_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    human_runtime_glb_path: str | Path,
    beagle_animal_manifest_path: str | Path,
    beagle_m2_request_path: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path | None = None,
) -> MP3DCaptureResult:
    """Run the independent 270-frame MP3D mixed visual gate."""

    route_path = Path(route_manifest_path).resolve()
    route = load_mp3d_route_manifest(route_path)
    navigation = validate_mp3d_paths_with_declared_navmesh(
        route=route,
        room_manifest_path=room_manifest_path,
        m1_request_path=m1_request_path,
        runtime_root=runtime_root,
    )
    route_provenance = {
        "route_manifest": _input_record(route_path),
        "route_id": route["route_id"],
        "path_generation": route["path_generation"],
        "human_trajectory_sha256": navigation.record["routes"]["human0"][
            "trajectory_sha256"
        ],
        "dog_trajectory_sha256": navigation.record["routes"]["dog0"][
            "trajectory_sha256"
        ],
        "path_consumption": "derived_once_from_manifest_endpoints_then_verbatim",
        "real_pathfinder_validation": navigation.record,
    }
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
        route_provenance=route_provenance,
        require_legacy_camera=False,
        human_semantic_id=int(route["semantic_ids"]["human0"]),
        beagle_semantic_id=int(route["semantic_ids"]["dog0"]),
    )
    visible = capture.evidence["readback"]["semantic_visible_frame_count"]
    if capture.evidence["frame_count"] != FRAME_COUNT:
        raise MP3DCaptureError("MP3D mixed capture returned the wrong frame count")
    semantic_visibility = _semantic_visibility_record(
        capture.semantic, route["semantic_ids"]
    )
    minimum_visible_frames = int(
        route["visibility_gate"]["minimum_visible_frames_per_actor"]
    )
    minimum_visible_pixels = int(
        route["visibility_gate"]["minimum_visible_pixels_per_frame"]
    )
    for actor_id in ("human0", "dog0"):
        record = semantic_visibility[actor_id]
        if record["visible_frame_count"] != int(visible[actor_id]):
            raise MP3DCaptureError(
                f"{actor_id} independent semantic count differs from mixed evidence"
            )
        if record["visible_frame_count"] < minimum_visible_frames:
            raise MP3DCaptureError(
                f"{actor_id} visible-frame count is below MP3D route gate"
            )
        if record["minimum_visible_pixels"] < minimum_visible_pixels:
            raise MP3DCaptureError(
                f"{actor_id} minimum semantic pixels are below MP3D route gate"
            )

    output = capture.output_dir
    contact_sheet_path = output / CONTACT_SHEET_NAME
    contact_sheet = write_mp3d_contact_sheet(
        rgb=capture.rgb,
        semantic=capture.semantic,
        semantic_ids=route["semantic_ids"],
        output_path=contact_sheet_path,
    )
    gates = [
        {
            "gate_id": "declared_navmesh_real_pathfinder_load",
            "status": "pass",
            "measured": navigation.record["declared_navmesh_loaded"],
            "required": True,
        },
        {
            "gate_id": "human_center_navigable_every_frame",
            "status": "pass",
            "measured": navigation.record["routes"]["human0"][
                "navigable_frame_count"
            ],
            "required": FRAME_COUNT,
        },
        {
            "gate_id": "dog_center_navigable_every_frame",
            "status": "pass",
            "measured": navigation.record["routes"]["dog0"][
                "navigable_frame_count"
            ],
            "required": FRAME_COUNT,
        },
        {
            "gate_id": "one_shared_navmesh_island",
            "status": "pass",
            "measured_island_id": navigation.record["shared_island_id"],
            "required": "same single nonnegative island for both routes",
        },
        {
            "gate_id": "human_segments_no_sliding",
            "status": "pass",
            "measured_passed_segments": navigation.record["routes"]["human0"][
                "no_sliding_passed_segment_count"
            ],
            "required": FRAME_COUNT - 1,
        },
        {
            "gate_id": "dog_segments_no_sliding",
            "status": "pass",
            "measured_passed_segments": navigation.record["routes"]["dog0"][
                "no_sliding_passed_segment_count"
            ],
            "required": FRAME_COUNT - 1,
        },
        {
            "gate_id": "actor_center_separation",
            "status": "pass",
            "measured_minimum_m": navigation.record["geometry"][
                "minimum_center_separation_m"
            ],
            "required_minimum_m": float(route["minimum_center_separation_m"]),
        },
        {
            "gate_id": "captured_actor_roots_match_route_paths",
            "status": "pass",
            "measured": True,
            "required": True,
            "verification": (
                "capture_human_beagle_paths exact actor_world_matrices readback"
            ),
        },
        {
            "gate_id": "human_actual_movement",
            "status": "pass",
            "measured": navigation.record["geometry"]["movement"]["human0"],
            "required_minimum_path_length_m": route["movement_gate"][
                "minimum_path_length_m"
            ],
            "required_minimum_endpoint_displacement_m": route["movement_gate"][
                "minimum_endpoint_displacement_m"
            ],
        },
        {
            "gate_id": "dog_actual_movement",
            "status": "pass",
            "measured": navigation.record["geometry"]["movement"]["dog0"],
            "required_minimum_path_length_m": route["movement_gate"][
                "minimum_path_length_m"
            ],
            "required_minimum_endpoint_displacement_m": route["movement_gate"][
                "minimum_endpoint_displacement_m"
            ],
        },
        {
            "gate_id": "no_actor_semantic_id_baseline_collision",
            "status": "pass",
            "measured": navigation.record["semantic_baseline"][
                "actor_id_pixel_counts"
            ],
            "required": {"human0": 0, "dog0": 0},
        },
        {
            "gate_id": "human_fixed_camera_visibility",
            "status": "pass",
            "measured_visible_frames": semantic_visibility["human0"][
                "visible_frame_count"
            ],
            "measured_minimum_pixels": semantic_visibility["human0"][
                "minimum_visible_pixels"
            ],
            "required_minimum_visible_frames": minimum_visible_frames,
            "required_minimum_pixels_per_frame": minimum_visible_pixels,
        },
        {
            "gate_id": "dog_fixed_camera_visibility",
            "status": "pass",
            "measured_visible_frames": semantic_visibility["dog0"][
                "visible_frame_count"
            ],
            "measured_minimum_pixels": semantic_visibility["dog0"][
                "minimum_visible_pixels"
            ],
            "required_minimum_visible_frames": minimum_visible_frames,
            "required_minimum_pixels_per_frame": minimum_visible_pixels,
        },
        {
            "gate_id": "contact_sheet_readback",
            "status": "pass",
            "measured": contact_sheet["readback_verified"],
            "required": True,
        },
    ]
    evidence: dict[str, Any] = {
        "schema": MP3D_GATE_SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "formal_view_ids": [],
        "review_view_ids": ["view0"],
        "route_id": route["route_id"],
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FRAME_RATE_HZ,
        "inputs": {
            "route_manifest": _input_record(route_path),
            "room_manifest": _input_record(room_manifest_path),
            "m1_request": _input_record(m1_request_path),
            "human_runtime_glb": _input_record(human_runtime_glb_path),
            "beagle_manifest": _input_record(beagle_animal_manifest_path),
            "beagle_m2_request": _input_record(beagle_m2_request_path),
        },
        "pathfinder": navigation.record,
        "mixed_capture": {
            "evidence": file_record(output / "evidence.json", relative_to=output),
            "evidence_content_sha256": capture.evidence["evidence_content_sha256"],
            "semantic_visible_frame_count": dict(visible),
            "independently_recomputed_semantic_visibility": semantic_visibility,
            "semantic_maximum_visible_pixels": dict(
                capture.evidence["readback"]["semantic_maximum_visible_pixels"]
            ),
            "maximum_articulation_readback_errors": dict(
                capture.evidence["readback"]["maximum_errors"]
            ),
        },
        "contact_sheet": contact_sheet,
        "gates": gates,
        "gate_count": len(gates),
        "passed_gate_count": len(gates),
        "claim_boundary": (
            "M5.1 MP3D mixed visual research canary; Pathfinder gates actor "
            "centers only; no mesh-clearance, room, asset, episode, acoustic, "
            "video, or dataset admission claim. Border-touch counts are "
            "diagnostic and this gate does not claim full-body framing."
        ),
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    gate_evidence_path = output / GATE_EVIDENCE_NAME
    _write_json(gate_evidence_path, evidence)
    if load_json(gate_evidence_path) != evidence:
        raise MP3DCaptureError("MP3D gate evidence differs on JSON readback")
    return MP3DCaptureResult(
        capture=capture,
        gate_evidence_path=gate_evidence_path,
        gate_evidence=evidence,
        contact_sheet_path=contact_sheet_path,
    )


__all__ = [
    "CONTACT_SHEET_NAME",
    "GATE_EVIDENCE_NAME",
    "MP3DCaptureError",
    "MP3DCaptureResult",
    "MP3D_GATE_SCHEMA",
    "MP3D_ROUTE_SCHEMA",
    "capture_mp3d_route",
    "derive_mp3d_route_paths",
    "load_mp3d_route_manifest",
    "validate_mp3d_paths_with_declared_navmesh",
    "write_mp3d_contact_sheet",
]
