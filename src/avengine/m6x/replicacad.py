"""Rebuild the retained ReplicaCAD review with live furnished-room obstacles.

This module intentionally does not recapture actors or rerender acoustics.  It
briefly loads the real ``apt_0`` scene and its declared navmesh, snapshots every
loaded rigid collision OBB, and uses that one live snapshot for both the
source-center gate and the diagnostic Topdown panel.  Retained RGB, articulated
emitter-link positions, and the existing binaural mixture are reused verbatim.

The collision claim is deliberately small: only the human and dog sound-source
centers are checked.  No body capsule or articulated-body volume is inferred.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from PIL import Image

from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import _make_configuration, discover_runtime_root
from avengine.m5_1.delivery import event_overlay_state, semantic_centroid_track
from avengine.m5_1.orientation import habitat_yaw_degrees_from_xyzw
from avengine.m5_1.replicacad_capture import (
    REPLICACAD_SCENE_ID,
    _assert_selected_closure,
    _replicacad_root_environment,
)
from avengine.m5_1.review import (
    SourceOverlayTrack,
    compose_annotated_frames,
    encode_annotated_review,
)
from avengine.m6x.geometry import (
    RuntimeObstacleMap,
    build_runtime_obstacle_map,
    evaluate_source_center_gate,
)
from avengine.m6x.topdown import render_runtime_topdown_frames


REPLICACAD_REVIEW_SCHEMA = "avengine_m6x_replicacad_obstacle_review_v1"
DEFAULT_EXPECTED_RIGID_COUNT = 113
_SOURCE_COLORS = ((42, 210, 220), (250, 120, 70))


class M6XReplicaCADError(RuntimeError):
    """The retained review or live ReplicaCAD room is inconsistent."""


@dataclass(frozen=True)
class RetainedReplicaCADReview:
    """Small retained-data surface needed to rebuild only the review media."""

    frame_count: int
    frame_rate_hz: int
    room_id: str
    rgb: np.ndarray
    semantic: np.ndarray
    trajectories_m: Mapping[str, np.ndarray]
    activity_by_frame: Mapping[str, np.ndarray]
    events_by_frame: Mapping[str, tuple[str | None, ...]]
    bindings: Mapping[str, Mapping[str, Any]]
    program_sources: Mapping[str, Mapping[str, Any]]
    listener_position_m: tuple[float, float, float]
    listener_yaw_deg: float
    camera_hfov_degrees: float
    mixture_wav: Path


@dataclass(frozen=True)
class ReplicaCADRuntimeReview:
    """In-memory result produced while the native scene is still live."""

    obstacle_map: RuntimeObstacleMap
    source_center_gate: Mapping[str, Any]
    topdown_frames: np.ndarray
    annotated_frames: np.ndarray


@dataclass(frozen=True)
class _LoadedReplicaCADScene:
    simulator: Any
    magnum: Any
    navmesh_path: Path
    expected_rigid_count: int
    expected_articulated_count: int


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise M6XReplicaCADError(f"{owner} must be a positive integer")
    return value


def _required_file(path: str | Path, *, owner: str) -> Path:
    result = Path(path).resolve()
    if not result.is_file():
        raise M6XReplicaCADError(f"{owner} is missing: {result}")
    return result


def _load_array(path: Path, *, owner: str) -> np.ndarray:
    try:
        value = np.load(_required_file(path, owner=owner), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise M6XReplicaCADError(f"{owner} cannot be read: {exc}") from exc
    return np.ascontiguousarray(value)


def _sound_class(source: Mapping[str, Any]) -> str:
    taxonomy = source.get("voice_taxonomy")
    field = "vocalization_type"
    if not isinstance(taxonomy, Mapping):
        taxonomy = source.get("call_taxonomy")
        field = "call_type"
    if not isinstance(taxonomy, Mapping):
        return "unknown"
    value = taxonomy.get(field)
    return str(value) if isinstance(value, str) and value else "unknown"


def load_retained_replicacad_review(
    *,
    capture_dir: str | Path,
    delivery_dir: str | Path,
    m1_request_path: str | Path,
) -> RetainedReplicaCADReview:
    """Load RGB and exact articulated emitter centers from retained M5.1 data."""

    capture = Path(capture_dir).resolve()
    delivery = Path(delivery_dir).resolve()
    evidence_path = _required_file(capture / "evidence.json", owner="capture evidence")
    binding_path = _required_file(
        delivery / "source_actor_bindings.json", owner="source bindings"
    )
    program_path = _required_file(
        delivery / "source_program_reuse.json", owner="source program"
    )
    request_path = _required_file(m1_request_path, owner="M1 request")

    try:
        evidence = load_json(evidence_path)
        binding_document = load_json(binding_path)
        program = load_json(program_path)
        request = load_json(request_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise M6XReplicaCADError(f"retained review JSON cannot be read: {exc}") from exc

    frame_count = _positive_int(evidence.get("frame_count"), owner="frame_count")
    frame_rate = _positive_int(evidence.get("frame_rate_hz"), owner="frame_rate_hz")
    if evidence.get("status") != "pass":
        raise M6XReplicaCADError("retained visual capture is not a pass")
    room_id = evidence.get("room_id")
    if not isinstance(room_id, str) or not room_id:
        raise M6XReplicaCADError("retained capture lacks room_id")

    rgb = _load_array(capture / "arrays/rgb.npy", owner="retained RGB")
    semantic = _load_array(
        capture / "arrays/semantic.npy", owner="retained semantic frames"
    )
    anchors = _load_array(
        capture / "arrays/anchor_positions_m.npy", owner="emitter anchors"
    ).astype(np.float64, copy=False)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 4
        or rgb.shape[0] != frame_count
        or rgb.shape[-1] != 3
    ):
        raise M6XReplicaCADError("retained RGB must be uint8 [frame,height,width,3]")
    if (
        semantic.ndim != 3
        or semantic.shape[0] != frame_count
        or semantic.dtype.kind not in "iu"
    ):
        raise M6XReplicaCADError("retained semantic data must be integer [frame,h,w]")
    if (
        anchors.ndim != 3
        or anchors.shape[0] != frame_count
        or anchors.shape[2] != 3
        or not np.all(np.isfinite(anchors))
    ):
        raise M6XReplicaCADError("emitter anchors must be finite [frame,anchor,3]")

    bindings_raw = binding_document.get("bindings")
    sources_raw = program.get("sources")
    if not isinstance(bindings_raw, Mapping) or not isinstance(sources_raw, list):
        raise M6XReplicaCADError("retained source bindings/program are malformed")
    bindings = {
        str(source_id): dict(value)
        for source_id, value in bindings_raw.items()
        if isinstance(source_id, str) and isinstance(value, Mapping)
    }
    program_sources = {
        str(value.get("source_id")): dict(value)
        for value in sources_raw
        if isinstance(value, Mapping) and isinstance(value.get("source_id"), str)
    }
    expected_sources = {"source0", "source1"}
    if set(bindings) != expected_sources or set(program_sources) != expected_sources:
        raise M6XReplicaCADError("ReplicaCAD review requires source0 and source1")
    if binding_document.get("room_id") != room_id:
        raise M6XReplicaCADError("source bindings and capture room differ")

    anchor_order = evidence.get("anchor_order")
    if not isinstance(anchor_order, list) or len(anchor_order) != anchors.shape[1]:
        raise M6XReplicaCADError("capture anchor_order differs from anchor array")
    trajectories: dict[str, np.ndarray] = {}
    activity: dict[str, np.ndarray] = {}
    events: dict[str, tuple[str | None, ...]] = {}
    for source_id in sorted(expected_sources):
        binding = bindings[source_id]
        if binding.get("source_id") != source_id:
            raise M6XReplicaCADError(f"{source_id} binding identity differs")
        index = binding.get("emitter_anchor_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < anchors.shape[1]
        ):
            raise M6XReplicaCADError(f"{source_id} emitter anchor index is invalid")
        if anchor_order[index] != binding.get("capture_anchor_id"):
            raise M6XReplicaCADError(f"{source_id} emitter anchor identity differs")
        trajectories[source_id] = np.ascontiguousarray(anchors[:, index, :])
        try:
            event_state, active_state = event_overlay_state(
                program_sources[source_id], frame_count
            )
        except ValueError as exc:
            raise M6XReplicaCADError(
                f"{source_id} event program is invalid: {exc}"
            ) from exc
        events[source_id] = event_state
        activity[source_id] = np.ascontiguousarray(active_state, dtype=np.bool_)

    clip = program.get("clip_time_and_audio_contract")
    if not isinstance(clip, Mapping) or clip.get("frame_count") != frame_count:
        raise M6XReplicaCADError("source program frame count differs from capture")
    rig = request.get("primary_camera_rig")
    if not isinstance(rig, Mapping) or request.get("room_id") != room_id:
        raise M6XReplicaCADError("M1 request and retained capture room differ")
    world_from_rig = rig.get("world_from_rig")
    calibration = rig.get("shared_calibration")
    if not isinstance(world_from_rig, Mapping) or not isinstance(calibration, Mapping):
        raise M6XReplicaCADError("M1 camera/listener contract is malformed")
    listener = np.asarray(world_from_rig.get("translation_m"), dtype=np.float64)
    rotation = np.asarray(world_from_rig.get("rotation_xyzw"), dtype=np.float64)
    if listener.shape != (3,) or not np.all(np.isfinite(listener)):
        raise M6XReplicaCADError("listener position must be finite [3]")
    try:
        listener_yaw = habitat_yaw_degrees_from_xyzw(rotation)
        hfov = float(calibration["hfov_degrees"])
    except (KeyError, TypeError, ValueError) as exc:
        raise M6XReplicaCADError(
            f"camera/listener orientation is invalid: {exc}"
        ) from exc
    if not 0.0 < hfov < 180.0:
        raise M6XReplicaCADError("camera HFOV must lie within (0,180)")

    mixture = _required_file(
        delivery / "audio/binaural/mixture.wav", owner="retained binaural mixture"
    )
    return RetainedReplicaCADReview(
        frame_count=frame_count,
        frame_rate_hz=frame_rate,
        room_id=room_id,
        rgb=rgb,
        semantic=semantic,
        trajectories_m=trajectories,
        activity_by_frame=activity,
        events_by_frame=events,
        bindings=bindings,
        program_sources=program_sources,
        listener_position_m=tuple(float(value) for value in listener),
        listener_yaw_deg=float(listener_yaw),
        camera_hfov_degrees=hfov,
        mixture_wav=mixture,
    )


def _source_tracks(
    retained: RetainedReplicaCADReview, gate: Mapping[str, Any]
) -> tuple[SourceOverlayTrack, ...]:
    tracks: list[SourceOverlayTrack] = []
    for color_index, source_id in enumerate(sorted(retained.trajectories_m)):
        binding = retained.bindings[source_id]
        source = retained.program_sources[source_id]
        semantic_id = binding.get("semantic_id")
        if isinstance(semantic_id, bool) or not isinstance(semantic_id, int):
            raise M6XReplicaCADError(f"{source_id} semantic ID is invalid")
        markers = semantic_centroid_track(retained.semantic, semantic_id)
        source_gate = gate.get("sources", {}).get(source_id)
        frames = source_gate.get("frames") if isinstance(source_gate, Mapping) else None
        if not isinstance(frames, list) or len(frames) != retained.frame_count:
            raise M6XReplicaCADError(f"{source_id} gate frame records are missing")
        clearances: list[float] = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise M6XReplicaCADError(f"{source_id} gate frame is malformed")
            nav = float(frame["navmesh_clearance_m"])
            rigid_raw = frame.get("rigid_obstacle_clearance_m")
            rigid = float(rigid_raw) if rigid_raw is not None else nav
            clearances.append(min(nav, rigid))
        actor_class = str(
            binding.get("actor_class", source.get("asset_class", "actor"))
        )
        label = "BEAGLE" if actor_class == "dog" else actor_class.upper()
        tracks.append(
            SourceOverlayTrack(
                source_id=source_id,
                label=label,
                asset_class=str(source.get("asset_class", actor_class)),
                sound_class=_sound_class(source),
                color_rgb=_SOURCE_COLORS[color_index % len(_SOURCE_COLORS)],
                positions_m=retained.trajectories_m[source_id],
                current_event_by_frame=retained.events_by_frame[source_id],
                active_by_frame=tuple(
                    bool(value) for value in retained.activity_by_frame[source_id]
                ),
                true_flags=("source_center_only", "runtime_obstacle_snapshot"),
                center_clearance_m=np.asarray(clearances, dtype=np.float64),
                main_marker_xy=markers,
            )
        )
    return tuple(tracks)


def inspect_replicacad_articulated_room_objects(
    articulated_object_manager: Any,
    pathfinder: Any,
    *,
    floor_height_m: float,
) -> Mapping[str, Any]:
    """Inventory room-native articulated furniture without treating actors as it.

    Habitat exposes a current-pose visual ``aabb`` for an articulated object
    and per-link ``SceneNode.cumulative_bb`` values, but not the same public
    collision-shape AABB used by rigid objects.  Turning those visual bounds
    into collision OBBs would therefore overstate the gate.  We retain the
    inventory and probe each root/link anchor against the declared navmesh,
    while leaving those six objects represented only by that navmesh.
    """

    try:
        objects = articulated_object_manager.get_objects_by_handle_substring().values()
    except (AttributeError, TypeError) as exc:
        raise M6XReplicaCADError(
            "Habitat articulated-object manager is unavailable"
        ) from exc
    records: list[dict[str, Any]] = []
    probe_count = 0
    navigable_probe_count = 0
    for value in sorted(objects, key=lambda item: str(item.handle).encode("utf-8")):
        try:
            bounds = value.aabb
            lower = np.asarray(bounds.min, dtype=np.float64)
            upper = np.asarray(bounds.max, dtype=np.float64)
            link_ids = tuple(int(link_id) for link_id in value.get_link_ids())
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise M6XReplicaCADError(
                f"articulated room object {value.handle} bounds API failed"
            ) from exc
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(upper < lower)
        ):
            raise M6XReplicaCADError(
                f"articulated room object {value.handle} has invalid visual bounds"
            )
        links: list[dict[str, Any]] = []
        for link_id in (-1, *link_ids):
            try:
                node = (
                    value.root_scene_node
                    if link_id == -1
                    else value.get_link_scene_node(link_id)
                )
                link_name = (
                    "BASE" if link_id == -1 else str(value.get_link_name(link_id))
                )
                position = np.asarray(node.absolute_translation, dtype=np.float64)
                link_bounds = node.cumulative_bb
                link_lower = np.asarray(link_bounds.min, dtype=np.float64)
                link_upper = np.asarray(link_bounds.max, dtype=np.float64)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise M6XReplicaCADError(
                    f"articulated room object {value.handle} link {link_id} failed"
                ) from exc
            if (
                position.shape != (3,)
                or link_lower.shape != (3,)
                or link_upper.shape != (3,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(link_lower))
                or not np.all(np.isfinite(link_upper))
            ):
                raise M6XReplicaCADError(
                    f"articulated room object {value.handle} link bounds are invalid"
                )
            floor_query = np.asarray(
                (position[0], float(floor_height_m), position[2]),
                dtype=np.float64,
            )
            navigable = bool(pathfinder.is_navigable(floor_query, 0.25))
            snapped = np.asarray(pathfinder.snap_point(floor_query), dtype=np.float64)
            clearance = float(pathfinder.distance_to_closest_obstacle(snapped, 10.0))
            probe_count += 1
            navigable_probe_count += int(navigable)
            links.append(
                {
                    "link_id": link_id,
                    "link_name": link_name,
                    "absolute_translation_m": position.tolist(),
                    "scene_node_cumulative_bb_local_m": {
                        "minimum": link_lower.tolist(),
                        "maximum": link_upper.tolist(),
                    },
                    "floor_anchor_navigable": navigable,
                    "floor_anchor_navmesh_clearance_m": clearance,
                }
            )
        records.append(
            {
                "object_id": int(value.object_id),
                "handle": str(value.handle),
                "link_count": len(link_ids),
                "available_bounds_api": [
                    "ArticulatedObject.aabb",
                    "SceneNode.cumulative_bb_per_link",
                ],
                "public_collision_shape_aabb_available": hasattr(
                    value, "collision_shape_aabb"
                ),
                "articulated_object_aabb_local_m": {
                    "minimum": lower.tolist(),
                    "maximum": upper.tolist(),
                },
                "links": links,
            }
        )
    all_non_navigable = probe_count > 0 and navigable_probe_count == 0
    return {
        "object_count": len(records),
        "objects": records,
        "loaded_before_scenario_actor_injection": True,
        "scenario_human_and_dog_included": False,
        "navmesh_anchor_probe_count": probe_count,
        "navmesh_navigable_anchor_probe_count": navigable_probe_count,
        "all_root_and_link_anchor_xz_non_navigable": all_non_navigable,
        "separate_collision_footprints_added": False,
        "representation": "declared_navmesh_only",
        "decision": (
            "not separately added: the public Python API exposes current-pose "
            "visual bounds, not rigid-equivalent collision-shape OBBs; root/link "
            "anchor probes are recorded instead"
        ),
    }


def build_replicacad_runtime_review(
    *,
    pathfinder: Any,
    object_manager: Any,
    magnum: Any,
    retained: RetainedReplicaCADReview,
    floor_height_m: float,
    meters_per_pixel: float = 0.02,
    expected_rigid_count: int = DEFAULT_EXPECTED_RIGID_COUNT,
) -> ReplicaCADRuntimeReview:
    """Build gate and review frames from one still-live ReplicaCAD snapshot."""

    expected = _positive_int(expected_rigid_count, owner="expected_rigid_count")
    obstacle_map = build_runtime_obstacle_map(
        pathfinder,
        object_manager,
        magnum,
        floor_height_m=floor_height_m,
        meters_per_pixel=meters_per_pixel,
    )
    if len(obstacle_map.rigid_obstacles) != expected:
        raise M6XReplicaCADError(
            "live ReplicaCAD rigid obstacle count "
            f"{len(obstacle_map.rigid_obstacles)} != {expected}"
        )
    gate = evaluate_source_center_gate(
        pathfinder,
        obstacle_map,
        retained.trajectories_m,
        maximum_floor_snap_xz_m=0.02,
        maximum_floor_y_delta_m=0.25,
        minimum_navmesh_clearance_m=0.0,
        minimum_rigid_clearance_m=0.0,
    )
    labels: dict[str, str] = {}
    colors: dict[str, tuple[int, int, int]] = {}
    for color_index, source_id in enumerate(sorted(retained.trajectories_m)):
        actor = str(retained.bindings[source_id].get("actor_class", "source"))
        labels[source_id] = "BEAGLE" if actor == "dog" else actor.upper()
        colors[source_id] = _SOURCE_COLORS[color_index % len(_SOURCE_COLORS)]
    topdown = render_runtime_topdown_frames(
        obstacle_map,
        retained.trajectories_m,
        listener_position_m=retained.listener_position_m,
        listener_yaw_deg=retained.listener_yaw_deg,
        camera_hfov_degrees=retained.camera_hfov_degrees,
        source_activity_by_frame=retained.activity_by_frame,
        source_labels=labels,
        source_colors=colors,
        size_wh=(640, 480),
        rigid_label_limit=0,
    )
    tracks = _source_tracks(retained, gate)
    annotated = compose_annotated_frames(
        main_rgb=retained.rgb,
        topdown_rgb=topdown,
        tracks=tracks,
        clip_id="replicacad_human_beagle_runtime_obstacles",
        room_id=retained.room_id,
        review_stage_label="M6.x",
        listener_position_m=retained.listener_position_m,
        listener_yaw_deg=retained.listener_yaw_deg,
        aggregate_true_flags=(
            "live_navmesh_and_rigid_obbs",
            "source_center_only",
            "audio_360_no_camera_fov_cutoff",
        ),
        center_gate_pass=gate.get("status") == "pass",
        fps=retained.frame_rate_hz,
    )
    return ReplicaCADRuntimeReview(
        obstacle_map=obstacle_map,
        source_center_gate=gate,
        topdown_frames=topdown,
        annotated_frames=annotated,
    )


@contextmanager
def _load_real_replicacad_scene(
    *,
    replicacad_root: Path,
    runtime_root: Path,
    room_manifest_path: Path,
    m1_request_path: Path,
    output_dir: Path,
) -> Iterator[_LoadedReplicaCADScene]:
    with _replicacad_root_environment(replicacad_root):
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
        closure = _assert_selected_closure(
            room_inputs=room_inputs,
            runtime=runtime_root,
            root=replicacad_root,
        )
        try:
            scene_instance = load_json(closure["scene_instance"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise M6XReplicaCADError(
                f"ReplicaCAD scene instance cannot be read: {exc}"
            ) from exc
        instances = scene_instance.get("object_instances")
        articulated_instances = scene_instance.get("articulated_object_instances")
        if not isinstance(instances, list) or not isinstance(
            articulated_instances, list
        ):
            raise M6XReplicaCADError("ReplicaCAD scene instance lacks room objects")

        # This import order is required by the pinned audio-enabled build.
        import quaternion as _quaternion  # noqa: F401

        import habitat_sim
        import magnum as mn

        configuration, _modalities, _listener, configured_scene = _make_configuration(
            room_inputs, runtime_root, output_dir
        )
        if configured_scene.get("scene_id") != REPLICACAD_SCENE_ID:
            raise M6XReplicaCADError("Habitat configuration did not select apt_0")
        with habitat_sim.Simulator(configuration) as simulator:
            if not simulator.pathfinder.load_nav_mesh(str(closure["navmesh"])):
                raise M6XReplicaCADError("declared apt_0 navmesh failed to load")
            if not simulator.pathfinder.is_loaded:
                raise M6XReplicaCADError("declared apt_0 navmesh is not live")
            if str(simulator.curr_scene_name) != REPLICACAD_SCENE_ID:
                raise M6XReplicaCADError("Habitat did not load the apt_0 scene")
            actual = len(
                simulator.get_rigid_object_manager().get_objects_by_handle_substring()
            )
            actual_articulated = len(
                simulator.get_articulated_object_manager().get_objects_by_handle_substring()
            )
            if actual != len(instances):
                raise M6XReplicaCADError(
                    f"live rigid object count {actual} != scene instance {len(instances)}"
                )
            if actual_articulated != len(articulated_instances):
                raise M6XReplicaCADError(
                    "live articulated room object count "
                    f"{actual_articulated} != scene instance "
                    f"{len(articulated_instances)}"
                )
            yield _LoadedReplicaCADScene(
                simulator=simulator,
                magnum=mn,
                navmesh_path=closure["navmesh"],
                expected_rigid_count=len(instances),
                expected_articulated_count=len(articulated_instances),
            )


def rebuild_replicacad_obstacle_review(
    *,
    replicacad_root: str | Path,
    runtime_root: str | Path,
    capture_dir: str | Path,
    delivery_dir: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    output_dir: str | Path,
    meters_per_pixel: float = 0.02,
) -> Mapping[str, Any]:
    """Create a new 18-second diagnostic MP4 without capture or RLR work."""

    dataset = Path(replicacad_root).resolve()
    runtime = discover_runtime_root(runtime_root)
    room_manifest = _required_file(room_manifest_path, owner="room manifest")
    request_path = _required_file(m1_request_path, owner="M1 request")
    output = Path(output_dir).resolve()
    if output.exists() or os.path.lexists(output):
        raise M6XReplicaCADError(f"refusing to replace output directory: {output}")
    if not dataset.is_dir():
        raise M6XReplicaCADError(f"ReplicaCAD dataset root is missing: {dataset}")

    retained = load_retained_replicacad_review(
        capture_dir=capture_dir,
        delivery_dir=delivery_dir,
        m1_request_path=request_path,
    )
    request = load_json(request_path)
    qa_views = [
        item
        for item in request.get("qa_views", [])
        if isinstance(item, Mapping) and item.get("kind") == "topdown"
    ]
    if len(qa_views) != 1:
        raise M6XReplicaCADError("M1 request must declare one Topdown QA view")
    try:
        floor_height = float(qa_views[0]["height_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise M6XReplicaCADError(f"Topdown floor height is invalid: {exc}") from exc

    output.mkdir(parents=True)
    with _load_real_replicacad_scene(
        replicacad_root=dataset,
        runtime_root=runtime,
        room_manifest_path=room_manifest,
        m1_request_path=request_path,
        output_dir=output,
    ) as loaded:
        if loaded.expected_rigid_count != DEFAULT_EXPECTED_RIGID_COUNT:
            raise M6XReplicaCADError(
                "selected apt_0 scene instance must contain "
                f"{DEFAULT_EXPECTED_RIGID_COUNT} rigid objects, found "
                f"{loaded.expected_rigid_count}"
            )
        result = build_replicacad_runtime_review(
            pathfinder=loaded.simulator.pathfinder,
            object_manager=loaded.simulator.get_rigid_object_manager(),
            magnum=loaded.magnum,
            retained=retained,
            floor_height_m=floor_height,
            meters_per_pixel=meters_per_pixel,
            expected_rigid_count=loaded.expected_rigid_count,
        )
        articulated_room_objects = inspect_replicacad_articulated_room_objects(
            loaded.simulator.get_articulated_object_manager(),
            loaded.simulator.pathfinder,
            floor_height_m=floor_height,
        )
        if (
            articulated_room_objects["object_count"]
            != loaded.expected_articulated_count
        ):
            raise M6XReplicaCADError(
                "articulated room object inventory changed during review"
            )
        obstacle_summary = result.obstacle_map.summary()

    room_dir = output / "room"
    video_path = output / "videos/replicacad_runtime_obstacles_diagnostic.mp4"
    write_json(room_dir / "runtime_obstacle_map.json", obstacle_summary)
    write_json(room_dir / "articulated_room_objects.json", articulated_room_objects)
    write_json(output / "source_center_gate.json", result.source_center_gate)
    room_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.topdown_frames[0], mode="RGB").save(
        room_dir / "runtime_obstacle_map.png"
    )
    video_record = dict(
        encode_annotated_review(
            result.annotated_frames,
            video_path,
            fps=retained.frame_rate_hz,
            audio_path=retained.mixture_wav,
        )
    )
    video_record["path"] = video_path.relative_to(output).as_posix()
    status = {
        "schema": REPLICACAD_REVIEW_SCHEMA,
        "status": result.source_center_gate.get("status"),
        "room_id": retained.room_id,
        "scene_id": REPLICACAD_SCENE_ID,
        "frame_count": retained.frame_count,
        "frame_rate_hz": retained.frame_rate_hz,
        "duration_seconds": retained.frame_count / retained.frame_rate_hz,
        "runtime_obstacle_authority": (
            "one live declared navmesh plus all loaded rigid collision OBBs"
        ),
        "rigid_obstacle_count": obstacle_summary["rigid_obstacle_count"],
        "articulated_room_object_count": articulated_room_objects["object_count"],
        "articulated_room_object_representation": articulated_room_objects[
            "representation"
        ],
        "articulated_room_objects_separately_drawn": False,
        "collision_semantics": "source_center_only",
        "full_body_collision_claim": False,
        "capture_rerun": False,
        "rlr_rerun": False,
        "audio_policy": "retained 360-degree binaural mixture; no camera-FOV cutoff",
        "outputs": {
            "diagnostic_video": video_record,
            "runtime_obstacle_map_json": "room/runtime_obstacle_map.json",
            "runtime_obstacle_map_png": "room/runtime_obstacle_map.png",
            "articulated_room_objects": "room/articulated_room_objects.json",
            "source_center_gate": "source_center_gate.json",
        },
    }
    write_json(output / "status.json", status)
    return status


__all__ = [
    "DEFAULT_EXPECTED_RIGID_COUNT",
    "M6XReplicaCADError",
    "REPLICACAD_REVIEW_SCHEMA",
    "ReplicaCADRuntimeReview",
    "RetainedReplicaCADReview",
    "build_replicacad_runtime_review",
    "inspect_replicacad_articulated_room_objects",
    "load_retained_replicacad_review",
    "rebuild_replicacad_obstacle_review",
]
