"""MP3D-specific bindings for the final M5.1 annotated review delivery.

The legacy Apartment source manifest is reused only as the authority for
taxonomy, event timing, and dry-audio programs.  Its observer, trajectories,
and spatial flag assessments are deliberately not copied into this room.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m5_1.delivery import (
    M51DeliveryError,
    event_overlay_state,
    semantic_centroid_track,
    source_binding_entries,
)
from avengine.m5_1.orientation import (
    M51OrientationError,
    habitat_basis_from_yaw_degrees,
    habitat_yaw_degrees_from_xyzw,
)
from avengine.m5_1.review import SourceOverlayTrack


MP3D_DELIVERY_SCHEMA = "avengine_m5_1_mp3d_delivery_v1"
MP3D_PROGRAM_REUSE_SCHEMA = "avengine_m5_1_mp3d_source_program_reuse_v1"
SOURCE_LABEL = {"source0": "HUMAN", "source1": "BEAGLE"}
SOURCE_COLOR = {"source0": (42, 210, 220), "source1": (250, 120, 70)}
MP3D_VISUAL_GATE_SCHEMA = "avengine_m5_1_mp3d_mixed_visual_gate_v1"
REPLICACAD_VISUAL_GATE_SCHEMA = "avengine_m5_1_replicacad_mixed_visual_gate_v1"
MP3D_REQUIRED_GATE_IDS = frozenset(
    {
        "declared_navmesh_real_pathfinder_load",
        "human_center_navigable_every_frame",
        "dog_center_navigable_every_frame",
        "one_shared_navmesh_island",
        "human_segments_no_sliding",
        "dog_segments_no_sliding",
        "actor_center_separation",
        "captured_actor_roots_match_route_paths",
        "human_actual_movement",
        "dog_actual_movement",
        "no_actor_semantic_id_baseline_collision",
        "human_fixed_camera_visibility",
        "dog_fixed_camera_visibility",
        "contact_sheet_readback",
    }
)
REPLICACAD_REQUIRED_GATE_IDS = frozenset(
    {
        "dataset_config_selected",
        "scene_instance_selected",
        "stage_surface_selected",
        "scene_instance_object_counts_match",
        "scene_lighting_count_matches",
        "declared_navmesh_loaded",
        "human_route_all_frames_navigable",
        "dog_route_all_frames_navigable",
        "routes_share_one_island",
        "routes_no_sliding",
        "actor_center_separation",
        "actor_route_clearance",
        "camera_listener_floor_placement",
        "camera_actor_line_of_sight",
        "semantic_ids_absent_before_actor_creation",
        "human_semantic_visibility",
        "dog_semantic_visibility",
        "articulated_state_readback",
    }
)


@dataclass(frozen=True)
class MP3DNavmeshQA:
    """Authenticated PathFinder map and per-frame center diagnostics."""

    binary_map: np.ndarray
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    clearance_m: Mapping[str, np.ndarray]
    navigable: Mapping[str, np.ndarray]
    island_id: Mapping[str, np.ndarray]
    navmesh_record: Mapping[str, Any]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def listener_yaw_degrees(rotation_xyzw: Sequence[float]) -> float:
    """Return Habitat Y-up yaw from a normalized XYZW quaternion."""

    try:
        return habitat_yaw_degrees_from_xyzw(rotation_xyzw)
    except M51OrientationError as exc:
        raise M51DeliveryError(f"listener rotation is invalid: {exc}") from exc


def source_program_reuse_record(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Describe exactly which legacy source-contract fields apply to MP3D."""

    sources = source_manifest.get("sources")
    relationships = source_manifest.get("relationships")
    clip = source_manifest.get("clip")
    if not isinstance(sources, list) or not isinstance(relationships, list) or not isinstance(clip, Mapping):
        raise M51DeliveryError("source manifest lacks clip/sources/relationships")
    if {item.get("source_id") for item in sources if isinstance(item, Mapping)} != {
        "source0",
        "source1",
    }:
        raise M51DeliveryError("MP3D delivery requires source0 and source1")
    reusable_sources: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise M51DeliveryError("source manifest source must be an object")
        provenance = source.get("provenance")
        audio_assets = (
            provenance.get("audio_assets")
            if isinstance(provenance, Mapping)
            else None
        )
        if not isinstance(audio_assets, list):
            raise M51DeliveryError("source manifest lacks audio provenance")
        reusable_sources.append(
            {
                "source_id": source["source_id"],
                "asset_class": source["asset_class"],
                "voice_taxonomy": source.get("voice_taxonomy"),
                "call_taxonomy": source.get("call_taxonomy"),
                "event_windows": source.get("event_windows"),
                "audio_provenance": {"audio_assets": audio_assets},
            }
        )
    record: dict[str, Any] = {
        "schema": MP3D_PROGRAM_REUSE_SCHEMA,
        "applicability": "taxonomy_event_timing_and_audio_program_only",
        "legacy_spatial_trajectory_applicable": False,
        "legacy_observer_applicable": False,
        "legacy_source_and_clip_flags_applicable": False,
        "excluded_legacy_fields": [
            "observer",
            "sources[*].trajectory",
            "sources[*].flags",
            "clip_flags",
            "relationships[*].flags",
            "sources[*].provenance.visual_asset",
            "sources[*].provenance.migration",
        ],
        "legacy_visual_provenance_applicable": False,
        "mp3d_visual_provenance_authority": "authenticated MP3D mixed capture evidence",
        "mp3d_spatial_authorities": [
            "captured articulated emitter-link world transforms",
            "M1 co-located camera/listener transform",
            "real Habitat PathFinder navmesh gate evidence",
            "captured semantic visibility",
        ],
        "clip_time_and_audio_contract": {
            key: clip[key]
            for key in (
                "fps_num",
                "fps_den",
                "frame_count",
                "sample_rate_hz",
                "sample_count",
            )
        },
        "sources": reusable_sources,
        "event_overlap_windows": [
            window
            for relationship in relationships
            if isinstance(relationship, Mapping)
            for window in relationship.get("event_overlap_windows", [])
        ],
    }
    record["record_content_sha256"] = canonical_json_sha256(record)
    return record


def validate_room_visual_gate(
    gate_evidence: Mapping[str, Any],
    route_manifest: Mapping[str, Any],
    *,
    room_family: str,
) -> tuple[str, ...]:
    """Require the exact schema and gate-ID set for one room family."""

    if (
        gate_evidence.get("status") != "pass"
        or gate_evidence.get("qualification_claim") is not False
        or gate_evidence.get("route_id") != route_manifest.get("route_id")
    ):
        raise M51DeliveryError(f"{room_family} visual gate identity/status differs")
    if room_family == "mp3d":
        if gate_evidence.get("schema") != MP3D_VISUAL_GATE_SCHEMA:
            raise M51DeliveryError("MP3D visual gate schema differs")
        raw_gates = gate_evidence.get("gates")
        if not isinstance(raw_gates, list) or not all(
            isinstance(item, Mapping) for item in raw_gates
        ):
            raise M51DeliveryError("MP3D visual gates must be an object array")
        gate_ids = [str(item.get("gate_id")) for item in raw_gates]
        if len(gate_ids) != len(set(gate_ids)) or set(gate_ids) != set(
            MP3D_REQUIRED_GATE_IDS
        ):
            raise M51DeliveryError("MP3D visual gate IDs differ from the frozen 14")
        if any(item.get("status") != "pass" for item in raw_gates):
            raise M51DeliveryError("MP3D visual gate includes a non-pass item")
        expected_count = len(MP3D_REQUIRED_GATE_IDS)
    elif room_family == "replicacad":
        if gate_evidence.get("schema") != REPLICACAD_VISUAL_GATE_SCHEMA:
            raise M51DeliveryError("ReplicaCAD visual gate schema differs")
        required = set(REPLICACAD_REQUIRED_GATE_IDS)
        placement = route_manifest.get("placement_gate")
        requires_rigid_clearance = isinstance(placement, Mapping) and placement.get(
            "require_rigid_object_center_clearance"
        ) is True
        if requires_rigid_clearance:
            required.add("actor_rigid_object_center_clearance")
        raw_gates = gate_evidence.get("gates")
        if not isinstance(raw_gates, Mapping) or set(raw_gates) != required:
            raise M51DeliveryError(
                "ReplicaCAD visual gate IDs differ from the route-required set"
            )
        if any(value is not True for value in raw_gates.values()):
            raise M51DeliveryError("ReplicaCAD visual gate includes a non-pass item")
        expected_count = len(required)
        if gate_evidence.get("room_id") != route_manifest.get("room_id"):
            raise M51DeliveryError("ReplicaCAD gate room_id differs from route")
        request_id = gate_evidence.get("request_id")
        if (
            request_id != route_manifest.get("request_id")
            if requires_rigid_clearance
            else request_id is not None
            and request_id != route_manifest.get("request_id")
        ):
            raise M51DeliveryError("ReplicaCAD gate request_id differs from route")
        gate_ids = sorted(required, key=lambda value: value.encode("ascii"))
    else:
        raise M51DeliveryError(f"unsupported room family: {room_family}")
    if (
        gate_evidence.get("gate_count") != expected_count
        or gate_evidence.get("passed_gate_count") != expected_count
    ):
        raise M51DeliveryError(
            f"{room_family} visual gate is not the expected {expected_count}/{expected_count}"
        )
    return tuple(sorted(gate_ids, key=lambda value: value.encode("ascii")))


def build_mp3d_overlay_tracks(
    source_manifest: Mapping[str, Any],
    *,
    anchor_positions_m: Any,
    semantic_frames: Any,
    clearance_m: Mapping[str, Any],
    gate_evidence: Mapping[str, Any],
    source_actor_bindings: Mapping[str, Any],
) -> tuple[SourceOverlayTrack, ...]:
    """Bind reusable source programs to MP3D capture and real navmesh facts."""

    anchors = np.asarray(anchor_positions_m, dtype=np.float64)
    semantic = np.asarray(semantic_frames)
    frame_count = int(source_manifest.get("clip", {}).get("frame_count", 0))
    if anchors.shape != (frame_count, 3, 3) or not np.all(np.isfinite(anchors)):
        raise M51DeliveryError("MP3D emitter anchors must be finite [frame,3,3]")
    if semantic.ndim != 3 or semantic.shape[0] != frame_count:
        raise M51DeliveryError("MP3D semantic frames differ from source program")
    if (
        gate_evidence.get("status") != "pass"
        or gate_evidence.get("qualification_claim") is not False
        or gate_evidence.get("passed_gate_count") != gate_evidence.get("gate_count")
    ):
        raise M51DeliveryError("MP3D center/visibility gate evidence is not a bounded pass")
    sources = source_manifest.get("sources")
    if not isinstance(sources, list):
        raise M51DeliveryError("source manifest sources must be an array")
    bindings = source_binding_entries(source_actor_bindings)
    tracks: list[SourceOverlayTrack] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise M51DeliveryError("source manifest source must be an object")
        source_id = str(source.get("source_id"))
        if source_id not in bindings:
            raise M51DeliveryError(f"unexpected MP3D source ID: {source_id}")
        binding = bindings[source_id]
        events, active = event_overlay_state(source, frame_count)
        taxonomy = (
            source.get("voice_taxonomy")
            if source.get("asset_class") == "human"
            else source.get("call_taxonomy")
        )
        if not isinstance(taxonomy, Mapping):
            raise M51DeliveryError(f"{source_id} reusable taxonomy is missing")
        sound_class = str(
            taxonomy.get("vocalization_type", taxonomy.get("call_type", "unknown"))
        )
        actor_id = str(binding.get("actor_id"))
        clearance = np.asarray(clearance_m.get(actor_id), dtype=np.float64)
        if clearance.shape != (frame_count,) or not np.all(np.isfinite(clearance)) or np.any(clearance < 0.0):
            raise M51DeliveryError(f"{actor_id} navmesh clearance is invalid")
        semantic_id = binding.get("semantic_id")
        anchor_index = binding.get("emitter_anchor_index")
        if (
            isinstance(semantic_id, bool)
            or not isinstance(semantic_id, int)
            or isinstance(anchor_index, bool)
            or not isinstance(anchor_index, int)
            or not 0 <= anchor_index < anchors.shape[1]
        ):
            raise M51DeliveryError(f"{source_id} source/actor binding is invalid")
        centroids = semantic_centroid_track(semantic, semantic_id)
        if not np.all(np.isfinite(centroids)):
            raise M51DeliveryError(f"{actor_id} is not semantically visible in every frame")
        tracks.append(
            SourceOverlayTrack(
                source_id=source_id,
                label=SOURCE_LABEL[source_id],
                asset_class=str(source["asset_class"]),
                sound_class=sound_class,
                color_rgb=SOURCE_COLOR[source_id],
                positions_m=np.ascontiguousarray(anchors[:, anchor_index, :]),
                current_event_by_frame=events,
                active_by_frame=active,
                true_flags=("center_navmesh_pass", "visible_all_frames"),
                center_clearance_m=np.ascontiguousarray(clearance),
                main_marker_xy=centroids,
            )
        )
    return tuple(sorted(tracks, key=lambda track: track.source_id.encode("ascii")))


def load_real_mp3d_navmesh_qa(
    *,
    navmesh_record: Mapping[str, Any],
    actor_center_paths_m: Mapping[str, Any],
    meters_per_pixel: float,
    height_m: float,
    maximum_y_delta_m: float,
) -> MP3DNavmeshQA:
    """Load the retained navmesh with Habitat and recompute frame diagnostics."""

    path_raw = navmesh_record.get("path")
    expected_hash = navmesh_record.get("sha256")
    expected_size = navmesh_record.get("byte_size")
    if not isinstance(path_raw, str) or not isinstance(expected_hash, str):
        raise M51DeliveryError("MP3D gate evidence lacks an authenticated navmesh")
    path = Path(path_raw).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != expected_hash
    ):
        raise M51DeliveryError("declared MP3D navmesh bytes changed")
    if not math.isfinite(float(meters_per_pixel)) or meters_per_pixel <= 0.0:
        raise M51DeliveryError("meters_per_pixel must be positive")
    if not math.isfinite(float(height_m)) or not math.isfinite(float(maximum_y_delta_m)):
        raise M51DeliveryError("navmesh query heights must be finite")

    # The pinned audio-enabled Habitat build requires numpy-quaternion first.
    import quaternion  # noqa: F401
    import habitat_sim

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(path)) or not pathfinder.is_loaded:
        raise M51DeliveryError("Habitat failed to load the declared MP3D navmesh")
    binary_map = np.asarray(
        pathfinder.get_topdown_view(float(meters_per_pixel), float(height_m)),
        dtype=np.uint8,
    )
    if binary_map.ndim != 2 or not np.any(binary_map) or np.any(~np.isin(binary_map, (0, 1))):
        raise M51DeliveryError("Habitat returned an invalid binary navmesh map")
    clearance: dict[str, np.ndarray] = {}
    navigable: dict[str, np.ndarray] = {}
    islands: dict[str, np.ndarray] = {}
    for actor_id in ("human0", "dog0"):
        path_points = np.asarray(actor_center_paths_m.get(actor_id), dtype=np.float64)
        if path_points.ndim != 2 or path_points.shape[1] != 3 or not np.all(np.isfinite(path_points)):
            raise M51DeliveryError(f"{actor_id} center path must be finite [frame,3]")
        navigable_values = np.asarray(
            [
                bool(pathfinder.is_navigable(point, float(maximum_y_delta_m)))
                for point in path_points
            ],
            dtype=np.bool_,
        )
        island_values = np.asarray(
            [int(pathfinder.get_island(point)) for point in path_points],
            dtype=np.int64,
        )
        clearance_values = np.asarray(
            [float(pathfinder.distance_to_closest_obstacle(point, 10.0)) for point in path_points],
            dtype=np.float64,
        )
        if (
            not np.all(navigable_values)
            or np.any(island_values < 0)
            or not np.all(np.isfinite(clearance_values))
            or np.any(clearance_values < 0.0)
        ):
            raise M51DeliveryError(f"{actor_id} failed independent navmesh review diagnostics")
        clearance[actor_id] = np.ascontiguousarray(clearance_values)
        navigable[actor_id] = np.ascontiguousarray(navigable_values)
        islands[actor_id] = np.ascontiguousarray(island_values)
    if len(np.unique(np.concatenate(tuple(islands.values())))) != 1:
        raise M51DeliveryError("MP3D review centers do not share one navmesh island")
    bounds = pathfinder.get_bounds()
    return MP3DNavmeshQA(
        binary_map=np.ascontiguousarray(binary_map),
        bounds_m=(
            tuple(float(component) for component in bounds[0]),
            tuple(float(component) for component in bounds[1]),
        ),
        clearance_m=clearance,
        navigable=navigable,
        island_id=islands,
        navmesh_record={
            "path": str(path),
            "byte_size": path.stat().st_size,
            "sha256": expected_hash,
            "meters_per_pixel": float(meters_per_pixel),
            "height_m": float(height_m),
            "shape": list(binary_map.shape),
            "navigable_pixel_count": int(np.count_nonzero(binary_map)),
        },
    )


def _focus_bounds_xz(
    paths: Mapping[str, np.ndarray], listener: np.ndarray, *, aspect: float
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        [np.asarray(path)[:, (0, 2)] for path in paths.values()]
        + [listener[None, (0, 2)]],
        axis=0,
    )
    low = np.min(points, axis=0) - 0.7
    high = np.max(points, axis=0) + 0.7
    center = (low + high) * 0.5
    span = np.maximum(high - low, (2.4, 2.4))
    if span[0] / span[1] < aspect:
        span[0] = span[1] * aspect
    else:
        span[1] = span[0] / aspect
    return center - span * 0.5, center + span * 0.5


def render_mp3d_topdown_frames(
    *,
    navmesh_binary_map: Any,
    navmesh_bounds_m: Sequence[Sequence[float]],
    actor_center_paths_m: Mapping[str, Any],
    listener_position_m: Sequence[float],
    listener_yaw_deg: float,
    camera_hfov_degrees: float,
    clearance_m: Mapping[str, Any],
    shared_island_id: int,
    source_actor_bindings: Mapping[str, Any],
    size_wh: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Render real-navmesh complete/current paths as a QA-only panel.

    The wedge is the visual camera HFOV.  It is not an acoustic audibility
    gate; this review contract intentionally defines no microphone cutoff.
    """

    navmesh = np.asarray(navmesh_binary_map, dtype=np.uint8)
    if navmesh.ndim != 2 or not np.any(navmesh) or np.any(~np.isin(navmesh, (0, 1))):
        raise M51DeliveryError("navmesh_binary_map must be a nonempty binary HxW map")
    bounds = np.asarray(navmesh_bounds_m, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0]):
        raise M51DeliveryError("navmesh bounds must be finite ordered XYZ bounds")
    paths = {
        actor_id: np.asarray(actor_center_paths_m.get(actor_id), dtype=np.float64)
        for actor_id in ("human0", "dog0")
    }
    bindings = source_binding_entries(source_actor_bindings)
    source_id_by_actor = {
        str(binding.get("actor_id")): source_id
        for source_id, binding in bindings.items()
    }
    if set(source_id_by_actor) != set(paths):
        raise M51DeliveryError("Topdown source/actor bindings differ from actor paths")
    if any(path.ndim != 2 or path.shape[1] != 3 or not np.all(np.isfinite(path)) for path in paths.values()):
        raise M51DeliveryError("actor center paths must be finite [frame,3]")
    frame_count = paths["human0"].shape[0]
    if paths["dog0"].shape != paths["human0"].shape:
        raise M51DeliveryError("human and dog path shapes differ")
    listener = np.asarray(listener_position_m, dtype=np.float64)
    if listener.shape != (3,) or not np.all(np.isfinite(listener)):
        raise M51DeliveryError("listener position must be finite [3]")
    if (
        isinstance(camera_hfov_degrees, bool)
        or not isinstance(camera_hfov_degrees, Real)
        or not math.isfinite(float(camera_hfov_degrees))
        or not 0.0 < float(camera_hfov_degrees) < 180.0
    ):
        raise M51DeliveryError("camera_hfov_degrees must be finite within (0,180)")
    try:
        listener_basis = habitat_basis_from_yaw_degrees(listener_yaw_deg)
    except M51OrientationError as exc:
        raise M51DeliveryError(f"listener yaw is invalid: {exc}") from exc
    width, height = size_wh
    if width < 320 or height < 240:
        raise M51DeliveryError("MP3D Topdown output must be at least 320x240")
    if not isinstance(shared_island_id, int) or shared_island_id < 0:
        raise M51DeliveryError("shared_island_id must be nonnegative")
    clearance_arrays = {
        actor_id: np.asarray(clearance_m.get(actor_id), dtype=np.float64)
        for actor_id in paths
    }
    if any(value.shape != (frame_count,) or not np.all(np.isfinite(value)) for value in clearance_arrays.values()):
        raise M51DeliveryError("Topdown clearance arrays differ from paths")

    focus_low, focus_high = _focus_bounds_xz(paths, listener, aspect=width / height)
    full_h, full_w = navmesh.shape

    def full_pixel(point_xz: Sequence[float]) -> np.ndarray:
        x, z = np.asarray(point_xz, dtype=np.float64)
        return np.asarray(
            (
                (x - bounds[0, 0]) / (bounds[1, 0] - bounds[0, 0]) * (full_w - 1),
                (z - bounds[0, 2]) / (bounds[1, 2] - bounds[0, 2]) * (full_h - 1),
            ),
            dtype=np.float64,
        )

    crop_a = full_pixel(focus_low)
    crop_b = full_pixel(focus_high)
    crop_low = np.floor(np.minimum(crop_a, crop_b)).astype(int)
    crop_high = np.ceil(np.maximum(crop_a, crop_b)).astype(int)
    crop_low = np.maximum(crop_low, (0, 0))
    crop_high = np.minimum(crop_high, (full_w - 1, full_h - 1))
    if np.any(crop_high - crop_low < 2):
        raise M51DeliveryError("MP3D Topdown focus crop is degenerate")
    rgb_map = np.where(navmesh[..., None] != 0, (210, 216, 222), (52, 58, 66)).astype(np.uint8)
    base = Image.fromarray(rgb_map, mode="RGB").crop(
        (int(crop_low[0]), int(crop_low[1]), int(crop_high[0] + 1), int(crop_high[1] + 1))
    ).resize((width, height), Image.Resampling.NEAREST)

    def panel_point(position: Sequence[float]) -> tuple[float, float]:
        pixel = full_pixel(np.asarray(position, dtype=np.float64)[(0, 2),])
        denominator = crop_high - crop_low
        return (
            float((pixel[0] - crop_low[0]) / denominator[0] * (width - 1)),
            float((pixel[1] - crop_low[1]) / denominator[1] * (height - 1)),
        )

    def panel_direction(
        direction_xz: Sequence[float], *, pixel_length: float
    ) -> np.ndarray:
        direction = np.asarray(direction_xz, dtype=np.float64)
        endpoint = listener + np.asarray((direction[0], 0.0, direction[1]))
        delta = np.asarray(panel_point(endpoint)) - np.asarray(panel_point(listener))
        norm = float(np.linalg.norm(delta))
        if norm <= 1.0e-12:
            raise M51DeliveryError("listener orientation has degenerate Topdown projection")
        return delta / norm * pixel_length

    projected = {
        actor_id: [panel_point(point) for point in path]
        for actor_id, path in paths.items()
    }
    styles = {
        "human0": (42, 210, 220, 255),
        "dog0": (250, 120, 70, 255),
    }
    listener_xy = panel_point(listener)
    forward_xz = np.asarray(listener_basis.forward_xz, dtype=np.float64)
    right_xz = np.asarray(listener_basis.right_xz, dtype=np.float64)
    half_fov = math.radians(float(camera_hfov_degrees) * 0.5)
    left_ray_xz = math.cos(half_fov) * forward_xz - math.sin(half_fov) * right_xz
    right_ray_xz = math.cos(half_fov) * forward_xz + math.sin(half_fov) * right_xz
    wedge_length = max(42.0, min(width, height) * 0.18)
    forward_delta = panel_direction(forward_xz, pixel_length=34.0)
    right_delta = panel_direction(right_xz, pixel_length=24.0)
    left_ray_delta = panel_direction(left_ray_xz, pixel_length=wedge_length)
    right_ray_delta = panel_direction(right_ray_xz, pixel_length=wedge_length)
    frames: list[np.ndarray] = []
    for frame_index in range(frame_count):
        image = base.convert("RGBA")
        lx, ly = listener_xy
        wedge_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        wedge_draw = ImageDraw.Draw(wedge_overlay, "RGBA")
        wedge = (
            (lx, ly),
            (lx + left_ray_delta[0], ly + left_ray_delta[1]),
            (lx + right_ray_delta[0], ly + right_ray_delta[1]),
        )
        wedge_draw.polygon(wedge, fill=(46, 154, 255, 48))
        wedge_draw.line((*wedge, wedge[0]), fill=(46, 154, 255, 190), width=2)
        image = Image.alpha_composite(image, wedge_overlay)
        draw = ImageDraw.Draw(image, "RGBA")
        for actor_id in ("human0", "dog0"):
            color = styles[actor_id]
            draw.line(projected[actor_id], fill=(*color[:3], 76), width=3)
            if frame_index > 0:
                draw.line(projected[actor_id][: frame_index + 1], fill=color, width=4)
            x, y = projected[actor_id][frame_index]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(0, 0, 0, 255), width=2)
            draw.text(
                (x + 10, y - 9),
                (
                    f"HUMAN [{source_id_by_actor[actor_id]}]"
                    if actor_id == "human0"
                    else f"BEAGLE [{source_id_by_actor[actor_id]}]"
                ),
                fill=color,
                font=_font(12),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
            )
        draw.ellipse((lx - 7, ly - 7, lx + 7, ly + 7), fill=(255, 224, 66, 255), outline=(0, 0, 0, 255), width=2)
        left_ear = (lx - right_delta[0], ly - right_delta[1])
        right_ear = (lx + right_delta[0], ly + right_delta[1])
        draw.line((*left_ear, *right_ear), fill=(210, 80, 220, 255), width=3)
        draw.text(
            (left_ear[0] - 5, left_ear[1] - 13),
            "L",
            fill=(255, 255, 255, 255),
            font=_font(10),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        draw.text(
            (right_ear[0] - 4, right_ear[1] - 13),
            "R",
            fill=(255, 255, 255, 255),
            font=_font(10),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        draw.line(
            (lx, ly, lx + forward_delta[0], ly + forward_delta[1]),
            fill=(20, 20, 20, 255),
            width=4,
        )
        draw.text(
            (lx + forward_delta[0] - 4, ly + forward_delta[1] - 13),
            "F",
            fill=(255, 255, 255, 255),
            font=_font(10),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        draw.text((lx + 10, ly + 5), "CAM/LISTENER", fill=(255, 235, 90, 255), font=_font(11), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        draw.rectangle((0, 0, width - 1, 45), fill=(0, 0, 0, 190))
        separation = float(np.linalg.norm(paths["human0"][frame_index] - paths["dog0"][frame_index]))
        draw.text(
            (8, 5),
            f"REAL HABITAT NAVMESH QA | frame {frame_index:03d}/{frame_count-1:03d} | island={shared_island_id} | centers=PASS",
            fill=(255, 255, 255, 255),
            font=_font(13),
        )
        draw.text(
            (8, 24),
            (
                f"nav-edge clearance H={clearance_arrays['human0'][frame_index]:.3f}m "
                f"D={clearance_arrays['dog0'][frame_index]:.3f}m | center separation={separation:.3f}m"
            ),
            fill=(225, 225, 225, 255),
            font=_font(12),
        )
        draw.text(
            (8, height - 37),
            f"VISUAL HFOV={float(camera_hfov_degrees):g} deg only | AUDIO: no mic-distance cutoff",
            fill=(255, 255, 255, 255),
            font=_font(11),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        draw.text(
            (8, height - 20),
            "QA-only: actor root centers, complete paths (faint), traversed paths (solid); not full-mesh clearance",
            fill=(255, 255, 255, 255),
            font=_font(11),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.ascontiguousarray(np.stack(frames, axis=0))


__all__ = [
    "MP3D_DELIVERY_SCHEMA",
    "MP3D_PROGRAM_REUSE_SCHEMA",
    "MP3DNavmeshQA",
    "build_mp3d_overlay_tracks",
    "listener_yaw_degrees",
    "load_real_mp3d_navmesh_qa",
    "render_mp3d_topdown_frames",
    "source_program_reuse_record",
    "validate_room_visual_gate",
]
