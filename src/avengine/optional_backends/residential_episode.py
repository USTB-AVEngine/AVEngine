"""Shared two-source episode planning for optional residential visual backends.

The room backend supplies only a Z-up room polygon, simple object bounds and
the UE map that will draw pixels.  AVEngine still owns the exact 75-frame
Timeline, source-center qualification, human/dog source identity, audio event
schedule, camera/listener pose and diagnostic Topdown facts.

This module is intentionally pure Python.  It never imports Habitat, SPEAR,
Unreal or USD, and it does not turn an optional room dataset into a second
navigation or audio authority.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from avengine.optional_backends.spear_apartment import (
    BEAGLE_ASSET_ID,
    DEFAULT_ACTOR_BINDINGS,
    HUMAN_ASSET_ID,
    component_frame_delta_for_asset,
)
from avengine.optional_backends.spear_visual import build_spear_visual_plan


EPISODE_SCHEMA = "avengine_optional_residential_two_source_episode_v1"
SCENE_METADATA_SCHEMA = "avengine_optional_residential_scene_metadata_v1"
PROFILE_SCHEMA = "avengine_optional_residential_source_profile_v1"
FRAME_COUNT = 75
FPS = 15
TIME_BASE_HZ = 48_000
TICKS_PER_FRAME = 3_200
SAMPLE_RATE_HZ = 16_000
SAMPLE_COUNT = 80_000

DOG_SOURCE_ID = "m6z_dog0_muzzle"
HUMAN_SOURCE_ID = "m6z_human0_mouth"

# The same simultaneous five-second dry program used by the reviewed M6.x S4
# episode.  Hash values identify those dry inputs and remain data, not locks on
# a room asset.
_AUDIO_EVENTS: tuple[dict[str, Any], ...] = (
    {
        "event_id": "m6z_human_speech",
        "actor_id": "human0",
        "event_type": "vocalization",
        "start_sample": 8_000,
        "end_sample": 56_000,
        "emitter_bone": "mouth",
        "emitter_path_sha256": (
            "4fb94574284f42e1518abc279725055ae3590e46de92f0080837436d13f54ad9"
        ),
        "audio_asset_sha256": (
            "e59d81b94066c9f15ef6ad61121d8aec0b10b3d5fbb1035f546c02d96bd11bda"
        ),
        "semantic_sync_required": True,
    },
    {
        "event_id": "m6z_dog_bark_0",
        "actor_id": "dog0",
        "event_type": "vocalization",
        "start_sample": 16_000,
        "end_sample": 20_800,
        "emitter_bone": "muzzle",
        "emitter_path_sha256": (
            "681bcc806ad5b020ece8b2f691a4b5f50b511f540b06509dbf56f57431e440ec"
        ),
        "audio_asset_sha256": (
            "12d9b3a2c9cd81852ddeb76d1abeef41ef623868b6731ff91ed511d474d2c634"
        ),
        "semantic_sync_required": True,
    },
    {
        "event_id": "m6z_dog_bark_1",
        "actor_id": "dog0",
        "event_type": "vocalization",
        "start_sample": 32_000,
        "end_sample": 36_800,
        "emitter_bone": "muzzle",
        "emitter_path_sha256": (
            "681bcc806ad5b020ece8b2f691a4b5f50b511f540b06509dbf56f57431e440ec"
        ),
        "audio_asset_sha256": (
            "12d9b3a2c9cd81852ddeb76d1abeef41ef623868b6731ff91ed511d474d2c634"
        ),
        "semantic_sync_required": True,
    },
    {
        "event_id": "m6z_dog_bark_2",
        "actor_id": "dog0",
        "event_type": "vocalization",
        "start_sample": 48_000,
        "end_sample": 52_800,
        "emitter_bone": "muzzle",
        "emitter_path_sha256": (
            "681bcc806ad5b020ece8b2f691a4b5f50b511f540b06509dbf56f57431e440ec"
        ),
        "audio_asset_sha256": (
            "12d9b3a2c9cd81852ddeb76d1abeef41ef623868b6731ff91ed511d474d2c634"
        ),
        "semantic_sync_required": True,
    },
)


class ResidentialEpisodeError(ValueError):
    """A room/profile cannot produce the bounded two-source episode."""


def _finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResidentialEpisodeError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ResidentialEpisodeError(f"{owner} must be a finite number")
    return result


def _vector(value: Any, size: int, *, owner: str) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != size
    ):
        raise ResidentialEpisodeError(f"{owner} must contain {size} numbers")
    return tuple(_finite(item, owner=f"{owner}[{index}]") for index, item in enumerate(value))


def _nonempty(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResidentialEpisodeError(f"{owner} must be a non-empty string")
    return value.strip()


def dataset_z_up_to_habitat(point_xyz_m: Sequence[float]) -> list[float]:
    """Map Z-up room coordinates to AVEngine/Habitat ``[right, up, back]``."""

    x, y, z = _vector(point_xyz_m, 3, owner="Z-up point")
    return [x, z, y]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_boundary(frame_boundary: int) -> int:
    # round(frame_boundary * 16000 / 15), ties away from zero (all positive)
    numerator = frame_boundary * SAMPLE_RATE_HZ
    quotient, remainder = divmod(numerator, FPS)
    return quotient + int(remainder * 2 >= FPS)


def _linear_route(start: Sequence[float], end: Sequence[float]) -> list[list[float]]:
    a = _vector(start, 3, owner="route start")
    b = _vector(end, 3, owner="route end")
    return [
        [a[axis] + (b[axis] - a[axis]) * index / (FRAME_COUNT - 1) for axis in range(3)]
        for index in range(FRAME_COUNT)
    ]


def _polyline_route(waypoints: Sequence[Sequence[float]]) -> list[list[float]]:
    if (
        isinstance(waypoints, (str, bytes))
        or not isinstance(waypoints, Sequence)
        or len(waypoints) < 2
    ):
        raise ResidentialEpisodeError("route waypoints require at least two points")
    points = [
        _vector(value, 3, owner=f"route waypoint {index}")
        for index, value in enumerate(waypoints)
    ]
    lengths = [
        math.sqrt(sum((b[axis] - a[axis]) ** 2 for axis in range(3)))
        for a, b in zip(points, points[1:])
    ]
    if any(length <= 1.0e-9 for length in lengths):
        raise ResidentialEpisodeError("consecutive route waypoints must differ")
    total = sum(lengths)
    route = []
    segment = 0
    elapsed_before = 0.0
    for frame_index in range(FRAME_COUNT):
        target = total * frame_index / (FRAME_COUNT - 1)
        while segment < len(lengths) - 1 and target > elapsed_before + lengths[segment]:
            elapsed_before += lengths[segment]
            segment += 1
        fraction = (target - elapsed_before) / lengths[segment]
        route.append(
            [
                points[segment][axis]
                + (points[segment + 1][axis] - points[segment][axis]) * fraction
                for axis in range(3)
            ]
        )
    return route


def _point_on_segment(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> bool:
    cross = (point[0] - a[0]) * (b[1] - a[1]) - (point[1] - a[1]) * (
        b[0] - a[0]
    )
    if abs(cross) > 1.0e-9:
        return False
    dot = (point[0] - a[0]) * (point[0] - b[0]) + (
        point[1] - a[1]
    ) * (point[1] - b[1])
    return dot <= 1.0e-9


def point_in_polygon_xy(
    point_xy: Sequence[float], polygon_xy: Sequence[Sequence[float]]
) -> bool:
    point = _vector(point_xy, 2, owner="point_xy")
    polygon = tuple(
        _vector(item, 2, owner=f"polygon[{index}]")
        for index, item in enumerate(polygon_xy)
    )
    if len(polygon) < 3:
        raise ResidentialEpisodeError("room polygon requires at least three points")
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        if _point_on_segment(point, current, previous):
            return True
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing_x = (previous[0] - current[0]) * (
                point[1] - current[1]
            ) / (previous[1] - current[1]) + current[0]
            if point[0] < crossing_x:
                inside = not inside
        j = i
    return inside


def classify_object_bounds(
    bounds_xyz_m: Sequence[Sequence[float]], *, floor_z_m: float = 0.0
) -> str:
    """Classify geometry for the deliberately center-only route gate.

    Floor coverings up to 10 cm are walkable.  Geometry whose lowest point is
    above the 15 cm root slice is visible in Topdown but does not block the
    source center.  This intentionally does not approximate an actor body.
    """

    if (
        isinstance(bounds_xyz_m, (str, bytes))
        or not isinstance(bounds_xyz_m, Sequence)
        or len(bounds_xyz_m) != 2
    ):
        raise ResidentialEpisodeError("object bounds must be [minimum, maximum]")
    minimum = _vector(bounds_xyz_m[0], 3, owner="bounds minimum")
    maximum = _vector(bounds_xyz_m[1], 3, owner="bounds maximum")
    if any(maximum[axis] < minimum[axis] for axis in range(3)):
        raise ResidentialEpisodeError("object bounds minimum exceeds maximum")
    floor = _finite(floor_z_m, owner="floor_z_m")
    if maximum[2] <= floor + 0.10:
        return "walkable_floor_covering"
    if minimum[2] > floor + 0.15:
        return "elevated_object"
    return "ground_blocker"


def object_footprint_rectangles_xy(
    item: Mapping[str, Any], *, owner: str = "object"
) -> list[list[list[float]]]:
    """Return the center-gate rectangles for one ground-level object.

    New InteriorAgent metadata may contain per-mesh XY footprints.  They avoid
    treating an L-shaped cabinet (or a group of separated furniture parts) as
    one solid parent AABB.  Older metadata remains valid and falls back to the
    top-level bounds.
    """

    def bounds_rectangle() -> list[list[list[float]]]:
        bounds = item.get("bounds_xyz_m")
        if (
            isinstance(bounds, (str, bytes))
            or not isinstance(bounds, Sequence)
            or len(bounds) != 2
        ):
            raise ResidentialEpisodeError(f"{owner}.bounds_xyz_m is invalid")
        minimum = _vector(bounds[0], 3, owner=f"{owner} bounds minimum")
        maximum = _vector(bounds[1], 3, owner=f"{owner} bounds maximum")
        return [[[minimum[0], minimum[1]], [maximum[0], maximum[1]]]]

    raw_parts = item.get("footprint_parts_xy_m")
    if raw_parts is None:
        return bounds_rectangle()
    if isinstance(raw_parts, (str, bytes)) or not isinstance(raw_parts, Sequence):
        raise ResidentialEpisodeError(f"{owner}.footprint_parts_xy_m must be a list")
    # A failed or empty optional mesh slice must never make a declared ground
    # blocker disappear.  The top-level AABB is conservative, but it is safer
    # and explicit; successful extraction supplies the tighter rectangles.
    if not raw_parts:
        return bounds_rectangle()
    rectangles: list[list[list[float]]] = []
    for index, raw in enumerate(raw_parts):
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != 2
        ):
            raise ResidentialEpisodeError(
                f"{owner}.footprint_parts_xy_m[{index}] is invalid"
            )
        minimum = _vector(raw[0], 2, owner=f"{owner} footprint {index} minimum")
        maximum = _vector(raw[1], 2, owner=f"{owner} footprint {index} maximum")
        if maximum[0] < minimum[0] or maximum[1] < minimum[1]:
            raise ResidentialEpisodeError(f"{owner} footprint minimum exceeds maximum")
        rectangles.append([list(minimum), list(maximum)])
    return rectangles


def _route_gate(
    *,
    route_xyz_m: Sequence[Sequence[float]],
    polygon_xy_m: Sequence[Sequence[float]],
    objects: Sequence[Mapping[str, Any]],
    center_margin_m: float,
) -> dict[str, Any]:
    margin = _finite(center_margin_m, owner="center_margin_m")
    if margin < 0.0:
        raise ResidentialEpisodeError("center_margin_m cannot be negative")
    blockers: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise ResidentialEpisodeError(f"objects[{index}] must be a mapping")
        bounds = item.get("bounds_xyz_m")
        role = item.get("navigation_role") or classify_object_bounds(bounds)
        if role != "ground_blocker":
            continue
        object_id = str(item.get("object_id", index))
        for part_index, rectangle in enumerate(
            object_footprint_rectangles_xy(item, owner=f"objects[{index}]")
        ):
            minimum = _vector(
                rectangle[0], 2, owner=f"objects[{index}] part {part_index} minimum"
            )
            maximum = _vector(
                rectangle[1], 2, owner=f"objects[{index}] part {part_index} maximum"
            )
            blockers.append((object_id, minimum, maximum))

    frames: list[dict[str, Any]] = []
    failed: list[int] = []
    for frame_index, raw_point in enumerate(route_xyz_m):
        point = _vector(raw_point, 3, owner=f"route[{frame_index}]")
        reasons: list[str] = []
        if not point_in_polygon_xy(point[:2], polygon_xy_m):
            reasons.append("outside_room_polygon")
        hit_ids = [
            object_id
            for object_id, minimum, maximum in blockers
            if minimum[0] - margin <= point[0] <= maximum[0] + margin
            and minimum[1] - margin <= point[1] <= maximum[1] + margin
        ]
        if hit_ids:
            reasons.append("blocking_object_center_overlap:" + ",".join(hit_ids))
        status = "pass" if not reasons else "fail"
        if reasons:
            failed.append(frame_index)
        frames.append(
            {
                "frame_index": frame_index,
                "position_xyz_m": list(point),
                "status": status,
                "reasons": reasons,
            }
        )
    return {
        "status": "pass" if not failed else "fail",
        "claim_boundary": "source_center_only",
        "center_margin_m": margin,
        "failed_frame_indices": failed,
        "frames": frames,
    }


def _yaw_quaternion_for_route(
    start_h: Sequence[float], end_h: Sequence[float], *, local_forward: str
) -> list[float]:
    start = _vector(start_h, 3, owner="route start Habitat")
    end = _vector(end_h, 3, owner="route end Habitat")
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    if math.hypot(dx, dz) <= 1.0e-9:
        raise ResidentialEpisodeError("walking route cannot be stationary")
    if local_forward == "+X":
        angle = math.atan2(-dz, dx)
    elif local_forward == "+Z":
        angle = math.atan2(dx, dz)
    else:
        raise ResidentialEpisodeError("unsupported local forward axis")
    return [0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0)]


def _vocalizing(actor_id: str, sample_start: int, sample_end: int) -> bool:
    return any(
        event["actor_id"] == actor_id
        and event["start_sample"] < sample_end
        and event["end_sample"] > sample_start
        for event in _AUDIO_EVENTS
    )


def _actor_speed(route: Sequence[Sequence[float]]) -> float:
    points = [_vector(value, 3, owner="route point") for value in route]
    distance = sum(
        math.sqrt(sum((b[index] - a[index]) ** 2 for index in range(3)))
        for a, b in zip(points, points[1:])
    )
    return distance / ((FRAME_COUNT - 1) / FPS)


def build_residential_source_episode(
    *, scene_metadata: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Compile one room into a closed human+Beagle review episode."""

    if scene_metadata.get("schema") != SCENE_METADATA_SCHEMA:
        raise ResidentialEpisodeError(f"scene metadata schema must be {SCENE_METADATA_SCHEMA}")
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ResidentialEpisodeError(f"source profile schema must be {PROFILE_SCHEMA}")

    scene_id = _nonempty(scene_metadata.get("scene_id"), owner="scene_id")
    if profile.get("scene_id") != scene_id:
        raise ResidentialEpisodeError("profile and metadata scene_id differ")
    dataset_id = _nonempty(scene_metadata.get("dataset_id"), owner="dataset_id")
    room_id = _nonempty(scene_metadata.get("room_id"), owner="room_id")
    map_path = _nonempty(profile.get("map_path"), owner="map_path")
    if not map_path.startswith("/Game/"):
        raise ResidentialEpisodeError("map_path must be a /Game asset path")
    polygon = scene_metadata.get("room_polygon_xy_m")
    if not isinstance(polygon, Sequence) or len(polygon) < 3:
        raise ResidentialEpisodeError("scene metadata lacks a room polygon")
    objects = scene_metadata.get("objects", [])
    if not isinstance(objects, list):
        raise ResidentialEpisodeError("scene metadata objects must be a list")

    camera = profile.get("camera")
    routes = profile.get("routes")
    if not isinstance(camera, Mapping) or not isinstance(routes, Mapping):
        raise ResidentialEpisodeError("profile camera/routes are required")
    camera_xyz = _vector(camera.get("position_xyz_m"), 3, owner="camera position")
    camera_yaw_ue = _finite(camera.get("yaw_ue_deg"), owner="camera yaw")
    camera_hfov = _finite(camera.get("horizontal_fov_deg"), owner="camera HFOV")
    if not 0.0 < camera_hfov < 180.0:
        raise ResidentialEpisodeError("camera HFOV must be in (0, 180)")
    listener_h = dataset_z_up_to_habitat(camera_xyz)
    listener_yaw_h = -90.0 - camera_yaw_ue
    lighting_profile_id = _nonempty(
        profile.get("lighting_profile_id", "explicit_review_lights"),
        owner="lighting_profile_id",
    )
    lighting_claim = _nonempty(
        profile.get(
            "lighting_claim_boundary",
            "generated visual review lights; not reconstructed physical lighting",
        ),
        owner="lighting_claim_boundary",
    )

    actor_routes_xyz: dict[str, list[list[float]]] = {}
    for actor_id in ("dog0", "human0"):
        route = routes.get(actor_id)
        if not isinstance(route, Mapping):
            raise ResidentialEpisodeError(f"route {actor_id} is missing")
        if "waypoints_xyz_m" in route:
            if "start_xyz_m" in route or "end_xyz_m" in route:
                raise ResidentialEpisodeError(
                    f"route {actor_id} cannot mix waypoints with start/end"
                )
            actor_routes_xyz[actor_id] = _polyline_route(route["waypoints_xyz_m"])
        else:
            actor_routes_xyz[actor_id] = _linear_route(
                route.get("start_xyz_m"), route.get("end_xyz_m")
            )

    margin = _finite(profile.get("source_center_margin_m", 0.03), owner="margin")
    gates = {
        actor_id: _route_gate(
            route_xyz_m=route,
            polygon_xy_m=polygon,
            objects=objects,
            center_margin_m=margin,
        )
        for actor_id, route in actor_routes_xyz.items()
    }
    failed_actors = [actor_id for actor_id, gate in gates.items() if gate["status"] != "pass"]
    if failed_actors:
        details = {actor_id: gates[actor_id]["failed_frame_indices"] for actor_id in failed_actors}
        raise ResidentialEpisodeError(f"source-center route gate failed: {details}")

    actor_routes_h = {
        actor_id: [dataset_z_up_to_habitat(point) for point in route]
        for actor_id, route in actor_routes_xyz.items()
    }
    local_forward = {"dog0": "+X", "human0": "+Z"}
    rotations = {
        actor_id: [
            _yaw_quaternion_for_route(
                route_h[frame_index],
                route_h[frame_index + 1]
                if frame_index + 1 < FRAME_COUNT
                else route_h[frame_index - 1],
                local_forward=local_forward[actor_id],
            )
            for frame_index in range(FRAME_COUNT)
        ]
        for actor_id, route_h in actor_routes_h.items()
    }
    # The last frame uses the preceding segment but must retain its direction,
    # not face backwards from the final point toward the previous point.
    for actor_id in rotations:
        rotations[actor_id][-1] = rotations[actor_id][-2]
    cycles_per_second = {"dog0": 1.5, "human0": 1.0}
    actors = [
        {
            "actor_id": "dog0",
            "asset_id": BEAGLE_ASSET_ID,
            "template_id": "rocketbox_dog_beagle_01",
            "body_plan_id": "quadruped_canine",
        },
        {
            "actor_id": "human0",
            "asset_id": HUMAN_ASSET_ID,
            "template_id": "rocketbox_human_male_adult_01",
            "body_plan_id": "biped_human",
        },
    ]
    view_pose_hash = _canonical_hash(
        {"listener_position_m": listener_h, "listener_yaw_deg": listener_yaw_h}
    )
    frames: list[dict[str, Any]] = []
    for frame_index in range(FRAME_COUNT):
        sample_start = _sample_boundary(frame_index)
        sample_end = _sample_boundary(frame_index + 1)
        states = []
        for actor in actors:
            actor_id = actor["actor_id"]
            phase = (frame_index / FPS * cycles_per_second[actor_id]) % 1.0
            state_core = {
                "actor_id": actor_id,
                "root_transform": {
                    "translation_m": actor_routes_h[actor_id][frame_index],
                    "rotation_xyzw": rotations[actor_id][frame_index],
                    "scale": [1.0, 1.0, 1.0],
                },
                "action_id": "walk",
                "action_time_ticks": frame_index * TICKS_PER_FRAME,
                "action_phase": phase,
                "contacts": {},
                "mouth_state": {
                    "open_ratio": 1.0 if _vocalizing(actor_id, sample_start, sample_end) else 0.0,
                    "vocalizing": _vocalizing(actor_id, sample_start, sample_end),
                },
            }
            states.append({**state_core, "pose_hash": _canonical_hash(state_core)})
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "actor_states": states,
                "view_pose_hashes": {"view0": view_pose_hash},
            }
        )

    timeline = {
        "schema": "avengine_authoritative_timeline_v2",
        "time_base_hz": TIME_BASE_HZ,
        "duration_ticks": FRAME_COUNT * TICKS_PER_FRAME,
        "video": {
            "fps_num": FPS,
            "fps_den": 1,
            "frame_count": FRAME_COUNT,
            "ticks_per_frame": TICKS_PER_FRAME,
            "view_ids": ["view0"],
        },
        "audio": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": SAMPLE_COUNT,
            "ticks_per_sample": 3,
            "channel_count": 2,
        },
        "actors": actors,
        "frames": frames,
        "audio_events": deepcopy(list(_AUDIO_EVENTS)),
    }

    emitter_heights = profile.get("emitter_heights_m", {"dog0": 0.45, "human0": 1.60})
    source_specs = (
        (DOG_SOURCE_ID, "dog0", "muzzle", "animal_vocalization"),
        (HUMAN_SOURCE_ID, "human0", "mouth", "human_speech"),
    )
    source_positions_h: dict[str, list[list[float]]] = {}
    sources = []
    for source_id, actor_id, anchor_id, sound_class in source_specs:
        height = _finite(emitter_heights.get(actor_id), owner=f"{actor_id} emitter height")
        positions = []
        for point in actor_routes_xyz[actor_id]:
            emitter_xyz = [point[0], point[1], point[2] + height]
            positions.append(dataset_z_up_to_habitat(emitter_xyz))
        source_positions_h[source_id] = positions
        actor = next(item for item in actors if item["actor_id"] == actor_id)
        sources.append(
            {
                "source_endpoint_id": source_id,
                "activation": "active",
                "endpoint": {
                    "source_endpoint_id": source_id,
                    "revision": "v1",
                    "admission_state": "research",
                    "source_visibility_mode": "visible_entity",
                    "allowed_sound_class_ids": [sound_class],
                    "directivity_profile_id": "point_emitter_v1",
                    "persistent_when_silent": True,
                    "binding": {
                        "kind": "entity_anchor",
                        "entity_instance_id": actor_id,
                        "entity_asset_id": actor["asset_id"],
                        "entity_asset_revision": "reviewed_runtime_asset",
                        "emitter_anchor_id": anchor_id,
                    },
                },
                "trajectory": {
                    "frame_count": FRAME_COUNT,
                    "position_authority": "timeline_root_plus_versioned_emitter_height",
                    "positions_m": positions,
                },
            }
        )
    source_manifest = {
        "schema": "avengine_optional_residential_source_manifest_v1",
        "scenario_id": "S4_residential_two_source",
        "variant_id": "A",
        "listener": {
            "listener_id": "listener0",
            "camera_listener_colocated": True,
            "camera_listener_cooriented": True,
            "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
        },
        "sources": sources,
    }

    failed_source_frame_indices = {
        source_id: [] for source_id, *_ in source_specs
    }
    gate_sources = {}
    for source_id, actor_id, *_ in source_specs:
        gate_sources[source_id] = {
            "status": "pass",
            "claim_boundary": "source_center_only",
            "failed_frame_indices": [],
            "frames": [
                {
                    "frame_index": frame["frame_index"],
                    "status": frame["status"],
                }
                for frame in gates[actor_id]["frames"]
            ],
        }
    qualification = {
        "schema": "avengine_optional_residential_qualification_v1",
        "status": "pass",
        "room_id": room_id,
        "listener": {
            "position_m": listener_h,
            "yaw_deg": listener_yaw_h,
            "orientation_wxyz": [
                math.cos(math.radians(listener_yaw_h) / 2.0),
                0.0,
                math.sin(math.radians(listener_yaw_h) / 2.0),
                0.0,
            ],
            "camera_hfov_degrees": camera_hfov,
            "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
        },
        "source_center_gate": {
            "status": "pass",
            "claim_boundary": "source_center_only",
            "failed_source_frame_indices": failed_source_frame_indices,
            "sources": gate_sources,
        },
    }
    flags = {
        "schema": "avengine_optional_residential_flag_report_v1",
        "source_flags": {source_id: {} for source_id, *_ in source_specs},
        "clip_flags": {
            "simultaneous_sources": {"status": "present", "value": True},
            "moving_sources": {"status": "present", "value": True},
            "camera_fov_audio_cutoff": {"status": "absent", "value": False},
        },
    }
    room_capsule = {
        "schema": "avengine_m6x_room_capsule_v1",
        "room_capsule_id": f"{scene_id}_optional_residential_v1",
        "revision": "v1",
        "room_registry_ref": {"room_id": room_id, "revision": "v1"},
        "source_scene_provenance": {
            "provider": dataset_id,
            "scene_id": scene_id,
            "external_data": True,
        },
        "camera_listener_rig": {
            "listener_id": "listener0",
            "view_id": "view0",
            "camera_listener_colocated": True,
            "camera_listener_cooriented": True,
        },
    }
    visual_plan = build_spear_visual_plan(
        timeline=timeline,
        source_manifest=source_manifest,
        flags=flags,
        room_capsule=room_capsule,
        qualification=qualification,
        actor_bindings=DEFAULT_ACTOR_BINDINGS,
    )
    for actor in visual_plan["actors"]:
        actor["ue_component_frame_delta"] = component_frame_delta_for_asset(actor["asset_id"])

    return {
        "schema": EPISODE_SCHEMA,
        "status": "pass",
        "scene": {
            "dataset_id": dataset_id,
            "scene_id": scene_id,
            "room_id": room_id,
            "map_path": map_path,
            "claim_boundary": scene_metadata.get("claim_boundary", "external_optional_scene"),
        },
        "clock": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "duration_seconds": FRAME_COUNT / FPS,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": SAMPLE_COUNT,
        },
        "routes_xyz_m": actor_routes_xyz,
        "routes_habitat_m": actor_routes_h,
        "source_trajectories_habitat_m": source_positions_h,
        "source_activity_by_frame": {
            source_id: [
                _vocalizing(actor_id, frame["sample_start"], frame["sample_end"])
                for frame in frames
            ]
            for source_id, actor_id, *_ in source_specs
        },
        "route_metrics": {
            actor_id: {"mean_speed_mps": _actor_speed(route), "gate": gates[actor_id]}
            for actor_id, route in actor_routes_xyz.items()
        },
        "room_polygon_xy_m": deepcopy(list(polygon)),
        "objects": deepcopy(objects),
        "review_lights": deepcopy(profile.get("review_lights", [])),
        "visual_lighting": {
            "profile_id": lighting_profile_id,
            "native_usd_rendering_scope_retained": True,
            "generated_review_light_count": len(profile.get("review_lights", [])),
            "claim_boundary": lighting_claim,
        },
        "acoustic_proxy": deepcopy(profile.get("acoustic_proxy")),
        "timeline": timeline,
        "source_manifest": source_manifest,
        "flags": flags,
        "room_capsule": room_capsule,
        "qualification": qualification,
        "visual_plan": visual_plan,
        "authority": {
            "timeline_navigation_source_logic_audio_topdown_metadata": "AVEngine",
            "room_backend": "comparison_visual_only",
            "backend_may_replan": False,
            "source_center_gate": "center_only_not_body_volume",
            "audio_camera_fov_cutoff": False,
        },
    }


__all__ = [
    "DOG_SOURCE_ID",
    "EPISODE_SCHEMA",
    "FRAME_COUNT",
    "FPS",
    "HUMAN_SOURCE_ID",
    "PROFILE_SCHEMA",
    "ResidentialEpisodeError",
    "SCENE_METADATA_SCHEMA",
    "build_residential_source_episode",
    "classify_object_bounds",
    "dataset_z_up_to_habitat",
    "object_footprint_rectangles_xy",
    "point_in_polygon_xy",
]
