"""Migrate the authoritative 18-second apartment route into Habitat space.

The formal collision primitive in this module is a zero-radius center point on
the Habitat XZ plane.  It is checked directly against the migrated horizontal
AABBs for every frame.  A Habitat navmesh query using a 0.2 m agent radius may
be retained as a diagnostic, but it is deliberately not this gate and cannot
stand in for it.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256


ROUTE_SCHEMA = "avengine_m5_1_legacy_apartment_route_v1"
FRAME_COUNT = 270
FRAME_RATE_HZ = 15
DURATION_SECONDS = 18.0
LEGACY_CAMERA_YAW_DEG = 145.0
HABITAT_CAMERA_YAW_DEG = 55.0
CAMERA_FOV_DEG = 105.0

APARTMENT_MIC_ORIGIN_UE_CM = (-120.0, 80.0, 120.0)
APARTMENT_FLOOR_Z_UE_CM = 27.1
SSOT_TO_HABITAT_EQUATION = "[x,y,z] -> [-1.2+x,0.271+z,0.8-y]"
DOG_REVERSE_OFFSET_SSOT_M = (-0.35, 0.0, 0.0)
MINIMUM_INTER_SOURCE_CENTER_SEPARATION_M = 0.3

BUILTIN_VISUAL_OBSTACLES_SSOT: tuple[
    tuple[str, tuple[float, float, float, float], float, float], ...
] = (
    ("kitchen_island_counter", (-3.35, -4.25, -0.35, -1.10), 0.0, 1.15),
    ("kitchen_peninsula_counter", (-3.95, -1.25, -2.55, 0.65), 0.0, 1.15),
    ("kitchen_sink_counter", (-1.05, -0.55, 1.55, 0.85), 0.0, 1.15),
)


class LegacyRouteError(ValueError):
    """The legacy route or its independently recomputed gate is invalid."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def _finite_vector3(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegacyRouteError([f"{owner} must contain three finite numbers"]) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise LegacyRouteError([f"{owner} must contain three finite numbers"])
    return result


def _finite_points(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegacyRouteError([f"{owner} must be a finite [270,3] array"]) from exc
    if result.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(result)):
        raise LegacyRouteError([f"{owner} must be a finite [270,3] array"])
    return np.ascontiguousarray(result)


def ssot_point_to_habitat(point_m: Sequence[float]) -> list[float]:
    """Apply the frozen legacy SSOT-to-Habitat axis/translation transform."""

    x, y, z = _finite_vector3(point_m, owner="SSOT point")
    return [float(-1.2 + x), float(0.271 + z), float(0.8 - y)]


def ssot_yaw_to_habitat_yaw(yaw_deg: float) -> float:
    """Convert SSOT +X-referenced yaw to Habitat yaw about +Y."""

    if (
        isinstance(yaw_deg, bool)
        or not isinstance(yaw_deg, Real)
        or not math.isfinite(float(yaw_deg))
    ):
        raise LegacyRouteError(["SSOT yaw must be finite"])
    converted = (float(yaw_deg) - 90.0) % 360.0
    return 0.0 if converted == 0.0 else converted


def _ssot_points_to_habitat(points: np.ndarray) -> np.ndarray:
    result = np.empty_like(points, dtype=np.float64)
    result[:, 0] = -1.2 + points[:, 0]
    result[:, 1] = 0.271 + points[:, 2]
    result[:, 2] = 0.8 - points[:, 1]
    return np.ascontiguousarray(result)


def _ue_point_to_ssot(point_ue_cm: Sequence[float]) -> list[float]:
    x, y, z = _finite_vector3(point_ue_cm, owner="UE point")
    return [
        float((x - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100.0),
        float(-(y - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100.0),
        float((z - APARTMENT_FLOOR_Z_UE_CM) / 100.0),
    ]


def _transform_aabb(
    minimum: Sequence[float],
    maximum: Sequence[float],
    transform: Any,
) -> tuple[list[float], list[float]]:
    low = _finite_vector3(minimum, owner="AABB minimum")
    high = _finite_vector3(maximum, owner="AABB maximum")
    if np.any(low > high):
        raise LegacyRouteError(["AABB minimum exceeds maximum"])
    corners = np.asarray(
        [
            transform((x, y, z))
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )
    return corners.min(axis=0).tolist(), corners.max(axis=0).tolist()


def _ue_aabb_to_ssot(
    minimum_ue_cm: Sequence[float], maximum_ue_cm: Sequence[float]
) -> tuple[list[float], list[float]]:
    return _transform_aabb(minimum_ue_cm, maximum_ue_cm, _ue_point_to_ssot)


def _ssot_aabb_to_habitat(
    minimum_ssot_m: Sequence[float], maximum_ssot_m: Sequence[float]
) -> tuple[list[float], list[float]]:
    return _transform_aabb(minimum_ssot_m, maximum_ssot_m, ssot_point_to_habitat)


def _obstacle_record(
    *,
    obstacle_id: str,
    source_kind: str,
    minimum_ssot_m: Sequence[float],
    maximum_ssot_m: Sequence[float],
    included_in_point_gate: bool,
    gate_reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    minimum = _finite_vector3(minimum_ssot_m, owner=f"{obstacle_id} minimum")
    maximum = _finite_vector3(maximum_ssot_m, owner=f"{obstacle_id} maximum")
    if np.any(minimum > maximum):
        raise LegacyRouteError([f"{obstacle_id} AABB minimum exceeds maximum"])
    habitat_minimum, habitat_maximum = _ssot_aabb_to_habitat(minimum, maximum)
    record: dict[str, Any] = {
        "obstacle_id": obstacle_id,
        "source_kind": source_kind,
        "included_in_point_gate": bool(included_in_point_gate),
        "gate_reason": gate_reason,
        "bbox_ssot_m": {"minimum": minimum.tolist(), "maximum": maximum.tolist()},
        "bbox_habitat_m": {
            "minimum": habitat_minimum,
            "maximum": habitat_maximum,
        },
        "horizontal_aabb_habitat_xz_m": {
            "minimum": [habitat_minimum[0], habitat_minimum[2]],
            "maximum": [habitat_maximum[0], habitat_maximum[2]],
        },
    }
    if extra:
        record.update(deepcopy(dict(extra)))
    return record


def _category_by_actor(categories: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for category in ("core", "decoration", "misc"):
        actors = categories.get(category)
        if not isinstance(actors, list) or not all(
            isinstance(actor, str) and actor for actor in actors
        ):
            raise LegacyRouteError([f"furniture category {category!r} is invalid"])
        for actor in actors:
            if actor in result:
                raise LegacyRouteError(
                    [f"furniture actor is multiply categorized: {actor}"]
                )
            result[actor] = category
    return result


def migrate_obstacle_records(
    legacy_spec: Mapping[str, Any],
    furniture_map: Mapping[str, Any],
    shell_map: Mapping[str, Any],
    furniture_categories: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Migrate every legacy map AABB and mark the exact clip gate subset."""

    category_by_actor = _category_by_actor(furniture_categories)
    mode = legacy_spec.get("furniture_mode")
    included_categories = set(legacy_spec.get("furniture_include_categories", []))
    extras = set(legacy_spec.get("furniture_include_actors_extra", []))
    exclusions = set(legacy_spec.get("furniture_exclude_actors", []))
    if mode not in {"full", "subset", "shell"}:
        raise LegacyRouteError(["legacy furniture_mode is invalid"])

    records: list[dict[str, Any]] = []
    furniture = furniture_map.get("furniture")
    if not isinstance(furniture, list):
        raise LegacyRouteError(["legacy furniture map has no furniture list"])
    seen_furniture: set[str] = set()
    for item in furniture:
        if not isinstance(item, Mapping) or not isinstance(item.get("actor_name"), str):
            raise LegacyRouteError(["legacy furniture record is invalid"])
        actor = item["actor_name"]
        if actor in seen_furniture or actor not in category_by_actor:
            raise LegacyRouteError(
                [f"furniture actor category closure failed: {actor}"]
            )
        seen_furniture.add(actor)
        category = category_by_actor[actor]
        selected = (
            mode == "full"
            or (mode == "subset" and category in included_categories)
            or actor in extras
        ) and actor not in exclusions
        ssot_minimum, ssot_maximum = _ue_aabb_to_ssot(
            item.get("bbox_min_ue_cm"), item.get("bbox_max_ue_cm")
        )
        records.append(
            _obstacle_record(
                obstacle_id=f"furniture:{actor}",
                source_kind="legacy_apartment_furniture_map",
                minimum_ssot_m=ssot_minimum,
                maximum_ssot_m=ssot_maximum,
                included_in_point_gate=selected,
                gate_reason=(
                    "selected_by_legacy_clip_furniture_policy"
                    if selected
                    else "retained_migration_record_not_selected_by_clip"
                ),
                extra={
                    "actor_name": actor,
                    "category": category,
                    "bbox_ue_cm": {
                        "minimum": list(item["bbox_min_ue_cm"]),
                        "maximum": list(item["bbox_max_ue_cm"]),
                    },
                },
            )
        )
    if seen_furniture != set(category_by_actor):
        raise LegacyRouteError(["furniture map/category actor sets differ"])

    shell_actors = shell_map.get("shell_actors")
    if not isinstance(shell_actors, list):
        raise LegacyRouteError(["legacy shell map has no shell_actors list"])
    seen_shell: set[str] = set()
    for item in shell_actors:
        if not isinstance(item, Mapping) or not isinstance(item.get("actor_name"), str):
            raise LegacyRouteError(["legacy shell record is invalid"])
        actor = item["actor_name"]
        label = item.get("shell_label")
        if actor in seen_shell or not isinstance(label, str):
            raise LegacyRouteError([f"legacy shell actor is invalid: {actor}"])
        seen_shell.add(actor)
        selected = label not in {"shell_floor", "shell_ceiling"}
        ssot_minimum, ssot_maximum = _ue_aabb_to_ssot(
            item.get("bbox_min_ue_cm"), item.get("bbox_max_ue_cm")
        )
        records.append(
            _obstacle_record(
                obstacle_id=f"shell:{actor}",
                source_kind="legacy_apartment_shell_map",
                minimum_ssot_m=ssot_minimum,
                maximum_ssot_m=ssot_maximum,
                included_in_point_gate=selected,
                gate_reason=(
                    "legacy_non_floor_shell_blocks_horizontal_motion"
                    if selected
                    else "floor_or_ceiling_retained_but_not_horizontal_obstacle"
                ),
                extra={
                    "actor_name": actor,
                    "shell_label": label,
                    "bbox_ue_cm": {
                        "minimum": list(item["bbox_min_ue_cm"]),
                        "maximum": list(item["bbox_max_ue_cm"]),
                    },
                },
            )
        )

    for name, (x0, y0, x1, y1), z0, z1 in BUILTIN_VISUAL_OBSTACLES_SSOT:
        records.append(
            _obstacle_record(
                obstacle_id=f"builtin:{name}",
                source_kind="legacy_builtin_visual_obstacle",
                minimum_ssot_m=(x0, y0, z0),
                maximum_ssot_m=(x1, y1, z1),
                included_in_point_gate=True,
                gate_reason="legacy_manual_apartment_obstacle_missing_from_static_maps",
                extra={"actor_name": name},
            )
        )
    return records


def _horizontal_clearance(
    point_habitat_m: Sequence[float], obstacle: Mapping[str, Any]
) -> tuple[float, bool]:
    point = _finite_vector3(point_habitat_m, owner="Habitat route point")
    horizontal = obstacle.get("horizontal_aabb_habitat_xz_m")
    if not isinstance(horizontal, Mapping):
        raise LegacyRouteError(["obstacle has no horizontal Habitat AABB"])
    minimum = np.asarray(horizontal.get("minimum"), dtype=np.float64)
    maximum = np.asarray(horizontal.get("maximum"), dtype=np.float64)
    if (
        minimum.shape != (2,)
        or maximum.shape != (2,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(minimum > maximum)
    ):
        raise LegacyRouteError(["obstacle horizontal Habitat AABB is invalid"])
    xz = point[[0, 2]]
    delta = np.maximum(np.maximum(minimum - xz, 0.0), xz - maximum)
    clearance = float(np.linalg.norm(delta))
    inside = bool(np.all(xz >= minimum) and np.all(xz <= maximum))
    return clearance, inside


def evaluate_center_point_gate(
    habitat_trajectory_m: Sequence[Sequence[float]],
    obstacles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the zero-radius, per-frame center-point AABB gate."""

    trajectory = _finite_points(habitat_trajectory_m, owner="Habitat trajectory")
    active = [item for item in obstacles if item.get("included_in_point_gate") is True]
    if not active:
        raise LegacyRouteError(["point gate has no active obstacle AABBs"])
    frames: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    minimum_clearance = math.inf
    closest: dict[str, Any] | None = None
    for frame_index, point in enumerate(trajectory):
        frame_minimum = math.inf
        frame_closest: str | None = None
        frame_collisions: list[str] = []
        for obstacle in active:
            obstacle_id = obstacle.get("obstacle_id")
            if not isinstance(obstacle_id, str) or not obstacle_id:
                raise LegacyRouteError(["point-gate obstacle ID is invalid"])
            clearance, inside = _horizontal_clearance(point, obstacle)
            if clearance < frame_minimum:
                frame_minimum = clearance
                frame_closest = obstacle_id
            if clearance < minimum_clearance:
                minimum_clearance = clearance
                closest = {
                    "frame_index": frame_index,
                    "obstacle_id": obstacle_id,
                    "clearance_m": clearance,
                }
            if inside:
                frame_collisions.append(obstacle_id)
                collisions.append(
                    {"frame_index": frame_index, "obstacle_id": obstacle_id}
                )
        frames.append(
            {
                "frame_index": frame_index,
                "status": "pass" if not frame_collisions else "fail",
                "minimum_clearance_m": frame_minimum,
                "closest_obstacle_id": frame_closest,
                "colliding_obstacle_ids": frame_collisions,
            }
        )
    passed = not collisions and minimum_clearance > 0.0
    return {
        "schema": "avengine_m5_1_center_point_aabb_gate_v1",
        "authority": "direct_horizontal_center_point_vs_migrated_aabb",
        "collision_primitive": "zero_radius_center_point_habitat_xz",
        "agent_radius_m": 0.0,
        "navmesh_is_gate": False,
        "frame_count": FRAME_COUNT,
        "active_obstacle_count": len(active),
        "collision_count": len(collisions),
        "minimum_clearance_m": minimum_clearance,
        "closest": closest,
        "collisions": collisions,
        "frames": frames,
        "status": "pass" if passed else "fail",
    }


def _inter_source_gate(
    human_habitat_m: np.ndarray, dog_habitat_m: np.ndarray
) -> dict[str, Any]:
    separations = np.linalg.norm(
        human_habitat_m[:, (0, 2)] - dog_habitat_m[:, (0, 2)], axis=1
    )
    minimum_frame = int(np.argmin(separations))
    minimum = float(separations[minimum_frame])
    passed = minimum >= MINIMUM_INTER_SOURCE_CENTER_SEPARATION_M
    return {
        "schema": "avengine_m5_1_inter_source_center_separation_v1",
        "authority": "per_frame_habitat_xz_center_distance",
        "minimum_required_m": MINIMUM_INTER_SOURCE_CENTER_SEPARATION_M,
        "minimum_observed_m": minimum,
        "minimum_frame_index": minimum_frame,
        "collision_count": int(np.count_nonzero(separations <= 0.0)),
        "per_frame_separation_m": separations.tolist(),
        "status": "pass" if passed else "fail",
    }


def _source_record(
    source_records: Mapping[str, Mapping[str, Any]], name: str
) -> dict[str, Any]:
    record = source_records.get(name)
    if not isinstance(record, Mapping):
        raise LegacyRouteError([f"source record {name!r} is missing"])
    if not isinstance(record.get("path"), str) or not record["path"]:
        raise LegacyRouteError([f"source record {name!r} path is invalid"])
    digest = record.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise LegacyRouteError([f"source record {name!r} SHA-256 is invalid"])
    if (
        isinstance(record.get("byte_size"), bool)
        or not isinstance(record.get("byte_size"), int)
        or record["byte_size"] <= 0
    ):
        raise LegacyRouteError([f"source record {name!r} byte size is invalid"])
    return deepcopy(dict(record))


def build_route_manifest(
    legacy_spec: Mapping[str, Any],
    furniture_map: Mapping[str, Any],
    shell_map: Mapping[str, Any],
    furniture_categories: Mapping[str, Any],
    *,
    source_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one self-contained M5.1 route manifest from legacy JSON inputs."""

    render = legacy_spec.get("render_config")
    if not isinstance(render, Mapping) or (
        render.get("n_frames") != FRAME_COUNT
        or render.get("fps") != FRAME_RATE_HZ
        or float(render.get("duration_s", -1.0)) != DURATION_SECONDS
    ):
        raise LegacyRouteError(["legacy route is not exactly 270 frames / 18 seconds"])
    sources = legacy_spec.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise LegacyRouteError(["legacy route must contain exactly one human source"])
    human_ssot = _finite_points(
        sources[0].get("trajectory_m"), owner="legacy human SSOT trajectory"
    )
    cameras = legacy_spec.get("camera_configs")
    if not isinstance(cameras, list) or len(cameras) != 1:
        raise LegacyRouteError(["legacy route must contain exactly one camera"])
    camera = cameras[0]
    if (
        not isinstance(camera, Mapping)
        or float(camera.get("yaw_deg", math.nan)) != LEGACY_CAMERA_YAW_DEG
        or float(camera.get("fov_deg", math.nan)) != CAMERA_FOV_DEG
    ):
        raise LegacyRouteError(["legacy camera must preserve yaw 145 and FOV 105"])
    camera_ssot = _finite_vector3(camera.get("pos_m"), owner="legacy camera position")
    if ssot_yaw_to_habitat_yaw(LEGACY_CAMERA_YAW_DEG) != HABITAT_CAMERA_YAW_DEG:
        raise LegacyRouteError(["camera yaw conversion contract changed"])

    dog_ssot = human_ssot[::-1].copy()
    dog_ssot += np.asarray(DOG_REVERSE_OFFSET_SSOT_M, dtype=np.float64)
    human_habitat = _ssot_points_to_habitat(human_ssot)
    dog_habitat = _ssot_points_to_habitat(dog_ssot)
    obstacles = migrate_obstacle_records(
        legacy_spec, furniture_map, shell_map, furniture_categories
    )
    human_gate = evaluate_center_point_gate(human_habitat, obstacles)
    dog_gate = evaluate_center_point_gate(dog_habitat, obstacles)
    separation_gate = _inter_source_gate(human_habitat, dog_habitat)
    status = (
        "pass"
        if human_gate["status"]
        == dog_gate["status"]
        == separation_gate["status"]
        == "pass"
        else "fail"
    )

    legacy_input = _source_record(source_records, "legacy_spec")
    legacy_input.update(
        {
            "trajectory_json_pointer": "/sources/0/trajectory_m",
            "trajectory_point_count": FRAME_COUNT,
            "ssot_trajectory_sha256": canonical_json_sha256(human_ssot.tolist()),
        }
    )
    manifest: dict[str, Any] = {
        "schema": ROUTE_SCHEMA,
        "route_id": "m5_1_legacy_apartment_human_dog_18s_v1",
        "status": status,
        "qualification_claim": False,
        "claim_boundary": (
            "M5.1 deterministic route migration and zero-radius center-point AABB "
            "canary; no body-volume, navmesh, mixed-capture, or dataset admission claim"
        ),
        "timebase": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_count": FRAME_COUNT,
            "duration_seconds": DURATION_SECONDS,
        },
        "authoritative_legacy_input": legacy_input,
        "input_maps": {
            "furniture_map": _source_record(source_records, "furniture_map"),
            "shell_map": _source_record(source_records, "shell_map"),
            "furniture_categories": _source_record(
                source_records, "furniture_categories"
            ),
        },
        "coordinate_transform": {
            "source_frame": "legacy_avengine_ssot_xyz_m",
            "target_frame": "habitat_world_xyz_m",
            "equation": SSOT_TO_HABITAT_EQUATION,
            "homogeneous_matrix_row_major": [
                [1.0, 0.0, 0.0, -1.2],
                [0.0, 0.0, 1.0, 0.271],
                [0.0, -1.0, 0.0, 0.8],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "yaw_equation": "habitat_yaw_deg = (ssot_yaw_deg - 90) mod 360",
        },
        "camera": {
            "camera_id": str(camera.get("name", "view0")),
            "ssot_position_m": camera_ssot.tolist(),
            "habitat_position_m": ssot_point_to_habitat(camera_ssot),
            "ssot_yaw_deg": LEGACY_CAMERA_YAW_DEG,
            "habitat_yaw_deg": HABITAT_CAMERA_YAW_DEG,
            "horizontal_fov_deg": CAMERA_FOV_DEG,
        },
        "obstacle_policy": {
            "formal_gate": "per_frame_zero_radius_center_point_vs_horizontal_aabb",
            "furniture_selection": {
                "mode": legacy_spec.get("furniture_mode"),
                "include_categories": list(
                    legacy_spec.get("furniture_include_categories", [])
                ),
                "include_actors_extra": list(
                    legacy_spec.get("furniture_include_actors_extra", [])
                ),
                "exclude_actors": list(legacy_spec.get("furniture_exclude_actors", [])),
            },
            "migrated_obstacle_count": len(obstacles),
            "active_point_gate_obstacle_count": sum(
                item["included_in_point_gate"] for item in obstacles
            ),
            "navmesh_diagnostic": {
                "status": "not_run",
                "formal_gate": False,
                "agent_radius_m": 0.2,
                "reason": (
                    "a radius-0.2 Habitat navmesh diagnostic is not the requested "
                    "zero-radius center-point AABB gate"
                ),
            },
        },
        "obstacles": obstacles,
        "routes": {
            "human_path": {
                "actor_class": "human",
                "derivation": "verbatim_legacy_sources_0_trajectory_m",
                "point_count": FRAME_COUNT,
                "ssot_trajectory_sha256": canonical_json_sha256(human_ssot.tolist()),
                "habitat_trajectory_sha256": canonical_json_sha256(
                    human_habitat.tolist()
                ),
                "ssot_trajectory_m": human_ssot.tolist(),
                "habitat_trajectory_m": human_habitat.tolist(),
            },
            "dog_path": {
                "actor_class": "dog",
                "derivation": "human_path_time_reverse_plus_validated_ssot_offset",
                "source_index_equation": "dog[i] = human[269-i] + offset_ssot_m",
                "offset_ssot_m": list(DOG_REVERSE_OFFSET_SSOT_M),
                "point_count": FRAME_COUNT,
                "ssot_trajectory_sha256": canonical_json_sha256(dog_ssot.tolist()),
                "habitat_trajectory_sha256": canonical_json_sha256(
                    dog_habitat.tolist()
                ),
                "ssot_trajectory_m": dog_ssot.tolist(),
                "habitat_trajectory_m": dog_habitat.tolist(),
            },
        },
        "gates": {
            "human_center_point_aabb": human_gate,
            "dog_center_point_aabb": dog_gate,
            "inter_source_center_separation": separation_gate,
        },
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def validate_route_manifest(value: Mapping[str, Any]) -> list[str]:
    """Independently recompute hashes, transforms, and every formal route gate."""

    errors: list[str] = []
    if not isinstance(value, Mapping):
        return ["route manifest root must be an object"]
    if value.get("schema") != ROUTE_SCHEMA:
        errors.append("route manifest schema differs")
    core = {
        key: item for key, item in value.items() if key != "manifest_content_sha256"
    }
    try:
        if value.get("manifest_content_sha256") != canonical_json_sha256(core):
            errors.append("manifest_content_sha256 differs")
    except (TypeError, ValueError):
        errors.append("manifest content is not canonical-JSON compatible")

    timebase = value.get("timebase")
    if not isinstance(timebase, Mapping) or timebase != {
        "frame_rate_hz": FRAME_RATE_HZ,
        "frame_count": FRAME_COUNT,
        "duration_seconds": DURATION_SECONDS,
    }:
        errors.append("route timebase differs from 270 frames / 18 seconds")
    transform = value.get("coordinate_transform")
    if not isinstance(transform, Mapping) or transform.get("equation") != (
        SSOT_TO_HABITAT_EQUATION
    ):
        errors.append("SSOT-to-Habitat transform equation differs")
    camera = value.get("camera")
    if not isinstance(camera, Mapping) or (
        camera.get("ssot_yaw_deg") != LEGACY_CAMERA_YAW_DEG
        or camera.get("habitat_yaw_deg") != HABITAT_CAMERA_YAW_DEG
        or camera.get("horizontal_fov_deg") != CAMERA_FOV_DEG
    ):
        errors.append("camera yaw/FOV contract differs")
    elif camera.get("habitat_position_m") != ssot_point_to_habitat(
        camera.get("ssot_position_m")
    ):
        errors.append("camera Habitat position was not recomputed from SSOT")

    routes = value.get("routes")
    obstacles = value.get("obstacles")
    if not isinstance(routes, Mapping) or not isinstance(obstacles, list):
        return errors + ["routes or obstacle records are missing"]
    try:
        human = routes["human_path"]
        dog = routes["dog_path"]
        human_ssot = _finite_points(
            human.get("ssot_trajectory_m"), owner="human SSOT trajectory"
        )
        human_habitat = _finite_points(
            human.get("habitat_trajectory_m"), owner="human Habitat trajectory"
        )
        dog_ssot = _finite_points(
            dog.get("ssot_trajectory_m"), owner="dog SSOT trajectory"
        )
        dog_habitat = _finite_points(
            dog.get("habitat_trajectory_m"), owner="dog Habitat trajectory"
        )
    except (KeyError, AttributeError, LegacyRouteError) as exc:
        return errors + [str(exc)]

    for name, route, ssot, habitat in (
        ("human_path", human, human_ssot, human_habitat),
        ("dog_path", dog, dog_ssot, dog_habitat),
    ):
        if route.get("point_count") != FRAME_COUNT:
            errors.append(f"{name} point count differs")
        if route.get("ssot_trajectory_sha256") != canonical_json_sha256(ssot.tolist()):
            errors.append(f"{name} SSOT trajectory hash differs")
        if route.get("habitat_trajectory_sha256") != canonical_json_sha256(
            habitat.tolist()
        ):
            errors.append(f"{name} Habitat trajectory hash differs")
        if not np.array_equal(habitat, _ssot_points_to_habitat(ssot)):
            errors.append(f"{name} Habitat points differ from SSOT transform")
    authority = value.get("authoritative_legacy_input")
    if not isinstance(authority, Mapping) or (
        authority.get("trajectory_point_count") != FRAME_COUNT
        or authority.get("ssot_trajectory_sha256")
        != canonical_json_sha256(human_ssot.tolist())
    ):
        errors.append("authoritative legacy input trajectory binding differs")
    expected_dog = human_ssot[::-1] + np.asarray(
        DOG_REVERSE_OFFSET_SSOT_M, dtype=np.float64
    )
    if not np.array_equal(dog_ssot, expected_dog):
        errors.append("dog path is not the declared reverse-plus-offset derivation")

    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, Mapping):
            errors.append(f"obstacle {index} is not an object")
            continue
        ssot_bbox = obstacle.get("bbox_ssot_m")
        habitat_bbox = obstacle.get("bbox_habitat_m")
        if not isinstance(ssot_bbox, Mapping) or not isinstance(habitat_bbox, Mapping):
            errors.append(f"obstacle {index} lacks migrated AABBs")
            continue
        try:
            expected_minimum, expected_maximum = _ssot_aabb_to_habitat(
                ssot_bbox.get("minimum"), ssot_bbox.get("maximum")
            )
        except LegacyRouteError as exc:
            errors.append(f"obstacle {index}: {exc}")
            continue
        if (
            habitat_bbox.get("minimum") != expected_minimum
            or habitat_bbox.get("maximum") != expected_maximum
        ):
            errors.append(f"obstacle {index} Habitat AABB transform differs")
        expected_horizontal = {
            "minimum": [expected_minimum[0], expected_minimum[2]],
            "maximum": [expected_maximum[0], expected_maximum[2]],
        }
        if obstacle.get("horizontal_aabb_habitat_xz_m") != expected_horizontal:
            errors.append(f"obstacle {index} horizontal point-gate AABB differs")

    gates = value.get("gates")
    if not isinstance(gates, Mapping):
        return errors + ["formal gates are missing"]
    try:
        recomputed_human = evaluate_center_point_gate(human_habitat, obstacles)
        recomputed_dog = evaluate_center_point_gate(dog_habitat, obstacles)
        recomputed_separation = _inter_source_gate(human_habitat, dog_habitat)
    except LegacyRouteError as exc:
        return errors + [str(exc)]
    for name, recomputed in (
        ("human_center_point_aabb", recomputed_human),
        ("dog_center_point_aabb", recomputed_dog),
        ("inter_source_center_separation", recomputed_separation),
    ):
        if gates.get(name) != recomputed:
            errors.append(f"{name} does not match independent recomputation")
        if recomputed["status"] != "pass":
            errors.append(f"{name} did not pass")
    expected_status = (
        "pass" if not any("did not pass" in item for item in errors) else "fail"
    )
    if value.get("status") != expected_status:
        errors.append("top-level route status differs from recomputed gates")
    navmesh = value.get("obstacle_policy", {}).get("navmesh_diagnostic", {})
    if not isinstance(navmesh, Mapping) or navmesh.get("formal_gate") is not False:
        errors.append("navmesh diagnostic must remain explicitly non-gating")
    return list(dict.fromkeys(errors))


def assert_valid_route_manifest(value: Mapping[str, Any]) -> None:
    errors = validate_route_manifest(value)
    if errors:
        raise LegacyRouteError(errors)


__all__ = [
    "APARTMENT_FLOOR_Z_UE_CM",
    "APARTMENT_MIC_ORIGIN_UE_CM",
    "BUILTIN_VISUAL_OBSTACLES_SSOT",
    "CAMERA_FOV_DEG",
    "DOG_REVERSE_OFFSET_SSOT_M",
    "DURATION_SECONDS",
    "FRAME_COUNT",
    "FRAME_RATE_HZ",
    "HABITAT_CAMERA_YAW_DEG",
    "LEGACY_CAMERA_YAW_DEG",
    "LegacyRouteError",
    "MINIMUM_INTER_SOURCE_CENTER_SEPARATION_M",
    "ROUTE_SCHEMA",
    "SSOT_TO_HABITAT_EQUATION",
    "assert_valid_route_manifest",
    "build_route_manifest",
    "evaluate_center_point_gate",
    "migrate_obstacle_records",
    "ssot_point_to_habitat",
    "ssot_yaw_to_habitat_yaw",
    "validate_route_manifest",
]
