"""Native qualification helpers for the fixed SPEAR Apartment canary.

The helpers in this module intentionally keep the placement claim small.  A
route is accepted only when its declared source-center points are navigable
and clear of every retained rigid collision OBB.  No articulated-body volume
is inferred.  The returned :class:`RuntimeObstacleMap` is the same object that
the M6.x Topdown renderer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import (
    InstalledHabitatRuntime,
    _make_configuration,
    discover_runtime_root,
)
from avengine.capture.orientation import habitat_basis_from_yaw_degrees
from avengine.routes.geometry import (
    RuntimeObstacleMap,
    build_runtime_obstacle_map,
    evaluate_source_center_gate,
)


FIXED_APARTMENT_QUALIFICATION_SCHEMA = (
    "avengine_m6x_fixed_apartment_native_qualification_v1"
)


class FixedApartmentQualificationError(RuntimeError):
    """The fixed room could not be qualified with native Habitat geometry."""


@dataclass(frozen=True)
class FixedApartmentQualification:
    """One self-consistent runtime geometry snapshot and its readbacks."""

    obstacle_map: RuntimeObstacleMap
    source_center_gate: Mapping[str, Any]
    anchor_qualification: Mapping[str, Any]
    record: Mapping[str, Any]


@dataclass(frozen=True)
class _MarkerTarget:
    """One live scene marker bound to its authored anchor position."""

    object_id: int
    handle: str
    position_m: tuple[float, float, float]


def _finite_point(value: Any, *, owner: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FixedApartmentQualificationError(
            f"{owner} must contain three finite numbers"
        ) from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise FixedApartmentQualificationError(
            f"{owner} must contain three finite numbers"
        )
    return point


def listener_yaw_degrees_from_request(request: Mapping[str, Any]) -> float:
    """Read the +Y Habitat yaw from the fixed co-located camera/listener rig."""

    try:
        quaternion_xyzw = np.asarray(
            request["primary_camera_rig"]["world_from_rig"]["rotation_xyzw"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise FixedApartmentQualificationError(
            "M1 request lacks a finite camera rotation"
        ) from exc
    if quaternion_xyzw.shape != (4,) or not np.all(np.isfinite(quaternion_xyzw)):
        raise FixedApartmentQualificationError(
            "M1 request lacks a finite camera rotation"
        )
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm <= 1.0e-12:
        raise FixedApartmentQualificationError("camera rotation is zero")
    x, y, z, w = quaternion_xyzw / norm
    # These fixed-room requests carry a pure +Y yaw.  Retain the complete
    # formula so a numerically equivalent unit quaternion has one readback.
    yaw = math.degrees(math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))
    if not math.isfinite(yaw):
        raise FixedApartmentQualificationError("camera yaw is not finite")
    return float(yaw)


def listener_orientation_wxyz(yaw_degrees: float) -> tuple[float, float, float, float]:
    """Return the fixed RLR listener quaternion for one Habitat +Y yaw."""

    if not math.isfinite(float(yaw_degrees)):
        raise FixedApartmentQualificationError("listener yaw must be finite")
    half = math.radians(float(yaw_degrees)) * 0.5
    return (float(math.cos(half)), 0.0, float(math.sin(half)), 0.0)


def _sector(azimuth_degrees: float) -> str:
    value = float(azimuth_degrees)
    if -22.5 <= value < 22.5:
        return "front"
    if 22.5 <= value < 67.5:
        return "front_right"
    if 67.5 <= value < 112.5:
        return "right"
    if 112.5 <= value < 157.5:
        return "rear_right"
    if value >= 157.5 or value < -157.5:
        return "rear"
    if -157.5 <= value < -112.5:
        return "rear_left"
    if -112.5 <= value < -67.5:
        return "left"
    return "front_left"


def _source_geometry(
    source_position_m: Sequence[float],
    listener_position_m: Sequence[float],
    listener_yaw_deg: float,
) -> tuple[float, float]:
    source = _finite_point(source_position_m, owner="source position")
    listener = _finite_point(listener_position_m, owner="listener position")
    delta = source - listener
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-9:
        raise FixedApartmentQualificationError("source coincides with listener")
    basis = habitat_basis_from_yaw_degrees(listener_yaw_deg)
    right = np.asarray(basis.right_xyz, dtype=np.float64)
    forward = np.asarray(basis.forward_xyz, dtype=np.float64)
    azimuth = math.degrees(
        math.atan2(float(np.dot(delta, right)), float(np.dot(delta, forward)))
    )
    return distance, float(azimuth)


def _marker_targets_by_anchor(
    object_manager: Any,
    anchors: Sequence[Mapping[str, Any]],
    *,
    position_tolerance_m: float = 0.035,
) -> dict[str, _MarkerTarget]:
    """Bind each marker anchor to one concrete live Habitat object.

    The anchor library already carries the authored world position and the
    scene instance carries each marker object's handle and translation.  A
    position join keeps this binding independent of Habitat's runtime object
    ID allocation while still requiring the subsequent ray hit to match that
    exact live object ID.
    """

    if not math.isfinite(position_tolerance_m) or position_tolerance_m < 0.0:
        raise FixedApartmentQualificationError(
            "marker position tolerance must be finite and nonnegative"
        )
    try:
        values = object_manager.get_objects_by_handle_substring().values()
    except (AttributeError, TypeError) as exc:
        raise FixedApartmentQualificationError(
            "Habitat rigid-object manager is unavailable"
        ) from exc
    marker_objects = tuple(
        _MarkerTarget(
            object_id=int(value.object_id),
            handle=str(value.handle),
            position_m=tuple(
                float(component)
                for component in _finite_point(
                    value.translation,
                    owner=f"marker object {value.handle!s} translation",
                )
            ),
        )
        for value in values
        if str(value.handle).startswith("source_marker_")
    )
    marker_anchors = tuple(
        anchor
        for anchor in anchors
        if str(anchor.get("anchor_id", "")).startswith("marker_")
    )
    bindings: dict[str, _MarkerTarget] = {}
    claimed_object_ids: set[int] = set()
    for anchor in marker_anchors:
        anchor_id = str(anchor["anchor_id"])
        anchor_position = _finite_point(
            anchor["position_m"], owner=f"anchor {anchor_id} position"
        )
        candidates = [
            item
            for item in marker_objects
            if float(
                np.linalg.norm(
                    np.asarray(item.position_m, dtype=np.float64) - anchor_position
                )
            )
            <= position_tolerance_m
        ]
        if len(candidates) != 1:
            raise FixedApartmentQualificationError(
                f"marker anchor {anchor_id!r} must match exactly one live scene "
                f"marker by position; observed {len(candidates)}"
            )
        target = candidates[0]
        if target.object_id in claimed_object_ids:
            raise FixedApartmentQualificationError(
                f"live scene marker {target.handle!r} is bound to multiple anchors"
            )
        bindings[anchor_id] = target
        claimed_object_ids.add(target.object_id)
    return bindings


def _anchor_probe_position(
    anchor: Mapping[str, Any], *, floor_height_m: float
) -> np.ndarray:
    position = _finite_point(anchor["position_m"], owner="anchor position")
    if anchor["kind"] == "camera_listener_pose":
        return position
    height = float(anchor["los_probe_height_m"])
    if not math.isfinite(height) or height < 0.0:
        raise FixedApartmentQualificationError(
            "anchor los_probe_height_m must be finite and nonnegative"
        )
    # Height is an offset from the declared source-center/spawn anchor.  This
    # keeps a marker whose center is already above the floor unchanged when
    # the offset is zero, while human/dog root anchors can name a mouth/head
    # probe height explicitly.
    result = position.copy()
    result[1] += height
    return result


def _qualify_anchors(
    simulator: Any,
    habitat_sim: Any,
    mn: Any,
    *,
    anchors: Sequence[Mapping[str, Any]],
    listener_position_m: Sequence[float],
    listener_yaw_deg: float,
    camera_hfov_degrees: float,
    floor_height_m: float,
    marker_targets_by_anchor: Mapping[str, _MarkerTarget],
    hit_tolerance_m: float = 0.035,
) -> dict[str, Any]:
    listener = _finite_point(listener_position_m, owner="listener position")
    records: list[dict[str, Any]] = []
    failed_ids: list[str] = []
    for anchor in anchors:
        if anchor["kind"] == "camera_listener_pose":
            continue
        target = _anchor_probe_position(anchor, floor_height_m=floor_height_m)
        distance, azimuth = _source_geometry(target, listener, listener_yaw_deg)
        direction = (target - listener) / distance
        cast = simulator.cast_ray(
            habitat_sim.geo.Ray(mn.Vector3(listener), mn.Vector3(direction)),
            buffer_distance=0.0,
        )
        nearest_distance = math.inf
        nearest_object_id: int | None = None
        if cast.has_hits():
            nearest_distance = float(cast.hits[0].ray_distance)
            nearest_object_id = int(cast.hits[0].object_id)
        target_is_marker = anchor["anchor_id"].startswith("marker_")
        marker_target = marker_targets_by_anchor.get(str(anchor["anchor_id"]))
        if target_is_marker and marker_target is None:
            raise FixedApartmentQualificationError(
                f"marker anchor {anchor['anchor_id']!r} has no live scene-object binding"
            )
        all_marker_object_ids = {
            item.object_id for item in marker_targets_by_anchor.values()
        }
        nearest_hit_is_marker = nearest_object_id in all_marker_object_ids
        target_object_hit = bool(
            target_is_marker
            and marker_target is not None
            and nearest_object_id is not None
            and nearest_object_id == marker_target.object_id
        )
        if target_is_marker and nearest_hit_is_marker:
            # A source-marker hit is transparent only for the marker bound to
            # this anchor.  Hitting any other marker is an occlusion, even if
            # it happens to fall within the endpoint-distance tolerance.
            los = target_object_hit
        else:
            los = bool(
                not math.isfinite(nearest_distance)
                or nearest_distance + hit_tolerance_m >= distance
            )
        observed_path = "los" if los else "nlos"
        observed_fov = (
            "in_fov"
            if abs(azimuth) <= float(camera_hfov_degrees) * 0.5 + 1.0e-9
            else "out_of_fov"
        )
        observed_sector = _sector(azimuth)
        passed = (
            observed_path == anchor["expected_acoustic_path"]
            and observed_fov == anchor["expected_camera_fov"]
            and observed_sector == anchor["listener_relative_sector"]
        )
        if not passed:
            failed_ids.append(str(anchor["anchor_id"]))
        records.append(
            {
                "anchor_id": anchor["anchor_id"],
                "status": "pass" if passed else "fail",
                "source_center_m": list(anchor["position_m"]),
                "los_probe_position_m": target.tolist(),
                "distance_m": distance,
                "azimuth_deg": azimuth,
                "observed_listener_relative_sector": observed_sector,
                "observed_camera_fov": observed_fov,
                "observed_acoustic_path": observed_path,
                "nearest_ray_hit_distance_m": (
                    None if not math.isfinite(nearest_distance) else nearest_distance
                ),
                "nearest_ray_hit_object_id": nearest_object_id,
                "target_marker_object_hit": target_object_hit,
                "expected_target_marker_object_id": (
                    None if marker_target is None else marker_target.object_id
                ),
                "expected_target_marker_object_handle": (
                    None if marker_target is None else marker_target.handle
                ),
                "expected": {
                    "listener_relative_sector": anchor["listener_relative_sector"],
                    "camera_fov": anchor["expected_camera_fov"],
                    "acoustic_path": anchor["expected_acoustic_path"],
                },
            }
        )
    return {
        "status": "fail" if failed_ids else "pass",
        "raycast_authority": "live_habitat_scene_collision",
        "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
        "failed_anchor_ids": failed_ids,
        "records": records,
    }


def qualify_fixed_apartment(
    *,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    anchor_library: Mapping[str, Any],
    source_center_trajectories_m: Mapping[str, Any],
    runtime_root: str | Path | None = None,
    installed_runtime: "InstalledHabitatRuntime | None" = None,
    meters_per_pixel: float = 0.02,
    maximum_floor_snap_xz_m: float = 0.02,
    maximum_floor_y_delta_m: float = 0.25,
    minimum_navmesh_clearance_m: float = 0.0,
    minimum_rigid_clearance_m: float = 0.0,
) -> FixedApartmentQualification:
    """Load the declared room once and produce all native placement evidence."""

    room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    if installed_runtime is not None:
        if runtime_root is not None:
            raise FixedApartmentQualificationError(
                "installed-prefix qualification does not accept runtime_root"
            )
        runtime = None
    else:
        runtime = discover_runtime_root(runtime_root)
    rig = room_inputs.request["primary_camera_rig"]
    listener = _finite_point(
        rig["world_from_rig"]["translation_m"], owner="listener position"
    )
    yaw = listener_yaw_degrees_from_request(room_inputs.request)
    hfov = float(rig["shared_calibration"]["hfov_degrees"])
    if not math.isfinite(hfov) or not 0.0 < hfov < 180.0:
        raise FixedApartmentQualificationError("camera HFOV is invalid")

    if installed_runtime is None:
        # The pinned Habitat build requires numpy-quaternion to be imported first.
        import quaternion  # noqa: F401

        import habitat_sim
        import magnum as mn
    else:
        habitat_sim = installed_runtime.habitat_sim
        mn = installed_runtime.magnum

    configuration, _modalities, _listener_uuid, resolved_scene = _make_configuration(
        room_inputs,
        runtime,
        Path(room_manifest_path).resolve().parent / ".m6x_scratch",
        include_audio_sensor=installed_runtime is None,
        physics_config_path=(
            None if installed_runtime is None else installed_runtime.physics_config_path
        ),
    )
    with habitat_sim.Simulator(configuration) as simulator:
        navmesh_path = resolved_scene.get("navmesh")
        if navmesh_path is None or not Path(navmesh_path).is_file():
            raise FixedApartmentQualificationError(
                "fixed Apartment declares no readable navmesh"
            )
        if not simulator.pathfinder.load_nav_mesh(str(navmesh_path)):
            raise FixedApartmentQualificationError(
                "Habitat could not load the declared Apartment navmesh"
            )
        object_manager = simulator.get_rigid_object_manager()
        obstacle_map = build_runtime_obstacle_map(
            simulator.pathfinder,
            object_manager,
            mn,
            floor_height_m=float(anchor_library.get("floor_height_m", 0.271)),
            meters_per_pixel=meters_per_pixel,
            excluded_handle_prefixes=("source_marker_",),
        )
        gate = evaluate_source_center_gate(
            simulator.pathfinder,
            obstacle_map,
            source_center_trajectories_m,
            maximum_floor_snap_xz_m=maximum_floor_snap_xz_m,
            maximum_floor_y_delta_m=maximum_floor_y_delta_m,
            minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
            minimum_rigid_clearance_m=minimum_rigid_clearance_m,
        )
        anchor_record = _qualify_anchors(
            simulator,
            habitat_sim,
            mn,
            anchors=anchor_library["anchors"],
            listener_position_m=listener,
            listener_yaw_deg=yaw,
            camera_hfov_degrees=hfov,
            floor_height_m=obstacle_map.floor_height_m,
            marker_targets_by_anchor=_marker_targets_by_anchor(
                object_manager, anchor_library["anchors"]
            ),
        )

    status = (
        "pass"
        if gate["status"] == "pass" and anchor_record["status"] == "pass"
        else "fail"
    )
    record = {
        "schema": FIXED_APARTMENT_QUALIFICATION_SCHEMA,
        "status": status,
        "room_id": room_inputs.room["room_id"],
        "runtime_backend": "habitat_sim_avengine",
        "listener": {
            "position_m": listener.tolist(),
            "yaw_deg": yaw,
            "orientation_wxyz": list(listener_orientation_wxyz(yaw)),
            "camera_hfov_degrees": hfov,
            "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
        },
        "obstacle_authority": obstacle_map.summary(),
        "source_center_gate": gate,
        "anchor_qualification": anchor_record,
        "claim_boundary": (
            "source-center placement only; no articulated-body-volume claim"
        ),
    }
    return FixedApartmentQualification(
        obstacle_map=obstacle_map,
        source_center_gate=gate,
        anchor_qualification=anchor_record,
        record=record,
    )


__all__ = [
    "FIXED_APARTMENT_QUALIFICATION_SCHEMA",
    "FixedApartmentQualification",
    "FixedApartmentQualificationError",
    "listener_orientation_wxyz",
    "listener_yaw_degrees_from_request",
    "qualify_fixed_apartment",
]
