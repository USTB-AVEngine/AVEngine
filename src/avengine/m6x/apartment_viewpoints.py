"""Deterministic 2-D raster prefilter for Apartment viewpoint authoring.

This module does not certify a camera, target pixel, composition, or native
render.  It reduces a geometry-derived candidate set using a retained
``RuntimeObstacleMap`` and explicit M1 horizontal calibration.  Its raster
sample signatures are deterministic proxies used for diversity selection.

Native first-person review remains required for target surface pixels,
vertical framing, rigid-object occlusion, lighting, and low/high camera poses.
Studio is not used by this prefilter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from numbers import Real
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from avengine.m5_1.orientation import (
    habitat_basis_from_yaw_degrees,
    habitat_yaw_degrees_from_xyzw,
)
from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.room_feasibility import (
    RoomFeasibilityCompiler,
    RoomFeasibilityError,
)
from avengine.optional_backends.spear_visual import (
    camera_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)


_NATIVE_PENDING_CHECKS = (
    "target_surface_pixel",
    "vertical_fov",
    "rigid_object_occlusion",
    "first_person_composition",
    "lighting",
)
_REVERSE_ROLES = frozenset(("living_reverse", "kitchen_reverse"))


class ApartmentRasterPrefilterError(ValueError):
    """Inputs cannot produce the requested deterministic raster prefilter."""


@dataclass(frozen=True)
class StraightCorridor:
    """One undirected chord with separate forward/reverse motion identities."""

    corridor_id: str
    start_xz_m: tuple[float, float]
    end_xz_m: tuple[float, float]
    length_m: float
    origin: str
    direction_ids: tuple[str, str]
    source_center_heights_m: tuple[float, ...] = ()
    sample_pixels_rc: tuple[tuple[int, int], ...] = field(
        default=(), repr=False, compare=False
    )

    def to_record(self) -> dict[str, Any]:
        return {
            "corridor_id": self.corridor_id,
            "coordinate_space": "habitat_xz_m",
            "start_xz_m": list(self.start_xz_m),
            "end_xz_m": list(self.end_xz_m),
            "length_m": self.length_m,
            "origin": self.origin,
            "direction_ids": list(self.direction_ids),
            "source_center_heights_m": list(self.source_center_heights_m),
        }


@dataclass(frozen=True)
class SemanticRasterTarget:
    """Surface aim point and a separately legal navmesh focus proxy."""

    target_id: str
    role: str
    surface_position_m: tuple[float, float, float]
    focus_position_m: tuple[float, float, float]
    focus_derivation: str
    focus_snap_distance_m: float

    @property
    def surface_xz_m(self) -> tuple[float, float]:
        return (self.surface_position_m[0], self.surface_position_m[2])

    @property
    def focus_xz_m(self) -> tuple[float, float]:
        return (self.focus_position_m[0], self.focus_position_m[2])

    def to_record(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "role": self.role,
            "surface_position_m": list(self.surface_position_m),
            "focus_position_m": list(self.focus_position_m),
            "focus_derivation": self.focus_derivation,
            "focus_snap_distance_m": self.focus_snap_distance_m,
            "target_surface_pixel_validation_status": "pending_native",
        }


@dataclass(frozen=True)
class SemanticFocusRasterCheck:
    """Horizontal/raster proxy checks for one semantic focus."""

    target_id: str
    focus_in_horizontal_frustum: bool
    focus_navmesh_los: bool
    prefilter_pass: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "focus_in_horizontal_frustum": self.focus_in_horizontal_frustum,
            "focus_navmesh_los": self.focus_navmesh_los,
            "prefilter_pass": self.prefilter_pass,
            "target_surface_pixel_validation_status": "pending_native",
        }


@dataclass(frozen=True)
class CorridorRasterCheck:
    """Endpoint/midpoint horizontal raster and lateral-cue proxy checks."""

    corridor_id: str
    point_in_horizontal_frustum: tuple[bool, bool, bool]
    point_navmesh_los: tuple[bool, bool, bool]
    point_prefilter_pass: tuple[bool, bool, bool]
    lateral_c: tuple[float, float, float]
    minimum_abs_lateral_c: float
    prefilter_pass: bool
    direction_ids: tuple[str, str]

    def to_record(self) -> dict[str, Any]:
        return {
            "corridor_id": self.corridor_id,
            "point_order": ["start", "midpoint", "end"],
            "point_in_horizontal_frustum": list(self.point_in_horizontal_frustum),
            "point_navmesh_los": list(self.point_navmesh_los),
            "point_prefilter_pass": list(self.point_prefilter_pass),
            "lateral_c": list(self.lateral_c),
            "minimum_abs_lateral_c": self.minimum_abs_lateral_c,
            "prefilter_pass": self.prefilter_pass,
            "direction_ids": list(self.direction_ids),
            "native_pixel_validation_status": "pending",
        }


@dataclass(frozen=True)
class RasterPosePrefilter:
    """Bounded horizontal raster proxy evidence for one camera pose."""

    frustum_sample_signature: frozenset[int]
    frustum_sample_count: int
    total_raster_sample_count: int
    frustum_sample_fraction: float
    semantic_focus_checks: tuple[SemanticFocusRasterCheck, ...]
    corridor_checks: tuple[CorridorRasterCheck, ...]
    grid_los_query_count: int

    @property
    def passed_semantic_focus_ids(self) -> tuple[str, ...]:
        return tuple(
            item.target_id for item in self.semantic_focus_checks if item.prefilter_pass
        )

    @property
    def passed_corridor_ids(self) -> tuple[str, ...]:
        return tuple(
            item.corridor_id for item in self.corridor_checks if item.prefilter_pass
        )


@dataclass(frozen=True)
class PrefilteredApartmentViewpoint:
    """One selected 2-D raster prefilter candidate, not a certified viewpoint."""

    viewpoint_id: str
    view_id: str
    roles: tuple[str, ...]
    target_ids: tuple[str, ...]
    height_role: str
    camera_position_m: tuple[float, float, float]
    camera_yaw_deg: float
    ue_position_cm: tuple[float, float, float]
    ue_yaw_deg: float
    hfov_degrees: float
    near_m: float
    far_m: float
    pose_difference_proxy_from_legacy: float
    frustum_sample_count: int
    total_raster_sample_count: int
    frustum_sample_fraction: float
    passed_semantic_focus_ids: tuple[str, ...]
    corridor_checks: tuple[CorridorRasterCheck, ...]
    passed_corridor_ids: tuple[str, ...]
    nearest_selected_difference: Mapping[str, float] | None
    native_pixel_validation_status: str
    native_validation_pending: tuple[str, ...]
    _frustum_sample_signature: frozenset[int] = field(
        default_factory=frozenset, repr=False, compare=False
    )

    def to_record(self) -> dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "view_id": self.view_id,
            "artifact_status": "deterministic_2d_raster_prefilter_only",
            "roles": list(self.roles),
            "target_ids": list(self.target_ids),
            "height_role": self.height_role,
            "camera": {
                "habitat_translation_m": list(self.camera_position_m),
                "habitat_yaw_deg": self.camera_yaw_deg,
                "ue_location_cm": list(self.ue_position_cm),
                "ue_yaw_deg": self.ue_yaw_deg,
                "hfov_degrees": self.hfov_degrees,
                "near_m": self.near_m,
                "far_m": self.far_m,
            },
            "pose_difference_proxy_from_legacy": (
                self.pose_difference_proxy_from_legacy
            ),
            "raster_frustum_sample": {
                "sample_count": self.frustum_sample_count,
                "total_sample_count": self.total_raster_sample_count,
                "fraction": self.frustum_sample_fraction,
                "semantic_focus_prefilter_pass_ids": list(
                    self.passed_semantic_focus_ids
                ),
            },
            "corridor_prefilter_pass_ids": list(self.passed_corridor_ids),
            "corridor_checks": [item.to_record() for item in self.corridor_checks],
            "nearest_selected_difference": (
                None
                if self.nearest_selected_difference is None
                else dict(self.nearest_selected_difference)
            ),
            "native_pixel_validation_status": self.native_pixel_validation_status,
            "native_validation_pending": list(self.native_validation_pending),
        }


@dataclass(frozen=True)
class ApartmentRasterPrefilterResult:
    """Selected candidates and bounded evidence from the 2-D prefilter."""

    viewpoints: tuple[PrefilteredApartmentViewpoint, ...]
    corridors: tuple[StraightCorridor, ...]
    semantic_targets: tuple[SemanticRasterTarget, ...]
    raw_candidate_count: int
    quality_filtered_candidate_count: int
    exact_deduplicated_candidate_count: int
    raster_sample_spacing_m: float
    raster_sample_count: int
    minimum_candidate_thresholds: Mapping[str, float | int]
    minimum_pairwise_thresholds: Mapping[str, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_status": "deterministic_2d_raster_prefilter_only",
            "claim_boundary": (
                "horizontal 2D navmesh sample and LOS proxies only; native pixels, "
                "vertical FOV, rigid occlusion, composition, lighting, and low/high "
                "camera validation remain pending"
            ),
            "room_family": "apartment",
            "formal_view_id": "view0",
            "candidate_counts": {
                "raw": self.raw_candidate_count,
                "candidate_quality_passed": self.quality_filtered_candidate_count,
                "exact_pose_signature_deduplicated": (
                    self.exact_deduplicated_candidate_count
                ),
                "pairwise_threshold_selected": len(self.viewpoints),
            },
            "raster_sample_spacing_m": self.raster_sample_spacing_m,
            "raster_sample_count": self.raster_sample_count,
            "minimum_candidate_thresholds": dict(self.minimum_candidate_thresholds),
            "minimum_pairwise_thresholds": dict(self.minimum_pairwise_thresholds),
            "semantic_targets": [item.to_record() for item in self.semantic_targets],
            "corridors": [item.to_record() for item in self.corridors],
            "viewpoints": [item.to_record() for item in self.viewpoints],
        }


@dataclass(frozen=True)
class _Grid:
    navmesh: np.ndarray
    bounds: np.ndarray
    pixel_x_m: float
    pixel_z_m: float

    def pixel_center_xz(self, row: int, col: int) -> tuple[float, float]:
        return (
            float(self.bounds[0, 0] + (col + 0.5) * self.pixel_x_m),
            float(self.bounds[0, 2] + (row + 0.5) * self.pixel_z_m),
        )

    def pixel_for_xz(self, point_xz: Sequence[float]) -> tuple[int, int] | None:
        x, z = _finite_xz(point_xz, owner="world point")
        col = int(math.floor((x - self.bounds[0, 0]) / self.pixel_x_m))
        row = int(math.floor((z - self.bounds[0, 2]) / self.pixel_z_m))
        if 0 <= row < self.navmesh.shape[0] and 0 <= col < self.navmesh.shape[1]:
            return row, col
        return None

    def navmesh_points(self) -> tuple[np.ndarray, np.ndarray]:
        pixels = np.argwhere(self.navmesh != 0).astype(np.int32, copy=False)
        x = self.bounds[0, 0] + (pixels[:, 1].astype(np.float64) + 0.5) * self.pixel_x_m
        z = self.bounds[0, 2] + (pixels[:, 0].astype(np.float64) + 0.5) * self.pixel_z_m
        return pixels, np.stack((x, z), axis=1)


@dataclass(frozen=True)
class _CameraContract:
    position_m: tuple[float, float, float]
    yaw_deg: float
    hfov_degrees: float
    near_m: float
    far_m: float


@dataclass(frozen=True)
class _Candidate:
    roles: tuple[str, ...]
    target_ids: tuple[str, ...]
    height_role: str
    position_m: tuple[float, float, float]
    yaw_deg: float
    pose_difference_proxy: float
    prefilter: RasterPosePrefilter
    is_legacy: bool = False

    @property
    def stable_key(self) -> tuple[Any, ...]:
        return (
            0 if self.is_legacy else 1,
            tuple(value.hex() for value in self.position_m),
            self.yaw_deg.hex(),
            self.roles,
            self.target_ids,
        )


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ApartmentRasterPrefilterError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ApartmentRasterPrefilterError(f"{owner} must be a finite number")
    return result


def _positive_number(value: Any, *, owner: str, allow_zero: bool = False) -> float:
    result = _finite_number(value, owner=owner)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ApartmentRasterPrefilterError(f"{owner} must be {qualifier}")
    return result


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ApartmentRasterPrefilterError(f"{owner} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ApartmentRasterPrefilterError(f"{owner} must be a nonnegative integer")
    return value


def _finite_xz(value: Any, *, owner: str) -> tuple[float, float]:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApartmentRasterPrefilterError(
            f"{owner} must contain two finite numbers"
        ) from exc
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ApartmentRasterPrefilterError(f"{owner} must contain two finite numbers")
    return float(point[0]), float(point[1])


def _finite_xyz_or_xz(
    value: Any, *, owner: str, default_y_m: float
) -> tuple[float, float, float]:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApartmentRasterPrefilterError(
            f"{owner} must contain finite XZ or XYZ coordinates"
        ) from exc
    if point.shape == (2,) and np.all(np.isfinite(point)):
        return float(point[0]), default_y_m, float(point[1])
    if point.shape == (3,) and np.all(np.isfinite(point)):
        return tuple(float(item) for item in point)
    raise ApartmentRasterPrefilterError(
        f"{owner} must contain finite XZ or XYZ coordinates"
    )


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not result:
        raise ApartmentRasterPrefilterError("semantic target ID is unusable")
    return result


def _wrap_yaw_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 0.0 if wrapped == 0.0 else wrapped


def circular_yaw_distance_degrees(left: Real, right: Real) -> float:
    """Return the shortest separation of two headings in ``[0, 180]``."""

    left_value = _finite_number(left, owner="left yaw")
    right_value = _finite_number(right, owner="right yaw")
    return abs(_wrap_yaw_degrees(left_value - right_value))


def _validated_grid(obstacle_map: RuntimeObstacleMap) -> _Grid:
    if not isinstance(obstacle_map, RuntimeObstacleMap):
        raise ApartmentRasterPrefilterError("obstacle_map must be a RuntimeObstacleMap")
    # Validate the original dtype/values before casting.  Casting first could
    # silently turn values such as 256 into a legal-looking zero.
    raw_navmesh = np.asarray(obstacle_map.binary_navmesh)
    if raw_navmesh.ndim != 2 or raw_navmesh.size == 0:
        raise ApartmentRasterPrefilterError(
            "obstacle_map contains an invalid binary navmesh"
        )
    try:
        binary_values = np.isin(raw_navmesh, (0, 1))
    except (TypeError, ValueError) as exc:
        raise ApartmentRasterPrefilterError(
            "obstacle_map contains an invalid binary navmesh"
        ) from exc
    if not np.all(binary_values) or not np.any(raw_navmesh == 1):
        raise ApartmentRasterPrefilterError(
            "obstacle_map contains an invalid binary navmesh"
        )
    bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[1] <= bounds[0])
    ):
        raise ApartmentRasterPrefilterError(
            "obstacle_map contains invalid world bounds"
        )
    navmesh = np.ascontiguousarray(raw_navmesh, dtype=np.uint8)
    return _Grid(
        navmesh=navmesh,
        bounds=np.ascontiguousarray(bounds),
        pixel_x_m=float((bounds[1, 0] - bounds[0, 0]) / navmesh.shape[1]),
        pixel_z_m=float((bounds[1, 2] - bounds[0, 2]) / navmesh.shape[0]),
    )


def _extract_camera_contract(base_m1_camera: Mapping[str, Any]) -> _CameraContract:
    if not isinstance(base_m1_camera, Mapping):
        raise ApartmentRasterPrefilterError("base M1 camera must be a mapping")
    rig: Any = base_m1_camera.get("primary_camera_rig", base_m1_camera)
    if not isinstance(rig, Mapping) or rig.get("view_id", "view0") != "view0":
        raise ApartmentRasterPrefilterError(
            "base M1 camera must retain formal view_id 'view0'"
        )
    try:
        transform = rig["world_from_rig"]
        calibration = rig["shared_calibration"]
        position = np.asarray(transform["translation_m"], dtype=np.float64)
        yaw = habitat_yaw_degrees_from_xyzw(transform["rotation_xyzw"])
        hfov = _finite_number(
            calibration["hfov_degrees"], owner="M1 camera hfov_degrees"
        )
        near_m = _positive_number(calibration["near_m"], owner="M1 camera near_m")
        far_m = _positive_number(calibration["far_m"], owner="M1 camera far_m")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ApartmentRasterPrefilterError(
            "base M1 camera lacks a finite pose and calibration"
        ) from exc
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ApartmentRasterPrefilterError(
            "base M1 camera translation_m must contain three finite numbers"
        )
    if not 0.0 < hfov < 180.0 or far_m <= near_m:
        raise ApartmentRasterPrefilterError("base M1 camera calibration is invalid")
    return _CameraContract(
        position_m=tuple(float(item) for item in position),
        yaw_deg=_wrap_yaw_degrees(yaw),
        hfov_degrees=hfov,
        near_m=near_m,
        far_m=far_m,
    )


def _inferred_target_role(target_id: str) -> str:
    normalized = _slug(target_id)
    if "living" in normalized:
        return "living_reverse"
    if "kitchen" in normalized:
        return "kitchen_reverse"
    if "window" in normalized:
        return "window_facing"
    return f"{normalized}_facing"


def _nearest_navmesh_focus(
    grid: _Grid, point_xz_m: Sequence[float]
) -> tuple[tuple[float, float], float]:
    _pixels, points = grid.navmesh_points()
    point = np.asarray(_finite_xz(point_xz_m, owner="semantic surface"))
    distances = np.linalg.norm(points - point, axis=1)
    index = int(np.argmin(distances))
    return tuple(float(item) for item in points[index]), float(distances[index])


def _points_in_polygon(points_xz: np.ndarray, polygon_xz: np.ndarray) -> np.ndarray:
    """Vectorized even-odd inclusion with polygon-boundary inclusion."""

    inside = np.zeros(points_xz.shape[0], dtype=np.bool_)
    on_boundary = np.zeros(points_xz.shape[0], dtype=np.bool_)
    x = points_xz[:, 0]
    z = points_xz[:, 1]
    for index in range(len(polygon_xz)):
        left = polygon_xz[index]
        right = polygon_xz[(index + 1) % len(polygon_xz)]
        edge = right - left
        relative = points_xz - left
        cross = edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        dot = relative[:, 0] * edge[0] + relative[:, 1] * edge[1]
        edge_norm_sq = float(np.dot(edge, edge))
        on_boundary |= (
            (np.abs(cross) <= 1.0e-9)
            & (dot >= -1.0e-9)
            & (dot <= edge_norm_sq + 1.0e-9)
        )
        crosses = (left[1] > z) != (right[1] > z)
        denominator = right[1] - left[1]
        x_intersection = np.full_like(x, math.inf)
        if abs(denominator) > 1.0e-15:
            x_intersection = (right[0] - left[0]) * (z - left[1]) / denominator + left[
                0
            ]
        inside ^= crosses & (x < x_intersection)
    return inside | on_boundary


def _region_polygon(raw: Any, *, owner: str) -> np.ndarray:
    try:
        polygon = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApartmentRasterPrefilterError(
            f"{owner} must contain finite XZ points"
        ) from exc
    if polygon.shape == (2, 2):
        low = np.minimum(polygon[0], polygon[1])
        high = np.maximum(polygon[0], polygon[1])
        polygon = np.asarray(
            (
                (low[0], low[1]),
                (high[0], low[1]),
                (high[0], high[1]),
                (low[0], high[1]),
            ),
            dtype=np.float64,
        )
    if (
        polygon.ndim != 2
        or polygon.shape[0] < 3
        or polygon.shape[1] != 2
        or not np.all(np.isfinite(polygon))
    ):
        raise ApartmentRasterPrefilterError(
            f"{owner} must contain at least three finite XZ points"
        )
    return polygon


def normalize_semantic_targets(
    obstacle_map: RuntimeObstacleMap,
    semantic_targets: Mapping[str, Any],
    *,
    maximum_focus_snap_distance_m: Real = 2.0,
) -> tuple[SemanticRasterTarget, ...]:
    """Derive a legal navmesh focus from an explicit focus, region, or surface."""

    grid = _validated_grid(obstacle_map)
    floor_height = _finite_number(
        obstacle_map.floor_height_m, owner="obstacle-map floor height"
    )
    maximum_snap = _positive_number(
        maximum_focus_snap_distance_m,
        owner="maximum semantic focus snap distance",
        allow_zero=True,
    )
    if not isinstance(semantic_targets, Mapping) or not semantic_targets:
        raise ApartmentRasterPrefilterError(
            "semantic_targets must be a nonempty mapping"
        )
    _nav_pixels, nav_points = grid.navmesh_points()
    result = []
    for target_id in sorted(
        semantic_targets, key=lambda item: str(item).encode("utf-8")
    ):
        if not isinstance(target_id, str) or not target_id.strip():
            raise ApartmentRasterPrefilterError(
                "semantic target IDs must be nonempty strings"
            )
        raw = semantic_targets[target_id]
        if isinstance(raw, Mapping):
            role = _slug(str(raw.get("role", _inferred_target_role(target_id))))
            surface_raw = raw.get("surface_position_m", raw.get("position_m"))
            focus_raw = raw.get("focus_position_m")
            region_raw = raw.get("region_polygon_xz_m", raw.get("region_xz_m"))
        else:
            role = _inferred_target_role(target_id)
            surface_raw = raw
            focus_raw = None
            region_raw = None
        if surface_raw is None and region_raw is None and focus_raw is None:
            raise ApartmentRasterPrefilterError(
                f"semantic target {target_id!r} needs a surface point or region"
            )
        if surface_raw is None and region_raw is None:
            surface = _finite_xyz_or_xz(
                focus_raw,
                owner=f"semantic target {target_id!r} focus/surface",
                default_y_m=floor_height,
            )
        elif surface_raw is None:
            polygon = _region_polygon(
                region_raw, owner=f"semantic target {target_id!r} region"
            )
            centroid = np.mean(polygon, axis=0)
            surface = (float(centroid[0]), floor_height, float(centroid[1]))
        else:
            surface = _finite_xyz_or_xz(
                surface_raw,
                owner=f"semantic target {target_id!r} surface",
                default_y_m=floor_height,
            )

        if focus_raw is not None:
            focus = _finite_xyz_or_xz(
                focus_raw,
                owner=f"semantic target {target_id!r} focus",
                default_y_m=floor_height,
            )
            focus_pixel = grid.pixel_for_xz((focus[0], focus[2]))
            if focus_pixel is None or not bool(grid.navmesh[focus_pixel]):
                raise ApartmentRasterPrefilterError(
                    f"semantic target {target_id!r} explicit focus is not navigable"
                )
            snap_distance = math.dist((focus[0], focus[2]), (surface[0], surface[2]))
            derivation = "explicit_navigable_focus_position_m"
        elif region_raw is not None:
            polygon = _region_polygon(
                region_raw, owner=f"semantic target {target_id!r} region"
            )
            members = np.flatnonzero(_points_in_polygon(nav_points, polygon))
            centroid = np.mean(polygon, axis=0)
            if len(members):
                distances = np.linalg.norm(nav_points[members] - centroid, axis=1)
                member = int(members[int(np.argmin(distances))])
                focus_xz = nav_points[member]
                snap_distance = math.dist(
                    tuple(float(item) for item in focus_xz),
                    (surface[0], surface[2]),
                )
                derivation = "nearest_region_navmesh_sample_to_centroid"
            else:
                focus_xz, snap_distance = _nearest_navmesh_focus(grid, centroid)
                focus_xz = np.asarray(focus_xz, dtype=np.float64)
                derivation = "region_centroid_nearest_navmesh_sample"
            focus = (
                float(focus_xz[0]),
                floor_height,
                float(focus_xz[1]),
            )
        else:
            focus_xz, snap_distance = _nearest_navmesh_focus(
                grid, (surface[0], surface[2])
            )
            focus = (
                focus_xz[0],
                floor_height,
                focus_xz[1],
            )
            derivation = "surface_nearest_navmesh_sample"
        if snap_distance > maximum_snap + 1.0e-12:
            raise ApartmentRasterPrefilterError(
                f"semantic target {target_id!r} focus snap exceeds {maximum_snap:g} m"
            )
        result.append(
            SemanticRasterTarget(
                target_id=target_id,
                role=role,
                surface_position_m=surface,
                focus_position_m=focus,
                focus_derivation=derivation,
                focus_snap_distance_m=float(snap_distance),
            )
        )
    return tuple(result)


def _supercover_line(
    start_rc: tuple[int, int], end_rc: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    """Return raster cells touched by a focus/corridor proxy ray."""

    start_row, start_col = start_rc
    end_row, end_col = end_rc
    delta_col = end_col - start_col
    delta_row = end_row - start_row
    count_col = abs(delta_col)
    count_row = abs(delta_row)
    step_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)
    step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
    row, col = start_row, start_col
    traversed = [(row, col)]
    seen = {(row, col)}
    col_steps = 0
    row_steps = 0
    while col_steps < count_col or row_steps < count_row:
        lhs = (1 + 2 * col_steps) * count_row
        rhs = (1 + 2 * row_steps) * count_col
        if lhs == rhs:
            for item in ((row, col + step_col), (row + step_row, col)):
                if item not in seen:
                    traversed.append(item)
                    seen.add(item)
            col += step_col
            row += step_row
            col_steps += 1
            row_steps += 1
        elif lhs < rhs:
            col += step_col
            col_steps += 1
        else:
            row += step_row
            row_steps += 1
        item = (row, col)
        if item not in seen:
            traversed.append(item)
            seen.add(item)
    return tuple(traversed)


def _navmesh_los(
    grid: _Grid,
    start_xz_m: Sequence[float],
    end_xz_m: Sequence[float],
) -> bool:
    start = grid.pixel_for_xz(start_xz_m)
    end = grid.pixel_for_xz(end_xz_m)
    if (
        start is None
        or end is None
        or not bool(grid.navmesh[start])
        or not bool(grid.navmesh[end])
    ):
        return False
    return all(bool(grid.navmesh[pixel]) for pixel in _supercover_line(start, end))


def _yaw_toward(start_xz_m: Sequence[float], target_xz_m: Sequence[float]) -> float:
    delta = np.asarray(target_xz_m, dtype=np.float64) - np.asarray(
        start_xz_m, dtype=np.float64
    )
    if float(np.linalg.norm(delta)) <= 1.0e-12:
        raise ApartmentRasterPrefilterError(
            "camera position coincides with semantic surface"
        )
    return _wrap_yaw_degrees(math.degrees(math.atan2(-delta[0], -delta[1])))


def _point_horizontal_proxy(
    point_xz_m: Sequence[float],
    *,
    camera_xz_m: np.ndarray,
    forward_xz: np.ndarray,
    right_xz: np.ndarray,
    half_fov_degrees: float,
    near_m: float,
    far_m: float,
) -> tuple[bool, float]:
    delta = np.asarray(point_xz_m, dtype=np.float64) - camera_xz_m
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        return False, 0.0
    unit = delta / distance
    azimuth = math.degrees(
        math.atan2(float(np.dot(unit, right_xz)), float(np.dot(unit, forward_xz)))
    )
    in_horizontal_frustum = bool(
        near_m <= distance <= far_m and abs(azimuth) <= half_fov_degrees + 1.0e-9
    )
    lateral_c = float((forward_xz[0] * delta[1] - forward_xz[1] * delta[0]) / distance)
    return in_horizontal_frustum, lateral_c


def _raster_samples(grid: _Grid, *, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    row_step = max(1, int(math.ceil(spacing_m / grid.pixel_z_m)))
    col_step = max(1, int(math.ceil(spacing_m / grid.pixel_x_m)))
    mask = np.zeros_like(grid.navmesh, dtype=np.bool_)
    mask[::row_step, ::col_step] = True
    pixels = np.argwhere((grid.navmesh != 0) & mask).astype(np.int32, copy=False)
    if not len(pixels):
        pixels = np.argwhere(grid.navmesh != 0).astype(np.int32, copy=False)
    x = grid.bounds[0, 0] + (pixels[:, 1].astype(np.float64) + 0.5) * grid.pixel_x_m
    z = grid.bounds[0, 2] + (pixels[:, 0].astype(np.float64) + 0.5) * grid.pixel_z_m
    ids = pixels[:, 0].astype(np.int64) * grid.navmesh.shape[1] + pixels[:, 1]
    return ids, np.stack((x, z), axis=1)


def evaluate_raster_pose_prefilter(
    obstacle_map: RuntimeObstacleMap,
    *,
    camera_position_m: Sequence[float],
    camera_yaw_deg: Real,
    hfov_degrees: Real,
    near_m: Real,
    far_m: Real,
    semantic_targets: Sequence[SemanticRasterTarget],
    corridors: Sequence[StraightCorridor] = (),
    raster_sample_spacing_m: Real = 0.25,
    minimum_lateral_c: Real = 0.2,
) -> RasterPosePrefilter:
    """Evaluate bounded horizontal proxies; no native pixel claim is made."""

    grid = _validated_grid(obstacle_map)
    try:
        camera = np.asarray(camera_position_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApartmentRasterPrefilterError(
            "camera_position_m must contain three finite numbers"
        ) from exc
    if camera.shape != (3,) or not np.all(np.isfinite(camera)):
        raise ApartmentRasterPrefilterError(
            "camera_position_m must contain three finite numbers"
        )
    yaw = _finite_number(camera_yaw_deg, owner="camera yaw")
    hfov = _finite_number(hfov_degrees, owner="camera HFOV")
    near = _positive_number(near_m, owner="camera near_m")
    far = _positive_number(far_m, owner="camera far_m")
    sample_spacing = _positive_number(
        raster_sample_spacing_m, owner="raster sample spacing"
    )
    lateral_threshold = _positive_number(
        minimum_lateral_c, owner="minimum_lateral_c", allow_zero=True
    )
    if not 0.0 < hfov < 180.0 or far <= near:
        raise ApartmentRasterPrefilterError("camera horizontal calibration is invalid")
    if lateral_threshold > 1.0:
        raise ApartmentRasterPrefilterError("minimum_lateral_c must not exceed 1")
    camera_xz = camera[[0, 2]]
    camera_pixel = grid.pixel_for_xz(camera_xz)
    if camera_pixel is None or not bool(grid.navmesh[camera_pixel]):
        raise ApartmentRasterPrefilterError(
            "camera XZ must lie on the retained navigable raster"
        )
    basis = habitat_basis_from_yaw_degrees(yaw)
    forward = np.asarray(basis.forward_xz, dtype=np.float64)
    right = np.asarray(basis.right_xz, dtype=np.float64)

    sample_ids, sample_xz = _raster_samples(grid, spacing_m=sample_spacing)
    deltas = sample_xz - camera_xz
    distances = np.linalg.norm(deltas, axis=1)
    safe = distances > 1.0e-12
    forward_projection = np.full(len(distances), -1.0, dtype=np.float64)
    forward_projection[safe] = deltas[safe] @ forward / distances[safe]
    in_frustum = (
        (distances >= near)
        & (distances <= far)
        & (forward_projection >= math.cos(math.radians(hfov * 0.5)) - 1.0e-12)
    )
    signature = frozenset(int(item) for item in sample_ids[in_frustum])

    focus_checks = []
    for target in sorted(
        semantic_targets, key=lambda item: item.target_id.encode("utf-8")
    ):
        in_horizontal, _lateral = _point_horizontal_proxy(
            target.focus_xz_m,
            camera_xz_m=camera_xz,
            forward_xz=forward,
            right_xz=right,
            half_fov_degrees=hfov * 0.5,
            near_m=near,
            far_m=far,
        )
        nav_los = _navmesh_los(grid, camera_xz, target.focus_xz_m)
        focus_checks.append(
            SemanticFocusRasterCheck(
                target_id=target.target_id,
                focus_in_horizontal_frustum=in_horizontal,
                focus_navmesh_los=nav_los,
                prefilter_pass=bool(in_horizontal and nav_los),
            )
        )

    corridor_checks = []
    for corridor in sorted(
        corridors, key=lambda item: item.corridor_id.encode("utf-8")
    ):
        start = np.asarray(corridor.start_xz_m, dtype=np.float64)
        end = np.asarray(corridor.end_xz_m, dtype=np.float64)
        points = (start, (start + end) * 0.5, end)
        horizontal_values = []
        los_values = []
        lateral_values = []
        for point in points:
            in_horizontal, lateral_c = _point_horizontal_proxy(
                point,
                camera_xz_m=camera_xz,
                forward_xz=forward,
                right_xz=right,
                half_fov_degrees=hfov * 0.5,
                near_m=near,
                far_m=far,
            )
            horizontal_values.append(in_horizontal)
            los_values.append(_navmesh_los(grid, camera_xz, point))
            lateral_values.append(lateral_c)
        point_pass = tuple(
            horizontal and los
            for horizontal, los in zip(horizontal_values, los_values, strict=True)
        )
        minimum_abs = min(abs(value) for value in lateral_values)
        corridor_checks.append(
            CorridorRasterCheck(
                corridor_id=corridor.corridor_id,
                point_in_horizontal_frustum=tuple(horizontal_values),
                point_navmesh_los=tuple(los_values),
                point_prefilter_pass=point_pass,
                lateral_c=tuple(lateral_values),
                minimum_abs_lateral_c=minimum_abs,
                prefilter_pass=bool(
                    all(point_pass) and minimum_abs + 1.0e-12 >= lateral_threshold
                ),
                direction_ids=corridor.direction_ids,
            )
        )
    return RasterPosePrefilter(
        frustum_sample_signature=signature,
        frustum_sample_count=len(signature),
        total_raster_sample_count=len(sample_ids),
        frustum_sample_fraction=len(signature) / len(sample_ids),
        semantic_focus_checks=tuple(focus_checks),
        corridor_checks=tuple(corridor_checks),
        grid_los_query_count=len(focus_checks) + 3 * len(corridor_checks),
    )


def _canonical_endpoints(
    start_xz_m: Sequence[float], end_xz_m: Sequence[float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    start = _finite_xz(start_xz_m, owner="corridor start")
    end = _finite_xz(end_xz_m, owner="corridor end")
    if start == end:
        raise ApartmentRasterPrefilterError("corridor endpoints must differ")
    return (start, end) if start < end else (end, start)


def normalize_corridors(
    corridor_segments: Sequence[Mapping[str, Any]],
    *,
    coordinate_space: str = "apartment_ue_cm",
) -> tuple[StraightCorridor, ...]:
    """Normalize and exactly deduplicate retained undirected geometry."""

    if isinstance(corridor_segments, (str, bytes)) or not isinstance(
        corridor_segments, Sequence
    ):
        raise ApartmentRasterPrefilterError("corridor_segments must be a sequence")
    if coordinate_space not in ("apartment_ue_cm", "habitat_xz_m"):
        raise ApartmentRasterPrefilterError(
            "corridor coordinate_space must be apartment_ue_cm or habitat_xz_m"
        )
    grouped: dict[tuple[tuple[float, float], tuple[float, float]], list[str]] = {}
    for index, item in enumerate(corridor_segments):
        if not isinstance(item, Mapping) or "start" not in item or "end" not in item:
            raise ApartmentRasterPrefilterError(
                f"corridor_segments[{index}] must contain start/end"
            )
        start = _finite_xz(item["start"], owner=f"corridor_segments[{index}].start")
        end = _finite_xz(item["end"], owner=f"corridor_segments[{index}].end")
        if coordinate_space == "apartment_ue_cm":
            start = (start[0] / 100.0, start[1] / 100.0)
            end = (end[0] / 100.0, end[1] / 100.0)
        key = _canonical_endpoints(start, end)
        explicit_id = item.get("corridor_id")
        if explicit_id is not None and (
            not isinstance(explicit_id, str) or not explicit_id.strip()
        ):
            raise ApartmentRasterPrefilterError(
                f"corridor_segments[{index}].corridor_id must be nonempty"
            )
        grouped.setdefault(key, [])
        if explicit_id is not None:
            grouped[key].append(explicit_id)
    result = []
    for ordinal, endpoints in enumerate(sorted(grouped)):
        explicit_ids = grouped[endpoints]
        corridor_id = (
            min(explicit_ids, key=lambda item: item.encode("utf-8"))
            if explicit_ids
            else f"input_corridor_{ordinal:04d}"
        )
        start, end = endpoints
        result.append(
            StraightCorridor(
                corridor_id=corridor_id,
                start_xz_m=start,
                end_xz_m=end,
                length_m=math.dist(start, end),
                origin="retained_input",
                direction_ids=(f"{corridor_id}:forward", f"{corridor_id}:reverse"),
            )
        )
    return tuple(result)


def _source_center_heights(value: Sequence[Real]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ApartmentRasterPrefilterError(
            "corridor_source_center_heights_m must be a nonempty sequence"
        )
    return tuple(
        sorted(
            {
                _finite_number(item, owner="corridor source-center height")
                for item in value
            }
        )
    )


def _stable_subset_pixels(
    pixels: Sequence[tuple[int, int]], *, maximum_count: int
) -> tuple[tuple[int, int], ...]:
    ordered = tuple(sorted(pixels))
    if len(ordered) <= maximum_count:
        return ordered
    selected = [ordered[0]]
    remaining = np.ones(len(ordered), dtype=np.bool_)
    remaining[0] = False
    points = np.asarray(ordered, dtype=np.float64)
    minimum_distance = np.linalg.norm(points - points[0], axis=1)
    while len(selected) < maximum_count:
        scores = np.where(remaining, minimum_distance, -1.0)
        index = int(np.argmax(scores))
        selected.append(ordered[index])
        remaining[index] = False
        minimum_distance = np.minimum(
            minimum_distance, np.linalg.norm(points - points[index], axis=1)
        )
    return tuple(sorted(selected))


def _exact_corridor_line_pixels(
    grid: _Grid,
    start_xz_m: tuple[float, float],
    end_xz_m: tuple[float, float],
) -> tuple[tuple[int, int], ...]:
    """Conservatively traverse every cell touched by a generated chord."""

    start = grid.pixel_for_xz(start_xz_m)
    end = grid.pixel_for_xz(end_xz_m)
    if start is None or end is None:
        return ()
    # Generated endpoints are raster-cell centres.  The exact integer
    # supercover includes both side cells when the segment crosses a corner,
    # unlike sub-pixel linspace sampling which can step over a thin blocker.
    return _supercover_line(start, end)


def _corridor_orientation_degrees(corridor: StraightCorridor) -> float:
    dx = corridor.end_xz_m[0] - corridor.start_xz_m[0]
    dz = corridor.end_xz_m[1] - corridor.start_xz_m[1]
    return math.degrees(math.atan2(dz, dx)) % 180.0


def _select_spatial_corridors(
    corridors: Sequence[StraightCorridor],
    *,
    maximum_count: int,
    room_diagonal_m: float,
) -> tuple[StraightCorridor, ...]:
    ordered = tuple(
        sorted(corridors, key=lambda item: (item.start_xz_m, item.end_xz_m))
    )
    if len(ordered) <= maximum_count:
        return ordered
    midpoints = np.asarray(
        [
            (
                (item.start_xz_m[0] + item.end_xz_m[0]) * 0.5,
                (item.start_xz_m[1] + item.end_xz_m[1]) * 0.5,
            )
            for item in ordered
        ],
        dtype=np.float64,
    )
    orientations = np.asarray(
        [_corridor_orientation_degrees(item) for item in ordered], dtype=np.float64
    )
    selected_indices = [0]
    remaining = np.ones(len(ordered), dtype=np.bool_)
    remaining[0] = False
    minimum_difference = np.full(len(ordered), math.inf, dtype=np.float64)
    while len(selected_indices) < maximum_count:
        retained = selected_indices[-1]
        position = np.minimum(
            1.0,
            np.linalg.norm(midpoints - midpoints[retained], axis=1) / room_diagonal_m,
        )
        raw_angle = np.abs(orientations - orientations[retained])
        angle = np.minimum(raw_angle, 180.0 - raw_angle) / 90.0
        minimum_difference = np.minimum(
            minimum_difference, 0.7 * position + 0.3 * angle
        )
        scores = np.where(remaining, minimum_difference, -1.0)
        index = int(np.argmax(scores))
        selected_indices.append(index)
        remaining[index] = False
    return tuple(
        sorted(
            (ordered[index] for index in selected_indices),
            key=lambda item: (item.start_xz_m, item.end_xz_m),
        )
    )


def generate_straight_corridors(
    obstacle_map: RuntimeObstacleMap,
    *,
    corridor_source_center_heights_m: Sequence[Real],
    feasible_mask: np.ndarray | None = None,
    minimum_length_m: Real = 3.0,
    maximum_length_m: Real = 3.8,
    minimum_clearance_m: Real = 0.2,
    endpoint_spacing_m: Real = 0.5,
    maximum_endpoint_count: int = 96,
    maximum_corridor_count: int = 64,
) -> tuple[StraightCorridor, ...]:
    """Generate chords passing feasibility at every explicit source height."""

    grid = _validated_grid(obstacle_map)
    heights = _source_center_heights(corridor_source_center_heights_m)
    minimum_length = _positive_number(minimum_length_m, owner="minimum corridor length")
    maximum_length = _positive_number(maximum_length_m, owner="maximum corridor length")
    clearance = _positive_number(
        minimum_clearance_m, owner="minimum corridor clearance", allow_zero=True
    )
    endpoint_spacing = _positive_number(endpoint_spacing_m, owner="endpoint spacing")
    maximum_endpoints = _positive_int(
        maximum_endpoint_count, owner="maximum_endpoint_count"
    )
    maximum_corridors = _positive_int(
        maximum_corridor_count, owner="maximum_corridor_count"
    )
    if maximum_length < minimum_length:
        raise ApartmentRasterPrefilterError(
            "maximum corridor length must not be below minimum length"
        )
    masks = []
    compiler = RoomFeasibilityCompiler(obstacle_map)
    try:
        for height in heights:
            masks.append(
                compiler.compile(
                    source_center_height_m=height,
                    minimum_navmesh_clearance_m=clearance,
                    minimum_rigid_clearance_m=clearance,
                    sample_spacing_m=endpoint_spacing,
                ).feasible_mask
            )
    except RoomFeasibilityError as exc:
        raise ApartmentRasterPrefilterError(
            f"corridor feasibility compilation failed: {exc}"
        ) from exc
    intersection = np.logical_and.reduce(masks)
    if feasible_mask is not None:
        explicit = np.asarray(feasible_mask, dtype=np.bool_)
        if explicit.shape != grid.navmesh.shape or np.any(
            explicit & (grid.navmesh == 0)
        ):
            raise ApartmentRasterPrefilterError(
                "corridor feasible_mask must match and be a navmesh subset"
            )
        intersection &= explicit

    row_step = max(1, int(math.ceil(endpoint_spacing / grid.pixel_z_m)))
    col_step = max(1, int(math.ceil(endpoint_spacing / grid.pixel_x_m)))
    sampled = np.zeros_like(intersection, dtype=np.bool_)
    sampled[::row_step, ::col_step] = True
    endpoint_pixels = [
        (int(row), int(col)) for row, col in np.argwhere(intersection & sampled)
    ]
    if len(endpoint_pixels) < 2:
        endpoint_pixels = [
            (int(row), int(col)) for row, col in np.argwhere(intersection)
        ]
    endpoints = _stable_subset_pixels(endpoint_pixels, maximum_count=maximum_endpoints)
    if len(endpoints) < 2:
        return ()
    endpoint_xz = np.asarray(
        [grid.pixel_center_xz(*pixel) for pixel in endpoints], dtype=np.float64
    )
    left_indices, right_indices = np.triu_indices(len(endpoints), k=1)
    pair_lengths = np.linalg.norm(
        endpoint_xz[right_indices] - endpoint_xz[left_indices], axis=1
    )
    retained_pairs = np.flatnonzero(
        (pair_lengths >= minimum_length - 1.0e-12)
        & (pair_lengths <= maximum_length + 1.0e-12)
    )
    raw = []
    for pair_index in retained_pairs:
        left_index = int(left_indices[pair_index])
        right_index = int(right_indices[pair_index])
        start_xz, end_xz = _canonical_endpoints(
            endpoint_xz[left_index], endpoint_xz[right_index]
        )
        line_pixels = _exact_corridor_line_pixels(grid, start_xz, end_xz)
        if not all(
            0 <= row < intersection.shape[0]
            and 0 <= col < intersection.shape[1]
            and bool(intersection[row, col])
            for row, col in line_pixels
        ):
            continue
        corridor_id = f"generated_corridor_{len(raw):04d}"
        raw.append(
            StraightCorridor(
                corridor_id=corridor_id,
                start_xz_m=start_xz,
                end_xz_m=end_xz,
                length_m=float(pair_lengths[pair_index]),
                origin="generated_multi_height_raster_chord",
                direction_ids=(
                    f"{corridor_id}:forward",
                    f"{corridor_id}:reverse",
                ),
                source_center_heights_m=heights,
                sample_pixels_rc=line_pixels,
            )
        )
    room_diagonal = math.hypot(
        float(grid.bounds[1, 0] - grid.bounds[0, 0]),
        float(grid.bounds[1, 2] - grid.bounds[0, 2]),
    )
    selected = _select_spatial_corridors(
        raw, maximum_count=maximum_corridors, room_diagonal_m=room_diagonal
    )
    result = []
    for index, corridor in enumerate(selected):
        corridor_id = f"generated_corridor_{index:04d}"
        result.append(
            replace(
                corridor,
                corridor_id=corridor_id,
                direction_ids=(
                    f"{corridor_id}:forward",
                    f"{corridor_id}:reverse",
                ),
            )
        )
    return tuple(result)


def _merge_corridors(
    retained: Sequence[StraightCorridor], generated: Sequence[StraightCorridor]
) -> tuple[StraightCorridor, ...]:
    by_geometry: dict[
        tuple[tuple[float, float], tuple[float, float]], StraightCorridor
    ] = {}
    for corridor in (*generated, *retained):
        by_geometry[_canonical_endpoints(corridor.start_xz_m, corridor.end_xz_m)] = (
            corridor
        )
    result = []
    for index, (_geometry, corridor) in enumerate(sorted(by_geometry.items())):
        prefix = "retained" if corridor.origin == "retained_input" else "generated"
        corridor_id = f"{prefix}_corridor_{index:04d}"
        result.append(
            replace(
                corridor,
                corridor_id=corridor_id,
                direction_ids=(
                    f"{corridor_id}:forward",
                    f"{corridor_id}:reverse",
                ),
            )
        )
    return tuple(result)


def _height_variants(
    grid: _Grid,
    base_position_m: tuple[float, float, float],
    *,
    floor_height_m: float,
) -> tuple[tuple[str, float], ...]:
    floor = _finite_number(floor_height_m, owner="obstacle-map floor height")
    ceiling = float(grid.bounds[1, 1])
    if not float(grid.bounds[0, 1]) <= floor < ceiling:
        raise ApartmentRasterPrefilterError(
            "obstacle-map floor height lies outside retained Y bounds"
        )
    base_height = base_position_m[1] - floor
    if base_height <= 0.0:
        raise ApartmentRasterPrefilterError(
            "base M1 camera must lie above the room floor"
        )
    upper = math.nextafter(ceiling, floor)
    low = min(max(floor + base_height * 0.72, math.nextafter(floor, ceiling)), upper)
    high = min(max(floor + base_height * 1.25, math.nextafter(floor, ceiling)), upper)
    values = (("base", base_position_m[1]), ("low", low), ("high", high))
    by_height: dict[float, tuple[str, float]] = {}
    for role, height in values:
        by_height.setdefault(height, (role, height))
    order = {"low": 0, "base": 1, "high": 2}
    return tuple(sorted(by_height.values(), key=lambda item: order[item[0]]))


def _pose_difference_proxy(
    *,
    position_xz_m: tuple[float, float],
    yaw_deg: float,
    legacy_position_xz_m: tuple[float, float],
    legacy_yaw_deg: float,
    room_diagonal_m: float,
) -> tuple[float, float]:
    """Score only XZ displacement and yaw; this is not an image metric."""

    yaw_difference = circular_yaw_distance_degrees(yaw_deg, legacy_yaw_deg)
    position_difference = min(
        1.0, math.dist(position_xz_m, legacy_position_xz_m) / room_diagonal_m
    )
    return 0.65 * yaw_difference / 180.0 + 0.35 * position_difference, yaw_difference


def _target_camera_positions(
    positions: Sequence[tuple[float, float]],
    *,
    target: SemanticRasterTarget,
    legacy_position_xz_m: tuple[float, float],
    legacy_yaw_deg: float,
    room_diagonal_m: float,
    near_m: float,
    far_m: float,
    minimum_reverse_yaw_difference_deg: float,
    minimum_reverse_pose_difference_proxy: float,
    maximum_count: int,
    grid: _Grid,
) -> tuple[tuple[tuple[float, float], float, float], ...]:
    ideal_distance = min(max(near_m * 2.0, room_diagonal_m * 0.3), far_m * 0.75)
    eligible = []
    for position in sorted(set(positions)):
        focus_distance = math.dist(position, target.focus_xz_m)
        if not near_m < focus_distance <= far_m:
            continue
        if not _navmesh_los(grid, position, target.focus_xz_m):
            continue
        yaw = _yaw_toward(position, target.surface_xz_m)
        pose_proxy, yaw_difference = _pose_difference_proxy(
            position_xz_m=position,
            yaw_deg=yaw,
            legacy_position_xz_m=legacy_position_xz_m,
            legacy_yaw_deg=legacy_yaw_deg,
            room_diagonal_m=room_diagonal_m,
        )
        if target.role in _REVERSE_ROLES and (
            yaw_difference + 1.0e-12 < minimum_reverse_yaw_difference_deg
            or pose_proxy + 1.0e-12 < minimum_reverse_pose_difference_proxy
        ):
            continue
        eligible.append((position, yaw, pose_proxy, focus_distance))
    if not eligible:
        return ()
    first = min(
        eligible,
        key=lambda item: (
            -item[2] if target.role in _REVERSE_ROLES else 0.0,
            abs(item[3] - ideal_distance),
            item[0],
        ),
    )
    selected = [first]
    remaining = [item for item in eligible if item is not first]
    while remaining and len(selected) < maximum_count:
        best = min(
            remaining,
            key=lambda item: (
                -min(math.dist(item[0], retained[0]) for retained in selected),
                -item[2] if target.role in _REVERSE_ROLES else 0.0,
                abs(item[3] - ideal_distance),
                item[0],
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return tuple((item[0], item[1], item[2]) for item in selected)


def _exact_deduplicate(candidates: Sequence[_Candidate]) -> tuple[_Candidate, ...]:
    retained: dict[tuple[Any, ...], _Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.stable_key):
        key = (
            candidate.position_m,
            candidate.yaw_deg,
            candidate.prefilter.frustum_sample_signature,
            candidate.prefilter.passed_corridor_ids,
        )
        existing = retained.get(key)
        if existing is None:
            retained[key] = candidate
            continue
        retained[key] = replace(
            existing,
            roles=tuple(sorted(set(existing.roles) | set(candidate.roles))),
            target_ids=tuple(
                sorted(set(existing.target_ids) | set(candidate.target_ids))
            ),
            pose_difference_proxy=max(
                existing.pose_difference_proxy, candidate.pose_difference_proxy
            ),
            is_legacy=existing.is_legacy or candidate.is_legacy,
        )
    return tuple(sorted(retained.values(), key=lambda item: item.stable_key))


def _jaccard_distance(left: frozenset[Any], right: frozenset[Any]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def _pairwise_differences(
    left: _Candidate,
    right: _Candidate,
    *,
    room_diagonal_m: float,
    height_span_m: float,
) -> dict[str, float]:
    position_m = math.dist(
        (left.position_m[0], left.position_m[2]),
        (right.position_m[0], right.position_m[2]),
    )
    yaw_degrees = circular_yaw_distance_degrees(left.yaw_deg, right.yaw_deg)
    height_m = abs(left.position_m[1] - right.position_m[1])
    raster_signature = _jaccard_distance(
        left.prefilter.frustum_sample_signature,
        right.prefilter.frustum_sample_signature,
    )
    corridor_signature = _jaccard_distance(
        frozenset(left.prefilter.passed_corridor_ids),
        frozenset(right.prefilter.passed_corridor_ids),
    )
    weighted_total = (
        0.25 * min(1.0, position_m / room_diagonal_m)
        + 0.20 * yaw_degrees / 180.0
        + 0.10 * min(1.0, height_m / height_span_m)
        + 0.30 * raster_signature
        + 0.15 * corridor_signature
    )
    return {
        "position_m": position_m,
        "circular_yaw_degrees": yaw_degrees,
        "height_m": height_m,
        "raster_signature_jaccard": raster_signature,
        "corridor_jaccard": corridor_signature,
        "weighted_total": weighted_total,
    }


def _passes_pairwise_thresholds(
    differences: Mapping[str, float], thresholds: Mapping[str, float]
) -> bool:
    return bool(
        differences["position_m"] + 1.0e-12 >= thresholds["position_m"]
        and differences["circular_yaw_degrees"] + 1.0e-12
        >= thresholds["circular_yaw_degrees"]
        and differences["raster_signature_jaccard"] + 1.0e-12
        >= thresholds["raster_signature_jaccard"]
        and differences["corridor_jaccard"] + 1.0e-12 >= thresholds["corridor_jaccard"]
    )


def _passes_candidate_quality(
    candidate: _Candidate,
    *,
    minimum_raster_sample_fraction: float,
    minimum_passed_corridor_count: int,
) -> bool:
    return bool(
        candidate.prefilter.frustum_sample_fraction + 1.0e-12
        >= minimum_raster_sample_fraction
        and len(candidate.prefilter.passed_corridor_ids)
        >= minimum_passed_corridor_count
    )


def _select_candidates(
    candidates: Sequence[_Candidate],
    *,
    count: int,
    required_roles: frozenset[str],
    thresholds: Mapping[str, float],
    room_diagonal_m: float,
    height_span_m: float,
) -> tuple[tuple[_Candidate, Mapping[str, float] | None], ...]:
    legacy = [item for item in candidates if item.is_legacy]
    if len(legacy) != 1:
        raise ApartmentRasterPrefilterError(
            "exactly one legacy view0 candidate must survive exact deduplication"
        )
    selected = [legacy[0]]
    evidence: list[Mapping[str, float] | None] = [None]
    remaining = [item for item in candidates if item is not legacy[0]]
    uncovered = set(required_roles) - set(legacy[0].roles)
    while len(selected) < count:
        scored = []
        for candidate in remaining:
            differences = [
                _pairwise_differences(
                    candidate,
                    retained,
                    room_diagonal_m=room_diagonal_m,
                    height_span_m=height_span_m,
                )
                for retained in selected
            ]
            if not all(
                _passes_pairwise_thresholds(item, thresholds) for item in differences
            ):
                continue
            nearest = min(differences, key=lambda item: item["weighted_total"])
            role_gain = len(uncovered & set(candidate.roles))
            scored.append((candidate, nearest, role_gain))
        if not scored:
            break
        candidate, nearest, _role_gain = min(
            scored,
            key=lambda item: (
                -int(item[2] > 0),
                -item[2],
                -item[1]["weighted_total"],
                -item[0].pose_difference_proxy,
                item[0].stable_key,
            ),
        )
        selected.append(candidate)
        evidence.append(nearest)
        remaining.remove(candidate)
        uncovered -= set(candidate.roles)
    if len(selected) < count:
        raise ApartmentRasterPrefilterError(
            f"only {len(selected)} candidates satisfy all pairwise difference "
            f"thresholds; {count} requested"
        )
    if uncovered:
        raise ApartmentRasterPrefilterError(
            "selected candidates do not cover required roles: "
            + ", ".join(sorted(uncovered))
        )
    return tuple(zip(selected, evidence, strict=True))


def _viewpoint_stem(candidate: _Candidate) -> str:
    semantic_roles = [
        role
        for role in candidate.roles
        if role not in ("legacy", "low", "base", "high")
    ]
    stem = semantic_roles[0] if semantic_roles else candidate.height_role
    if candidate.height_role != "base":
        stem = f"{stem}_{candidate.height_role}"
    return f"apartment_{_slug(stem)}"


def prefilter_apartment_viewpoints(
    obstacle_map: RuntimeObstacleMap,
    *,
    base_m1_camera: Mapping[str, Any],
    semantic_targets: Mapping[str, Any],
    corridor_segments: Sequence[Mapping[str, Any]] = (),
    corridor_coordinate_space: str = "apartment_ue_cm",
    generate_navmesh_corridors: bool = False,
    corridor_source_center_heights_m: Sequence[Real] | None = None,
    corridor_feasible_mask: np.ndarray | None = None,
    selection_count: int = 6,
    candidate_spacing_m: Real = 0.75,
    raster_sample_spacing_m: Real = 0.25,
    minimum_camera_navmesh_clearance_m: Real = 0.02,
    minimum_camera_rigid_clearance_m: Real = 0.0,
    maximum_positions_per_target: int = 10,
    minimum_raster_sample_fraction: Real = 0.0,
    minimum_passed_corridor_count: int = 0,
    minimum_lateral_c: Real = 0.2,
    maximum_focus_snap_distance_m: Real = 2.0,
    minimum_reverse_yaw_difference_deg: Real = 30.0,
    minimum_reverse_pose_difference_proxy: Real = 0.25,
    minimum_pairwise_position_m: Real = 0.75,
    minimum_pairwise_yaw_degrees: Real = 15.0,
    minimum_pairwise_raster_signature_jaccard: Real = 0.02,
    minimum_pairwise_corridor_jaccard: Real = 0.0,
    required_roles: Iterable[str] | None = None,
    generated_corridor_minimum_length_m: Real = 3.0,
    generated_corridor_maximum_length_m: Real = 3.8,
    generated_corridor_minimum_clearance_m: Real = 0.2,
    generated_corridor_endpoint_spacing_m: Real = 0.5,
    generated_corridor_endpoint_limit: int = 96,
    generated_corridor_limit: int = 64,
) -> ApartmentRasterPrefilterResult:
    """Select 4--8 candidates using deterministic horizontal raster proxies."""

    grid = _validated_grid(obstacle_map)
    camera = _extract_camera_contract(base_m1_camera)
    targets = normalize_semantic_targets(
        obstacle_map,
        semantic_targets,
        maximum_focus_snap_distance_m=maximum_focus_snap_distance_m,
    )
    if isinstance(selection_count, bool) or not isinstance(selection_count, int):
        raise ApartmentRasterPrefilterError("selection_count must be an integer")
    if not 4 <= selection_count <= 8:
        raise ApartmentRasterPrefilterError(
            "selection_count must lie within the 4--8 prefilter range"
        )
    candidate_spacing = _positive_number(candidate_spacing_m, owner="candidate spacing")
    raster_spacing = _positive_number(
        raster_sample_spacing_m, owner="raster sample spacing"
    )
    camera_nav_clearance = _positive_number(
        minimum_camera_navmesh_clearance_m,
        owner="minimum camera navmesh clearance",
        allow_zero=True,
    )
    camera_rigid_clearance = _positive_number(
        minimum_camera_rigid_clearance_m,
        owner="minimum camera rigid clearance",
        allow_zero=True,
    )
    maximum_positions = _positive_int(
        maximum_positions_per_target, owner="maximum_positions_per_target"
    )
    minimum_sample_fraction = _positive_number(
        minimum_raster_sample_fraction,
        owner="minimum raster sample fraction",
        allow_zero=True,
    )
    minimum_corridor_count = _nonnegative_int(
        minimum_passed_corridor_count,
        owner="minimum_passed_corridor_count",
    )
    reverse_yaw_threshold = _positive_number(
        minimum_reverse_yaw_difference_deg,
        owner="minimum reverse yaw difference",
        allow_zero=True,
    )
    reverse_pose_proxy_threshold = _positive_number(
        minimum_reverse_pose_difference_proxy,
        owner="minimum reverse pose-difference proxy",
        allow_zero=True,
    )
    if minimum_sample_fraction > 1.0 or reverse_pose_proxy_threshold > 1.0:
        raise ApartmentRasterPrefilterError("fraction thresholds must not exceed 1")
    if reverse_yaw_threshold > 180.0:
        raise ApartmentRasterPrefilterError(
            "minimum reverse yaw difference must not exceed 180"
        )
    thresholds = {
        "position_m": _positive_number(
            minimum_pairwise_position_m,
            owner="minimum pairwise position difference",
            allow_zero=True,
        ),
        "circular_yaw_degrees": _positive_number(
            minimum_pairwise_yaw_degrees,
            owner="minimum pairwise yaw difference",
            allow_zero=True,
        ),
        "raster_signature_jaccard": _positive_number(
            minimum_pairwise_raster_signature_jaccard,
            owner="minimum pairwise raster signature difference",
            allow_zero=True,
        ),
        "corridor_jaccard": _positive_number(
            minimum_pairwise_corridor_jaccard,
            owner="minimum pairwise corridor difference",
            allow_zero=True,
        ),
    }
    if (
        thresholds["circular_yaw_degrees"] > 180.0
        or thresholds["raster_signature_jaccard"] > 1.0
        or thresholds["corridor_jaccard"] > 1.0
    ):
        raise ApartmentRasterPrefilterError(
            "pairwise yaw/Jaccard thresholds exceed their valid range"
        )

    retained_corridors = normalize_corridors(
        corridor_segments, coordinate_space=corridor_coordinate_space
    )
    generated_corridors: tuple[StraightCorridor, ...] = ()
    if generate_navmesh_corridors:
        if corridor_source_center_heights_m is None:
            raise ApartmentRasterPrefilterError(
                "corridor_source_center_heights_m is required for generation"
            )
        generated_corridors = generate_straight_corridors(
            obstacle_map,
            corridor_source_center_heights_m=corridor_source_center_heights_m,
            feasible_mask=corridor_feasible_mask,
            minimum_length_m=generated_corridor_minimum_length_m,
            maximum_length_m=generated_corridor_maximum_length_m,
            minimum_clearance_m=generated_corridor_minimum_clearance_m,
            endpoint_spacing_m=generated_corridor_endpoint_spacing_m,
            maximum_endpoint_count=generated_corridor_endpoint_limit,
            maximum_corridor_count=generated_corridor_limit,
        )
    elif (
        corridor_feasible_mask is not None
        or corridor_source_center_heights_m is not None
    ):
        raise ApartmentRasterPrefilterError(
            "corridor generation inputs require generate_navmesh_corridors=True"
        )
    corridors = _merge_corridors(retained_corridors, generated_corridors)

    height_variants = _height_variants(
        grid,
        camera.position_m,
        floor_height_m=obstacle_map.floor_height_m,
    )
    compiler = RoomFeasibilityCompiler(obstacle_map)
    camera_regions = {}
    try:
        for height_role, height in height_variants:
            camera_regions[height_role] = compiler.compile(
                source_center_height_m=height,
                minimum_navmesh_clearance_m=camera_nav_clearance,
                minimum_rigid_clearance_m=camera_rigid_clearance,
                sample_spacing_m=candidate_spacing,
            )
    except RoomFeasibilityError as exc:
        raise ApartmentRasterPrefilterError(
            f"camera feasibility compilation failed: {exc}"
        ) from exc
    base_region = camera_regions.get("base")
    if base_region is None:
        raise ApartmentRasterPrefilterError(
            "base camera height collapses with another derived height"
        )
    legacy_pixel = grid.pixel_for_xz((camera.position_m[0], camera.position_m[2]))
    if legacy_pixel is None or not bool(base_region.feasible_mask[legacy_pixel]):
        raise ApartmentRasterPrefilterError(
            "legacy view0 camera fails requested navmesh/rigid clearance"
        )

    legacy_prefilter = evaluate_raster_pose_prefilter(
        obstacle_map,
        camera_position_m=camera.position_m,
        camera_yaw_deg=camera.yaw_deg,
        hfov_degrees=camera.hfov_degrees,
        near_m=camera.near_m,
        far_m=camera.far_m,
        semantic_targets=targets,
        corridors=corridors,
        raster_sample_spacing_m=raster_spacing,
        minimum_lateral_c=minimum_lateral_c,
    )
    candidates = [
        _Candidate(
            roles=("base", "legacy"),
            target_ids=(),
            height_role="base",
            position_m=camera.position_m,
            yaw_deg=camera.yaw_deg,
            pose_difference_proxy=0.0,
            prefilter=legacy_prefilter,
            is_legacy=True,
        )
    ]
    room_diagonal = math.hypot(
        float(grid.bounds[1, 0] - grid.bounds[0, 0]),
        float(grid.bounds[1, 2] - grid.bounds[0, 2]),
    )
    legacy_xz = (camera.position_m[0], camera.position_m[2])
    prefilter_cache: dict[tuple[tuple[float, float], float], RasterPosePrefilter] = {}
    for height_role, height in height_variants:
        positions = tuple(
            (float(point[0]), float(point[2]))
            for point in camera_regions[height_role].sample_points_m()
        )
        for target in targets:
            placements = _target_camera_positions(
                positions,
                target=target,
                legacy_position_xz_m=legacy_xz,
                legacy_yaw_deg=camera.yaw_deg,
                room_diagonal_m=room_diagonal,
                near_m=camera.near_m,
                far_m=camera.far_m,
                minimum_reverse_yaw_difference_deg=reverse_yaw_threshold,
                minimum_reverse_pose_difference_proxy=reverse_pose_proxy_threshold,
                maximum_count=maximum_positions,
                grid=grid,
            )
            for position_xz, yaw, pose_proxy in placements:
                cache_key = (position_xz, yaw)
                pose_prefilter = prefilter_cache.get(cache_key)
                if pose_prefilter is None:
                    pose_prefilter = evaluate_raster_pose_prefilter(
                        obstacle_map,
                        camera_position_m=(position_xz[0], height, position_xz[1]),
                        camera_yaw_deg=yaw,
                        hfov_degrees=camera.hfov_degrees,
                        near_m=camera.near_m,
                        far_m=camera.far_m,
                        semantic_targets=targets,
                        corridors=corridors,
                        raster_sample_spacing_m=raster_spacing,
                        minimum_lateral_c=minimum_lateral_c,
                    )
                    prefilter_cache[cache_key] = pose_prefilter
                if target.target_id not in pose_prefilter.passed_semantic_focus_ids:
                    continue
                candidates.append(
                    _Candidate(
                        roles=tuple(sorted((height_role, target.role))),
                        target_ids=(target.target_id,),
                        height_role=height_role,
                        position_m=(position_xz[0], height, position_xz[1]),
                        yaw_deg=yaw,
                        pose_difference_proxy=pose_proxy,
                        prefilter=pose_prefilter,
                    )
                )

    raw_candidate_count = len(candidates)
    legacy_candidate = candidates[0]
    if not _passes_candidate_quality(
        legacy_candidate,
        minimum_raster_sample_fraction=minimum_sample_fraction,
        minimum_passed_corridor_count=minimum_corridor_count,
    ):
        raise ApartmentRasterPrefilterError(
            "legacy view0 fails candidate quality thresholds: "
            f"raster_fraction={legacy_prefilter.frustum_sample_fraction:.6f} "
            f"passed_corridors={len(legacy_prefilter.passed_corridor_ids)}; "
            f"required_fraction={minimum_sample_fraction:.6f} "
            f"required_corridors={minimum_corridor_count}"
        )
    quality_candidates = tuple(
        candidate
        for candidate in candidates
        if _passes_candidate_quality(
            candidate,
            minimum_raster_sample_fraction=minimum_sample_fraction,
            minimum_passed_corridor_count=minimum_corridor_count,
        )
    )
    deduplicated = _exact_deduplicate(quality_candidates)
    if len(deduplicated) < 4:
        raise ApartmentRasterPrefilterError(
            "fewer than four exact raster candidates survive generation"
        )
    if required_roles is None:
        required = {target.role for target in targets}
        required.update(
            role for role, _height in height_variants if role in ("low", "high")
        )
        required_roles_set = frozenset(required)
    else:
        if isinstance(required_roles, (str, bytes)):
            raise ApartmentRasterPrefilterError(
                "required_roles must be an iterable of role names"
            )
        required_roles_set = frozenset(_slug(str(role)) for role in required_roles)
    available_roles = {role for item in deduplicated for role in item.roles}
    missing_roles = required_roles_set - available_roles
    if missing_roles:
        raise ApartmentRasterPrefilterError(
            "generated candidates do not provide required roles: "
            + ", ".join(sorted(missing_roles))
        )
    height_span = max(
        1.0e-9,
        max(height for _role, height in height_variants)
        - min(height for _role, height in height_variants),
    )
    selected = _select_candidates(
        deduplicated,
        count=selection_count,
        required_roles=required_roles_set,
        thresholds=thresholds,
        room_diagonal_m=room_diagonal,
        height_span_m=height_span,
    )

    used_ids: dict[str, int] = {}
    viewpoints = []
    for candidate, nearest in selected:
        if candidate.is_legacy:
            viewpoint_id = "apartment_view0"
        else:
            stem = _viewpoint_stem(candidate)
            used_ids[stem] = used_ids.get(stem, 0) + 1
            viewpoint_id = stem if used_ids[stem] == 1 else f"{stem}_{used_ids[stem]}"
        ue_position = habitat_point_to_apartment_ue_cm(candidate.position_m)
        viewpoints.append(
            PrefilteredApartmentViewpoint(
                viewpoint_id=viewpoint_id,
                view_id="view0",
                roles=candidate.roles,
                target_ids=candidate.target_ids,
                height_role=candidate.height_role,
                camera_position_m=candidate.position_m,
                camera_yaw_deg=candidate.yaw_deg,
                ue_position_cm=tuple(float(item) for item in ue_position),
                ue_yaw_deg=camera_ue_yaw_degrees(candidate.yaw_deg),
                hfov_degrees=camera.hfov_degrees,
                near_m=camera.near_m,
                far_m=camera.far_m,
                pose_difference_proxy_from_legacy=(candidate.pose_difference_proxy),
                frustum_sample_count=candidate.prefilter.frustum_sample_count,
                total_raster_sample_count=(
                    candidate.prefilter.total_raster_sample_count
                ),
                frustum_sample_fraction=(candidate.prefilter.frustum_sample_fraction),
                passed_semantic_focus_ids=(
                    candidate.prefilter.passed_semantic_focus_ids
                ),
                corridor_checks=candidate.prefilter.corridor_checks,
                passed_corridor_ids=candidate.prefilter.passed_corridor_ids,
                nearest_selected_difference=nearest,
                native_pixel_validation_status="pending",
                native_validation_pending=_NATIVE_PENDING_CHECKS,
                _frustum_sample_signature=(
                    candidate.prefilter.frustum_sample_signature
                ),
            )
        )
    _sample_ids, _sample_xz = _raster_samples(grid, spacing_m=raster_spacing)
    return ApartmentRasterPrefilterResult(
        viewpoints=tuple(viewpoints),
        corridors=corridors,
        semantic_targets=targets,
        raw_candidate_count=raw_candidate_count,
        quality_filtered_candidate_count=len(quality_candidates),
        exact_deduplicated_candidate_count=len(deduplicated),
        raster_sample_spacing_m=raster_spacing,
        raster_sample_count=len(_sample_ids),
        minimum_candidate_thresholds={
            "raster_sample_fraction": minimum_sample_fraction,
            "passed_corridor_count": minimum_corridor_count,
        },
        minimum_pairwise_thresholds=thresholds,
    )


__all__ = [
    "ApartmentRasterPrefilterError",
    "ApartmentRasterPrefilterResult",
    "CorridorRasterCheck",
    "PrefilteredApartmentViewpoint",
    "RasterPosePrefilter",
    "SemanticFocusRasterCheck",
    "SemanticRasterTarget",
    "StraightCorridor",
    "camera_ue_yaw_degrees",
    "circular_yaw_distance_degrees",
    "evaluate_raster_pose_prefilter",
    "generate_straight_corridors",
    "normalize_corridors",
    "normalize_semantic_targets",
    "prefilter_apartment_viewpoints",
]
