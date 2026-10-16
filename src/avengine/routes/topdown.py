"""Generic diagnostic Topdown panels for the fixed-room M6.x canary.

The renderer consumes :class:`~avengine.routes.geometry.RuntimeObstacleMap`
directly.  Consequently, the floor map used by placement QA and the map
shown to a reviewer have one authority: the live Habitat navmesh plus every
retained rigid collision OBB.  Source paths and dots represent source-center
points only; they must not be read as articulated-body collision volumes.
"""

from __future__ import annotations

from functools import lru_cache
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avengine.capture.orientation import (
    M51OrientationError,
    habitat_basis_from_yaw_degrees,
)
from avengine.routes.geometry import (
    ELEVATED_OBJECT,
    GROUND_BLOCKER,
    UNKNOWN_OBSTACLE_ROLE,
    WALKABLE_FLOOR_COVERING,
    RuntimeObstacleMap,
)


TOPDOWN_SCHEMA = "avengine_m6x_runtime_obstacle_topdown_v3"


class M6XTopdownError(ValueError):
    """Runtime geometry or source centers cannot be rendered unambiguously."""


_SOURCE_COLORS = (
    (42, 210, 220, 255),
    (250, 120, 70, 255),
    (167, 121, 255, 255),
    (120, 220, 112, 255),
    (255, 196, 66, 255),
    (255, 105, 180, 255),
    (89, 156, 255, 255),
    (232, 232, 232, 255),
)

_OBSTACLE_ROLE_STYLES: Mapping[
    str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
] = {
    GROUND_BLOCKER: ((230, 139, 58, 116), (255, 186, 90, 235)),
    WALKABLE_FLOOR_COVERING: ((44, 155, 137, 88), (76, 222, 194, 235)),
    ELEVATED_OBJECT: ((104, 135, 168, 36), (151, 184, 218, 175)),
    UNKNOWN_OBSTACLE_ROLE: ((181, 67, 73, 94), (244, 102, 110, 235)),
}

_MINIMUM_INFERRED_HEADING_DISPLACEMENT_M = 0.03
_NAVMESH_INTERIOR_BLOCKER_COLOR = np.asarray((196, 113, 52), dtype=np.uint8)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _positive_int(value: Any, *, owner: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise M6XTopdownError(f"{owner} must be an integer >= {minimum}")
    return value


def _finite_point(value: Any, *, owner: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M6XTopdownError(f"{owner} must contain three finite numbers") from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise M6XTopdownError(f"{owner} must contain three finite numbers")
    return point


def _source_paths(value: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], int]:
    if not isinstance(value, Mapping) or not value:
        raise M6XTopdownError("at least one source-center trajectory is required")
    paths: dict[str, np.ndarray] = {}
    frame_count: int | None = None
    for source_id in sorted(value, key=lambda item: str(item).encode("utf-8")):
        if not isinstance(source_id, str) or not source_id:
            raise M6XTopdownError("source IDs must be nonempty strings")
        try:
            points = np.asarray(value[source_id], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise M6XTopdownError(
                f"{source_id} trajectory must be finite [frame,3]"
            ) from exc
        if (
            points.ndim != 2
            or points.shape[0] < 1
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise M6XTopdownError(f"{source_id} trajectory must be finite [frame,3]")
        if frame_count is None:
            frame_count = int(points.shape[0])
        elif points.shape[0] != frame_count:
            raise M6XTopdownError("source-center trajectory frame counts differ")
        paths[source_id] = np.ascontiguousarray(points)
    assert frame_count is not None
    return paths, frame_count


def _listener_pose_series(
    *,
    listener_position_m: Sequence[float],
    listener_yaw_deg: Real,
    listener_positions_m_by_frame: Sequence[Sequence[float]] | None,
    listener_yaws_deg_by_frame: Sequence[Real] | None,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Resolve fixed or per-frame listener poses on the visual frame clock."""

    fixed_position = _finite_point(listener_position_m, owner="listener position")
    if (
        isinstance(listener_yaw_deg, bool)
        or not isinstance(listener_yaw_deg, Real)
        or not math.isfinite(float(listener_yaw_deg))
    ):
        raise M6XTopdownError("listener_yaw_deg must be a finite number")
    dynamic = (
        listener_positions_m_by_frame is not None
        or listener_yaws_deg_by_frame is not None
    )
    if not dynamic:
        return (
            np.repeat(fixed_position[None, :], frame_count, axis=0),
            np.full(frame_count, float(listener_yaw_deg), dtype=np.float64),
            False,
        )
    if (
        listener_positions_m_by_frame is None
        or listener_yaws_deg_by_frame is None
    ):
        raise M6XTopdownError(
            "per-frame listener positions and yaws must be provided together"
        )
    try:
        positions = np.asarray(listener_positions_m_by_frame, dtype=np.float64)
        yaws = np.asarray(listener_yaws_deg_by_frame, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M6XTopdownError(
            "per-frame listener poses must contain finite numbers"
        ) from exc
    if positions.shape != (frame_count, 3) or not np.all(np.isfinite(positions)):
        raise M6XTopdownError(
            "listener_positions_m_by_frame must be finite [frame,3]"
        )
    if yaws.shape != (frame_count,) or not np.all(np.isfinite(yaws)):
        raise M6XTopdownError(
            "listener_yaws_deg_by_frame must be finite [frame]"
        )
    return (
        np.ascontiguousarray(positions),
        np.ascontiguousarray(yaws),
        True,
    )


def _source_activity(
    value: Mapping[str, Any] | None,
    *,
    source_ids: Sequence[str],
    frame_count: int,
) -> dict[str, np.ndarray] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(source_ids):
        raise M6XTopdownError(
            "source_activity_by_frame keys must equal the trajectory source IDs"
        )
    result: dict[str, np.ndarray] = {}
    for source_id in source_ids:
        flags = np.asarray(value[source_id])
        if flags.shape != (frame_count,) or flags.dtype != np.bool_:
            raise M6XTopdownError(
                f"{source_id} activity must be a boolean [frame] array"
            )
        result[source_id] = np.ascontiguousarray(flags)
    return result


def _source_headings(
    value: Mapping[str, Any] | None,
    *,
    source_ids: Sequence[str],
    frame_count: int,
) -> dict[str, np.ndarray] | None:
    """Validate explicit world-XZ source orientation vectors.

    These vectors describe entity/emitter orientation, not displacement of an
    articulated mouth or muzzle anchor.  That distinction keeps an idle
    source's arrow stable even when its animated anchor has tiny pose motion.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(source_ids):
        raise M6XTopdownError(
            "source_heading_xz_by_frame keys must equal the trajectory source IDs"
        )
    result: dict[str, np.ndarray] = {}
    for source_id in source_ids:
        try:
            headings = np.asarray(value[source_id], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise M6XTopdownError(
                f"{source_id} headings must be finite [frame,2]"
            ) from exc
        if headings.shape != (frame_count, 2) or not np.all(np.isfinite(headings)):
            raise M6XTopdownError(f"{source_id} headings must be finite [frame,2]")
        norms = np.linalg.norm(headings, axis=1)
        if np.any(norms <= 1.0e-12):
            raise M6XTopdownError(f"{source_id} headings must be nonzero")
        result[source_id] = np.ascontiguousarray(headings / norms[:, None])
    return result


def _source_label_map(
    value: Mapping[str, Any] | None, *, source_ids: Sequence[str]
) -> dict[str, str]:
    if value is None:
        return {source_id: source_id for source_id in source_ids}
    if not isinstance(value, Mapping) or not set(value).issubset(source_ids):
        raise M6XTopdownError("source_labels contains an unknown source ID")
    result = {source_id: source_id for source_id in source_ids}
    for source_id, label in value.items():
        if not isinstance(label, str) or not label.strip():
            raise M6XTopdownError("source labels must be nonempty strings")
        result[source_id] = label.strip()
    return result


def _source_color_map(
    value: Mapping[str, Any] | None, *, source_ids: Sequence[str]
) -> dict[str, tuple[int, int, int, int]]:
    if value is None:
        return {
            source_id: _SOURCE_COLORS[index % len(_SOURCE_COLORS)]
            for index, source_id in enumerate(source_ids)
        }
    if not isinstance(value, Mapping) or set(value) != set(source_ids):
        raise M6XTopdownError("source_colors keys must equal the trajectory source IDs")
    result: dict[str, tuple[int, int, int, int]] = {}
    for source_id in source_ids:
        raw = value[source_id]
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or not 0 <= channel <= 255
                for channel in raw
            )
        ):
            raise M6XTopdownError(
                f"{source_id} color must contain three uint8 integers"
            )
        result[source_id] = (int(raw[0]), int(raw[1]), int(raw[2]), 255)
    return result


def _validate_obstacle_map(value: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, RuntimeObstacleMap):
        raise M6XTopdownError("obstacle_map must be a RuntimeObstacleMap")
    navmesh = np.asarray(value.binary_navmesh, dtype=np.uint8)
    if (
        navmesh.ndim != 2
        or navmesh.size == 0
        or not np.any(navmesh)
        or np.any(~np.isin(navmesh, (0, 1)))
    ):
        raise M6XTopdownError("obstacle_map contains an invalid binary navmesh")
    bounds = np.asarray(value.bounds_m, dtype=np.float64)
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[1] <= bounds[0])
    ):
        raise M6XTopdownError("obstacle_map contains invalid world bounds")
    return np.ascontiguousarray(navmesh), bounds


@lru_cache(maxsize=16)
def _cached_interior_navmesh_exclusions(
    shape: tuple[int, int], packed_navmesh: bytes
) -> tuple[bytes, int]:
    navmesh = np.frombuffer(packed_navmesh, dtype=np.uint8).reshape(shape)
    blocked = np.asarray(navmesh == 0, dtype=np.bool_)

    def grow(seed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
        current = seed
        while True:
            expanded = current.copy()
            expanded[1:] |= current[:-1]
            expanded[:-1] |= current[1:]
            expanded[:, 1:] |= current[:, :-1]
            expanded[:, :-1] |= current[:, 1:]
            expanded &= allowed
            if np.array_equal(expanded, current):
                return current
            current = expanded

    exterior_seed = np.zeros(blocked.shape, dtype=np.bool_)
    exterior_seed[0, :] = blocked[0, :]
    exterior_seed[-1, :] = blocked[-1, :]
    exterior_seed[:, 0] |= blocked[:, 0]
    exterior_seed[:, -1] |= blocked[:, -1]
    exterior = grow(exterior_seed, blocked)
    interior = np.ascontiguousarray(blocked & ~exterior)

    remaining = interior.copy()
    component_count = 0
    while np.any(remaining):
        component_count += 1
        seed = np.zeros(remaining.shape, dtype=np.bool_)
        row, column = np.argwhere(remaining)[0]
        seed[row, column] = True
        component = grow(seed, remaining)
        remaining &= ~component
    return interior.tobytes(), component_count


def _interior_navmesh_exclusions(navmesh: np.ndarray) -> tuple[np.ndarray, int]:
    """Return non-navigable regions enclosed by the navigable floor.

    Apartment furniture is part of the baked stage mesh, so Habitat exposes
    its center-point exclusion through NavMesh holes rather than separate
    rigid-object OBBs.  Treat only non-navigable pixels disconnected from the
    image border as baked floor blockers.  Border-connected pixels remain the
    room exterior/wall background, while rugs and elevated objects remain
    walkable unless an explicit rigid-object role says otherwise.
    """

    binary = np.ascontiguousarray(navmesh, dtype=np.uint8)
    mask_bytes, component_count = _cached_interior_navmesh_exclusions(
        tuple(int(value) for value in binary.shape),
        binary.tobytes(),
    )
    return (
        np.frombuffer(mask_bytes, dtype=np.bool_).reshape(binary.shape).copy(),
        component_count,
    )


def _nearest_motion_heading_xz(
    points: np.ndarray, frame_index: int
) -> np.ndarray | None:
    horizontal = points[:, (0, 2)]
    if (
        float(np.max(np.linalg.norm(horizontal - horizontal[0], axis=1)))
        < _MINIMUM_INFERRED_HEADING_DISPLACEMENT_M
    ):
        # Animated mouth/muzzle anchors can wobble by millimetres while the
        # entity root is idle.  Without an explicit authored heading, drawing
        # an arrow from that wobble is more misleading than drawing no arrow.
        return None
    for radius in range(1, points.shape[0]):
        left = max(0, frame_index - radius)
        right = min(points.shape[0] - 1, frame_index + radius)
        delta = points[right, (0, 2)] - points[left, (0, 2)]
        norm = float(np.linalg.norm(delta))
        if norm > 1.0e-9:
            return delta / norm
    return None


class _PreparedTopdown:
    def __init__(
        self,
        obstacle_map: RuntimeObstacleMap,
        paths: Mapping[str, np.ndarray],
        listener: np.ndarray,
        listener_yaw_deg: float,
        camera_hfov_degrees: float,
        size_wh: tuple[int, int],
        *,
        rigid_label_limit: int,
    ) -> None:
        navmesh, self.bounds = _validate_obstacle_map(obstacle_map)
        self.obstacle_map = obstacle_map
        self.paths = paths
        self.listener = listener
        self.width, self.height = size_wh
        self.rigid_label_limit = rigid_label_limit
        (
            self.baked_navmesh_blocker_mask,
            self.baked_navmesh_blocker_count,
        ) = _interior_navmesh_exclusions(navmesh)

        # Keep one uniform pixel scale so the room and OBB footprints are not
        # stretched and the visual FOV wedge remains geometrically meaningful.
        header_height = 49
        footer_height = 42
        margin = 8
        available_w = self.width - 2 * margin
        available_h = self.height - header_height - footer_height - 2 * margin
        if available_w < 2 or available_h < 2:
            raise M6XTopdownError("Topdown output has no drawable map area")
        map_h, map_w = navmesh.shape
        scale = min(available_w / map_w, available_h / map_h)
        draw_w = max(2, int(round(map_w * scale)))
        draw_h = max(2, int(round(map_h * scale)))
        left = (self.width - draw_w) // 2
        top = header_height + margin + (available_h - draw_h) // 2
        self.map_rect = (left, top, left + draw_w, top + draw_h)

        rgb_map = np.where(
            navmesh[..., None] != 0,
            np.asarray((209, 219, 225), dtype=np.uint8),
            np.asarray((43, 50, 59), dtype=np.uint8),
        ).astype(np.uint8)
        rgb_map[self.baked_navmesh_blocker_mask] = (
            _NAVMESH_INTERIOR_BLOCKER_COLOR
        )
        base = Image.new("RGBA", (self.width, self.height), (27, 31, 37, 255))
        map_image = Image.fromarray(rgb_map, mode="RGB").resize(
            (draw_w, draw_h), Image.Resampling.NEAREST
        )
        base.paste(map_image, (left, top))

        obstacle_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        obstacle_draw = ImageDraw.Draw(obstacle_overlay, "RGBA")
        obstacle_outlines: list[
            tuple[list[tuple[float, float]], tuple[int, int, int, int]]
        ] = []
        for obstacle in obstacle_map.rigid_obstacles:
            try:
                footprint = np.asarray(obstacle["footprint_xz_m"], dtype=np.float64)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise M6XTopdownError("rigid obstacle footprint is invalid") from exc
            if (
                footprint.ndim != 2
                or footprint.shape[0] < 3
                or footprint.shape[1] != 2
                or not np.all(np.isfinite(footprint))
            ):
                raise M6XTopdownError("rigid obstacle footprint is invalid")
            role = obstacle.get("obstacle_role", UNKNOWN_OBSTACLE_ROLE)
            if role not in _OBSTACLE_ROLE_STYLES:
                role = UNKNOWN_OBSTACLE_ROLE
            fill, outline = _OBSTACLE_ROLE_STYLES[str(role)]
            polygon = [self.panel_point_xz(point) for point in footprint]
            obstacle_draw.polygon(
                polygon,
                fill=fill,
            )
            obstacle_outlines.append((polygon, outline))

        # ImageDraw replaces RGBA pixels; a later RGB conversion would simply
        # discard their alpha instead of blending it.  Composite explicitly so
        # rugs and elevated objects stay contextual overlays rather than
        # opaque floor blockers.
        base = Image.alpha_composite(base, obstacle_overlay)
        draw = ImageDraw.Draw(base, "RGBA")
        for polygon, outline in obstacle_outlines:
            draw.line(
                (*polygon, polygon[0]),
                fill=(*outline[:3], 255),
                width=2,
                joint="curve",
            )

        if len(obstacle_map.rigid_obstacles) <= rigid_label_limit:
            for obstacle in obstacle_map.rigid_obstacles:
                footprint = np.asarray(obstacle["footprint_xz_m"], dtype=np.float64)
                x, y = self.panel_point_xz(np.mean(footprint, axis=0))
                handle = str(
                    obstacle.get("handle", obstacle.get("object_id", "object"))
                )
                role = obstacle.get("obstacle_role", UNKNOWN_OBSTACLE_ROLE)
                if role not in _OBSTACLE_ROLE_STYLES:
                    role = UNKNOWN_OBSTACLE_ROLE
                _fill, outline = _OBSTACLE_ROLE_STYLES[str(role)]
                draw.text(
                    (x + 3, y + 2),
                    handle,
                    fill=(*outline[:3], 255),
                    font=_font(9),
                    stroke_width=2,
                    stroke_fill=(30, 24, 18, 230),
                )

        try:
            basis = habitat_basis_from_yaw_degrees(listener_yaw_deg)
        except M51OrientationError as exc:
            raise M6XTopdownError(f"listener yaw is invalid: {exc}") from exc
        self.forward_xz = np.asarray(basis.forward_xz, dtype=np.float64)
        self.right_xz = np.asarray(basis.right_xz, dtype=np.float64)
        self.listener_xy = self.panel_point(listener)
        half_fov = math.radians(camera_hfov_degrees * 0.5)
        left_ray_xz = (
            math.cos(half_fov) * self.forward_xz - math.sin(half_fov) * self.right_xz
        )
        right_ray_xz = (
            math.cos(half_fov) * self.forward_xz + math.sin(half_fov) * self.right_xz
        )
        wedge_length = max(42.0, min(draw_w, draw_h) * 0.17)
        self.forward_delta = self.panel_direction(self.forward_xz, pixel_length=34.0)
        self.right_delta = self.panel_direction(self.right_xz, pixel_length=23.0)
        self.left_ray_delta = self.panel_direction(
            left_ray_xz, pixel_length=wedge_length
        )
        self.right_ray_delta = self.panel_direction(
            right_ray_xz, pixel_length=wedge_length
        )
        self.projected_paths = {
            source_id: tuple(self.panel_point(point) for point in points)
            for source_id, points in paths.items()
        }
        self.base = base

    def panel_point_xz(self, point_xz: Sequence[float]) -> tuple[float, float]:
        point = np.asarray(point_xz, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise M6XTopdownError("Topdown XZ point must contain two finite numbers")
        low, high = self.bounds[:, (0, 2)]
        u, v = (point - low) / (high - low)
        left, top, right, bottom = self.map_rect
        return (
            float(left + u * (right - left - 1)),
            float(top + v * (bottom - top - 1)),
        )

    def panel_point(self, point_xyz: Sequence[float]) -> tuple[float, float]:
        point = np.asarray(point_xyz, dtype=np.float64)
        return self.panel_point_xz(point[(0, 2),])

    def panel_direction(
        self, direction_xz: Sequence[float], *, pixel_length: float
    ) -> np.ndarray:
        direction = np.asarray(direction_xz, dtype=np.float64)
        endpoint = self.listener + np.asarray((direction[0], 0.0, direction[1]))
        delta = np.asarray(self.panel_point(endpoint)) - np.asarray(self.listener_xy)
        norm = float(np.linalg.norm(delta))
        if norm <= 1.0e-12:
            raise M6XTopdownError(
                "listener orientation has degenerate Topdown projection"
            )
        return delta / norm * pixel_length


def _validate_common(
    obstacle_map: RuntimeObstacleMap,
    source_center_trajectories_m: Mapping[str, Any],
    *,
    listener_position_m: Sequence[float],
    listener_yaw_deg: Real,
    camera_hfov_degrees: Real,
    source_activity_by_frame: Mapping[str, Any] | None,
    source_heading_xz_by_frame: Mapping[str, Any] | None,
    source_labels: Mapping[str, Any] | None,
    source_colors: Mapping[str, Any] | None,
    size_wh: tuple[int, int],
    rigid_label_limit: int,
) -> tuple[
    dict[str, np.ndarray],
    int,
    np.ndarray,
    float,
    float,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, str],
    dict[str, tuple[int, int, int, int]],
    _PreparedTopdown,
]:
    paths, frame_count = _source_paths(source_center_trajectories_m)
    listener = _finite_point(listener_position_m, owner="listener position")
    if (
        isinstance(listener_yaw_deg, bool)
        or not isinstance(listener_yaw_deg, Real)
        or not math.isfinite(float(listener_yaw_deg))
    ):
        raise M6XTopdownError("listener_yaw_deg must be a finite number")
    if (
        isinstance(camera_hfov_degrees, bool)
        or not isinstance(camera_hfov_degrees, Real)
        or not math.isfinite(float(camera_hfov_degrees))
        or not 0.0 < float(camera_hfov_degrees) < 180.0
    ):
        raise M6XTopdownError("camera_hfov_degrees must lie within (0,180)")
    try:
        width, height = size_wh
    except (TypeError, ValueError) as exc:
        raise M6XTopdownError("size_wh must contain width and height") from exc
    width = _positive_int(width, owner="Topdown width", minimum=320)
    height = _positive_int(height, owner="Topdown height", minimum=240)
    label_limit = _positive_int(rigid_label_limit, owner="rigid_label_limit", minimum=0)
    source_ids = tuple(paths)
    activity = _source_activity(
        source_activity_by_frame,
        source_ids=source_ids,
        frame_count=frame_count,
    )
    headings = _source_headings(
        source_heading_xz_by_frame,
        source_ids=source_ids,
        frame_count=frame_count,
    )
    labels = _source_label_map(source_labels, source_ids=source_ids)
    colors = _source_color_map(source_colors, source_ids=source_ids)
    prepared = _PreparedTopdown(
        obstacle_map,
        paths,
        listener,
        float(listener_yaw_deg),
        float(camera_hfov_degrees),
        (width, height),
        rigid_label_limit=label_limit,
    )
    return (
        paths,
        frame_count,
        listener,
        float(listener_yaw_deg),
        float(camera_hfov_degrees),
        activity,
        headings,
        labels,
        colors,
        prepared,
    )


def _render_prepared_frame(
    prepared: _PreparedTopdown,
    frame_index: int,
    *,
    paths: Mapping[str, np.ndarray],
    frame_count: int,
    camera_hfov_degrees: float,
    activity: Mapping[str, np.ndarray] | None,
    headings: Mapping[str, np.ndarray] | None,
    labels: Mapping[str, str],
    colors: Mapping[str, tuple[int, int, int, int]],
) -> np.ndarray:
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or not 0 <= frame_index < frame_count
    ):
        raise M6XTopdownError("frame_index lies outside the source trajectories")

    image = prepared.base.copy()
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    lx, ly = prepared.listener_xy
    wedge = (
        (lx, ly),
        (lx + prepared.left_ray_delta[0], ly + prepared.left_ray_delta[1]),
        (lx + prepared.right_ray_delta[0], ly + prepared.right_ray_delta[1]),
    )
    overlay_draw.polygon(wedge, fill=(46, 154, 255, 48))
    overlay_draw.line((*wedge, wedge[0]), fill=(46, 154, 255, 255), width=2)
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image, "RGBA")

    for source_id in paths:
        points = paths[source_id]
        color = colors[source_id]
        projected = prepared.projected_paths[source_id]
        if len(projected) > 1:
            draw.line(projected, fill=(*color[:3], 72), width=3)
            if frame_index > 0:
                draw.line(projected[: frame_index + 1], fill=color, width=4)
        x, y = projected[frame_index]
        is_active = None if activity is None else bool(activity[source_id][frame_index])
        fill = color if is_active is not False else (*color[:3], 88)
        outline = (0, 0, 0, 255) if is_active is not False else (230, 230, 230, 230)
        draw.ellipse(
            (x - 7, y - 7, x + 7, y + 7),
            fill=fill,
            outline=outline,
            width=2,
        )
        heading = (
            headings[source_id][frame_index]
            if headings is not None
            else _nearest_motion_heading_xz(points, frame_index)
        )
        if heading is not None:
            endpoint = prepared.panel_point_xz(points[frame_index, (0, 2)] + heading)
            delta = np.asarray(endpoint) - np.asarray((x, y))
            norm = float(np.linalg.norm(delta))
            if norm > 1.0e-12:
                delta = delta / norm * 16.0
                draw.line((x, y, x + delta[0], y + delta[1]), fill=outline, width=2)
        state = "" if is_active is None else (" ACTIVE" if is_active else " SILENT")
        draw.text(
            (x + 10, y - 10),
            f"{labels[source_id]} [{source_id}]{state}",
            fill=color if is_active is not False else (210, 210, 210, 255),
            font=_font(11),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    draw.ellipse(
        (lx - 7, ly - 7, lx + 7, ly + 7),
        fill=(255, 224, 66, 255),
        outline=(0, 0, 0, 255),
        width=2,
    )
    left_ear = (lx - prepared.right_delta[0], ly - prepared.right_delta[1])
    right_ear = (lx + prepared.right_delta[0], ly + prepared.right_delta[1])
    draw.line((*left_ear, *right_ear), fill=(210, 80, 220, 255), width=3)
    for marker, point in (("L", left_ear), ("R", right_ear)):
        draw.text(
            (point[0] - 4, point[1] - 13),
            marker,
            fill=(255, 255, 255, 255),
            font=_font(10),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    draw.line(
        (lx, ly, lx + prepared.forward_delta[0], ly + prepared.forward_delta[1]),
        fill=(20, 20, 20, 255),
        width=4,
    )
    draw.text(
        (lx + prepared.forward_delta[0] - 4, ly + prepared.forward_delta[1] - 13),
        "F",
        fill=(255, 255, 255, 255),
        font=_font(10),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    draw.text(
        (lx + 10, ly + 5),
        "CAM/LISTENER",
        fill=(255, 235, 90, 255),
        font=_font(11),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )

    draw.rectangle((0, 0, prepared.width - 1, 48), fill=(0, 0, 0, 205))
    draw.text(
        (8, 5),
        (
            f"LIVE ROOM OBSTACLES | frame {frame_index:03d}/{frame_count - 1:03d} | "
            f"sources={len(paths)} | "
            f"nav blocks={prepared.baked_navmesh_blocker_count} | "
            f"rigid OBBs={len(prepared.obstacle_map.rigid_obstacles)}"
        ),
        fill=(255, 255, 255, 255),
        font=_font(13),
    )
    draw.text(
        (8, 25),
        "ORANGE=NAV/OBB BLOCK | TEAL=RUG/MAT | BLUE=ELEVATED | RED=UNKNOWN",
        fill=(230, 230, 230, 255),
        font=_font(11),
    )
    draw.rectangle(
        (0, prepared.height - 41, prepared.width - 1, prepared.height - 1),
        fill=(0, 0, 0, 205),
    )
    draw.text(
        (8, prepared.height - 38),
        f"VISUAL HFOV={camera_hfov_degrees:g} deg only | AUDIO: 360 deg; no FOV cutoff",
        fill=(255, 255, 255, 255),
        font=_font(11),
    )
    draw.text(
        (8, prepared.height - 20),
        "QA ONLY: source-center paths/points; no body-volume collision claim",
        fill=(255, 255, 255, 255),
        font=_font(11),
    )
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def render_runtime_topdown_frame(
    obstacle_map: RuntimeObstacleMap,
    source_center_trajectories_m: Mapping[str, Any],
    frame_index: int,
    *,
    listener_position_m: Sequence[float],
    listener_yaw_deg: Real,
    camera_hfov_degrees: Real,
    listener_positions_m_by_frame: Sequence[Sequence[float]] | None = None,
    listener_yaws_deg_by_frame: Sequence[Real] | None = None,
    source_activity_by_frame: Mapping[str, Any] | None = None,
    source_heading_xz_by_frame: Mapping[str, Any] | None = None,
    source_labels: Mapping[str, Any] | None = None,
    source_colors: Mapping[str, Any] | None = None,
    size_wh: tuple[int, int] = (640, 480),
    rigid_label_limit: int = 16,
) -> np.ndarray:
    """Render one diagnostic frame from the runtime obstacle snapshot.

    ``source_activity_by_frame`` is optional visual metadata.  It changes the
    dot/label styling only and never controls acoustic rendering or audibility.
    ``source_heading_xz_by_frame`` is the preferred arrow authority; small
    anchor-only displacement is never treated as entity orientation.
    """

    (
        paths,
        frame_count,
        _listener,
        _listener_yaw,
        hfov,
        activity,
        headings,
        labels,
        colors,
        prepared,
    ) = _validate_common(
        obstacle_map,
        source_center_trajectories_m,
        listener_position_m=listener_position_m,
        listener_yaw_deg=listener_yaw_deg,
        camera_hfov_degrees=camera_hfov_degrees,
        source_activity_by_frame=source_activity_by_frame,
        source_heading_xz_by_frame=source_heading_xz_by_frame,
        source_labels=source_labels,
        source_colors=source_colors,
        size_wh=size_wh,
        rigid_label_limit=rigid_label_limit,
    )
    listener_positions, listener_yaws, dynamic_listener = _listener_pose_series(
        listener_position_m=listener_position_m,
        listener_yaw_deg=listener_yaw_deg,
        listener_positions_m_by_frame=listener_positions_m_by_frame,
        listener_yaws_deg_by_frame=listener_yaws_deg_by_frame,
        frame_count=frame_count,
    )
    if dynamic_listener:
        prepared = _PreparedTopdown(
            obstacle_map,
            paths,
            listener_positions[frame_index],
            float(listener_yaws[frame_index]),
            hfov,
            (prepared.width, prepared.height),
            rigid_label_limit=prepared.rigid_label_limit,
        )
    return _render_prepared_frame(
        prepared,
        frame_index,
        paths=paths,
        frame_count=frame_count,
        camera_hfov_degrees=hfov,
        activity=activity,
        headings=headings,
        labels=labels,
        colors=colors,
    )


def render_runtime_topdown_frames(
    obstacle_map: RuntimeObstacleMap,
    source_center_trajectories_m: Mapping[str, Any],
    *,
    listener_position_m: Sequence[float],
    listener_yaw_deg: Real,
    camera_hfov_degrees: Real,
    listener_positions_m_by_frame: Sequence[Sequence[float]] | None = None,
    listener_yaws_deg_by_frame: Sequence[Real] | None = None,
    source_activity_by_frame: Mapping[str, Any] | None = None,
    source_heading_xz_by_frame: Mapping[str, Any] | None = None,
    source_labels: Mapping[str, Any] | None = None,
    source_colors: Mapping[str, Any] | None = None,
    size_wh: tuple[int, int] = (640, 480),
    rigid_label_limit: int = 16,
) -> np.ndarray:
    """Render every source-center frame while reusing one room base image.

    Explicit source headings keep idle articulated anchors from turning their
    pose wobble into a false changing orientation arrow.
    """

    (
        paths,
        frame_count,
        _listener,
        _listener_yaw,
        hfov,
        activity,
        headings,
        labels,
        colors,
        prepared,
    ) = _validate_common(
        obstacle_map,
        source_center_trajectories_m,
        listener_position_m=listener_position_m,
        listener_yaw_deg=listener_yaw_deg,
        camera_hfov_degrees=camera_hfov_degrees,
        source_activity_by_frame=source_activity_by_frame,
        source_heading_xz_by_frame=source_heading_xz_by_frame,
        source_labels=source_labels,
        source_colors=source_colors,
        size_wh=size_wh,
        rigid_label_limit=rigid_label_limit,
    )
    listener_positions, listener_yaws, dynamic_listener = _listener_pose_series(
        listener_position_m=listener_position_m,
        listener_yaw_deg=listener_yaw_deg,
        listener_positions_m_by_frame=listener_positions_m_by_frame,
        listener_yaws_deg_by_frame=listener_yaws_deg_by_frame,
        frame_count=frame_count,
    )
    return np.ascontiguousarray(
        np.stack(
            [
                _render_prepared_frame(
                    (
                        _PreparedTopdown(
                            obstacle_map,
                            paths,
                            listener_positions[frame_index],
                            float(listener_yaws[frame_index]),
                            hfov,
                            (prepared.width, prepared.height),
                            rigid_label_limit=prepared.rigid_label_limit,
                        )
                        if dynamic_listener
                        else prepared
                    ),
                    frame_index,
                    paths=paths,
                    frame_count=frame_count,
                    camera_hfov_degrees=hfov,
                    activity=activity,
                    headings=headings,
                    labels=labels,
                    colors=colors,
                )
                for frame_index in range(frame_count)
            ],
            axis=0,
        )
    )


__all__ = [
    "M6XTopdownError",
    "TOPDOWN_SCHEMA",
    "render_runtime_topdown_frame",
    "render_runtime_topdown_frames",
]
