"""Legacy-Apartment Topdown QA panels bound to the migrated M5.1 route."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avengine.m5_1.legacy_route import assert_valid_route_manifest


TOPDOWN_SCHEMA = "avengine_m5_1_legacy_topdown_v1"


class M51TopdownError(ValueError):
    """The route cannot be represented as an auditable Topdown panel."""


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


def _positions(route: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        value = np.asarray(
            route["routes"][key]["habitat_trajectory_m"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M51TopdownError(f"missing route positions for {key}") from exc
    if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
        raise M51TopdownError(f"{key} positions must be finite [frame,3]")
    return value


def _nearest_heading_xz(points: np.ndarray, frame_index: int) -> np.ndarray:
    for radius in range(1, points.shape[0]):
        left = max(0, frame_index - radius)
        right = min(points.shape[0] - 1, frame_index + radius)
        delta = points[right, (0, 2)] - points[left, (0, 2)]
        norm = float(np.linalg.norm(delta))
        if norm > 1.0e-9:
            return delta / norm
    return np.asarray((0.0, -1.0), dtype=np.float64)


def _world_bounds(
    route: Mapping[str, Any], human: np.ndarray, dog: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = [human[:, (0, 2)], dog[:, (0, 2)]]
    camera = np.asarray(route["camera"]["habitat_position_m"], dtype=np.float64)
    points.append(camera[None, (0, 2)])
    for obstacle in route.get("obstacles", []):
        box = obstacle.get("horizontal_aabb_habitat_xz_m", {})
        try:
            points.append(np.asarray([box["minimum"], box["maximum"]], dtype=np.float64))
        except (KeyError, TypeError, ValueError):
            continue
    merged = np.concatenate(points, axis=0)
    low = np.min(merged, axis=0) - 0.4
    high = np.max(merged, axis=0) + 0.4
    if np.any(high - low < 1.0):
        raise M51TopdownError("route Topdown world bounds are degenerate")
    return low, high


def render_legacy_topdown_frame(
    route_manifest: Mapping[str, Any],
    frame_index: int,
    *,
    size_wh: tuple[int, int] = (320, 240),
    _prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Render one complete-path QA frame; Topdown never becomes a formal view."""

    if _prepared is None:
        assert_valid_route_manifest(route_manifest)
        human = _positions(route_manifest, "human_path")
        dog = _positions(route_manifest, "dog_path")
        low, high = _world_bounds(route_manifest, human, dog)
    else:
        human, dog, low, high = _prepared
    if human.shape != dog.shape:
        raise M51TopdownError("human and dog route lengths differ")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index < human.shape[0]:
        raise M51TopdownError("frame_index lies outside the route")
    width, height = size_wh
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 160
        or height < 120
    ):
        raise M51TopdownError("Topdown size must be integer and at least 160x120")

    margin = 12.0

    def project_xz(xz: Sequence[float]) -> tuple[float, float]:
        point = np.asarray(xz, dtype=np.float64)
        u = (point[0] - low[0]) / (high[0] - low[0])
        v = (point[1] - low[1]) / (high[1] - low[1])
        return (
            margin + u * (width - 2.0 * margin),
            height - margin - v * (height - 2.0 * margin),
        )

    image = Image.new("RGB", (width, height), (239, 241, 244))
    draw = ImageDraw.Draw(image, "RGBA")
    for obstacle in route_manifest.get("obstacles", []):
        box = obstacle.get("horizontal_aabb_habitat_xz_m")
        if not isinstance(box, Mapping):
            continue
        try:
            a = project_xz(box["minimum"])
            b = project_xz(box["maximum"])
        except (KeyError, TypeError, ValueError):
            continue
        bounds = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        kind = str(obstacle.get("source_kind", ""))
        active = bool(obstacle.get("included_in_point_gate"))
        if "shell" in kind:
            draw.rectangle(bounds, outline=(65, 72, 80, 155), width=1)
        elif active:
            draw.rectangle(bounds, fill=(222, 185, 116, 62), outline=(151, 116, 55, 140), width=1)
        else:
            draw.rectangle(bounds, fill=(160, 166, 176, 30), outline=(130, 136, 146, 65), width=1)

    colors = {
        "human": (28, 151, 169, 255),
        "dog": (231, 103, 62, 255),
    }
    for name, points in (("human", human), ("dog", dog)):
        projected = [project_xz(point[[0, 2]]) for point in points]
        draw.line(projected, fill=(*colors[name][:3], 70), width=2)
        if frame_index > 0:
            draw.line(projected[: frame_index + 1], fill=colors[name], width=3)
        current = projected[frame_index]
        draw.ellipse(
            (current[0] - 5, current[1] - 5, current[0] + 5, current[1] + 5),
            fill=colors[name],
            outline=(0, 0, 0, 255),
            width=1,
        )
        heading = _nearest_heading_xz(points, frame_index)
        # Projection flips vertical screen direction relative to +Z.
        end = (current[0] + 12.0 * heading[0], current[1] - 12.0 * heading[1])
        draw.line((current[0], current[1], end[0], end[1]), fill=(0, 0, 0, 255), width=2)
        draw.text(
            (current[0] + 7, current[1] - 8),
            name.upper(),
            font=_font(10),
            fill=colors[name],
            stroke_width=1,
            stroke_fill=(255, 255, 255, 230),
        )

    camera = np.asarray(route_manifest["camera"]["habitat_position_m"], dtype=np.float64)
    camera_xy = project_xz(camera[[0, 2]])
    yaw = math.radians(float(route_manifest["camera"]["habitat_yaw_deg"]))
    forward_xz = np.asarray((math.sin(yaw), -math.cos(yaw)), dtype=np.float64)
    camera_end = (
        camera_xy[0] + 15.0 * forward_xz[0],
        camera_xy[1] - 15.0 * forward_xz[1],
    )
    draw.ellipse(
        (camera_xy[0] - 5, camera_xy[1] - 5, camera_xy[0] + 5, camera_xy[1] + 5),
        fill=(255, 224, 66, 255),
        outline=(0, 0, 0, 255),
    )
    draw.line((*camera_xy, *camera_end), fill=(20, 20, 20, 255), width=3)
    draw.text(
        (camera_xy[0] + 7, camera_xy[1] + 2),
        "CAM/LISTENER",
        font=_font(9),
        fill=(30, 30, 30, 255),
    )

    gate_h = route_manifest["gates"]["human_center_point_aabb"]["frames"][frame_index]["status"]
    gate_d = route_manifest["gates"]["dog_center_point_aabb"]["frames"][frame_index]["status"]
    draw.rectangle((3, 3, width - 4, 22), fill=(0, 0, 0, 178))
    draw.text(
        (7, 6),
        f"TOPDOWN QA {frame_index:03d}/{human.shape[0]-1:03d}  point gate H={gate_h} D={gate_d}",
        font=_font(10),
        fill=(255, 255, 255, 255),
    )
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def render_legacy_topdown_frames(
    route_manifest: Mapping[str, Any],
    *,
    size_wh: tuple[int, int] = (320, 240),
) -> np.ndarray:
    """Render every frame in the exact 18-second legacy route manifest."""

    assert_valid_route_manifest(route_manifest)
    human = _positions(route_manifest, "human_path")
    dog = _positions(route_manifest, "dog_path")
    if human.shape != dog.shape:
        raise M51TopdownError("human and dog route lengths differ")
    low, high = _world_bounds(route_manifest, human, dog)
    prepared = (human, dog, low, high)
    count = int(route_manifest["timebase"]["frame_count"])
    return np.ascontiguousarray(
        np.stack(
            [
                render_legacy_topdown_frame(
                    route_manifest,
                    frame_index,
                    size_wh=size_wh,
                    _prepared=prepared,
                )
                for frame_index in range(count)
            ],
            axis=0,
        )
    )


__all__ = [
    "M51TopdownError",
    "TOPDOWN_SCHEMA",
    "render_legacy_topdown_frame",
    "render_legacy_topdown_frames",
]
