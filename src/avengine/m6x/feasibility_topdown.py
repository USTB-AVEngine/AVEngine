"""Static Topdown overview for complete feasibility and sampled trajectories."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avengine.m5_1.orientation import habitat_basis_from_yaw_degrees
from avengine.m6x.geometry import (
    ELEVATED_OBJECT,
    GROUND_BLOCKER,
    UNKNOWN_OBSTACLE_ROLE,
    WALKABLE_FLOOR_COVERING,
)
from avengine.m6x.room_feasibility import (
    MOTION_CASES,
    FeasibleRegionIndex,
    TrajectoryBank,
    TrajectoryCoverage,
)


FEASIBILITY_TOPDOWN_SCHEMA = "avengine_room_feasibility_topdown_v1"


class FeasibilityTopdownError(ValueError):
    """The feasibility bank cannot be rendered unambiguously."""


_ACTOR_COLORS = {
    "human0": (32, 212, 235, 42),
    "dog0": (255, 132, 61, 42),
}

_OBSTACLE_COLORS = {
    GROUND_BLOCKER: ((230, 139, 58, 105), (255, 186, 90, 235)),
    WALKABLE_FLOOR_COVERING: ((44, 155, 137, 64), (76, 222, 194, 220)),
    ELEVATED_OBJECT: ((104, 135, 168, 32), (151, 184, 218, 165)),
    UNKNOWN_OBSTACLE_ROLE: ((181, 67, 73, 82), (244, 102, 110, 225)),
}


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


def render_feasibility_topdown(
    region_by_actor: Mapping[str, FeasibleRegionIndex],
    trajectory_bank: TrajectoryBank,
    *,
    source_to_actor: Mapping[str, str],
    trajectory_coverage: TrajectoryCoverage | None = None,
    listener_position_m: Sequence[float],
    listener_yaw_deg: Real,
    camera_hfov_degrees: Real,
    room_label: str = "Apartment",
    navigation_authority_label: str | None = None,
    size_wh: tuple[int, int] = (1800, 1400),
) -> np.ndarray:
    """Draw the complete raster region, sample nodes, and every bank path."""

    if set(region_by_actor) != {"human0", "dog0"}:
        raise FeasibilityTopdownError(
            "region_by_actor must contain exactly human0 and dog0"
        )
    human_region = region_by_actor["human0"]
    dog_region = region_by_actor["dog0"]
    if human_region.obstacle_map is not dog_region.obstacle_map:
        raise FeasibilityTopdownError("regions do not share one obstacle map")
    if human_region.feasible_mask.shape != dog_region.feasible_mask.shape:
        raise FeasibilityTopdownError("region mask shapes differ")
    if set(source_to_actor.values()) != {"human0", "dog0"}:
        raise FeasibilityTopdownError("source_to_actor must cover human0 and dog0")
    if any(actor_id not in _ACTOR_COLORS for actor_id in source_to_actor.values()):
        raise FeasibilityTopdownError("source_to_actor contains an unknown actor")
    if not isinstance(room_label, str) or not room_label.strip():
        raise FeasibilityTopdownError("room_label must be a nonempty string")
    listener = np.asarray(listener_position_m, dtype=np.float64)
    if listener.shape != (3,) or not np.all(np.isfinite(listener)):
        raise FeasibilityTopdownError("listener position is invalid")
    if (
        isinstance(listener_yaw_deg, bool)
        or not isinstance(listener_yaw_deg, Real)
        or not math.isfinite(float(listener_yaw_deg))
    ):
        raise FeasibilityTopdownError("listener yaw is invalid")
    if (
        isinstance(camera_hfov_degrees, bool)
        or not isinstance(camera_hfov_degrees, Real)
        or not 0.0 < float(camera_hfov_degrees) < 180.0
    ):
        raise FeasibilityTopdownError("camera HFOV is invalid")
    try:
        width, height = (int(size_wh[0]), int(size_wh[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise FeasibilityTopdownError("size_wh is invalid") from exc
    if width < 800 or height < 600:
        raise FeasibilityTopdownError("overview must be at least 800x600")

    obstacle_map = human_region.obstacle_map
    authority_label = navigation_authority_label
    if authority_label is None:
        authority_label = (
            "Habitat navmesh + loaded collision OBBs"
            if obstacle_map.authority.startswith("live_habitat")
            else obstacle_map.authority.replace("_", " ")
        )
    if not isinstance(authority_label, str) or not authority_label.strip():
        raise FeasibilityTopdownError(
            "navigation_authority_label must be a nonempty string"
        )
    navmesh = np.asarray(obstacle_map.binary_navmesh, dtype=np.bool_)
    feasible = human_region.feasible_mask & dog_region.feasible_mask
    if not np.any(feasible):
        raise FeasibilityTopdownError("human/dog feasible-region intersection is empty")
    rgb = np.empty((*navmesh.shape, 3), dtype=np.uint8)
    rgb[:] = (32, 38, 46)
    rgb[navmesh] = (110, 122, 132)
    rgb[feasible] = (182, 224, 190)
    if trajectory_coverage is not None:
        coverage_distance = np.asarray(
            trajectory_coverage.distance_to_trajectory_m, dtype=np.float64
        )
        if coverage_distance.shape != feasible.shape:
            raise FeasibilityTopdownError("trajectory coverage shape differs")
        rgb[feasible & (coverage_distance <= 0.25)] = (164, 222, 179)
        rgb[feasible & (coverage_distance > 0.25) & (coverage_distance <= 0.50)] = (
            199,
            225,
            161,
        )
        rgb[feasible & (coverage_distance > 0.50) & (coverage_distance <= 1.00)] = (
            238,
            214,
            137,
        )
        rgb[feasible & (coverage_distance > 1.00)] = (238, 155, 137)

    header_height = 190
    footer_height = 92
    margin = 22
    available_width = width - 2 * margin
    available_height = height - header_height - footer_height - 2 * margin
    map_height, map_width = navmesh.shape
    scale = min(available_width / map_width, available_height / map_height)
    draw_width = max(2, int(round(map_width * scale)))
    draw_height = max(2, int(round(map_height * scale)))
    left = (width - draw_width) // 2
    top = header_height + margin + (available_height - draw_height) // 2

    image = Image.new("RGBA", (width, height), (24, 28, 34, 255))
    map_image = Image.fromarray(rgb, mode="RGB").resize(
        (draw_width, draw_height), Image.Resampling.NEAREST
    )
    image.paste(map_image, (left, top))
    bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)

    def point_xz(value: Sequence[float]) -> tuple[float, float]:
        point = np.asarray(value, dtype=np.float64)
        low = bounds[0, (0, 2)]
        high = bounds[1, (0, 2)]
        uv = (point - low) / (high - low)
        return (
            float(left + uv[0] * (draw_width - 1)),
            float(top + uv[1] * (draw_height - 1)),
        )

    obstacle_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    obstacle_draw = ImageDraw.Draw(obstacle_overlay, "RGBA")
    obstacle_outlines: list[tuple[list[tuple[float, float]], tuple[int, ...]]] = []
    for obstacle in obstacle_map.rigid_obstacles:
        footprint = np.asarray(obstacle["footprint_xz_m"], dtype=np.float64)
        role = obstacle.get("obstacle_role", UNKNOWN_OBSTACLE_ROLE)
        if role not in _OBSTACLE_COLORS:
            role = UNKNOWN_OBSTACLE_ROLE
        fill, outline = _OBSTACLE_COLORS[str(role)]
        polygon = [point_xz(point) for point in footprint]
        obstacle_draw.polygon(polygon, fill=fill)
        obstacle_outlines.append((polygon, outline))
    image = Image.alpha_composite(image, obstacle_overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    for polygon, outline in obstacle_outlines:
        draw.line((*polygon, polygon[0]), fill=outline, width=2, joint="curve")

    # Sampling nodes are the finite candidate set used for route endpoints.
    sample_pixels = human_region.sample_pixels_rc
    sample_pixels = sample_pixels[
        dog_region.feasible_mask[sample_pixels[:, 0], sample_pixels[:, 1]]
    ]
    for row, col in sample_pixels:
        point = human_region.pixel_to_world((int(row), int(col)))
        x, y = point_xz(point[[0, 2]])
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(31, 108, 49, 145))

    # All finite bank paths are intentionally faint; the first route of each
    # motion case is redrawn strongly as a representative example.
    first_by_case: dict[str, Any] = {}
    for episode in trajectory_bank.episodes:
        first_by_case.setdefault(episode.motion_case, episode)
        for source_id, path in episode.source_center_paths_m.items():
            actor_id = source_to_actor[source_id]
            color = _ACTOR_COLORS[actor_id]
            projected = [point_xz(point[[0, 2]]) for point in np.asarray(path)]
            if np.allclose(path[:, (0, 2)], path[0, (0, 2)], atol=1.0e-12):
                x, y = projected[0]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            else:
                draw.line(projected, fill=color, width=2, joint="curve")
    for motion_case in MOTION_CASES:
        episode = first_by_case.get(motion_case)
        if episode is None:
            continue
        for source_id, path in episode.source_center_paths_m.items():
            actor_id = source_to_actor[source_id]
            base = _ACTOR_COLORS[actor_id]
            projected = [point_xz(point[[0, 2]]) for point in np.asarray(path)]
            if len(projected) > 1:
                draw.line(projected, fill=(*base[:3], 230), width=5, joint="curve")
            for x, y in (projected[0], projected[-1]):
                draw.ellipse(
                    (x - 5, y - 5, x + 5, y + 5),
                    fill=(*base[:3], 240),
                    outline=(12, 12, 12, 255),
                    width=1,
                )

    basis = habitat_basis_from_yaw_degrees(float(listener_yaw_deg))
    listener_xy = point_xz(listener[[0, 2]])
    forward = np.asarray(basis.forward_xz, dtype=np.float64)
    right = np.asarray(basis.right_xz, dtype=np.float64)
    half_fov = math.radians(float(camera_hfov_degrees) * 0.5)
    left_direction = math.cos(half_fov) * forward - math.sin(half_fov) * right
    right_direction = math.cos(half_fov) * forward + math.sin(half_fov) * right

    def direction_delta(direction: np.ndarray, length: float) -> np.ndarray:
        endpoint = point_xz(listener[[0, 2]] + direction)
        delta = np.asarray(endpoint) - np.asarray(listener_xy)
        return delta / np.linalg.norm(delta) * length

    left_delta = direction_delta(left_direction, 115.0)
    right_delta = direction_delta(right_direction, 115.0)
    forward_delta = direction_delta(forward, 72.0)
    lx, ly = listener_xy
    wedge_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    wedge_draw = ImageDraw.Draw(wedge_overlay, "RGBA")
    wedge_draw.polygon(
        (
            (lx, ly),
            (lx + left_delta[0], ly + left_delta[1]),
            (lx + right_delta[0], ly + right_delta[1]),
        ),
        fill=(45, 148, 255, 42),
    )
    image = Image.alpha_composite(image, wedge_overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.line(
        (
            lx + left_delta[0],
            ly + left_delta[1],
            lx,
            ly,
            lx + right_delta[0],
            ly + right_delta[1],
        ),
        fill=(45, 148, 255, 245),
        width=3,
    )
    draw.line(
        (lx, ly, lx + forward_delta[0], ly + forward_delta[1]),
        fill=(255, 232, 64, 255),
        width=5,
    )
    draw.ellipse(
        (lx - 8, ly - 8, lx + 8, ly + 8),
        fill=(255, 232, 64, 255),
        outline=(0, 0, 0, 255),
        width=2,
    )

    feasible_pixels = int(np.count_nonzero(feasible))
    pixel_area = human_region.pixel_size_x_m * human_region.pixel_size_z_m
    counts = {
        motion_case: sum(
            episode.motion_case == motion_case for episode in trajectory_bank.episodes
        )
        for motion_case in MOTION_CASES
    }
    draw.rectangle((0, 0, width - 1, header_height), fill=(0, 0, 0, 220))
    draw.text(
        (24, 14),
        f"{room_label.strip().upper()} COMPLETE FEASIBLE REGION + FINITE TRAJECTORY BANK",
        fill=(255, 255, 255, 255),
        font=_font(28),
    )
    draw.text(
        (24, 54),
        (
            f"feasible={feasible_pixels * pixel_area:.2f} m² / "
            f"{feasible_pixels} pixels | samples={len(sample_pixels)} | "
            f"components={len(human_region.components)} | "
            f"trajectories={len(trajectory_bank.episodes)} x 2 sources"
        ),
        fill=(235, 235, 235, 255),
        font=_font(19),
    )
    draw.text(
        (24, 87),
        (
            "motion cases: "
            + " | ".join(f"{case}={counts[case]}" for case in MOTION_CASES)
        ),
        fill=(215, 215, 215, 255),
        font=_font(17),
    )
    if trajectory_coverage is None:
        coverage_text = "coverage audit: not supplied"
    else:
        coverage_record = trajectory_coverage.record
        half_meter = coverage_record["coverage_fraction_by_threshold"].get(
            "within_0.50m"
        )
        coverage_text = (
            f"coverage={coverage_record['status'].upper()} | within 0.50m="
            f"{100.0 * float(half_meter):.2f}% | "
            f"p95 gap={coverage_record['p95_gap_m']:.3f}m | "
            f"max gap={coverage_record['maximum_gap_m']:.3f}m"
        )
    draw.text((24, 119), coverage_text, fill=(255, 231, 151, 255), font=_font(17))
    draw.text(
        (24, 151),
        (
            "COVERAGE: GREEN<=0.25m | LIME<=0.50m | AMBER<=1.00m | RED>1.00m "
            "| CYAN=human | ORANGE=dog | YELLOW=listener/FOV"
        ),
        fill=(225, 235, 225, 255),
        font=_font(16),
    )

    draw.rectangle(
        (0, height - footer_height, width - 1, height - 1),
        fill=(0, 0, 0, 220),
    )
    draw.text(
        (24, height - footer_height + 12),
        (
            "Complete means every raster cell in the declared source-center feasible "
            "region; continuous paths are infinite, so the overlaid bank is sampled."
        ),
        fill=(255, 255, 255, 255),
        font=_font(16),
    )
    draw.text(
        (24, height - footer_height + 43),
        (
            f"QA ONLY — source centers; authority: {authority_label.strip()}; "
            "no human/dog body-volume collision claim."
        ),
        fill=(255, 211, 94, 255),
        font=_font(16),
    )
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


__all__ = [
    "FEASIBILITY_TOPDOWN_SCHEMA",
    "FeasibilityTopdownError",
    "render_feasibility_topdown",
]
