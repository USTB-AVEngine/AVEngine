"""Pure rendering helpers for text-selective 360-degree DoA review videos."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np


PANEL_WIDTH = 640
PANEL_HEIGHT = 480
FRAME_RATE_HZ = 15.0
SOURCE_COLORS = ((32, 102, 184), (224, 112, 31))
INK = (28, 32, 38)
MUTED_INK = (94, 101, 112)
GRID = (218, 222, 228)
BACKGROUND = (248, 249, 251)
CARD_BACKGROUND = (255, 255, 255)


def normalize_360(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map angles to [0, 360), preserving array shape."""

    result = np.mod(np.asarray(values, dtype=np.float64), 360.0)
    result[np.isclose(result, 360.0)] = 0.0
    return result


def signed_degrees(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map angles to [-180, 180), for a readable vertical plot axis."""

    result = normalize_360(values)
    return (result + 180.0) % 360.0 - 180.0


def circular_error_deg(
    predicted: np.ndarray | Sequence[float],
    target: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return shortest-path absolute angular error in degrees."""

    predicted_array = normalize_360(predicted)
    target_array = normalize_360(target)
    difference = np.abs(predicted_array - target_array)
    return np.minimum(difference, 360.0 - difference)


def compass_point(
    angle_deg: float,
    *,
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    """Project AVEngine azimuth to screen coordinates.

    The native convention is front=0 degrees, right=+90 degrees,
    rear=180 degrees and left=-90/270 degrees.
    """

    radians = math.radians(float(angle_deg))
    return (
        center[0] + radius * math.sin(radians),
        center[1] - radius * math.cos(radians),
    )


def continuous_segments(
    values: np.ndarray | Sequence[float],
) -> list[tuple[int, int]]:
    """Split a signed angular track at the +/-180-degree display seam."""

    signed = signed_degrees(values)
    if signed.size == 0:
        return []
    breaks = np.flatnonzero(np.abs(np.diff(signed)) > 180.0) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [signed.size]))
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if stop > start
    ]


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    family = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(family, size=size)
    except OSError:
        return ImageFont.load_default()


def _dashed_line(
    draw,
    start: tuple[float, float],
    stop: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int,
    dash: float = 7.0,
    gap: float = 5.0,
) -> None:
    dx = stop[0] - start[0]
    dy = stop[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance <= 0.0:
        return
    ux = dx / distance
    uy = dy / distance
    cursor = 0.0
    while cursor < distance:
        segment_stop = min(cursor + dash, distance)
        draw.line(
            (
                start[0] + ux * cursor,
                start[1] + uy * cursor,
                start[0] + ux * segment_stop,
                start[1] + uy * segment_stop,
            ),
            fill=fill,
            width=width,
        )
        cursor += dash + gap


def _draw_compass(
    draw,
    *,
    bounds: tuple[int, int, int, int],
    title: str,
    caption: str,
    target_deg: float,
    predicted_deg: float,
    mean_error_deg: float,
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=10, fill=CARD_BACKGROUND, outline=GRID)
    draw.text(
        (left + 10, top + 7),
        title,
        fill=color,
        font=_font(15, bold=True),
    )
    draw.text(
        (left + 10, top + 27),
        caption,
        fill=MUTED_INK,
        font=_font(12),
    )
    center = ((left + right) / 2.0, top + 126.0)
    radius = 72.0
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline=GRID,
        width=2,
    )
    draw.line(
        (center[0] - radius, center[1], center[0] + radius, center[1]),
        fill=GRID,
        width=1,
    )
    draw.line(
        (center[0], center[1] - radius, center[0], center[1] + radius),
        fill=GRID,
        width=1,
    )
    draw.text(
        (center[0] - 5, center[1] - radius - 16),
        "F",
        fill=INK,
        font=_font(11),
    )
    draw.text((center[0] + radius + 4, center[1] - 7), "R", fill=INK, font=_font(11))
    draw.text((center[0] - 5, center[1] + radius + 2), "B", fill=INK, font=_font(11))
    draw.text((center[0] - radius - 13, center[1] - 7), "L", fill=INK, font=_font(11))

    target_point = compass_point(target_deg, center=center, radius=radius - 4)
    predicted_point = compass_point(
        predicted_deg,
        center=center,
        radius=radius - 4,
    )
    draw.line((*center, *target_point), fill=color, width=4)
    draw.ellipse(
        (
            target_point[0] - 5,
            target_point[1] - 5,
            target_point[0] + 5,
            target_point[1] + 5,
        ),
        fill=color,
    )
    _dashed_line(
        draw,
        center,
        predicted_point,
        fill=color,
        width=3,
    )
    draw.ellipse(
        (
            predicted_point[0] - 6,
            predicted_point[1] - 6,
            predicted_point[0] + 6,
            predicted_point[1] + 6,
        ),
        fill=CARD_BACKGROUND,
        outline=color,
        width=3,
    )
    frame_error = float(circular_error_deg([predicted_deg], [target_deg])[0])
    footer = (
        f"GT {target_deg:6.1f}°   Pred {predicted_deg:6.1f}°   "
        f"Err {frame_error:5.1f}°"
    )
    draw.text((left + 10, bottom - 37), footer, fill=INK, font=_font(11))
    draw.text(
        (left + 10, bottom - 19),
        f"5s mean error: {mean_error_deg:.2f}°",
        fill=MUTED_INK,
        font=_font(10),
    )


def _plot_point(
    index: int,
    angle_deg: float,
    *,
    frame_count: int,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    left, top, right, bottom = bounds
    x = left if frame_count <= 1 else left + index * (right - left) / (frame_count - 1)
    signed = float(signed_degrees([angle_deg])[0])
    y = top + (180.0 - signed) * (bottom - top) / 360.0
    return x, y


def _draw_track(
    draw,
    values: np.ndarray,
    *,
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int,
    dashed: bool,
) -> None:
    for start, stop in continuous_segments(values):
        points = [
            _plot_point(
                index,
                values[index],
                frame_count=len(values),
                bounds=bounds,
            )
            for index in range(start, stop)
        ]
        if len(points) == 1:
            draw.point(points[0], fill=color)
            continue
        if dashed:
            for first, second in zip(points[:-1], points[1:]):
                _dashed_line(
                    draw,
                    first,
                    second,
                    fill=color,
                    width=width,
                    dash=5.0,
                    gap=4.0,
                )
        else:
            draw.line(points, fill=color, width=width)


def render_panel_frame(
    *,
    frame_index: int,
    targets_deg: np.ndarray,
    predictions_deg: np.ndarray,
    captions: Sequence[str],
    sample_id: str,
    output_path: Path,
) -> None:
    """Render one 640x480 GT-versus-prediction review panel."""

    from PIL import Image, ImageDraw

    targets = normalize_360(targets_deg)
    predictions = normalize_360(predictions_deg)
    if targets.shape != predictions.shape or targets.shape[0] != 2:
        raise ValueError("expected matching [2, frames] target/prediction arrays")
    if len(captions) != 2 or not 0 <= frame_index < targets.shape[1]:
        raise ValueError("caption count or frame index is invalid")

    image = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text(
        (12, 8),
        "Text-selective binaural DoA",
        fill=INK,
        font=_font(18, bold=True),
    )
    draw.text(
        (12, 31),
        f"{sample_id}  |  frame {frame_index + 1:02d}/{targets.shape[1]}",
        fill=MUTED_INK,
        font=_font(10),
    )

    mean_errors = [
        float(np.mean(circular_error_deg(predictions[index], targets[index])))
        for index in range(2)
    ]
    compass_bounds = ((8, 50, 316, 276), (324, 50, 632, 276))
    for source_index, bounds in enumerate(compass_bounds):
        _draw_compass(
            draw,
            bounds=bounds,
            title=f"Query {source_index + 1}",
            caption=str(captions[source_index]),
            target_deg=float(targets[source_index, frame_index]),
            predicted_deg=float(predictions[source_index, frame_index]),
            mean_error_deg=mean_errors[source_index],
            color=SOURCE_COLORS[source_index],
        )

    plot_bounds = (54, 314, 626, 453)
    draw.rounded_rectangle((8, 285, 632, 472), radius=10, fill=CARD_BACKGROUND, outline=GRID)
    draw.text((18, 292), "DoA trajectory over 5 seconds", fill=INK, font=_font(13, bold=True))
    for angle in (-180, -90, 0, 90, 180):
        _, y = _plot_point(
            0,
            angle,
            frame_count=targets.shape[1],
            bounds=plot_bounds,
        )
        draw.line((plot_bounds[0], y, plot_bounds[2], y), fill=GRID, width=1)
        draw.text((14, y - 6), f"{angle:+d}", fill=MUTED_INK, font=_font(9))
    draw.text((54, 454), "0s", fill=MUTED_INK, font=_font(9))
    draw.text((603, 454), "5s", fill=MUTED_INK, font=_font(9))

    for source_index, color in enumerate(SOURCE_COLORS):
        _draw_track(
            draw,
            targets[source_index],
            bounds=plot_bounds,
            color=color,
            width=3,
            dashed=False,
        )
        _draw_track(
            draw,
            predictions[source_index],
            bounds=plot_bounds,
            color=color,
            width=2,
            dashed=True,
        )
    cursor_x, _ = _plot_point(
        frame_index,
        0.0,
        frame_count=targets.shape[1],
        bounds=plot_bounds,
    )
    draw.line(
        (cursor_x, plot_bounds[1], cursor_x, plot_bounds[3]),
        fill=INK,
        width=2,
    )
    draw.text(
        (440, 292),
        "solid/filled = GT   dashed/open = prediction",
        fill=MUTED_INK,
        font=_font(9),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
