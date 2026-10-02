"""Read existing overheads and use the saved camera matrices, not axis guesses."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import GeometryCollection


def polygons(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for g in geometry.geoms:
            yield from polygons(g)


def project_xz(coords, floor_y, entry, image_record, size):
    points = np.asarray([[x, floor_y, z, 1] for x, z in coords])
    clip = (
        np.asarray(image_record["projection"]) @ np.asarray(entry["view"]) @ points.T
    ).T
    ndc = clip[:, :2] / clip[:, 3, None]
    return np.column_stack(
        [(ndc[:, 0] + 1) * size[0] / 2, (1 - ndc[:, 1]) * size[1] / 2]
    ).tolist()


def polygon_mask(geometry, floor_y, entry, image_record, size):
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for poly in polygons(geometry):
        d.polygon(
            [
                tuple(v)
                for v in project_xz(
                    poly.exterior.coords, floor_y, entry, image_record, size
                )
            ],
            fill=255,
        )
        for ring in poly.interiors:
            d.polygon(
                [
                    tuple(v)
                    for v in project_xz(ring.coords, floor_y, entry, image_record, size)
                ],
                fill=0,
            )
    return np.asarray(mask) > 0


def load_overhead(house, label, cache, base):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = cache / f"{house}__{label}.json"
    png = cache / f"{house}__{label}.png"
    origin = base.rstrip("/") + "/overheads/"
    if not meta.exists():
        with urllib.request.urlopen(
            origin + f"{house}/{label}.json", timeout=20
        ) as response:
            data = response.read()
        # JSON must decode before any cache is published.
        json.loads(data)
        meta.write_bytes(data)
    entry = json.loads(meta.read_text())
    im = next(v for v in entry["images"] if v["path"].endswith("_overview.png"))
    if not png.exists():
        with urllib.request.urlopen(origin + im["path"], timeout=20) as response:
            data = response.read()
        image = Image.open(io.BytesIO(data))
        image.load()
        png.write_bytes(data)
    image = Image.open(png).convert("RGB")
    return (
        entry,
        im,
        image,
        dict(
            metadata_url=origin + f"{house}/{label}.json",
            image_url=origin + im["path"],
            metadata_cache=str(meta),
            image_cache=str(png),
            orientation=entry["orientation"],
        ),
    )


def black_metric(geometry, floor_y, entry, im, image, p):
    if abs(floor_y - entry["floor_y_m"]) > p["floor_height_separation_m"]:
        return dict(
            black_fraction=None,
            status="OVERHEAD_FLOOR_MISMATCH",
            image_floor_y_m=entry["floor_y_m"],
        )
    mask = polygon_mask(geometry, floor_y, entry, im, image.size)
    rgb = np.asarray(image)
    if not mask.any():
        return dict(black_fraction=None, status="OVERHEAD_MASK_EMPTY")
    return dict(
        black_fraction=float(
            np.all(rgb[mask] < p["black_pixel_channel_lt"], axis=1).mean()
        ),
        status="measured",
        polygon_pixel_count=int(mask.sum()),
        px_per_m=image.width / im["span_m"],
        method="overview RGB; world polygon projected with stored projection/view matrices; polygon holes excluded",
    )


def overlay(path, scope, floor_y, parts, overhead):
    geometry_only = overhead is None
    if geometry_only:
        # A plan of measured polygons is useful even when no scan RGB exists.
        # The virtual mapping is for drawing ONLY: it is never cached as a
        # camera measurement and must never enter black_metric().
        x0, z0, x1, z1 = scope.bounds
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        span = max(x1 - x0, z1 - z0) * 1.1
        entry = {
            "view": [
                [1, 0, 0, -cx],
                [0, 0, 1, -cz],
                [0, -1, 0, floor_y + 30],
                [0, 0, 0, 1],
            ]
        }
        im = {"projection": np.diag([2 / span, 2 / span, 1, 1]).tolist()}
        image = Image.new("RGB", (1024, 1024), "white")
        mask = Image.fromarray(
            (polygon_mask(scope, floor_y, entry, im, image.size) * 255).astype("uint8")
        )
        image.paste((225, 225, 225), mask=mask)
    else:
        entry, im, image, _ = overhead
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    colours = [(45, 170, 235), (240, 80, 50), (60, 200, 100), (180, 80, 225)]
    for i, g in enumerate(parts):
        rgb = colours[i % len(colours)]
        # Raster masks retain holes and disconnected components.
        mask = Image.fromarray(
            (polygon_mask(g, floor_y, entry, im, image.size) * 65).astype("uint8")
        )
        paint = Image.new("RGBA", image.size, (*rgb, 0))
        paint.putalpha(mask)
        layer.alpha_composite(paint)
        d = ImageDraw.Draw(layer)
        for poly in polygons(g):
            xy = project_xz(poly.exterior.coords, floor_y, entry, im, image.size)
            d.line([tuple(q) for q in xy], fill=(*rgb, 255), width=3)
    canvas = Image.alpha_composite(canvas, layer)
    annotations = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), 20)
        if font_path.exists()
        else ImageFont.load_default()
    )
    title = (
        "GEOMETRY ONLY - no matching scan overhead; scan quality UNKNOWN"
        if geometry_only
        else "CPU split proposal; independent human review required"
    )
    annotations.text(
        (12, 12),
        title,
        fill="black" if geometry_only else "white",
        font=font,
        stroke_width=1,
        stroke_fill="white" if geometry_only else "black",
    )
    for i, g in enumerate(parts):
        point = g.representative_point()
        pixel = project_xz([(point.x, point.y)], floor_y, entry, im, image.size)[0]
        annotations.text(
            tuple(pixel),
            f"S{i}",
            fill="white",
            font=font,
            stroke_width=2,
            stroke_fill="black",
        )
    if geometry_only:
        annotations.text(
            (12, image.height - 32),
            f"World plan: +X right, +Z up; floor_y={floor_y:.3f} m",
            fill="black",
            font=font,
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path)
    return True
