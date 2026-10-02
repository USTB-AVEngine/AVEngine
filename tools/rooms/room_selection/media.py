"""Read existing overheads and use the saved camera matrices, not axis guesses."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
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
    if overhead is None:
        return False
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
    ImageDraw.Draw(canvas).text(
        (12, 12),
        "CPU geodesic / narrow-channel split proposal; human review required",
        fill="white",
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path)
    return True
