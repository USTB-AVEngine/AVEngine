#!/usr/bin/env python3
"""Approximate visual overlay of R8 nav geometry on the overhead RGB image."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.rooms.runtime_config import (
    RUNTIME_PREFIX,
    MAGNUM_SITE,
    RLR_SDK_ROOT,
    MP3D_ROOT,
    TASKS_ROOT,
    MEDIA_ROOT,
    ROOM_PYTHON,
)
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/smy/room_split_experiments/00006_20260907_v9")
GEOM = ROOT / "R8_geometry_overlay.png"
RGB = ROOT / "R8_overhead_rgb.png"
OUT = ROOT / "R8_geometry_rgb_overlay.png"


def main():
    rgb = Image.open(RGB).convert("RGBA")
    geom = Image.open(GEOM).convert("RGBA")
    # Remove the geometry title strip and fit the remaining map to the RGB canvas.
    geom_map = geom.crop((0, 110, geom.width, geom.height))
    geom_map = geom_map.resize(rgb.size, Image.NEAREST)
    # White background in the geometry layer should remain transparent; retain colored nav/boxes.
    pix = geom_map.load()
    for y in range(geom_map.height):
        for x in range(geom_map.width):
            r, g, b, a = pix[x, y]
            if r > 242 and g > 242 and b > 242:
                pix[x, y] = (255, 255, 255, 0)
            else:
                pix[x, y] = (r, g, b, 105)
    result = Image.alpha_composite(rgb, geom_map).convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        f = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 24
        )
    except OSError:
        f = ImageFont.load_default()
    draw.rectangle((10, 10, 660, 55), fill=(255, 255, 255))
    draw.text(
        (20, 18), "R8 几何/导航区域与真实 RGB 俯视图叠加（近似）", fill="black", font=f
    )
    result.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
