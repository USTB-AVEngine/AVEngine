#!/usr/bin/env python3
"""Project R8 world-coordinate boxes onto the matching overhead RGB image."""

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
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/smy/room_split_experiments/00006_20260907_v9")
RGB = ROOT / "R8_overhead_rgb.png"
OUT = ROOT / "R8_real_rgb_with_boxes.png"


def main():
    image = Image.open(RGB).convert("RGBA")
    W, H = image.size
    # Same overhead camera used to make R8_overhead_rgb.png.
    cx, cy, cz = (-4.897 + 0.775) / 2, 10.0, (-5.236 + 4.707) / 2
    floor_y = 2.836
    fx = W / (2.0 * math.tan(math.radians(75.0) / 2.0))
    fy = fx

    def project(x, z):
        depth = cy - floor_y
        return (W / 2 - fx * (x - cx) / depth, H / 2 + fy * (z - cz) / depth)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    def box(box, colour, width=6, alpha=0):
        x0, z0, x1, z1 = box
        pts = [
            project(x0, z0),
            project(x1, z0),
            project(x1, z1),
            project(x0, z1),
            project(x0, z0),
        ]
        if alpha:
            odraw.polygon(pts[:-1], fill=tuple(colour) + (alpha,))
        odraw.line(pts, fill=tuple(colour) + (255,), width=width, joint="curve")

    draw = ImageDraw.Draw(image)
    original = (-4.897, -5.236, 0.775, 4.707)
    sofa = (-4.5, -0.8, -1.0, 2.2)
    dining = (-4.5, -4.5, -1.0, -0.8)
    box(original, (35, 35, 35), 7)
    box(sofa, (20, 150, 55), 9, 65)
    box(dining, (25, 80, 220), 9, 65)
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    label_font = (
        ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)
        if Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").is_file()
        else ImageFont.load_default()
    )
    for (x, z), colour, label in [
        ((-2.75, 0.70), (20, 100, 40), "绿框中心"),
        ((-2.397, -1.986), (20, 55, 170), "蓝框中心"),
    ]:
        px, py = project(x, z)
        draw.ellipse(
            (px - 10, py - 10, px + 10, py + 10), fill=colour, outline="white", width=3
        )
        draw.text((px + 14, py - 18), label, fill=colour, font=label_font)
    draw.rectangle((8, 8, 430, 38), fill="white")
    draw.text(
        (16, 15),
        "真实 RGB 俯视图 + R8 区域边界（投影叠加）",
        fill="black",
        font=label_font,
    )
    draw.text((16, 55), "绿色框=餐桌侧；蓝色框=沙发侧", fill="black", font=label_font)
    image.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
