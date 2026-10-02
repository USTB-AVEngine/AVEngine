#!/usr/bin/env python3
"""Temporary R8 geometry/navmesh overlay; does not modify source data."""

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
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

SCENE = Path(
    "/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.glb"
)
NAV = SCENE.with_suffix(".basis.navmesh")
OUT = Path("/data/smy/room_split_experiments/00006_20260907_v9/R8_geometry_overlay.png")


def fnt(size):
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=RUNTIME_PREFIX,
        magnum_python_site=MAGNUM_SITE,
        rlr_sdk_root=RLR_SDK_ROOT,
        mp3d_root=MP3D_ROOT,
        allow_mp3d_environment=False,
    )
    hs = runtime.habitat_sim
    config = hs.SimulatorConfiguration()
    config.scene_id = str(SCENE)
    config.load_semantic_mesh = False
    config.enable_physics = True
    if runtime.physics_config_path:
        config.physics_config_file = str(runtime.physics_config_path)
    sim = hs.Simulator(hs.Configuration(config, [hs.agent.AgentConfiguration()]))
    pf = sim.pathfinder
    print("nav", NAV, NAV.is_file(), "pre", pf.is_loaded)
    ok = pf.load_nav_mesh(str(NAV))
    print("load", ok, "post", pf.is_loaded)
    if not ok or not pf.is_loaded:
        raise SystemExit("navmesh load failed")
    mpp, floor = 0.05, 2.836
    raw = np.asarray(pf.get_topdown_view(mpp, floor), dtype=np.uint8)
    low, _high = pf.get_bounds()
    low_x, low_z = float(low[0]), float(low[2])
    original = (-4.897, -5.236, 0.775, 4.707)
    sofa = (-4.5, -0.8, -1.0, 2.2)
    dining = (-4.5, -4.5, -1.0, -0.8)
    camera_sofa, camera_dining = (-2.75, 0.70), (-2.397, -1.986)
    # Make a white/green navmesh image. Raw rows increase with world Z.
    base = np.where(
        raw[..., None] > 0,
        np.array([210, 235, 215], dtype=np.uint8),
        np.array([250, 250, 250], dtype=np.uint8),
    )
    image = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(image)

    def p(x, z):
        return (int((x - low_x) / mpp), int((z - low_z) / mpp))

    def rectangle(box, colour, width=6):
        a, b = p(box[0], box[1]), p(box[2], box[3])
        draw.rectangle((a[0], a[1], b[0], b[1]), outline=colour, width=width)

    rectangle(original, (40, 40, 40), 8)
    rectangle(sofa, (30, 140, 60), 8)
    rectangle(dining, (30, 80, 210), 8)
    for point, colour, label in (
        (camera_sofa, (20, 100, 40), "相机 S"),
        (camera_dining, (20, 55, 170), "相机 D"),
    ):
        x, y = p(*point)
        draw.ellipse(
            (x - 10, y - 10, x + 10, y + 10), fill=colour, outline="white", width=2
        )
        draw.text((x + 14, y - 20), label, fill=colour, font=fnt(18))
    # Crop to R8 with a modest margin, using PIL's top-left coordinate convention.
    a, b = p(original[0] - 0.8, original[1] - 0.8), p(
        original[2] + 0.8, original[3] + 0.8
    )
    left, right = max(0, min(a[0], b[0])), min(image.width, max(a[0], b[0]))
    top, bottom = max(0, min(a[1], b[1])), min(image.height, max(a[1], b[1]))
    crop = image.crop((left, top, right, bottom)).resize(
        ((right - left) * 4, (bottom - top) * 4), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (crop.width, crop.height + 110), "white")
    canvas.paste(crop, (0, 110))
    cd = ImageDraw.Draw(canvas)
    cd.text((12, 12), "R8 实际导航几何叠加图", fill="black", font=fnt(26))
    cd.text(
        (12, 55),
        "绿色=可行走网格；白色=非导航像素；黑框=原始包围盒；彩框=切分区",
        fill=(50, 50, 50),
        font=fnt(20),
    )
    canvas.save(OUT)
    sim.close()
    print(OUT)


if __name__ == "__main__":
    main()
