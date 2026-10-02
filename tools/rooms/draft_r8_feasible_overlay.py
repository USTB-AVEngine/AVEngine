#!/usr/bin/env python3
"""Draft R8 overlay using only floor-level navmesh cells as placeable regions."""
from pathlib import Path
import sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

SCENE = Path("/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.glb")
RGB = Path("/data/smy/room_split_experiments/00006_20260907_v9/R8_overhead_rgb.png")
OUT = Path("/data/smy/room_split_experiments/00006_20260907_v9/R8_feasible_draft_overlay.png")

def main():
    rt = prepare_installed_habitat_runtime(
        runtime_prefix="/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z",
        magnum_python_site="/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages",
        rlr_sdk_root="/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg",
        mp3d_root="/data/datasets/habitat_data", allow_mp3d_environment=False)
    hs = rt.habitat_sim; pf = hs.PathFinder()
    nav = SCENE.with_suffix(".basis.navmesh")
    if not pf.load_nav_mesh(str(nav)) or not pf.is_loaded: raise SystemExit("navmesh load failed")
    floor, mpp = 2.836, 0.05
    mask = np.asarray(pf.get_topdown_view(mpp, floor), dtype=np.uint8)
    lower, _upper = pf.get_bounds(); lx, lz = float(lower[0]), float(lower[2])
    image = Image.open(RGB).convert("RGBA"); W,H=image.size
    layer = Image.new("RGBA", image.size, (0,0,0,0)); ld=ImageDraw.Draw(layer)
    cx,cy,cz=(-4.897+0.775)/2,10.0,(-5.236+4.707)/2
    fx=W/(2*math.tan(math.radians(75)/2)); fy=fx; depth=cy-floor
    def proj(x,z): return (W/2-fx*(x-cx)/depth, H/2+fy*(z-cz)/depth)
    sofa=(-4.5,-0.8,-1.0,2.2); dining=(-4.5,-4.5,-1.0,-0.8); original=(-4.897,-5.236,0.775,4.707)
    # First show all navigable cells inside the original R8 bbox as unassigned gray.
    # Then color only the cells assigned to the two candidate subregions.
    for row,col in zip(*np.where(mask > 0)):
        x0=lx+col*mpp; z0=lz+row*mpp; x=x0+mpp/2; z=z0+mpp/2
        if not (original[0] <= x <= original[2] and original[1] <= z <= original[3]): continue
        box = sofa if sofa[0] <= x <= sofa[2] and sofa[1] <= z <= sofa[3] else dining if dining[0] <= x <= dining[2] and dining[1] <= z <= dining[3] else None
        colour=(20,150,55,145) if box is sofa else (25,85,220,145) if box is dining else (150,150,150,115)
        pts=[proj(x0,z0),proj(x0+mpp,z0),proj(x0+mpp,z0+mpp),proj(x0,z0+mpp)]
        ld.polygon(pts, fill=colour)
    image=Image.alpha_composite(image,layer); draw=ImageDraw.Draw(image)
    def outline(box,colour):
        pts=[proj(box[0],box[1]),proj(box[2],box[1]),proj(box[2],box[3]),proj(box[0],box[3]),proj(box[0],box[1])]
        draw.line(pts,fill=colour,width=5,joint="curve")
    outline(original,(40,40,40)); outline(sofa,(20,150,55)); outline(dining,(25,85,220))
    font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font=ImageFont.truetype(font_path,22) if Path(font_path).is_file() else ImageFont.load_default()
    draw.rectangle((8,8,690,82),fill="white")
    draw.text((16,14),"R8 可行域切分草图（只填充导航网格可行走点）",fill="black",font=font)
    draw.text((16,48),"绿/蓝=已分配可行域；灰=可行但未分配；白=无导航网格",fill="black",font=font)
    image.convert("RGB").save(OUT); print(OUT)

if __name__ == "__main__": main()
