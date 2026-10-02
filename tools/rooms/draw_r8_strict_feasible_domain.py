#!/usr/bin/env python3
"""Strict R8 draft: original room bbox intersected with floor navmesh only."""
from pathlib import Path
import sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

SCENE=Path("/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.glb")
RGB=Path("/data/smy/room_split_experiments/00006_20260907_v9/R8_overhead_rgb.png")
OUT=Path("/data/smy/room_split_experiments/00006_20260907_v9/R8_strict_feasible_domain.png")

def main():
    rt=prepare_installed_habitat_runtime(runtime_prefix="/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z",magnum_python_site="/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages",rlr_sdk_root="/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg",mp3d_root="/data/datasets/habitat_data",allow_mp3d_environment=False)
    hs=rt.habitat_sim; pf=hs.PathFinder(); nav=SCENE.with_suffix(".basis.navmesh")
    if not pf.load_nav_mesh(str(nav)) or not pf.is_loaded: raise SystemExit("navmesh load failed")
    floor,mpp=2.836,0.05; mask=np.asarray(pf.get_topdown_view(mpp,floor),dtype=np.uint8); lower,_=pf.get_bounds(); lx,lz=float(lower[0]),float(lower[2])
    original=(-4.897,-5.236,0.775,4.707); image=Image.open(RGB).convert("RGBA"); W,H=image.size
    cx,cy,cz=(-4.897+0.775)/2,10.0,(-5.236+4.707)/2; f=W/(2*math.tan(math.radians(75)/2)); depth=cy-floor
    # Habitat camera's image-right axis is opposite world +X for this pose.
    x_offset_px = -55.0
    def proj(x,z): return (W/2-f*(x-cx)/depth+x_offset_px,H/2+f*(z-cz)/depth)
    cells=[]
    for row,col in zip(*np.where(mask>0)):
        x=lx+(col+0.5)*mpp; z=lz+(row+0.5)*mpp
        if original[0]<=x<=original[2] and original[1]<=z<=original[3]: cells.append((row,col))
    cellset=set(cells); components=[]
    while cellset:
        stack=[cellset.pop()]; comp=[]
        while stack:
            r,c=stack.pop(); comp.append((r,c))
            for n in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if n in cellset: cellset.remove(n); stack.append(n)
        components.append(comp)
    layer=Image.new("RGBA",image.size,(0,0,0,0)); ld=ImageDraw.Draw(layer); colors=[(30,150,80,145),(220,130,30,145),(150,60,180,145),(30,130,190,145)]
    for idx,comp in enumerate(components):
        colour=colors[idx%len(colors)]
        for row,col in comp:
            x0=lx+col*mpp; z0=lz+row*mpp; x1=x0+mpp; z1=z0+mpp
            ld.polygon([proj(x0,z0),proj(x1,z0),proj(x1,z1),proj(x0,z1)],fill=colour)
    image=Image.alpha_composite(image,layer); draw=ImageDraw.Draw(image)
    p=[proj(original[0],original[1]),proj(original[2],original[1]),proj(original[2],original[3]),proj(original[0],original[3]),proj(original[0],original[1])]; draw.line(p,fill=(35,35,35),width=7)
    fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; font=ImageFont.truetype(fp,22) if Path(fp).is_file() else ImageFont.load_default()
    draw.rectangle((8,8,710,82),fill="white"); draw.text((16,14),"R8 严格可行域草图",fill="black",font=font); draw.text((16,48),f"仅显示：原始 R8 范围内 ∩ {floor:.3f}m 楼层导航网格；连通块={len(components)}",fill="black",font=font)
    image.convert("RGB").save(OUT); print(OUT); print("cells",len(cells),"components",len(components))

if __name__=="__main__": main()
