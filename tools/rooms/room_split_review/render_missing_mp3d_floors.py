"""Render whole-house MP3D overhead frames for floors that a review-site build had no base image for.
Reuses tools/rooms/room_split_auto/mp3d_render.render_house (CPU raster of the raw GLB, no GPU, no Habitat).
Set AVENGINE_BASIS_CPU_DECODER as in the MP3D split cpu_env.sh: without it every texture is unavailable and the frames are black.
usage: python tools/rooms/room_split_review/render_missing_mp3d_floors.py <build_receipt.json> <mp3d rooms json> <mp3d plan.json> <new out root> [workers]
Then add <out root>/house_renders to the MP3D house_render_dirs of the review config and build again.
"""
import json, sys, multiprocessing, traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tools.rooms.room_split_auto.mp3d_render import render_house

receipt, rooms_path, plan_path, out = sys.argv[1:5]
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
out = Path(out)
for d in ("house_renders", "region_renders"):
    (out / d).mkdir(parents=True, exist_ok=False)
missing = json.load(open(receipt))["families"]["MP3D"]["cards_missing_image"]
rooms = json.load(open(rooms_path))["rooms"]
plan = json.load(open(plan_path))
want = defaultdict(list)
for key in missing:
    _, house, region, floor = key.split("/")
    ys = [r["floor_y_m"] for r in rooms if r["house"] == house and r["source_region"] == region and r["floor_id"] == floor]
    if ys:
        want[house].append(float(np.median(ys)))
jobs = []
for house, ys in sorted(want.items()):
    levels = []
    for y in sorted(ys):
        if not any(abs(y - l["floor_y_m"]) <= 0.05 for l in levels):
            levels.append(dict(floor_y_m=y, rooms=[]))
    scan = house.split("_", 1)[1]
    jobs.append(dict(house=house, scene_glb=f"/data/datasets/habitat_data/scene_datasets/mp3d/{scan}/{scan}.glb",
                     frame_prefix="Yreview_v1_", levels=levels))
print("houses", len(jobs), "levels", sum(len(j["levels"]) for j in jobs), flush=True)
receipts = []
with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
    fs = {pool.submit(render_house, j, out, plan["adapter_root"], plan["parameters"]): j for j in jobs}
    for f in as_completed(fs):
        try:
            receipts.append(f.result())
        except Exception:
            receipts.append(dict(house=fs[f]["house"], error=traceback.format_exc()))
json.dump(dict(jobs=jobs, receipts=receipts), open(out / "render_receipt.json", "w"), indent=1)
print("errors", [r["house"] for r in receipts if "error" in r], flush=True)
