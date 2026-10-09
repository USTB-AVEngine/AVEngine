"""JPEG whole-house CPU frames and metadata consumed by the owner's overlay tool."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time
import traceback
import numpy as np
from PIL import Image
from tools.rooms.room_split_auto.mp3d_run import dump, bounded, now


def projection(cx, cz, fy, span, size):
    return dict(view=[[1, 0, 0, -cx], [0, 0, -1, cz], [0, 1, 0, -fy], [0, 0, 0, 1]],
                projection=np.diag([2 / span, 2 / span, 1, 1]).tolist(),
                floor_y_m=float(fy), span_m=float(span), size_px=[size, size])


def cpu_renderer(adapter_root):
    path = Path(adapter_root) / "tools/rooms/room_selection/cpu_overhead.py"
    spec = importlib.util.spec_from_file_location("_room_split_mp3d_cpu_overhead", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def region_meta(room, frame, directory):
    record = dict(images=[dict(path=frame["path"], projection=frame["projection"], span_m=frame["span_m"])],
                  view=frame["view"], size_px=frame["size_px"], floor_y_m=room["floor_y_m"],
                  span_m=frame["span_m"], whole_house_frame_metadata=frame.get("metadata_path"),
                  renderer=frame["renderer"], black_source_path=frame.get("black_source_path", frame["path"]),
                  section_floor_y_m=frame["floor_y_m"], section_vertical_range_m=frame.get("section_vertical_range_m"),
                  source_room_id=room["room_id"], full_house_loaded=True,
                  room_geometry_crop=False, outside_room_mask=False, GPU_used=False)
    stem = room["house"] + "__" + room["room_label"] + "__" + room["selected_floor_id"]
    dump(Path(directory) / (stem + ".json"), record)
    return str(Path(directory) / (stem + ".json"))


def render_house(job, root, adapter_root, p):
    bounded()
    start = time.time()
    root = Path(root)
    module = cpu_renderer(adapter_root)
    renderer = module.CPUOverhead(job["scene_glb"])
    if renderer.basis_error:
        raise RuntimeError(renderer.basis_error)
    lows = np.stack([t[:, :, [0, 2]].min(axis=(0, 1)) for t, _, _, _ in renderer.items]).min(0)
    highs = np.stack([t[:, :, [0, 2]].max(axis=(0, 1)) for t, _, _, _ in renderer.items]).max(0)
    cx, cz = ((lows + highs) / 2).tolist()
    span = float((highs - lows).max() * 1.08)
    size = 1400
    frames = []
    for i, level in enumerate(job["levels"]):
        fy = level["floor_y_m"]
        rgb = np.zeros((size, size, 3), np.uint8)
        height = np.full((size, size), -np.inf)
        known = np.ones((size, size), bool)
        low, high = fy - p["floor_height_separation_m"], fy + p["camera_height_m"]
        for t, uv, tex, solid in renderer.items:
            module.raster(t, uv, tex, solid, span, cx, cz, low, high, rgb, height, known)
        path = root / "house_renders" / (job["house"] + f"__Y{i:03d}.jpg")
        with path.open("xb") as f:
            Image.fromarray(rgb).save(f, format="JPEG", quality=88)
        record = dict(projection(cx, cz, fy, span, size), path=str(path),
                      whole_house_xz_bounds=[lows.tolist(), highs.tolist()],
                      renderer="existing MP3D room_selection CPUOverhead/raster; original GLB UV textures",
                      scene_glb=job["scene_glb"], input_scope="complete raw whole-house mesh",
                      section_vertical_range_m=[low, high], black_source_path=str(path),
                      unavailable_texture_pixels_full_house=int((~known & np.isfinite(height)).sum()),
                      basis_decoder_error=renderer.basis_error, GPU_used=False, Simulator_created=False,
                      habitat_rgb_equivalence="unverified", pid=os.getpid(),
                      nice=os.getpriority(os.PRIO_PROCESS, 0), generated_at_utc=now())
        record["metadata_path"] = str(path.with_suffix(".json"))
        dump(path.with_suffix(".json"), record)
        for room in level["rooms"]:
            region_meta(room, record, root / "region_renders")
        frames.append(record["metadata_path"])
    print("MP3D_RENDER_HOUSE", job["house"], "FRAMES", len(frames), flush=True)
    return dict(house=job["house"], frames=frames, elapsed_s=time.time() - start,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0))


def run(root, workers=2):
    bounded()
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    for folder in ("region_renders", "house_renders"):
        (root / folder).mkdir(exist_ok=False)
    originals = [r for j in plan["jobs"] for r in j["rooms"]]
    existing = root / "mp3d_existing_rooms_connectivity_v1/regions"
    wanted = [r for r in originals if r["floor_area_m2"] > 35 + 1e-8 or
              json.loads((existing / (r["house"] + "__" + r["room_label"] + ".json")).read_text())
              .get("metrics", {}).get("affected", False)]
    cached = {}
    # Reuse the already verified whole-house cap35 raw renders, converting only our copy to JPEG.
    for room in originals:
        path = Path(plan["prep"]) / "renders_cap35_v1" / (room["house"] + "__" + room["room_label"]) / "render.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        fy = room["floor_y_m"]
        cx, cz = d["image_center_xz_m"]
        raw = Path(d["raw_png"])
        size = d["image_size_hw"][0]
        target = root / "house_renders" / (room["house"] + "__Yreuse_" + room["room_label"] + ".jpg")
        Image.open(raw).convert("RGB").save(target, quality=88)
        frame = dict(projection(cx, cz, fy, d["span_m"], size), path=str(target),
                     black_source_path=str(raw), metadata_path=str(target.with_suffix(".json")),
                     renderer=d["renderer"], source_render_metadata=str(path),
                     source_render_readonly=True, full_house_loaded=True, GPU_used=False,
                     section_vertical_range_m=d["current_floor_vertical_section_m"])
        dump(target.with_suffix(".json"), frame)
        cached.setdefault(room["house"], []).append(frame)
    jobs = {}
    reused = []
    for room in wanted:
        candidates = [(abs(x["floor_y_m"] - room["floor_y_m"]), x)
                      for x in cached.get(room["house"], []) if abs(x["floor_y_m"] - room["floor_y_m"]) <= .05]
        if candidates:
            _, frame = min(candidates, key=lambda q: q[0])
            path = region_meta(room, frame, root / "region_renders")
            reused.append(dict(room=room["room_id"], metadata=path, source=frame.get("source_render_metadata")))
            continue
        job = jobs.setdefault(room["house"], dict(house=room["house"], scene_glb=room["whole_house_visual_source"], levels=[]))
        level = next((x for x in job["levels"] if abs(x["floor_y_m"] - room["floor_y_m"]) <= .05), None)
        if level is None:
            level = dict(floor_y_m=room["floor_y_m"], rooms=[])
            job["levels"].append(level)
        level["rooms"].append(room)
    receipts = []
    dump(root / "render_plan.json", dict(wanted_regions=len(wanted), reused=reused, jobs=list(jobs.values()),
                                       workers=workers, original_pixels="whole raw textured house; floor roof section"))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = {pool.submit(render_house, j, root, plan["adapter_root"], plan["parameters"]): j for j in jobs.values()}
        for f in as_completed(futures):
            try:
                receipts.append(f.result())
            except Exception:
                receipts.append(dict(house=futures[f]["house"], error=traceback.format_exc()))
    dump(root / "render_receipt.json", dict(reused=reused, newly_rendered=receipts))
    missing = []
    for room in wanted:
        path = root / "region_renders" / (room["house"] + "__" + room["room_label"] + "__" + room["selected_floor_id"] + ".json")
        if not path.exists():
            missing.append(room["room_id"])
    dump(root / "render_validation.json", dict(wanted_regions=len(wanted), metadata_count=len(list((root / "region_renders").glob("*.json"))),
                                              missing_regions=missing, passed=not missing,
                                              habitat_rgb_equivalence="unverified; renderer matches authorised MP3D preparation",
                                              GPU_used=False, finished_at_utc=now()))
    print("MP3D_RENDER_DONE", len(wanted), "MISSING", missing, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error("workers must be 1..8")
    run(args.root, args.workers)
