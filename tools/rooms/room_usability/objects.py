#!/usr/bin/env python3
"""Write the objects that stand in each listed room as a table, one row per object (read-only inputs, new output dir).

HM3D and MP3D objects keep the region they are annotated in. A region that was split hands each object to the nearest of
its rooms, measured to the hole-filled floor polygon and at most 0.75 m away; HM3D floor polygons have holes under
furniture, so a centre-in-polygon test would miss most of it. Kujiale furniture has no region label: each footprint goes
to the nearest room on its storey within 0.3 m, and its area is the minimum rotated rectangle, because the floor-height
slice cuts a table down to its legs. Habitat axes, y up.

usage: python tools/rooms/room_usability/objects.py --family hm3d|mp3d|kujiale --rooms LIST.json --out NEW_DIR
           [--hm3d-root DIR] [--mp3d-adapter-root DIR] [--kujiale-adapters DIR] [--workers N]
writes objects.csv (room, cat, role, foot = footprint m^2, top = height above the room floor, empty for Kujiale) and
objects_receipt.json. MP3D needs the authorised adapter checkout that holds tools/rooms/room_screening/mp3d_geometry.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union

SCHEMA = "avengine_room_usability_objects_v1"
REGION_GAP_M = 0.75
KUJIALE_GAP_M = 0.3
KUJIALE_STOREY_M = 0.3
STOREY_HEIGHT_M = 2.3
HM3D_ROOT = "/data/datasets/habitat_data/scene_datasets/hm3d"
COLUMNS = ("room", "cat", "role", "foot", "top")


def filled(geometry):
    """The room outline: floor polygons carry holes wherever furniture hid the floor from the scanner."""
    return unary_union([Polygon(g.exterior) for g in getattr(geometry, "geoms", [geometry])])


def box_object(category, role, region, triangles):
    v = triangles.reshape(-1, 3)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return dict(cat=category, role=role, region=int(region), cx=(lo[0] + hi[0]) / 2, cz=(lo[2] + hi[2]) / 2,
                ylo=float(lo[1]), yhi=float(hi[1]), foot=float((hi[0] - lo[0]) * (hi[2] - lo[2])))


def assign_by_region(rooms, objects, gap_m=REGION_GAP_M):
    """room id -> objects annotated in the room's source region on its storey; split regions go to the nearest room."""
    outline = {r["id"]: filled(shape(r["floor_polygon_xz_m"])) for r in rooms}
    assigned = defaultdict(list)
    for o in objects:
        cands = [r for r in rooms if int(r["source_region_id"]) == o["region"]
                 and o["yhi"] > float(r["floor_y_m"]) + 0.05 and o["ylo"] < float(r["floor_y_m"]) + STOREY_HEIGHT_M]
        if not cands:
            continue
        d, best = min((outline[r["id"]].distance(Point(o["cx"], o["cz"])), r["id"]) for r in cands)
        if d <= gap_m:
            assigned[best].append(o)
    return assigned


def assign_kujiale(rooms, furniture_by_height, gap_m=KUJIALE_GAP_M):
    """room id -> Kujiale furniture footprints, nearest room on the same storey; foot is the minimum rotated rectangle."""
    from tools.rooms.room_screening.geometry import classify_category
    outline = {r["id"]: filled(shape(r["floor_polygon_xz_m"])) for r in rooms}
    assigned = defaultdict(list)
    for key, items in furniture_by_height.items():
        storey = [r for r in rooms if abs(float(r["floor_y_m"]) - float(key)) <= KUJIALE_STOREY_M]
        for item in items:
            g = shape(item["geometry"])
            if g.is_empty or not storey:
                continue
            c = g.centroid if g.centroid.within(g) else g.representative_point()
            d, best = min((outline[r["id"]].distance(c), r["id"]) for r in storey)
            if d <= gap_m:
                cat = item["category"].replace("_", " ").lower()
                assigned[best].append(dict(cat=cat, role=classify_category(cat), foot=float(g.minimum_rotated_rectangle.area),
                                           yhi=None))
    return assigned


def table(rooms, assigned):
    return [dict(room=r["id"], cat=o["cat"], role=o["role"], foot=round(o["foot"], 3),
                 top="" if o["yhi"] is None else round(o["yhi"] - float(r["floor_y_m"]), 2))
            for r in rooms for o in assigned[r["id"]]]


def hm3d_house(job):
    from tools.rooms.room_screening.geometry import load_semantic_ground_and_instances
    scene = Path(job["scene_directory"])
    _, _, instances, *_ = load_semantic_ground_and_instances(scene, scene.name.split("-", 1)[1])
    objects = [box_object(i["category"], i["role"], i["region_id"], i["triangles"]) for i in instances]
    return table(job["rooms"], assign_by_region(job["rooms"], objects))


def mp3d_house(job):
    from tools.rooms.room_split_auto.mp3d_adapter import selection_adapter
    scene = selection_adapter(job["adapter_root"]).load_mp3d_scene(Path(job["scene_directory"]))
    objects = [box_object(i["category"], i["role"], i["region_id"], i["triangles"])
               for i in scene.instances if i["role"] in ("blocker", "review")]
    return table(job["rooms"], assign_by_region(job["rooms"], objects))


def kujiale_house(job):
    adapter = json.loads(Path(job["adapter"]).read_text())
    return table(job["rooms"], assign_kujiale(job["rooms"], adapter["furniture_by_height"]))


def jobs_for(family, rooms, args):
    by_house = defaultdict(list)
    for r in rooms:
        by_house[r["house"]].append(r)
    jobs = []
    for house, rs in sorted(by_house.items()):
        if family == "hm3d":
            _, split, num, scan = house.split("_", 3)
            scene = Path(args.hm3d_root) / split / f"{num}-{scan}"
            if not (scene / f"{scan}.semantic.txt").is_file():
                raise SystemExit(f"missing HM3D semantics for {house}: {scene}")
            jobs.append(dict(scene_directory=str(scene), rooms=rs))
        elif family == "mp3d":
            scenes = {str(Path(r["measurement_source"]).parent) for r in rs}
            if len(scenes) != 1:
                raise SystemExit(f"{house}: rooms point at {sorted(scenes)}")
            jobs.append(dict(adapter_root=args.mp3d_adapter_root, scene_directory=scenes.pop(), rooms=rs))
        else:
            adapter = Path(args.kujiale_adapters) / f"{house}.json"
            if not adapter.is_file():
                raise SystemExit(f"missing Kujiale scene adapter {adapter}")
            jobs.append(dict(adapter=str(adapter), rooms=rs))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, choices=("hm3d", "mp3d", "kujiale"))
    ap.add_argument("--rooms", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="new directory; refuses to overwrite an existing one")
    ap.add_argument("--hm3d-root", default=HM3D_ROOT)
    ap.add_argument("--mp3d-adapter-root")
    ap.add_argument("--kujiale-adapters")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.family == "mp3d" and not args.mp3d_adapter_root:
        ap.error("--mp3d-adapter-root is required for mp3d")
    if args.family == "kujiale" and not args.kujiale_adapters:
        ap.error("--kujiale-adapters is required for kujiale")
    raw = args.rooms.read_bytes()
    rooms = json.loads(raw)["rooms"]
    jobs = jobs_for(args.family, rooms, args)
    worker = dict(hm3d=hm3d_house, mp3d=mp3d_house, kujiale=kujiale_house)[args.family]
    if args.out.exists():
        sys.exit(f"refuse: {args.out} exists")
    args.out.mkdir(parents=True)
    rows = []
    if args.family == "kujiale" or args.workers <= 1:
        for job in jobs:
            rows.extend(worker(job))
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            for part in ex.map(worker, jobs):
                rows.extend(part)
    with (args.out / "objects.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    receipt = dict(schema=SCHEMA, family=args.family, rooms=str(args.rooms), rooms_sha256=hashlib.sha256(raw).hexdigest(),
                   room_count=len(rooms), house_count=len(jobs), object_rows=len(rows),
                   rooms_with_objects=len({r["room"] for r in rows}), region_gap_m=REGION_GAP_M,
                   kujiale_gap_m=KUJIALE_GAP_M, storey_height_m=STOREY_HEIGHT_M,
                   mp3d_adapter_root=args.mp3d_adapter_root, kujiale_adapters=args.kujiale_adapters,
                   hm3d_root=args.hm3d_root if args.family == "hm3d" else None)
    (args.out / "objects_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=1))
    print(f"{args.family}: {len(rows)} objects in {receipt['rooms_with_objects']}/{len(rooms)} rooms, {len(jobs)} houses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
