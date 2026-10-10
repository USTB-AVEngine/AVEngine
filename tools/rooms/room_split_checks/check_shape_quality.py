"""Claude 10-10 v2 (= v1 + NODISK + WIDE_DROPPED): shape-quality check of split rooms.
All tests run on the room's own outline: raw parts, floor seams <= 0.05 m closed, holes filled. No navmesh, no neighbour floor.
  NECK      0.6 m opening (mitre) splits off a piece >= 0.5 m2  -> the room hangs a piece behind a passage narrower than 0.6 m
  CORRIDOR  the part narrower than 1.5 m (outline minus its 0.75 m mitre opening) has a connected piece >= 3 m2 whose
            minimum rotated rectangle is >= 3.5 m long  -> a hallway strip is kept inside the room
  WRAP      the room's convex hull covers >= 30% of another kept room of the same region and floor  -> L/U room around another
  FURNITURE an active cut line crosses > 0.5 m of furniture (furniture_intersection_length_m in the delivery)
  NODISK    (v2) a kept room's own outline cannot hold a 2.4 m disk (maximum inscribed circle radius < 1.2 m)
  WIDE_DROPPED (v2) a discarded piece (any reason but STAIRS) still holds a wide body: its 0.75 m mitre opening has a
            connected piece >= 6 m2 that holds a 2.4 m disk -> real room floor was thrown away (e.g. as CORRIDOR)
usage: python tools/rooms/room_split_checks/check_shape_quality.py <delivery dir> <out.json>
"""
import json
import math
import os
import sys

import shapely
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

DELIV, OUT = sys.argv[1:3]


def parts(g):
    return [g] if g.geom_type == "Polygon" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]


def outline(x):
    g = shape(json.loads(x) if isinstance(x, str) else x).buffer(0)
    p = unary_union([Polygon(q.exterior) for q in parts(g)])
    p = p.buffer(0.03, join_style=2).buffer(-0.03, join_style=2)
    return unary_union([Polygon(q.exterior) for q in parts(p)])


def long_side(p):
    xy = list(p.minimum_rotated_rectangle.exterior.coords)
    return max(math.dist(xy[i], xy[i + 1]) for i in range(4))


def disk_radius(p):
    return max((shapely.maximum_inscribed_circle(q, tolerance=0.0005).length for q in parts(p)), default=0.0)


flags = []
furn = []
nodisk = []
wide = []
n_regions = n_kept = 0
for name in sorted(os.listdir(os.path.join(DELIV, "regions"))):
    reg = json.load(open(os.path.join(DELIV, "regions", name)))
    if not reg.get("requires_split"):
        continue
    n_regions += 1
    key = f'{reg["house"]}/{reg["source_region"]}'
    kept = [b for b in reg.get("blocks", []) if b.get("decision") == "retain"]
    n_kept += len(kept)
    P = {b["id"]: outline(b["floor_polygon_xz_m"]) for b in kept}
    for b in kept:
        p = P[b["id"]]
        o = sorted([q for q in parts(p.buffer(-0.3, join_style=2).buffer(0.3, join_style=2)) if q.area >= 0.05], key=lambda q: -q.area)
        for q in o[1:]:
            if q.area >= 0.5:
                flags.append(dict(kind="NECK", region=key, room=b["id"], area=round(b["floor_area_m2"], 1), piece_m2=round(q.area, 2)))
        nar = [q for q in parts(p.difference(p.buffer(-0.75, join_style=2).buffer(0.75, join_style=2))) if q.area >= 3]
        for q in nar:
            L = long_side(q)
            if L >= 3.5:
                flags.append(dict(kind="CORRIDOR", region=key, room=b["id"], area=round(b["floor_area_m2"], 1), piece_m2=round(q.area, 2), length_m=round(L, 1)))
        hull = p.convex_hull
        for c in kept:
            if c is b or c["floor_id"] != b["floor_id"]:
                continue
            f = hull.intersection(P[c["id"]]).area / P[c["id"]].area
            if f >= 0.3:
                flags.append(dict(kind="WRAP", region=key, room=b["id"], area=round(b["floor_area_m2"], 1), covers=c["id"], covered_fraction=round(f, 2)))
    for b in kept:
        r = disk_radius(P[b["id"]])
        if r < 1.2:
            nodisk.append(dict(kind="NODISK", region=key, room=b["id"], area=round(b["floor_area_m2"], 1), radius_m=round(r, 3)))
    for b in reg.get("blocks", []):
        rs = b.get("discard_reasons") or []
        if b.get("decision") != "discard" or "STAIRS" in rs or (b.get("floor_area_m2") or 0) < 6:
            continue
        p = outline(b["floor_polygon_xz_m"])
        for q in parts(p.buffer(-0.75, join_style=2).buffer(0.75, join_style=2)):
            if q.area >= 6:
                r = disk_radius(q)
                if r >= 1.2:
                    wide.append(dict(kind="WIDE_DROPPED", region=key, block=b["id"], reasons=rs, area=round(b["floor_area_m2"], 1), wide_m2=round(q.area, 1), radius_m=round(r, 3)))
    for c in reg.get("cut_lines", []):
        if c.get("active_in_final_partition", True) and (c.get("furniture_intersection_length_m") or 0) > 0.5:
            fi = c.get("furniture_intersections")
            fi = json.loads(fi) if isinstance(fi, str) else (fi or [])
            furn.append(dict(kind="FURNITURE", region=key, cut=c["id"], stage=c.get("stage"), furniture_m=round(c["furniture_intersection_length_m"], 2),
                             categories=sorted({x.get("category") for x in fi})))
rooms = sorted({f["room"] for f in flags})
regions = sorted({f["region"] for f in flags} | {f["region"] for f in furn})
regions_v2 = sorted(set(regions) | {f["region"] for f in nodisk + wide})
out = dict(delivery=DELIV, split_regions=n_regions, kept=n_kept,
           counts={k: sum(1 for f in flags if f["kind"] == k) for k in ("NECK", "CORRIDOR", "WRAP")} | {"FURNITURE": len(furn)},
           flagged_rooms=len(rooms), flagged_regions=len(regions), rooms=rooms, regions=regions, flags=flags, furniture=furn,
           counts_v2=dict(NODISK=len(nodisk), WIDE_DROPPED=len(wide)), regions_v2=regions_v2, nodisk=nodisk, wide_dropped=wide)
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("split_regions", "kept", "counts", "flagged_rooms", "flagged_regions", "counts_v2")}, ensure_ascii=False))
