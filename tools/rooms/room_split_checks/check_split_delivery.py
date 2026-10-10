"""Claude 10-10 v4 (= v3 + MP3D navmesh path + NAVMESH_MAP override): independent check of a split delivery under the owner's current rules.
- kept rooms: 6 <= area <= CAP (default 35), short side >= 2.4
- connectivity, the seam/walkable rule: raw parts within 0.05 m of the main group are merged (floor seams);
  each remaining "far" part >= 0.3 m2 must be reachable from the main group by a short walk on the original navmesh
  (geodesic <= straight gap + 1.0 m); a part with no navmesh support or only a long detour is a real enclave
- 0.6 m passage test on the hole-filled outline (same as v2), reported with whether the cut-off piece is a far part
- designed cut lines: legs after merging collinear pieces (2 deg), furniture crossing length, stage
usage: python tools/rooms/room_split_checks/check_split_delivery.py <delivery dir> <out.json>
"""
import collections
import json
import math
import os
import sys

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import nearest_points, unary_union

CAP = float(os.environ.get("CAP", 35))
DELIV, OUT = sys.argv[1:3]


def geom(x):
    return shape(json.loads(x) if isinstance(x, str) else x)


def parts(g):
    return [g] if g.geom_type == "Polygon" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]


def groups(g):
    """main group = largest raw part plus everything chained to it by gaps <= 0.05 m; returns (main, far parts)"""
    ps = sorted(parts(g.buffer(0)), key=lambda p: -p.area)
    main, rest = [ps[0]], ps[1:]
    changed = True
    while changed:
        changed = False
        blob = unary_union(main)
        for p in list(rest):
            if p.distance(blob) <= 0.05:
                main.append(p)
                rest.remove(p)
                changed = True
    return unary_union(main), rest


def legs(line):
    dirs = []
    for c in ([line] if line.geom_type == "LineString" else list(line.geoms)):
        xy = list(c.coords)
        for a, b in zip(xy, xy[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if math.hypot(dx, dy) < 1e-6:
                continue
            ang = math.degrees(math.atan2(dy, dx)) % 180
            if dirs and min(abs(ang - dirs[-1]), 180 - abs(ang - dirs[-1])) < 2:
                continue
            dirs.append(ang)
    return len(dirs)


_pf = {}


def pathfinder(house):
    if house not in _pf:
        # same installed runtime as tools/acoustics/check_split_room_escape.py (habitat_sim is not in any env)
        from avengine.acoustics.runtime import RUNTIME_MODE_CURRENT_INSTALLED, load_habitat_runtime
        habitat_sim = _pf.get("_hs")
        habitat_sim = habitat_sim or load_habitat_runtime(
            runtime_prefix="/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z",
            magnum_python_site="/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages",
            rlr_sdk_root="/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg",
            runtime_mode=RUNTIME_MODE_CURRENT_INSTALLED)[0]
        _pf["_hs"] = habitat_sim
        # navmesh: NAVMESH_MAP json {house: path} first (Kujiale etc.), then MP3D and HM3D dataset layouts
        nav_map = json.load(open(os.environ["NAVMESH_MAP"])) if os.environ.get("NAVMESH_MAP") else {}
        if house in nav_map:
            nav = nav_map[house]
        elif house.startswith("mp3d_"):
            scan = house[len("mp3d_"):]
            nav = f"/data/datasets/habitat_data/scene_datasets/mp3d/{scan}/{scan}.navmesh"
        else:
            _, split, num, sid = house.split("_", 3)
            nav = f"/data/datasets/habitat_data/scene_datasets/hm3d/{split}/{num}-{sid}/{sid}.basis.navmesh"
        pf = habitat_sim.PathFinder()
        if not pf.load_nav_mesh(nav):
            raise RuntimeError(nav)
        _pf[house] = (pf, habitat_sim)
    return _pf[house]


def nav_points(pf, poly, y, step=0.15):
    """navigable points inside poly (snap within 0.1 m horizontally, same floor)"""
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    for x in np.arange(minx + step / 2, maxx, step):
        for z in np.arange(miny + step / 2, maxy, step):
            from shapely.geometry import Point
            if not poly.contains(Point(x, z)):
                continue
            s = np.array(pf.snap_point(np.array([x, y, z], dtype=np.float32)))
            if np.all(np.isfinite(s)) and math.hypot(s[0] - x, s[2] - z) <= 0.1 and abs(s[1] - y) < 0.5:
                pts.append(s)
    return pts


def walk_test(house, y, main, far):
    pf, hs = pathfinder(house)
    a, b = nearest_points(far, main)
    gap = a.distance(b)
    fp = nav_points(pf, far, y)
    if not fp:
        return dict(gap=round(gap, 2), nav_in_far=0, verdict="NO_NAV_IN_PART")
    mp = nav_points(pf, main.intersection(b.buffer(2.0)), y) or nav_points(pf, main, y, 0.3)
    if not mp:
        return dict(gap=round(gap, 2), nav_in_far=len(fp), verdict="NO_NAV_NEAR_MAIN")
    fp = sorted(fp, key=lambda s: math.hypot(s[0] - b.x, s[2] - b.y))[:6]
    mp = sorted(mp, key=lambda s: math.hypot(s[0] - a.x, s[2] - a.y))[:6]
    best = None
    for s in fp:
        for t in mp:
            sp = hs.ShortestPath()
            sp.requested_start, sp.requested_end = s, t
            if pf.find_path(sp) and math.isfinite(sp.geodesic_distance):
                eu = math.hypot(s[0] - t[0], s[2] - t[2])
                extra = sp.geodesic_distance - eu
                if best is None or extra < best[0]:
                    best = (extra, sp.geodesic_distance, eu)
    if best is None:
        return dict(gap=round(gap, 2), nav_in_far=len(fp), verdict="NO_PATH")
    extra, geo, eu = best
    return dict(gap=round(gap, 2), geodesic=round(geo, 2), straight=round(eu, 2), detour=round(extra, 2),
                verdict="DIRECT" if extra <= 1.0 else "DETOUR")


def opened(g):
    filled = unary_union([Polygon(p.exterior) for p in parts(g.buffer(0))])
    # close floor seams (gaps <= 0.05 m) first, as the seam rule says; then the 0.6 m passage test
    filled = filled.buffer(0.03, join_style=2).buffer(-0.03, join_style=2)
    filled = unary_union([Polygon(p.exterior) for p in parts(filled)])
    o = filled.buffer(-0.3, join_style=2).buffer(0.3, join_style=2)
    return sorted([p for p in parts(o) if p.area >= 0.05], key=lambda p: -p.area)


regs = [json.load(open(os.path.join(DELIV, "regions", n))) for n in sorted(os.listdir(os.path.join(DELIV, "regions")))]
split = [r for r in regs if r.get("requires_split")]
kept = [(r, b) for r in split for b in r.get("blocks", []) if b.get("decision") == "retain"]
out = dict(delivery=DELIV, cap=CAP, regions=len(regs), split_regions=len(split), kept=len(kept),
           kept_area_m2=round(sum(b["floor_area_m2"] for _, b in kept), 1))
bad_area, far_rows, narrow_rows = [], [], []
for r, b in kept:
    if not (6 - 1e-6 <= b["floor_area_m2"] <= CAP + 1e-6) or (b.get("short_side_m") or 0) < 2.4 - 1e-6:
        bad_area.append((b["id"], round(b["floor_area_m2"], 2), round(b.get("short_side_m") or 0, 2)))
    g = geom(b["floor_polygon_xz_m"])
    main, far = groups(g)
    for p in far:
        if p.area < 0.3:
            continue
        row = dict(id=b["id"], area=round(b["floor_area_m2"], 1), part=round(p.area, 2))
        row.update(walk_test(b["house"], float(b["floor_y_m"]), main, p))
        far_rows.append(row)
    comps = opened(g)
    if len(comps) > 1:
        for c in comps[1:]:
            far_hit = any(c.intersection(p).area > 0.5 * min(c.area, p.area) for p in far)
            narrow_rows.append(dict(id=b["id"], area=round(b["floor_area_m2"], 1), main=round(comps[0].area, 1),
                                    cut_off=round(c.area, 2), is_far_part=far_hit))
out["kept_outside_area_or_short_side"] = bad_area
out["far_parts_ge_0p3m2"] = len(far_rows)
out["far_parts_by_verdict"] = dict(collections.Counter(x["verdict"] for x in far_rows))
out["far_parts_not_direct"] = [x for x in far_rows if x["verdict"] != "DIRECT"]
out["far_parts_all"] = far_rows
out["narrow_0p6_cut_off"] = narrow_rows
drop, unres = collections.Counter(), []
for r in split:
    for b in r.get("blocks", []):
        if b.get("decision") == "discard":
            rs = b.get("discard_reasons") or ["?"]
            key = "STAIRS" if "STAIRS" in rs else ("DETACHED_FRAGMENT" if "DETACHED_FRAGMENT" in rs else
                  ("CORRIDOR" if any("CORRIDOR" in x for x in rs) else ("SMALL" if "FLOOR_AREA_BELOW_6" in rs else rs[0])))
            drop[key] += b["floor_area_m2"]
        elif b.get("decision") == "unresolved":
            unres.append((b["id"], round(b["floor_area_m2"], 1), b.get("unresolved_reasons")))
out["dropped_area_by_reason_m2"] = {k: round(v, 1) for k, v in drop.most_common()}
out["unresolved"] = unres
lines = [c for r in split for c in r.get("cut_lines", []) if c.get("active_in_final_partition", True)]
L = []
for c in lines:
    g = c.get("line_geometry_xz_m")
    L.append(dict(id=c["id"], stage=c.get("stage"), legs=legs(geom(g)) if g else None,
                  furniture_m=round(c.get("furniture_intersection_length_m") or 0, 2)))
out["active_cut_lines"] = len(L)
out["cut_legs_hist"] = dict(collections.Counter(x["legs"] for x in L))
out["cut_stage_hist"] = dict(collections.Counter(x["stage"] for x in L))
out["cut_furniture_over_0p5m"] = sum(1 for x in L if x["furniture_m"] > 0.5)
out["cut_furniture_over_1m"] = sum(1 for x in L if x["furniture_m"] > 1.0)
out["cut_furniture_max_m"] = max((x["furniture_m"] for x in L), default=None)
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
short = {k: v for k, v in out.items() if k not in ("far_parts_all",)}
print(json.dumps(short, ensure_ascii=False, indent=1)[:6000])
