#!/usr/bin/env python3
"""Build the tiled review site for final room lists (HM3D / MP3D / Kujiale) so one reviewer can mark every room.

One card per source region: the real overhead render cropped around the region, every listed room in its own colour with a
large number, dropped pieces hatched, unresolved or unlisted pieces cross-hatched, neighbouring listed rooms as thin outlines.
Under the image each listed room has OK / wrong / unsure buttons, reason chips and a note. The page works opened as a file
(browser storage plus JSON export/import) and, when served by serve_review.py, autosaves every change to a JSON file.

usage: python tools/rooms/room_split_review/build_review_site.py --config CONFIG.json --out NEW_DIR [--workers N]
config:
  {"title": str, "build_id": str, "lead_html": str (optional),
   "families": [{"name": "HM3D", "rooms": "<json with a rooms array>",
                 "region_dirs": [...], "house_render_dirs": [...], "region_render_dirs": [...],
                 "leakage_csvs": [...], "prior_verdict_dir": "<optional dir of {house}__{region}.json>"}]}
Region JSON files carry blocks with id, decision (retain/discard/unresolved), floor_id, floor_y_m, floor_polygon_xz_m.
Renders are orthographic overhead images with view/projection/size_px/floor_y_m (house renders also span_m and path).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import html
import json
import os
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box as shapely_box
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "review_page.html"
SERVER = HERE / "serve_review.py"
SCHEMA = "avengine_room_split_review_site_v1"
PALETTE = [(31, 119, 230), (240, 120, 30), (40, 170, 90), (150, 70, 210), (230, 190, 0), (0, 180, 200), (220, 60, 150),
           (140, 90, 40), (100, 150, 30), (60, 60, 200)]
FONT_CANDIDATES = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
FLOOR_MATCH_M = 0.5
MARGIN_M = 1.5
REASONS = {
    "two_rooms": "其实是两间或更多间，应该再切",
    "bad_cut": "切线位置不对（穿家具、不在墙或门口）",
    "should_merge": "应该和旁边的块并成一间",
    "not_room": "不是房间（窄过道、楼梯、阳台、室外）",
    "bad_shape": "形状不对（缺一块、多一块、边没贴墙）",
    "too_small": "太小或太窄，站不下相机和两个人",
    "unclear": "图看不清，没法判断",
    "other": "其他（写在备注里）",
}
DROP_REASON_ZH = {
    "CORRIDOR": "走廊窄条", "CORRIDOR_BODY_WIDTH_BELOW_1_5": "走廊窄条（宽不到 1.5 m）", "NECK_FRAGMENT": "窄口后挂着的小块",
    "DETACHED_FRAGMENT": "隔开、走不过去的碎块", "SMALL": "太小", "FLOOR_AREA_BELOW_6": "不到 6 m²",
    "SHORT_SIDE_BELOW_2_4": "短边不到 2.4 m", "STAIRS": "楼梯", "SCAN_BLACK_FRACTION_ABOVE_15_PERCENT": "扫描缺洞超过 15%",
    "NO_2_4M_DISK": "放不下直径 2.4 m 的圆", "OUTDOOR": "室外", "NOT_LISTED": "不在最终名单（待定）",
}
TYPE_ZH = {"kitchen": "厨房", "dining": "餐厅", "dining room": "餐厅", "living": "客厅", "living room": "客厅", "livingroom": "客厅",
           "bedroom": "卧室", "bathroom": "卫生间", "office": "书房", "study": "书房", "family room": "家庭室",
           "familyroom/lounge": "家庭室", "lounge": "休息室", "hallway": "过道", "corridor": "过道", "toilet": "卫生间",
           "closet": "储物间", "garage": "车库", "laundryroom/mudroom": "洗衣间", "rec/game": "娱乐室", "tv": "影音室",
           "workout/gym/exercise": "健身房", "other room": "其他", "unknown": ""}
PRIOR_ZH = {"use": "可用", "skip": "不用", "unsure": "存疑"}
CUT_SOURCES = {"new_cut", "cap_cut_new", "cut"}   # room sources that mean "cut out of a larger region" in the three lists


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size, index=2 if path.endswith(".ttc") else 0)
    return ImageFont.load_default(size=size)


def polys(geom):
    g = shape(geom)
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]


def matrix(value):
    return np.array(json.loads(value) if isinstance(value, str) else value, dtype=float)


def projector(meta, crop=(0.0, 0.0, 1.0)):
    img = meta["images"][0] if meta.get("images") else meta
    V, P = matrix(meta["view"]), matrix(img["projection"])
    W, H = meta["size_px"]
    y = float(meta["floor_y_m"])
    ox, oy, s = crop

    def f(coords):
        out = []
        for x, z in coords:
            c = P @ (V @ np.array([x, y, z, 1.0]))
            out.append((((c[0] / c[3] + 1) * W / 2 - ox) * s, ((1 - c[1] / c[3]) * H / 2 - oy) * s))
        return out
    return f


def image_path(meta):
    img = meta["images"][0] if meta.get("images") else meta
    return img["path"]


def read_leakage(paths):
    out = {}
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                out[r["room"]] = (float(r["max_escape_fraction"]), r["band"])
    return out


def index_renders(dirs, pattern):
    out = defaultdict(list)
    for d in dirs:
        for mp in sorted(glob.glob(os.path.join(d, pattern))):
            m = json.load(open(mp))
            key = os.path.basename(mp).split("__")[0]
            out[key].append((float(m["floor_y_m"]), mp))
    return out


def region_floor_key(house, region, floor):
    return f"{house}__{region}__{floor}"


def build_cards(fam):
    """Group the family's listed rooms by the region file that holds them; returns (cards, problems)."""
    rooms = json.load(open(fam["rooms"]))["rooms"]
    listed = {r["id"]: r for r in rooms}
    if len(listed) != len(rooms):
        raise ValueError(f'{fam["name"]}: duplicate room ids in {fam["rooms"]}')
    holder = {}
    regions = {}
    for d in fam.get("region_dirs", []):
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            reg = json.load(open(f))
            ids = {b["id"] for b in reg.get("blocks", [])}
            if not ids & set(listed):
                continue
            regions[f] = reg
            for i in ids & set(listed):
                holder.setdefault(i, f)
    leak = read_leakage(fam.get("leakage_csvs", []))
    prior_dir = fam.get("prior_verdict_dir")
    cards = {}
    problems = {"no_region_file": [], "no_leakage": []}
    for r in rooms:
        f = holder.get(r["id"])
        key = f or f'{r["house"]}/{r["source_region"]}'
        if f is None:
            problems["no_region_file"].append(r["id"])
        c = cards.setdefault(key, dict(region_file=f, house=r["house"], region=r["source_region"], rooms=[]))
        c["rooms"].append(r)
        if r["id"] not in leak:
            problems["no_leakage"].append(r["id"])
    out = []
    for key, c in cards.items():
        reg = regions.get(c["region_file"]) if c["region_file"] else None
        blocks = reg.get("blocks", []) if reg else []
        kept = sorted(c["rooms"], key=lambda r: -r["floor_area_m2"])
        kept_ids = {r["id"] for r in kept}
        others = []
        for b in blocks:
            if b["id"] in kept_ids:
                continue
            kind = b.get("decision")
            reasons = list(b.get("discard_reasons") or [])
            if kind == "retain":            # retained by the split but not in the final list (pending / dropped later)
                kind, reasons = "unresolved", ["NOT_LISTED"]
            elif kind != "discard":
                kind = "unresolved"
                reasons = list(b.get("unresolved_reasons") or []) or ["NOT_LISTED"]
            others.append(dict(b, _kind=kind, _reasons=reasons))
        others.sort(key=lambda b: -(b.get("floor_area_m2") or 0))
        block_type = {b["id"]: b.get("room_type") for b in blocks}
        prior = None
        if prior_dir:
            pf = os.path.join(prior_dir, f'{c["house"]}__{c["region"]}.json')
            if os.path.exists(pf):
                pv = json.load(open(pf))
                prior = dict(verdict=pv.get("verdict"), verdict_zh=PRIOR_ZH.get(pv.get("verdict"), pv.get("verdict")),
                             note=pv.get("note") or "", author=pv.get("author") or "", written_at=pv.get("written_at") or "")
        src_area = (reg or {}).get("source_floor_area_m2")
        if src_area is None:
            src_area = sum(r["floor_area_m2"] for r in kept) + sum(b.get("floor_area_m2") or 0 for b in others)
        card_rooms = []
        for i, r in enumerate(kept):
            fr, band = leak.get(r["id"], (None, None))
            t = (r.get("room_type") or block_type.get(r["id"]) or "").strip()
            card_rooms.append(dict(id=r["id"], no=i + 1, colour="#%02x%02x%02x" % PALETTE[i % len(PALETTE)],
                                   area=round(r["floor_area_m2"], 2), short=round(r.get("short_side_m") or 0, 2),
                                   type=TYPE_ZH.get(t.lower(), t), floor=r.get("floor_id"), source=r.get("source"),
                                   leak=None if fr is None else round(fr, 4), band=band))
        dropped = [dict(id=b["id"], no=len(kept) + j + 1, kind=b["_kind"], area=round(b.get("floor_area_m2") or 0, 2),
                        reasons=list(dict.fromkeys(DROP_REASON_ZH.get(x, x) for x in b["_reasons"])))
                   for j, b in enumerate(others)]
        split = any((r.get("source") or "") in CUT_SOURCES for r in kept)
        out.append(dict(card=f'{fam["name"].lower()}/{c["house"]}/{c["region"]}', family=fam["name"], house=c["house"],
                        region=c["region"], split=split, source_area=round(src_area, 2), prior=prior,
                        rooms=card_rooms, dropped=dropped,
                        _draw=dict(kept=[(r["id"], r["floor_id"], r["floor_y_m"], r["floor_polygon_xz_m"]) for r in kept],
                                   others=[(b["id"], b["_kind"], b.get("floor_id"), b.get("floor_y_m"), b["floor_polygon_xz_m"])
                                           for b in others if b.get("floor_polygon_xz_m")])))
    return out, problems, rooms


def hatch(size, cross, step):
    W, H = size
    h = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(h)
    w = max(2, step // 5)
    for k in range(-H, W, step):
        d.line([(k, 0), (k + H, H)], fill=(35, 35, 35, 215), width=w)
        if cross:
            d.line([(k, H), (k + H, 0)], fill=(35, 35, 35, 215), width=w)
    return h


def render_card_floor(task):
    """Draw one card on one floor; task is a plain dict so it can cross process boundaries."""
    meta, crop_mode, out_path, size = task["meta"], task["crop_mode"], task["out"], task["size"]
    full = Image.open(image_path(meta)).convert("RGB")
    p0 = projector(meta)
    rings = [list(p.exterior.coords) for _, _, g in task["kept"] for p in polys(g)]
    rings += [list(p.exterior.coords) for _, _, g in task["others"] for p in polys(g)]
    draw_scale = 1.5
    S = int(size * draw_scale)
    if crop_mode == "house":
        xy = np.array([q for ring in rings for q in p0(ring)])
        px_per_m = meta["size_px"][0] / float(meta["span_m"])
        cx, cy = (xy.min(0) + xy.max(0)) / 2
        half = max(xy.max(0) - xy.min(0)) / 2 + MARGIN_M * px_per_m
        box = (cx - half, cy - half, cx + half, cy + half)
        base = full.crop(tuple(int(round(v)) for v in box)).resize((S, S), Image.LANCZOS)
        prj = projector(meta, (round(box[0]), round(box[1]), S / (2 * half)))
    else:
        base = full.resize((S, S), Image.LANCZOS)
        prj = projector(meta, (0.0, 0.0, S / float(meta["size_px"][0])))
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for no, colour, g in task["kept"]:
        for p in polys(g):
            d.polygon(prj(list(p.exterior.coords)), fill=tuple(colour) + (115,))
            for hole in p.interiors:
                d.polygon(prj(list(hole.coords)), fill=(0, 0, 0, 0))
    masks = {"discard": Image.new("L", base.size, 0), "unresolved": Image.new("L", base.size, 0)}
    for no, kind, g in task["others"]:
        dm = ImageDraw.Draw(masks["unresolved" if kind == "unresolved" else "discard"])
        for p in polys(g):
            dm.polygon(prj(list(p.exterior.coords)), fill=255)
            for hole in p.interiors:
                dm.polygon(prj(list(hole.coords)), fill=0)
    step = max(10, S // 64)
    for kind, cross in (("discard", False), ("unresolved", True)):
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        layer.paste(hatch(base.size, cross, step), (0, 0), masks[kind])
        over = Image.alpha_composite(over, layer)
    d = ImageDraw.Draw(over)
    lw = max(3, S // 270)
    for g in task["neighbours"]:
        for p in polys(g):
            ring = prj(list(p.exterior.coords))
            d.line(ring, fill=(30, 30, 30, 150), width=lw + 2, joint="curve")
            d.line(ring, fill=(255, 255, 255, 210), width=max(2, lw - 1), joint="curve")
    for no, colour, g in task["kept"]:
        for p in polys(g):
            d.line(prj(list(p.exterior.coords)), fill=tuple(colour) + (255,), width=lw + 1, joint="curve")
    for no, kind, g in task["others"]:
        for p in polys(g):
            d.line(prj(list(p.exterior.coords)), fill=(40, 40, 40, 255), width=lw, joint="curve")
    img = Image.alpha_composite(base.convert("RGBA"), over)
    d2 = ImageDraw.Draw(img)
    f_big, f_small = load_font(max(18, S // 16)), load_font(max(12, S // 30))
    stroke = max(3, S // 200)
    for no, colour, g in task["kept"]:
        ps = sorted(polys(g), key=lambda p: -p.area)
        if not ps:
            continue
        rp = ps[0].representative_point()
        (x, y), = prj([(rp.x, rp.y)])
        d2.text((x, y - S // 40), f"{no}", font=f_big, fill=(255, 255, 255), stroke_width=stroke + 2, stroke_fill=tuple(colour), anchor="mm")
        d2.text((x, y + S // 28), f'{task["areas"][str(no)]:.1f} m²', font=f_small, fill=(20, 20, 20), stroke_width=stroke,
                stroke_fill=(255, 255, 255), anchor="mm")
        for p in ps[1:]:
            if p.area >= 0.5:
                q = p.representative_point()
                (fx, fy), = prj([(q.x, q.y)])
                d2.text((fx, fy), f"{no}", font=f_small, fill=(255, 255, 255), stroke_width=stroke, stroke_fill=tuple(colour), anchor="mm")
    for no, kind, g in task["others"]:
        ps = sorted(polys(g), key=lambda p: -p.area)
        if not ps or ps[0].area < 0.5:
            continue
        rp = ps[0].representative_point()
        (x, y), = prj([(rp.x, rp.y)])
        d2.text((x, y), ("?" if kind == "unresolved" else "×") + f"{no}", font=f_small, fill=(255, 255, 255), stroke_width=stroke,
                stroke_fill=(60, 60, 60), anchor="mm")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.convert("RGB").resize((size, size), Image.LANCZOS).save(out_path, quality=74, optimize=True)
    return out_path


def plan_images(fam, cards, all_rooms, out_dir):
    house_r = index_renders(fam.get("house_render_dirs", []), "*__Y*.json")
    region_r = {}
    for d in fam.get("region_render_dirs", []):
        for mp in glob.glob(os.path.join(d, "*__*__*.json")):
            region_r.setdefault(os.path.basename(mp)[:-5], mp)
    by_floor = defaultdict(list)
    for r in all_rooms:
        by_floor[r["house"]].append(r)
    tasks, missing, dropped_only = [], [], []
    for c in cards:
        draw = c.pop("_draw")
        floors = defaultdict(lambda: dict(kept=[], others=[], ys=[]))
        nos = {r["id"]: (r["no"], r["colour"]) for r in c["rooms"]}
        for rid, fl, y, g in draw["kept"]:
            no, colour = nos[rid]
            floors[fl]["kept"].append((no, [int(colour[i:i + 2], 16) for i in (1, 3, 5)], g))
            floors[fl]["ys"].append(y)
        drop_no = {b["id"]: b["no"] for b in c["dropped"]}
        for bid, kind, fl, y, g in draw["others"]:
            floors[fl]["others"].append((drop_no[bid], kind, g))
            if y is not None and not floors[fl]["kept"]:
                floors[fl]["ys"].append(y)
        c["images"] = []
        for fl in sorted(floors):
            F = floors[fl]
            if not F["kept"] and not any(shape(g).area >= 0.5 for _, _, g in F["others"]):
                continue
            y = float(np.median(F["ys"])) if F["ys"] else None
            meta, mode = None, None
            cands = [(abs(fy - y), mp) for fy, mp in house_r.get(c["house"], [])] if y is not None else []
            cands = sorted(t for t in cands if t[0] < FLOOR_MATCH_M)
            if cands:
                meta, mode = json.load(open(cands[0][1])), "house"
            elif region_floor_key(c["house"], c["region"], fl) in region_r:
                meta, mode = json.load(open(region_r[region_floor_key(c["house"], c["region"], fl)])), "region"
            if meta is None:
                # a floor holding only dropped pieces (e.g. a stair landing) is listed in text; only missing room images count
                (missing if F["kept"] else dropped_only).append(f'{c["card"]}/{fl}')
                continue
            kept_ids = {r["id"] for r in c["rooms"]}
            neighbours = [r["floor_polygon_xz_m"] for r in by_floor[c["house"]]
                          if r["id"] not in kept_ids and y is not None and abs(r["floor_y_m"] - y) < FLOOR_MATCH_M]
            multi = len(F["kept"]) > 1 or any(shape(g).area >= 0.5 for _, _, g in F["others"])
            rel = f'img/{c["family"].lower()}/{c["house"]}__{c["region"]}__{fl}.jpg'
            tasks.append(dict(meta=meta, crop_mode=mode, out=os.path.join(out_dir, rel), size=720 if multi else 560,
                              kept=F["kept"], others=F["others"], neighbours=neighbours,
                              areas={str(no): next(r["area"] for r in c["rooms"] if r["no"] == no) for no, _, _ in F["kept"]}))
            c["images"].append(rel)
    return tasks, missing, dropped_only


def crop_neighbours(task):
    """Keep only neighbour outlines near the card, so workers do not draw whole houses."""
    if task["crop_mode"] != "house":
        return task
    shp = [p for _, _, g in task["kept"] for p in polys(g)] + [p for _, _, g in task["others"] for p in polys(g)]
    minx = min(p.bounds[0] for p in shp) - MARGIN_M * 2
    minz = min(p.bounds[1] for p in shp) - MARGIN_M * 2
    maxx = max(p.bounds[2] for p in shp) + MARGIN_M * 2
    maxz = max(p.bounds[3] for p in shp) + MARGIN_M * 2
    win = shapely_box(minx, minz, maxx, maxz)
    task["neighbours"] = [g for g in task["neighbours"] if shape(g).intersects(win)]
    return task


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="new directory; refuses to overwrite an existing one")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    cfg = json.load(open(args.config))
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refuse: {out} already exists")
    out.mkdir(parents=True)
    all_cards, receipt = [], dict(schema=SCHEMA, build_id=cfg["build_id"], families={}, inputs=[])
    tasks = []
    for fam in cfg["families"]:
        cards, problems, rooms = build_cards(fam)
        t, missing, dropped_only = plan_images(fam, cards, rooms, str(out))
        tasks += [crop_neighbours(x) for x in t]
        bands = defaultdict(int)
        for c in cards:
            for r in c["rooms"]:
                bands[r["band"] or "未测"] += 1
        receipt["families"][fam["name"]] = dict(rooms=len(rooms), cards=len(cards), split_cards=sum(c["split"] for c in cards),
                                                images=len(t), cards_missing_image=missing, bands=dict(bands),
                                                dropped_only_floors_without_image=dropped_only,
                                                rooms_without_region_file=problems["no_region_file"],
                                                rooms_without_leakage=problems["no_leakage"],
                                                area_m2=round(sum(r["floor_area_m2"] for r in rooms), 1))
        for p in [fam["rooms"]] + list(fam.get("leakage_csvs", [])):
            receipt["inputs"].append(dict(path=p, sha256=sha256(p)))
        all_cards += cards
    order = {f["name"]: i for i, f in enumerate(cfg["families"])}

    def region_no(c):
        tail = c["region"].lstrip("R")
        return int(tail) if tail.isdigit() else 10 ** 6
    all_cards.sort(key=lambda c: (order[c["family"]], c["house"], not c["split"], region_no(c), c["region"]))
    if args.workers <= 1:
        for task in tasks:
            render_card_floor(task)
    else:
        with Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(render_card_floor, tasks, chunksize=4)):
                if (i + 1) % 100 == 0:
                    print(f"rendered {i + 1}/{len(tasks)}", flush=True)
    data = dict(schema=SCHEMA, build_id=cfg["build_id"], title=cfg["title"], reasons=REASONS,
                families=[dict(name=f["name"], **{k: receipt["families"][f["name"]][k] for k in ("rooms", "cards", "split_cards", "area_m2")})
                          for f in cfg["families"]],
                cards=all_cards)
    (out / "site_data.json").write_text(json.dumps(data, ensure_ascii=False))
    page = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace("__TITLE__", html.escape(cfg["title"])).replace("__LEAD_HTML__", cfg.get("lead_html", "")).replace("__DATA__", payload)
    (out / "index.html").write_text(page, encoding="utf-8")
    shutil.copy2(SERVER, out / "serve_review.py")
    receipt["room_total"] = sum(len(c["rooms"]) for c in all_cards)
    receipt["card_total"] = len(all_cards)
    receipt["image_total"] = len(tasks)
    (out / "build_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=1))
    print(json.dumps({k: receipt[k] for k in ("room_total", "card_total", "image_total")}),
          {f: {k: (len(v) if isinstance(v, list) else v) for k, v in s.items()} for f, s in receipt["families"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
