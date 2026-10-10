#!/usr/bin/env python3
"""Apply the reviewer's usability rules (smy, 2026-10-10) to the final room lists; reviewed rooms keep the human verdict.

A room nobody reviewed is dropped when any rule fires:
  bathroom  labelled bathroom or toilet, or a toilet/bathtub/shower/urinal/bidet stands in a room under 10 m^2 (bigger
            rooms with a fixture are mostly bedrooms whose en-suite shares the annotated region); bathrooms are narrow
            and may not fit the camera and the people
  small     floor area under 7.5 m^2 (the reviewer kept uncertain rooms from about 7.5 m^2 up)
  bare      fewer than 2 pieces of furniture with a footprint of at least 0.25 m^2: empty rooms and bare corridors,
            where there is nothing to ask distances to and nothing occludes
Furniture is the room-screening blocker word list plus labels it misses: MP3D's coarse mpcat40 classes ('furniture',
'gym equipment', 'appliances', 'seating', 'chest of drawers', 'fireplace') and a few HM3D ones ('pouffe', 'drawer', ...).

usage: python tools/rooms/room_usability/rules.py --site-data SITE/site_data.json --config REVIEW_CONFIG.json
           --feedback review_feedback.json --objects A/objects.csv [B/objects.csv ...] --out NEW_DIR
The review site supplies cards, bands and HM3D room types; the config names each family's source list. Writes
room_decisions.csv, summary.json and <family>_room_list_usability.json: the source records of the kept rooms plus a
usability_filter receipt naming every dropped room and why.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "avengine_room_usability_v1"
RULES = dict(bathroom="卫生间", small="面积小于 7.5 m²", bare="大件家具少于 2 件（空房间或过道）")
BATH_TYPES = {"bathroom", "toilet", "卫生间"}
BATH_WORDS = ("toilet", "bathtub", "shower", "urinal", "bidet", "tub")
EXTRA_FURNITURE = {"furniture", "gym equipment", "appliances", "seating", "chest of drawers", "fireplace",  # MP3D mpcat40
                   "pouffe", "drawer", "closet", "dishwasher", "kitchen top", "exercise equipment", "footrest",
                   "beanbag", "storage", "stovetop"}  # HM3D labels the blocker word list misses
MIN_AREA_M2, MIN_FURNITURE, MIN_FOOTPRINT_M2, FIXTURE_ONLY_BELOW_M2 = 7.5, 2, 0.25, 10.0
FAMILY_ORDER = ("HM3D", "MP3D", "Kujiale")


def is_furniture(o):
    return (o["role"] == "blocker" or o["cat"] in EXTRA_FURNITURE) and float(o["foot"]) >= MIN_FOOTPRINT_M2


def is_fixture(o):
    return any(w in o["cat"].split() for w in BATH_WORDS)


def rules_fired(labelled_bathroom, area_m2, objects):
    fired = []
    fixtures = sum(1 for o in objects if is_fixture(o))
    if labelled_bathroom or (fixtures and area_m2 < FIXTURE_ONLY_BELOW_M2):
        fired.append("bathroom")
    if area_m2 < MIN_AREA_M2:
        fired.append("small")
    if sum(1 for o in objects if is_furniture(o)) < MIN_FURNITURE:
        fired.append("bare")
    return fired


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site-data", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--feedback", required=True, type=Path)
    ap.add_argument("--objects", required=True, type=Path, nargs="+")
    ap.add_argument("--out", required=True, type=Path, help="new directory; refuses to overwrite an existing one")
    args = ap.parse_args(argv)
    if args.out.exists():
        sys.exit(f"refuse: {args.out} exists")
    site = json.loads(args.site_data.read_text())
    config = json.loads(args.config.read_text())
    feedback = json.loads(args.feedback.read_text())["feedback"]
    objects = defaultdict(list)
    for path in args.objects:
        for o in csv.DictReader(path.open()):
            objects[o["room"]].append(o)
    source = {}
    for fam in config["families"]:
        for r in json.loads(Path(fam["rooms"]).read_text())["rooms"]:
            source[r["id"]] = r
    rows = []
    for card in site["cards"]:
        for r in card["rooms"]:
            rec = source[r["id"]]
            objs = objects.get(r["id"], [])
            room_type = str(rec.get("room_type") or "").lower() or r["type"]
            area = float(rec["floor_area_m2"])
            fired = rules_fired(room_type in BATH_TYPES or r["type"] == "卫生间", area, objs)
            verdict = feedback.get(r["id"], {}).get("assessment", "")
            if verdict in ("ok", "bad"):
                decision, basis = ("keep" if verdict == "ok" else "drop"), "smy"
            else:
                decision, basis = ("drop" if fired else "keep"), "rules"
            furniture = Counter(o["cat"] for o in objs if is_furniture(o))
            rows.append(dict(id=r["id"], family=card["family"], card=card["card"], split=card["split"], band=r["band"],
                             area=round(area, 2), type=room_type, n_big=sum(furniture.values()), n_objects=len(objs),
                             n_fixture=sum(1 for o in objs if is_fixture(o)),
                             big_furniture="|".join(f"{k}×{v}" for k, v in furniture.most_common()),
                             smy=verdict, rules_fired="|".join(fired), decision=decision, basis=basis))
    args.out.mkdir(parents=True)
    with (args.out / "room_decisions.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = dict(schema=SCHEMA, rules=RULES, min_area_m2=MIN_AREA_M2, min_big_furniture=MIN_FURNITURE,
                   min_footprint_m2=MIN_FOOTPRINT_M2, fixture_only_below_m2=FIXTURE_ONLY_BELOW_M2,
                   extra_furniture_labels=sorted(EXTRA_FURNITURE),
                   inputs={str(p): sha256(p) for p in [args.site_data, args.config, args.feedback, *args.objects]},
                   families={})
    for fam in FAMILY_ORDER:
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            continue
        unreviewed = [r for r in rs if r["basis"] == "rules"]
        summary["families"][fam] = dict(
            rooms=len(rs), bands_before=Counter(r["band"] for r in rs),
            bands_after=Counter(r["band"] for r in rs if r["decision"] == "keep"),
            kept=sum(r["decision"] == "keep" for r in rs),
            dropped_by_smy=sum(r["basis"] == "smy" and r["decision"] == "drop" for r in rs),
            unreviewed=len(unreviewed), dropped_by_rules=sum(r["decision"] == "drop" for r in unreviewed),
            rule_hits_unreviewed={k: sum(k in r["rules_fired"].split("|") for r in unreviewed) for k in RULES},
            rule_combos_unreviewed=Counter(r["rules_fired"] for r in unreviewed if r["rules_fired"]),
            kept_types=Counter(r["type"] for r in rs if r["decision"] == "keep"),
            rules_vs_smy=dict(bad_caught=sum(1 for r in rs if r["smy"] == "bad" and r["rules_fired"]),
                              bad_total=sum(1 for r in rs if r["smy"] == "bad"),
                              ok_hit=sum(1 for r in rs if r["smy"] == "ok" and r["rules_fired"]),
                              ok_total=sum(1 for r in rs if r["smy"] == "ok"),
                              ok_hit_by_rule={k: sum(1 for r in rs if r["smy"] == "ok" and k in r["rules_fired"].split("|"))
                                              for k in RULES}))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    keep = {r["id"] for r in rows if r["decision"] == "keep"}
    why = {r["id"]: dict(basis=r["basis"], smy=r["smy"], rules=r["rules_fired"].split("|") if r["rules_fired"] else [])
           for r in rows}
    for fam in config["families"]:
        src = Path(fam["rooms"])
        raw = src.read_bytes()
        doc = json.loads(raw)
        kept = [r for r in doc["rooms"] if r["id"] in keep]
        new = {k: v for k, v in doc.items() if k != "rooms"}
        new.update(schema=str(doc.get("schema", "")) + "+usability_v1", usability_filter=dict(
            schema=SCHEMA, source_list=str(src), source_sha256=hashlib.sha256(raw).hexdigest(),
            feedback=str(args.feedback), feedback_sha256=sha256(args.feedback), rules=RULES, min_area_m2=MIN_AREA_M2,
            min_big_furniture=MIN_FURNITURE, min_footprint_m2=MIN_FOOTPRINT_M2,
            fixture_only_below_m2=FIXTURE_ONLY_BELOW_M2, human_verdicts_override_rules=True, kept=len(kept),
            dropped=[dict(id=r["id"], **why[r["id"]]) for r in doc["rooms"] if r["id"] not in keep]), rooms=kept)
        (args.out / f"{fam['name'].lower()}_room_list_usability.json").write_text(json.dumps(new, ensure_ascii=False, indent=1))
    for fam, s in summary["families"].items():
        print(fam, f"{s['rooms']} -> {s['kept']}", "bands after", dict(s["bands_after"]), "dropped by smy", s["dropped_by_smy"],
              "by rules", s["dropped_by_rules"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
