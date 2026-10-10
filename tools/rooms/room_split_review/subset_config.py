#!/usr/bin/env python3
"""Write a review config that shows only chosen cards, e.g. a second round on the rooms a screener could not settle.

usage: python tools/rooms/room_split_review/subset_config.py --config REVIEW_CONFIG.json --lists DIR
           --round ROUND.json --site-data SITE/site_data.json --out NEW_DIR
ROUND.json: {"title": str, "build_id": str, "lead_html": str, "cards": {"hm3d/<house>/<region>": "<category>", ...}}.
--lists holds <family>_room_list_usability.json (written by tools/rooms/room_usability/rules.py). Every room of a chosen
card that is still in that list goes into the subset, so a split region is shown whole. --site-data is the site the
cards come from (it maps rooms to cards). Writes <family>_round_rooms.json and review_config.json with absolute paths;
build the pages with build_review_site.py --config NEW_DIR/review_config.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--lists", required=True, type=Path)
    ap.add_argument("--round", required=True, type=Path)
    ap.add_argument("--site-data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="new directory; refuses to overwrite an existing one")
    args = ap.parse_args(argv)
    if args.out.exists():
        sys.exit(f"refuse: {args.out} exists")
    config = json.loads(args.config.read_text())
    spec = json.loads(args.round.read_text())
    cards = spec["cards"]
    card_of = {r["id"]: c["card"] for c in json.loads(args.site_data.read_text())["cards"] for r in c["rooms"]}
    unknown = sorted(set(cards) - set(card_of.values()))
    if unknown:
        sys.exit(f"cards not on the site: {unknown[:5]}")
    out = args.out.resolve()
    out.mkdir(parents=True)
    counts = {}
    for fam in config["families"]:
        src = args.lists.resolve() / f"{fam['name'].lower()}_room_list_usability.json"
        doc = json.loads(src.read_text())
        doc["rooms"] = [r for r in doc["rooms"] if card_of.get(r["id"]) in cards]
        doc["round_selection"] = dict(source=str(src), round=str(args.round.resolve()),
                                      cards={c: k for c, k in cards.items() if c.startswith(fam["name"].lower() + "/")})
        path = out / f"{fam['name'].lower()}_round_rooms.json"
        path.write_text(json.dumps(doc, ensure_ascii=False))
        fam["rooms"] = str(path)
        counts[fam["name"]] = dict(cards=len({card_of[r["id"]] for r in doc["rooms"]}), rooms=len(doc["rooms"]))
    config.update(title=spec["title"], build_id=spec["build_id"], lead_html=spec["lead_html"])
    (out / "review_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    print(json.dumps(counts, ensure_ascii=False), "cards chosen", len(cards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
