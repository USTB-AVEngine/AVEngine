"""Extend the acoustic material database to HM3D's category vocabulary.

The database shipped with SoundSpaces carries 64 labels written against MP3D's
categories. HM3D names things far more finely - kitchen cabinet, door frame,
window curtain - so most of its vocabulary falls through to the default
material. Measured by surface area over the 36 annotated val scenes, 78.8
percent is already matched and the missing fifth is dominated by compounds and
synonyms of things that are matched already.

Three rules keep this reviewable:

  * only labels are added. No absorption, scattering or transmission figure is
    touched. Those are measured material data and this is a naming exercise; if
    a category's acoustics are wrong, the fix is to point it at a different
    existing material, not to invent numbers.
  * every mapping names the label it is a synonym or compound of, so a reviewer
    can check the judgement rather than take it on trust.
  * a mapping whose category does not appear in the measured area report is an
    error, not a harmless extra. Dead entries are how a map drifts away from the
    dataset it claims to cover.

Three defects in the shipped file are repaired, because they make labels
unreachable rather than merely untidy:

  * "piperefrigerator" is one label in Steel. Two category names were plainly
    concatenated when the list was authored, so neither pipe nor refrigerator
    matches anything - together 0.35 percent of HM3D's annotated area. Split.
  * "floor" is listed twice in Carpet. Deduplicated.
  * "carpet" matches nothing at all while three carpet materials exist, and the
    most absorptive of them, Carpet Heavy Padded at 0.08, carries no labels and
    is unreachable. Given a label.

One label is re-pointed rather than merely extended, and it is the single
largest acoustic change in this file. "ceiling" pointed at Acoustic Tile, a
suspended commercial absorber whose mid-band coefficient is 0.667. Ceilings are
17 percent of HM3D's annotated area, and calibrated with
calibrate_surface_materials.py the scenes came out at 0.237 s of mid-band T60
against the 0.3 to 0.6 s a furnished domestic room occupies - too dead for a
house by half. A house ceiling is painted plasterboard, the same material as its
walls, and pointing the label at Gypsum Board puts the same scenes at 0.518 s.
The decay is straight to 0.996 either way, so those extrapolations mean something.

A correction to an earlier version of this note, which said "floor" pointed at a
material whose absorption was 0.010 and called it a hard reflector. That misread
the data: absorption is a flat [frequency, alpha, ...] list, so absorption[1] is
the coefficient at the lowest frequency in the table, usually 125 Hz. The floor's
Carpet is 0.25 across the mid band and Gypsum Board is 0.053, not 0.29. Both are
sensible for a house and neither needed changing; the ceiling was the outlier all
along.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# category -> (existing material, the label it follows)
MAPPING = {
    # ceiling surfaces follow the ceiling, which REPOINT moves to Gypsum Board.
    # Naming Acoustic Tile here would leave a plasterboard ceiling with
    # commercial acoustic tile in its recesses.
    "ceiling lower": ("Gypsum Board", "ceiling"),
    "ceiling under stairs": ("Gypsum Board", "ceiling"),
    # wall-like structure in a house is drywall
    "wall panel": ("Gypsum Board", "wall"),
    "partition": ("Gypsum Board", "wall"),
    "arch": ("Gypsum Board", "wall"),
    "pillar": ("Gypsum Board", "wall"),
    "fireplace wall": ("Gypsum Board", "wall"),
    "shower wall": ("Tile, Ceramic", "shower-stall"),
    "shower cabin": ("Tile, Ceramic", "shower-stall"),
    "shower floor": ("Tile, Ceramic", "shower-stall"),
    # the only carpet label there is, for the carpet material nothing reached
    "carpet": ("Carpet, Heavy Padded", "(none: this material had no labels)"),
    # joinery and case goods follow door / cabinet / table / shelf
    "door frame": ("wood, Thick", "door"),
    "window frame": ("wood, Thick", "door"),
    "kitchen cabinet": ("wood, Thick", "cabinet"),
    "bathroom cabinet": ("wood, Thick", "cabinet"),
    "kitchen lower cabinet": ("wood, Thick", "cabinet"),
    "storage cabinet": ("wood, Thick", "cabinet"),
    "display cabinet": ("wood, Thick", "cabinet"),
    "sink cabinet": ("wood, Thick", "cabinet"),
    "kitchen counter": ("wood, Thick", "countertop"),
    "kitchen island": ("wood, Thick", "countertop"),
    "kitchen shelf": ("wood, Thick", "shelf"),
    "book rack": ("wood, Thick", "shelving"),
    "rack": ("wood, Thick", "shelving"),
    "bookshelf": ("wood, Thick", "shelving"),
    "dining table": ("wood, Thick", "table"),
    "coffee table": ("wood, Thick", "table"),
    "dining chair": ("wood, Thick", "chair"),
    "bench": ("wood, Thick", "seating"),
    "dresser": ("wood, Thick", "chest_of_drawers"),
    "chest of drawers": ("wood, Thick", "chest_of_drawers"),
    "drawer": ("wood, Thick", "chest_of_drawers"),
    "board": ("wood, Thick", "board_panel"),
    "rafter": ("wood, Thick", "beam"),
    # upholstery and textile follow sofa / cushion / curtain / clothes
    "couch": ("Curtain", "sofa"),
    "sofa chair": ("Curtain", "sofa"),
    "l-shaped sofa": ("Curtain", "sofa"),
    "armchair": ("Curtain", "sofa"),
    "pillow": ("Curtain", "cushion"),
    "window curtain": ("Curtain", "curtain"),
    "shower curtain": ("Curtain", "curtain"),
    "hanging clothes": ("Curtain", "clothes"),
    # metal fixtures and white goods follow railing / appliances
    "banister": ("Steel", "handrail"),
    "stairs railing": ("Steel", "railing"),
    "staircase handrail": ("Steel", "handrail"),
    "balustrade": ("Steel", "railing"),
    "washing machine": ("Steel", "major-appliance"),
    "oven and stove": ("Steel", "major-appliance"),
    "radiator": ("Steel", "appliances"),
    "heater": ("Steel", "appliances"),
    # glazed and framed surfaces follow window / mirror / lighting
    "picture": ("Glass", "mirror"),
    "painting": ("Glass", "mirror"),
    "exhibition window": ("Glass", "window"),
    "lamp": ("Glass", "lighting"),
    "ceiling lamp": ("Glass", "lighting"),
    "lighting fixture": ("Glass", "lighting"),
    "tv": ("Glass", "tv_monitor"),
    "decorative plant": ("Foliage", "plant"),
}

# Named so that leaving them to the default is a decision on the record rather
# than an oversight. Small, cluttered, or of no single material.
LEFT_TO_DEFAULT = (
    "unknown", "clutter", "decoration", "box", "container", "basket", "toy",
    "book", "vase", "flower vase", "ceiling fan",
)

SPLIT_LABELS = {"piperefrigerator": ("pipe", "refrigerator")}

# label -> (material it should point at, material it pointed at, why)
REPOINT = {
    "ceiling": (
        "Gypsum Board",
        "Acoustic Tile",
        "a house ceiling is painted plasterboard, not a suspended commercial "
        "absorber. Acoustic Tile is 0.667 across the mid band and put these "
        "scenes at 0.237 s of T60 against a residential 0.3 to 0.6; Gypsum "
        "Board is 0.053 and puts them at 0.518 s",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-materials", required=True, type=Path)
    parser.add_argument("--area-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    base = json.loads(args.base_materials.read_text(encoding="utf-8"))
    area = json.loads(args.area_report.read_text(encoding="utf-8"))
    share = {c["category"]: c["share"] for c in area["categories"]}

    by_name = {m["name"]: m for m in base["materials"]}

    def mid_band_alpha(material):
        flat = material["absorption"]
        pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        mid = [a for f, a in pairs if 400.0 <= f <= 2000.0]
        return sum(mid) / len(mid) if mid else float("nan")

    unknown_targets = sorted(
        {target for target, _ in MAPPING.values() if target not in by_name}
    )
    if unknown_targets:
        raise SystemExit(
            f"mapping names materials that do not exist: {unknown_targets}"
        )
    dead = sorted(c for c in MAPPING if c not in share)
    if dead:
        raise SystemExit(
            "these categories are not in the measured area report, so the "
            f"mapping would be claiming coverage it does not have: {dead}"
        )

    before = set()
    for material in base["materials"]:
        before.update(material.get("labels", []))
    clashes = sorted(c for c in MAPPING if c in before)
    if clashes:
        raise SystemExit(
            f"these categories are already mapped; adding them again would "
            f"move them silently: {clashes}"
        )

    repointed = []
    for label, (target, expected, reason) in REPOINT.items():
        if target not in by_name:
            raise SystemExit(f"repoint names a material that does not exist: {target}")
        held = [m["name"] for m in base["materials"] if label in m.get("labels", [])]
        if held != [expected]:
            raise SystemExit(
                f"{label!r} was expected to point at {expected!r} but points at "
                f"{held}; the base file has changed and this repoint needs "
                "re-deciding rather than applying blind"
            )
        for material in base["materials"]:
            material["labels"] = [
                l for l in material.get("labels", []) if l != label
            ]
        by_name[target].setdefault("labels", []).append(label)
        repointed.append(
            {
                "label": label,
                "from": expected,
                "to": target,
                "from_mid_band_alpha": round(mid_band_alpha(by_name[expected]), 3),
                "to_mid_band_alpha": round(mid_band_alpha(by_name[target]), 3),
                "area_share": round(share.get(label, 0.0), 6),
                "reason": reason,
            }
        )
        print(f"repointed {label!r}: {expected} -> {target}")

    repairs = []
    for material in base["materials"]:
        labels = list(material.get("labels", []))
        for joined, parts in SPLIT_LABELS.items():
            if joined in labels:
                labels = [l for l in labels if l != joined] + list(parts)
                repairs.append(
                    {
                        "material": material["name"],
                        "removed": joined,
                        "added": list(parts),
                        "reason": "two category names were concatenated, so "
                        "neither matched anything",
                    }
                )
        deduped = list(dict.fromkeys(labels))
        if len(deduped) != len(labels):
            repairs.append(
                {
                    "material": material["name"],
                    "deduplicated": sorted(
                        {l for l in labels if labels.count(l) > 1}
                    ),
                }
            )
        material["labels"] = deduped

    added = {}
    for category, (target, follows) in MAPPING.items():
        by_name[target].setdefault("labels", []).append(category)
        added.setdefault(target, []).append(
            {"category": category, "follows": follows,
             "area_share": round(share[category], 6)}
        )

    after = set()
    for material in base["materials"]:
        after.update(material.get("labels", []))
    covered_before = sum(s for c, s in share.items() if c in before)
    covered_after = sum(s for c, s in share.items() if c in after)

    base["provenance"] = {
        "schema": "avengine_hm3d_material_map_v1",
        "derived_from": str(args.base_materials),
        "area_report": str(args.area_report),
        "scenes_measured": area["scenes"],
        "rule": (
            "labels only. No absorption, scattering or transmission value is "
            "touched; every category points at a material that already existed"
        ),
        "labels_before": len(before),
        "labels_after": len(after),
        "area_share_covered_before": round(covered_before, 5),
        "area_share_covered_after": round(covered_after, 5),
        "repointed": repointed,
        "repairs": repairs,
        "added_by_material": added,
        "left_to_default": [
            {"category": c, "area_share": round(share.get(c, 0.0), 6)}
            for c in LEFT_TO_DEFAULT
        ],
        "how_to_read_absorption": (
            "absorption is a flat [frequency, alpha, ...] list, so absorption[1] "
            "is the coefficient at the lowest frequency in the table, usually "
            "125 Hz, not the mid band. Misreading it made Gypsum Board look like "
            "0.29 when its mid band is 0.053, and the floor's Carpet look like "
            "0.010 when its mid band is 0.25"
        ),
    }
    args.output.write_text(
        json.dumps(base, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print(f"labels {len(before)} -> {len(after)}")
    print(f"area share covered {100 * covered_before:.1f}% -> "
          f"{100 * covered_after:.1f}%")
    for material, entries in sorted(added.items()):
        total = sum(e["area_share"] for e in entries)
        print(f"  {material:<22} +{len(entries):>2} labels  "
              f"+{100 * total:5.2f}% of area")
    for repair in repairs:
        print(f"  repaired {repair}")
    print(f"wrote {args.output}")
    if args.report:
        args.report.write_text(
            json.dumps(base["provenance"], ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
