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

One thing is deliberately reported and not changed: "floor" maps to a material
named Carpet whose absorption at 1 kHz is 0.010, which is a hard reflective
surface. Whether that is right for a carpeted house is a question about the
material data, not about naming, so it is raised rather than quietly rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# category -> (existing material, the label it follows)
MAPPING = {
    # ceiling surfaces follow the ceiling
    "ceiling lower": ("Acoustic Tile", "ceiling"),
    "ceiling under stairs": ("Acoustic Tile", "ceiling"),
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
        "repairs": repairs,
        "added_by_material": added,
        "left_to_default": [
            {"category": c, "area_share": round(share.get(c, 0.0), 6)}
            for c in LEFT_TO_DEFAULT
        ],
        "raised_not_changed": (
            "the label 'floor' points at a material named Carpet whose "
            "absorption at 1 kHz is 0.010, a hard reflective surface. Whether "
            "that suits a carpeted house is a question about the material data, "
            "not about naming"
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
