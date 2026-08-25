#!/usr/bin/env python3
"""Collect the assets a ladder run accepted into one place, with their evidence.

A ladder workdir holds every rung it tried, which is what you want while tuning
and not what you want to hand downstream. This copies the accepted rung's outputs
into a stable directory per breed and writes one manifest recording, for each,
which rung won and what the gate read - so an asset can be traced back to the
preparation that produced it without keeping the whole search.

Nothing is overwritten: an existing destination is an error, because a second run
of the same breed is a new asset and not a replacement for the reviewed one.

Example::

  python tools/assets/collect_accepted_animals.py \\
    --out /data/avengine_external/assets/generated_animals_v1 \\
    --asset siamese=/path/to/ladder_workdir \\
    --asset jack_russell=/path/to/other_workdir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "avengine_generated_animal_ladder_collection_v1"
CARRIED = (
    "animated.glb", "prepared.glb", "prepared.json", "walk_deformation.json",
    "ladder.json", "heading.json", "level.json", "retarget.json",
    "heading_probe.png",
)
RENDER_DIRS = ("walk", "turntable")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--asset", action="append", required=True,
                        metavar="NAME=LADDER_WORKDIR")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"destination exists, refusing to overwrite: {out}")
    out.mkdir(parents=True)

    entries = []
    for spec in args.asset:
        name, _, source = spec.partition("=")
        src = Path(source)
        ladder = json.loads((src / "ladder.json").read_text(encoding="utf-8"))
        walk = json.loads((src / "walk_deformation.json").read_text(encoding="utf-8"))
        prepared = json.loads((src / "prepared.json").read_text(encoding="utf-8"))
        dest = out / name
        dest.mkdir()
        carried = []
        for item in CARRIED:
            if (src / item).is_file():
                shutil.copy2(src / item, dest / item)
                carried.append(item)
        for folder in RENDER_DIRS:
            if (src / folder).is_dir():
                shutil.copytree(src / folder, dest / folder)
        entries.append({
            "asset": name,
            "source_workdir": str(src),
            "accepted_rung": ladder["accepted_rung"],
            "ladder_tried": ladder["ladder"],
            "pick": ladder.get("pick"),
            "faces": walk["faces"],
            "worst_share_area_shards": walk["worst_share_area_shards"],
            "worst_share_area_shards_visible":
                walk.get("worst_share_area_shards_visible"),
            "worst_frame": walk["worst_frame_by_shards"],
            "head_third_survival": prepared.get("band_survival", {}).get("front"),
            "source_mesh": prepared.get("input"),
            "animated_glb_sha256": digest(dest / "animated.glb"),
            "carried": carried,
        })

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate": {
            "tool": "tools/assets/gate_rigged_asset.py",
            "criterion": "worst_share_area_shards <= 0.025",
            "calibration": "owner judgement on rigged walk videos at ordinary "
                           "viewing distance; accepted versions reach 0.0192 and "
                           "the rejected one sits at 0.0371",
            "known_variance": "the same rung of the same ladder has measured "
                              "0.01867 to 0.02628 across six rigs, so a reading "
                              "within about 17 percent of the threshold is "
                              "inside the noise",
        },
        "formal_dataset_registration_authorized": False,
        "assets": entries,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("COLLECTED " + json.dumps(
        {"out": str(out), "assets": [e["asset"] for e in entries]},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
