#!/usr/bin/env python3
"""Publish accepted generated animals into a tree a program can consume.

Layout mirrors the lineage the assets actually have, coarse to fine:

    <root>/index.json                        the only file a consumer must read
    <root>/<body_plan_id>/                   which animations can bind at all
              <breed>/                       the source asset the breed came from
                  <size>_<coat_value>/       one realized attribute combination
                      asset.json             the same record, kept beside the mesh
                      animated.glb
                      prepared.glb
                      walk/  turntable/      review renders
                      evidence/              gate reports and stage manifests

The top level is the body plan rather than the species because it records which
donor gait was retargeted into the asset, which is the thing a consumer must not
get wrong. It is not a structural claim: the felid and canid donors are the same
rig - 34 bones, identical names, identical hierarchy depth, identical 41-frame
action range - differing only in their rotation curves. Nothing in the engine
branches on which body plan an asset has either; the id is checked for equality
against the template and the actor, and that is all. So a cat asset can be
re-animated with the canid gait and the reverse, and the folder says which one it
currently carries rather than which one it could accept.

Species and breed are in the index and in every asset record, so nothing is lost
by not foldering on them. The leaf is size and coat value only: body build and
life stage were dropped as instance axes by owner decision, being visually
indistinguishable.

index.json carries the gate criterion and its calibration alongside the assets,
so a consumer reading one file knows both what it has and on what basis it was
accepted. Every asset also carries its own asset.json, so a directory moved out
of this tree still describes itself.

Each record also carries the generation that produced it: the prompt as written,
its token accounting against the 512 window, the image model and snapshot, the
seed and sampling parameters, the clay pose guide with its hash, and the one-shot
policy that was in force. Without those an accepted asset cannot be reproduced or
varied deliberately, and the reason it looks the way it does is lost.

Nothing is overwritten. Republishing the same combination is a new asset that
needs a new version, not a silent replacement for a reviewed one.

Example::

  python tools/assets/publish_animal_assets.py \\
    --root /data/avengine_external/assets/generated_animals_v1 \\
    --asset LADDER_WORKDIR=GENERATION_MANIFEST_JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "avengine_generated_animal_asset_index_v1"
ASSET_SCHEMA = "avengine_generated_animal_asset_v1"
EVIDENCE = (
    "prepared.json", "walk_deformation.json", "ladder.json", "heading.json",
    "level.json", "retarget.json", "heading_probe.png",
)
MESHES = ("animated.glb", "prepared.glb")
RENDERS = ("walk", "turntable")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def generation_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def generation_record(manifest: dict) -> dict:
    """Everything needed to reproduce the image the mesh was built from."""
    prompt = dict(manifest.get("prompt") or {})
    return {
        "manifest_schema": manifest.get("schema"),
        "prompt": prompt,
        "token_counts": manifest.get("token_counts"),
        "model": manifest.get("model"),
        "parameters": manifest.get("parameters"),
        "clay_guide": manifest.get("clay_guide"),
        "one_shot_execution": manifest.get("one_shot_execution"),
        "candidate_image": manifest.get("candidate"),
        "gate_at_generation": manifest.get("gate"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--asset", action="append", required=True,
                        metavar="LADDER_WORKDIR=GENERATION_MANIFEST_JSON")
    parser.add_argument("--admission-state", default="research")
    parser.add_argument("--revision", default="v1")
    args = parser.parse_args()

    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"root exists, refusing to overwrite: {root}")

    records = []
    for spec in args.asset:
        workdir, _, manifest_file = spec.partition("=")
        src = Path(workdir)
        manifest = generation_manifest(Path(manifest_file))
        identity = manifest.get("identity", manifest)
        coat = identity["coat_profile"]
        coat_value = coat.get("baseline_value")
        size = identity["realized_attributes"]["size"]
        breed = identity["breed"]
        body_plan = identity["body_plan_id"]

        ladder = json.loads((src / "ladder.json").read_text(encoding="utf-8"))
        walk = json.loads((src / "walk_deformation.json").read_text(encoding="utf-8"))
        prepared = json.loads((src / "prepared.json").read_text(encoding="utf-8"))

        asset_id = (f"generated_{breed}_{coat_value}_{size}_"
                    f"{args.admission_state}_{args.revision}")
        relative = Path(body_plan) / breed / f"{size}_{coat_value}"
        dest = root / relative
        if dest.exists():
            raise SystemExit(f"{relative} already published; bump --revision")
        dest.mkdir(parents=True)

        for item in MESHES:
            shutil.copy2(src / item, dest / item)
        (dest / "evidence").mkdir()
        for item in EVIDENCE:
            if (src / item).is_file():
                shutil.copy2(src / item, dest / "evidence" / item)
        for folder in RENDERS:
            if (src / folder).is_dir():
                shutil.copytree(src / folder, dest / folder)

        record = {
            "schema": ASSET_SCHEMA,
            "asset_id": asset_id,
            "path": str(relative),
            "entity_class": "articulated_animal",
            "identity": {
                "species": identity["species"],
                "breed": breed,
                "morphotype": identity.get("morphotype"),
                "body_plan_id": body_plan,
                "motion_family_id": identity.get("motion_family_id"),
            },
            "realized_attributes": {
                "size": size,
                "coat_profile": {"profile_id": coat.get("profile_id"),
                                 "value": coat_value},
            },
            "acoustic_profile": identity.get("acoustic_profile"),
            "generation": generation_record(manifest),
            "generation_manifest_path": str(Path(manifest_file)),
            "geometry": {
                "animated_glb": "animated.glb",
                "animated_glb_sha256": digest(dest / "animated.glb"),
                "faces": walk["faces"],
                "source_mesh": prepared.get("input"),
                "preparation_rung": ladder["accepted_rung"],
                "ladder_tried": ladder["ladder"],
                "pick": ladder.get("pick"),
            },
            "acceptance": {
                "worst_share_area_shards": walk["worst_share_area_shards"],
                "worst_share_area_shards_visible":
                    walk.get("worst_share_area_shards_visible"),
                "worst_frame": walk["worst_frame_by_shards"],
                "head_third_survival":
                    prepared.get("band_survival", {}).get("front"),
            },
            "admission_state": args.admission_state,
            "formal_dataset_registration_authorized": False,
        }
        (dest / "asset.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        records.append(record)

    index = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revision": args.revision,
        "layout": "<body_plan_id>/<breed>/<size>_<coat_value>",
        "layout_note": (
            "the top level records which donor gait is baked into the asset, not "
            "a skeletal difference: the felid and canid donors are the same rig "
            "down to bone names, hierarchy depth and action length, and differ "
            "only in their rotation curves. No engine code branches on the body "
            "plan; it is checked for equality between template and actor. Assets "
            "under one body plan can therefore be re-animated from another's "
            "donor without rig work"),
        "instance_axes": ["size", "coat_profile"],
        "instance_axes_note": (
            "body_build and life_stage were dropped as axes by owner decision, "
            "being visually indistinguishable in a rendered frame"),
        "acceptance_gate": {
            "pre_rig": {
                "tool": "tools/assets/gate_retopology.py",
                "criteria": ["head-third survival >= 0.7", "face target reached"],
                "why": ("a reduction that starved the head is invisible to any "
                        "tearing measurement: the asset whose face collapsed "
                        "tears less than one that looks fine"),
            },
            "post_rig": {
                "tool": "tools/assets/gate_rigged_asset.py",
                "criterion": "worst_share_area_shards <= 0.025",
                "calibration": ("owner judgement on rigged walk videos at "
                                "ordinary viewing distance; accepted versions "
                                "reach 0.0192 and the rejected one sits at 0.0371"),
                "known_variance": ("the same ladder rung has measured 0.0187 to "
                                   "0.0263 across six rigs, so a reading within "
                                   "about 17 percent of the threshold is inside "
                                   "the noise"),
            },
        },
        "formal_dataset_registration_authorized": False,
        "assets": records,
    }
    (root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("PUBLISHED " + json.dumps(
        {"root": str(root), "assets": [r["asset_id"] for r in records]},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
