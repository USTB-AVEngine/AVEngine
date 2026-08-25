#!/usr/bin/env python3
"""Publish accepted generated animals into the shared sound-source asset tree.

    <root>/index.json                      the only file a consumer must read
    <root>/<category>/                     what kind of thing this is
              <type>/                      the breed, model or product
                  <variant>/               one realized attribute combination
                      asset.json           the same record, kept beside the mesh
                      animated.glb
                      prepared.glb
                      walk/  turntable/    review renders
                      evidence/            gate reports and stage manifests

The tree holds every kind of sound source, not only animals: humans, the
universal speaker, the static appliances, the building fixtures. So the top level
is the category an engine asks for first - cat, dog, human, speaker, appliance -
and everything about how the asset is driven is a field inside its record, read
after the category has already narrowed the choice.

An earlier version of this layout put the body plan on top. That was wrong twice
over. It cannot hold a speaker or a human at all, and between the two animal body
plans it does not even distinguish anything: the felid and canid donors are the
same rig, 34 bones with identical names, hierarchy depth and action range,
differing only in their rotation curves, and nothing in the engine branches on
the id - it is checked for equality between template and actor and never read
otherwise. body_plan_id and motion_family_id stay in every record, which is where
a caller wants them, and an asset under one can be re-animated from another's
donor without rig work.

The variant is size and coat value for animals; body build and life stage were
dropped as instance axes by owner decision, being visually indistinguishable.

Several publishers write into one tree - this one handles generated animals, and
humans and static sources need their own - so the root may already exist and the
index is merged rather than rewritten. A leaf that already exists is still an
error: republishing a combination is a new version, not a silent replacement for
a reviewed asset.

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
    --root /data/avengine_external/assets/sound_source_assets_v1 \\
    --asset LADDER_WORKDIR=GENERATION_MANIFEST_JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "avengine_sound_source_asset_index_v1"
ASSET_SCHEMA = "avengine_sound_source_asset_v1"
PIPELINE = "generated_animal_ladder_v1"
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
    parser.add_argument(
        "--category", default=None,
        help="top-level category; defaults to the species, which is what an "
             "engine asks for first")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

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

        category = args.category or identity["species"]
        asset_id = (f"generated_{breed}_{coat_value}_{size}_"
                    f"{args.admission_state}_{args.revision}")
        relative = Path(category) / breed / f"{size}_{coat_value}"
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
            "category": category,
            "entity_class": "articulated_animal",
            "pipeline": PIPELINE,
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
        "layout": "<category>/<type>/<variant>",
        "layout_note": (
            "the top level is the category an engine asks for first, so the tree "
            "holds humans, the universal speaker and the static appliances "
            "alongside animals. How an asset is driven - body_plan_id, "
            "motion_family_id - is a field in its record, read after the "
            "category has narrowed the choice. Those two ids do not partition "
            "anything structural among the animals here: the felid and canid "
            "donors are the same rig down to bone names, hierarchy depth and "
            "action length, so an asset under one can be re-animated from the "
            "other's donor without rig work"),
        "instance_axes": {
            "articulated_animal": ["size", "coat_profile"],
        },
        "instance_axes_note": (
            "body_build and life_stage were dropped as animal axes by owner "
            "decision, being visually indistinguishable in a rendered frame"),
        "acceptance_gates": {PIPELINE: {
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
        }},
        "formal_dataset_registration_authorized": False,
        "assets": records,
    }

    index_path = root / "index.json"
    if index_path.is_file():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        known = {entry["asset_id"] for entry in index["assets"]}
        index["assets"] = sorted(
            index["assets"] + [entry for entry in existing.get("assets", [])
                               if entry["asset_id"] not in known],
            key=lambda entry: entry["asset_id"])
        merged_gates = dict(existing.get("acceptance_gates") or {})
        merged_gates.update(index["acceptance_gates"])
        index["acceptance_gates"] = merged_gates
        merged_axes = dict(existing.get("instance_axes") or {})
        merged_axes.update(index["instance_axes"])
        index["instance_axes"] = merged_axes
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("PUBLISHED " + json.dumps(
        {"root": str(root), "assets": [r["asset_id"] for r in records]},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
