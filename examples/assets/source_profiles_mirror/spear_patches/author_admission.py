#!/usr/bin/env python3
"""Author the heading evidence, anchor authorities and admission plan.

Both authorities bind the approved raw Pixal GLB by hash, so they can only be
written after the static-object decision batch exists.  The reviewed front yaw
is read off the rendered five-view sheet: the reviews runner renders with
--front-axis negative-y, so the camera named "front" sits at -Y, "side" at +X,
"back" at +Y and "quarter" between +X and -Y.  The finalizer rotates by
-reviewed_source_front_yaw_deg about Z to bring the front onto +X, so the yaw
to declare is the Blender-space yaw of the face that is actually the front:
  the front render shows it  -> -90
  the side render shows it   ->   0
  the back render shows it   ->  90
  the quarter render shows it-> -45
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SPEAR = Path("/data/jzy/code/SPEAR-lead-b")
ANCHOR_AUTHORITY_SCHEMA = "avengine_static_emitter_anchor_authority_v1"
HEADING_SCHEMA = "avengine_static_heading_review_v1"
PLAN_SCHEMA = "avengine_controlled_static_object_admission_plan_v1"

# Rigid manufactured objects: the raw I23D surface is the truth, so shrinkwrap
# fully back onto it and smooth as little as possible - a cabinet is boxy and
# smoothing rounds the corners a reviewer uses to recognise it.  The two
# animal-only repairs stay off.
WATERTIGHT_PARAMETERS = {
    "voxel_resolution": 256,
    "target_faces": 30000,
    "smooth_iterations": 1,
    "shrinkwrap_strength": 1.0,
    "post_shrinkwrap_smooth_iterations": 0,
    "torso_fold_repair_iterations": 0,
    "attribute_transfer_backend": "bake",
    "bake_resolution": 2048,
    "base_color_encoding_policy": "preserve-bake",
    "base_color_gain": [1.0, 1.0, 1.0],
    "double_sided": False,
}

# object_type -> (anchor_id, target_fraction_xyz in the finalized frame
#                 x=forward y=up z=right, why)
ANCHORS = {
    "bookshelf_speaker": (
        "woofer_cone_front_baffle",
        [1.0, 0.30, 0.5],
        "the woofer is the dominant radiator and it sits low on the baffle",
    ),
    "floorstanding_speaker": (
        "driver_array_front_baffle",
        [1.0, 0.65, 0.5],
        "centre of the four-driver column, which spans the upper two thirds",
    ),
    "soundbar": (
        "front_grille_centre",
        [1.0, 0.50, 0.5],
        "the whole front face is one grille, so its centre is the outlet",
    ),
    "smart_speaker": (
        "fabric_wrap_radiating_band",
        [1.0, 0.50, 0.5],
        "an omnidirectional wrap radiates at driver height; the +X side is "
        "one of many equivalent azimuths and keeps the frame convention",
    ),
    "television": (
        "bottom_bezel_grille_strip",
        [1.0, 0.08, 0.5],
        "the front grille strip under the screen",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def hash_without(value, key: str) -> str:
    return hashlib.sha256(
        canonical_json({k: v for k, v in value.items() if k != key}).encode("utf-8")
    ).hexdigest()


def file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-batch", required=True, type=Path)
    parser.add_argument("--decision-root", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--yaw",
        action="append",
        required=True,
        metavar="INSTANCE_SUFFIX=DEGREES",
        help="reviewed source front yaw per instance, read off the five views",
    )
    args = parser.parse_args()

    yaws = {}
    for item in args.yaw:
        suffix, _, value = item.partition("=")
        yaws[suffix] = float(value)

    batch = json.loads(args.decision_batch.read_text(encoding="utf-8"))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    instances = []
    for entry in batch["decisions"]:
        if entry["decision"] != "approved_for_watertight_finalization":
            continue
        instance_id = entry["instance_id"]
        decision = json.loads(
            (args.decision_root / entry["record"]["path"]).read_text(encoding="utf-8")
        )
        pixal = Path(decision["pixal_output"]["path"])
        pixal_sha = decision["pixal_output"]["sha256"]
        if sha256_file(pixal) != pixal_sha:
            raise SystemExit(f"{instance_id}: raw Pixal GLB hash moved")

        matched = [suffix for suffix in yaws if instance_id.endswith(suffix)]
        if len(matched) != 1:
            raise SystemExit(f"{instance_id}: {len(matched)} yaw entries matched")
        yaw = yaws[matched[0]]

        object_type = next(
            key for key in ANCHORS
            if instance_id.startswith(f"audio_playback_{key}_")
        )
        anchor_id, fraction, _why = ANCHORS[object_type]

        # The evidence for both authorities is the rendered five-view contact
        # sheet: it is the image the front direction and the outlet were read
        # off.  One artifact, two authorities, no second rendering pass.
        sheet = args.review_root / instance_id / "contact_sheet.png"
        if not sheet.is_file():
            raise SystemExit(f"{instance_id}: no contact sheet at {sheet}")
        evidence = file_record(sheet)

        heading = {
            "schema": HEADING_SCHEMA,
            "instance_id": instance_id,
            "request_sha256": decision["request_sha256"],
            "profile_sha256": decision["profile_sha256"],
            "input_glb_sha256": pixal_sha,
            "review_artifact": evidence,
            "reviewed_source_front_yaw_deg": yaw,
            "target_front_axis": "positive-x",
            "decision": "approved_for_positive_x_normalization",
            "formal_dataset_registration_authorized": False,
        }
        heading_path = out / instance_id / "heading.json"
        heading_path.parent.mkdir(parents=True, exist_ok=True)
        heading_path.write_text(
            json.dumps(heading, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )

        anchor = {
            "schema": ANCHOR_AUTHORITY_SCHEMA,
            "instance_id": instance_id,
            "request_sha256": decision["request_sha256"],
            "profile_sha256": decision["profile_sha256"],
            "input_glb_sha256": pixal_sha,
            "anchor_id": anchor_id,
            "anchor_type": "object_speaker",
            "semantic_role": "primary_sound_source",
            "selection": {
                "method": "reviewed_bbox_fraction_nearest_surface_v1",
                "samples": [{"target_fraction_xyz": fraction, "weight": 1.0}],
                "aggregation": "weighted_centroid",
                "maximum_search_distance_fraction": 0.5,
            },
            "review_evidence": evidence,
            "formal_dataset_registration_authorized": False,
        }
        anchor_path = out / instance_id / "anchor.json"
        anchor_path.write_text(
            json.dumps(anchor, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )

        instances.append(
            {
                "instance_id": instance_id,
                "heading_evidence_path": str(heading_path.resolve()),
                "anchor_spec_path": str(anchor_path.resolve()),
                "watertight_parameters": dict(WATERTIGHT_PARAMETERS),
            }
        )

    plan = {
        "schema": PLAN_SCHEMA,
        "decision_batch_sha256": batch["decision_batch_sha256"],
        "instances": instances,
        "formal_dataset_registration_authorized": False,
    }
    plan["plan_sha256"] = hash_without(plan, "plan_sha256")
    plan_path = out / "admission_plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"authored {len(instances)} instances -> {plan_path}")
    for item in instances:
        print(f"  {item['instance_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
