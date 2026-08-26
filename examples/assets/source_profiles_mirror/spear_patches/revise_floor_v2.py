#!/usr/bin/env python3
"""The tower profile carries the same two faults the bookshelf one did.

Its fixed form_factor is three_way_floor_tower, which names a crossover
topology rather than a form, and its positive template pins four drivers by
name and position. The bookshelf profile made both promises and could only keep
them one time in three; there is no reason to wait for the tower to prove the
same thing on its own batch.

The level camera comes along too, measured to reduce reconstruction tilt on
flat-sided objects. All three v1 towers failed the 3D gate outright - a
hallucinated bottle fused to the plinth, dangling wire geometry, and one
cabinet reconstructed split open with the baffle detached - so this batch has
to be regenerated regardless of the declaration.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SPEAR = Path("/data/jzy/code/SPEAR-lead-b")
DATA = SPEAR / "data/controlled_source_attributes_v1"
PROFILE_DIR = DATA / "candidate_profiles/static_object"
REVISION_DIR = DATA / "candidate_profile_revisions/static_object"
REVISION = "2026_08_26_v2_form_not_crossover_and_level_camera"
SCHEMA_ID = "audio_playback_floorstanding_speaker_product_view_v1"

sys.path.insert(0, str(SPEAR))
from tools import controlled_source_asset_schema as contracts  # noqa: E402

OLD_VIEW = (
    "Three-quarter product view from slightly above eye level so the front and "
    "one side are both visible."
)
NEW_VIEW = (
    "Three-quarter product view with the camera exactly level at the object's "
    "own mid-height, looking straight ahead and never downward, so the front "
    "and one side are both visible while the object's vertical edges stay "
    "parallel to the sides of the frame and its base line stays horizontal."
)
NEW_POSITIVE = (
    "One {finish} {material} {form_factor} {object_type}, a household "
    "{category}: a tall slim upright loudspeaker column resting directly on the "
    "floor on a rectangular plinth, roughly five times taller than it is wide. "
    "Its front baffle carries round drivers stacked on the vertical centerline, "
    "large woofer cones low down and a small dome tweeter at the very top. The "
    "side and back panels are plain and flat with no controls and no cloth, and "
    "the cabinet is one closed sealed box with nothing hanging off it."
)
EXTRA_NEGATIVE = (
    "high angle, looking down at the object, elevated viewpoint, top-down view, "
    "tilted camera, dutch angle, leaning object, open cabinet, detached panel, "
    "exploded view, loose baffle, cables, wires, objects on the plinth, "
    "bottle, vase, ornament beside the speaker"
)

path = PROFILE_DIR / f"{SCHEMA_ID}.json"
old = json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(contracts.canonical_json(payload).encode("utf-8")).hexdigest()


def file_record(target: Path, profile: dict) -> dict:
    data = target.read_bytes()
    return {
        "path": str(target.relative_to(SPEAR)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "profile_schema_id": profile["profile_schema_id"],
        "profile_revision": profile["profile_revision"],
        "canonical_sha256": canonical_sha256(profile),
    }


old_record = file_record(path, old)
new = json.loads(json.dumps(old))
new["profile_revision"] = REVISION
new["fixed_attributes"]["form_factor"] = "floor_standing_tower"
contract = new["generation_contract"]
contract["prompt_template_id"] = "static_product_view_t2i_v2_floor_standing_tower"
contract["positive_template"] = NEW_POSITIVE
contract["pose_guard_prompt"] = contract["pose_guard_prompt"].replace(OLD_VIEW, NEW_VIEW)
contract["negative_prompt"] = f"{contract['negative_prompt']}, {EXTRA_NEGATIVE}"
contract["value_labels"]["form_factor"] = {
    "floor_standing_tower": "floor-standing tower"
}
contracts.validate_attribute_profile(new)
path.write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

provenance = {
    "schema": "avengine_controlled_profile_method_revision_provenance_v1",
    "profile": file_record(path, new),
    "method_revision": {
        "kind": "declared_attribute_correction_and_level_camera",
        "supersedes_profile_revision": old["profile_revision"],
        "superseded_profile_canonical_sha256": old_record["canonical_sha256"],
        "reason": (
            "Three faults in one revision. The declared fixed attribute "
            "form_factor=three_way_floor_tower names a crossover topology "
            "rather than a form, the same mis-specification that held the "
            "bookshelf profile to a one in three yield across two prompt "
            "methods; the positive template pinned four drivers by name and "
            "position, which is the count constraint that generator could not "
            "keep; and all three v1 towers failed the 3D gate outright, so the "
            "batch has to be regenerated in any case. The level camera is "
            "carried in because it measurably reduces reconstruction tilt on "
            "flat-sided objects. The negative prompt now also guards the three "
            "specific 3D failures seen: an object fused to the plinth, hanging "
            "wire geometry, and a cabinet reconstructed split open."
        ),
        "this_is_not_a_seed_retry": True,
        "same_candidate_retry_allowed": False,
        "same_generation_seed_retry_allowed": False,
        "candidate_ranking_allowed": False,
        "next_execution_requires_new_profile_bound_request": True,
        "next_execution_requires_new_batch_seed": True,
    },
    "triggering_failure": {
        "reviewer": (
            "first_author_claude_visual_review_under_project_lead_delegation_20260826"
        ),
        "rejected_instances": {
            "audio_playback_floorstanding_speaker_product_view_0dcc002103d7": (
                "5.88 percent debris including a hallucinated bottle on an "
                "extended base slab; 22.5 degrees off upright"
            ),
            "audio_playback_floorstanding_speaker_product_view_995cd26ea881": (
                "3.03 percent debris as dangling wire and rod geometry at the base"
            ),
            "audio_playback_floorstanding_speaker_product_view_e78b6e647ac6": (
                "cabinet reconstructed split open, baffle detached, 10013 "
                "non-manifold edges"
            ),
        },
    },
}
out = REVISION_DIR / REVISION / f"{SCHEMA_ID}.provenance.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(provenance, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"revised {SCHEMA_ID} -> {REVISION}")
