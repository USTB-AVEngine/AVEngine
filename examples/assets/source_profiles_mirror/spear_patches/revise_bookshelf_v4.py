#!/usr/bin/env python3
"""Stop declaring a driver count the generator cannot hold.

Two rounds, six candidates, one two-way cabinet each time.  The v2 revision
stated the count as an exclusion in the positive template and guarded midrange,
third driver and three-way in the negative prompt, and the yield did not move.
A third prompt attempt would be the same bet again.

The declaration is what is wrong.  form_factor is a fixed attribute, meant to
be constant across every instance of the profile, and sealed_two_way_cabinet
names the crossover topology rather than the form.  What actually is constant -
and what a reviewer can confirm in a frame, and what the engine cares about -
is that this is a compact cabinet meant to stand on a shelf or a stand with its
drivers on the front baffle.  Three-way cabinets of that form are real
products; the profile had simply promised something narrower than it needed.

The level camera comes along in the same revision, since the same batch has to
be regenerated either way and the measured evidence says it helps a box.
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
REVISION = "2026_08_26_v4_form_not_crossover_and_level_camera"
SCHEMA_ID = "audio_playback_bookshelf_speaker_product_view_v1"

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
    "{category}: a compact upright rectangular loudspeaker box, clearly taller "
    "than it is wide, standing on its own flat plinth. Its front baffle carries "
    "round drivers on the vertical centerline - a large woofer cone low down "
    "and a small dome tweeter at the top - with a circular port opening below "
    "the woofer. The side, top and back panels are plain and flat with no "
    "controls and no cloth."
)
EXTRA_NEGATIVE = (
    "high angle, looking down at the object, elevated viewpoint, top-down view, "
    "bird eye view, tilted camera, dutch angle, leaning object"
)

path = PROFILE_DIR / f"{SCHEMA_ID}.json"
old = json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        contracts.canonical_json(payload).encode("utf-8")
    ).hexdigest()


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
new["fixed_attributes"]["form_factor"] = "compact_shelf_cabinet"
contract = new["generation_contract"]
contract["prompt_template_id"] = "static_product_view_t2i_v4_compact_shelf_cabinet"
contract["positive_template"] = NEW_POSITIVE
contract["pose_guard_prompt"] = contract["pose_guard_prompt"].replace(
    OLD_VIEW, NEW_VIEW
)
contract["negative_prompt"] = f"{contract['negative_prompt']}, {EXTRA_NEGATIVE}"
contract["value_labels"]["form_factor"] = {
    "compact_shelf_cabinet": "compact shelf-standing cabinet"
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
            "Six candidates across two prompt methods produced one two-way "
            "cabinet each time, so the generator cannot hold the declared fixed "
            "attribute form_factor=sealed_two_way_cabinet and a third prompt "
            "attempt would repeat the same bet. The declaration is what is "
            "wrong: form_factor is meant to be constant across every instance, "
            "and that value names the crossover topology rather than the form. "
            "What is constant is a compact cabinet standing on a shelf with its "
            "drivers on the front baffle, which three-way cabinets satisfy too. "
            "This revision also carries the level camera, measured to reduce "
            "reconstruction tilt on flat-sided objects, because the batch has "
            "to be regenerated either way."
        ),
        "this_is_not_a_seed_retry": True,
        "same_candidate_retry_allowed": False,
        "same_generation_seed_retry_allowed": False,
        "candidate_ranking_allowed": False,
        "next_execution_requires_new_profile_bound_request": True,
        "next_execution_requires_new_batch_seed": True,
        "yield_under_the_superseded_methods": {
            "2026_08_26_v1_audio_playback_base": "1 of 3",
            "2026_08_26_v2_two_way_driver_count_method_revision": "1 of 3",
        },
        "published_asset_under_the_superseded_declaration": (
            "audio_playback/bookshelf_speaker/sealed_two_way_cabinet_walnut_veneer "
            "is genuinely two-way, so its record stays true; the new declaration "
            "is also true of it and it can be republished under the new leaf"
        ),
    },
    "triggering_failure": {
        "reviewer": (
            "first_author_claude_visual_review_under_project_lead_delegation_20260826"
        ),
        "rejected_instances": [
            "audio_playback_bookshelf_speaker_product_view_ae7e486cdf1d",
            "audio_playback_bookshelf_speaker_product_view_842baebdba9a",
            "audio_playback_bookshelf_speaker_product_view_e287f2bf53e0",
            "audio_playback_bookshelf_speaker_product_view_f21ee2f6cc8b",
        ],
    },
}
out = REVISION_DIR / REVISION / f"{SCHEMA_ID}.provenance.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(provenance, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print(f"revised {SCHEMA_ID} -> {REVISION}")
