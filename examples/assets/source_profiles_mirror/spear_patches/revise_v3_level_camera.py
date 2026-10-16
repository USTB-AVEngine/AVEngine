#!/usr/bin/env python3
"""Test whether the tilt can be cured at the cause: the camera in the prompt.

Every reconstruction arrives pitched 5 to 19 degrees, and the pose guard asks
for a "three-quarter product view from slightly above eye level".  Pixal
reconstructs in a frame tied to the input view, so the camera's downward pitch
is a plausible source.  If it is the source, a level camera fixes the whole
family for free and no chain change is needed.

Two profiles carry the test: the smart speaker leans most (16.4 to 18.7) and
the soundbar pays most (a flat bar tilted 15 degrees loses 40 percent of its
height to its own bounding box).
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
REVISION = "2026_08_26_v3_level_camera_method_revision"

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
EXTRA_NEGATIVE = (
    "high angle, looking down at the object, elevated viewpoint, top-down view, "
    "bird eye view, tilted camera, dutch angle, leaning object, object tipping over"
)

TARGETS = (
    "audio_playback_smart_speaker_product_view_v1",
    "audio_playback_soundbar_product_view_v1",
)


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        contracts.canonical_json(payload).encode("utf-8")
    ).hexdigest()


def file_record(path: Path, profile: dict) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(SPEAR)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "profile_schema_id": profile["profile_schema_id"],
        "profile_revision": profile["profile_revision"],
        "canonical_sha256": canonical_sha256(profile),
    }


REVISION_DIR.mkdir(parents=True, exist_ok=True)
for schema_id in TARGETS:
    path = PROFILE_DIR / f"{schema_id}.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    old_record = file_record(path, old)

    new = json.loads(json.dumps(old))
    new["profile_revision"] = REVISION
    contract = new["generation_contract"]
    if OLD_VIEW not in contract["pose_guard_prompt"]:
        raise SystemExit(f"{schema_id}: pose guard does not carry the old view text")
    contract["pose_guard_prompt"] = contract["pose_guard_prompt"].replace(
        OLD_VIEW, NEW_VIEW
    )
    contract["prompt_template_id"] = (
        contract["prompt_template_id"].replace("_v1_", "_v3_").replace("_v2_", "_v3_")
        if "_v1_" in contract["prompt_template_id"]
        or "_v2_" in contract["prompt_template_id"]
        else contract["prompt_template_id"] + "_v3"
    )
    contract["negative_prompt"] = f"{contract['negative_prompt']}, {EXTRA_NEGATIVE}"
    contracts.validate_attribute_profile(new)
    path.write_text(
        json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    provenance = {
        "schema": "avengine_controlled_profile_method_revision_provenance_v1",
        "profile": file_record(path, new),
        "method_revision": {
            "kind": "level_camera_upright_reconstruction_method_revision",
            "supersedes_profile_revision": old["profile_revision"],
            "superseded_profile_canonical_sha256": old_record["canonical_sha256"],
            "reason": (
                "Nine reconstructions from the v1 pose guard arrived 5.3 to 18.7 "
                "degrees off upright, measured by "
                "tools/assets/measure_static_upright_correction.py with two "
                "independent authorities agreeing within 1.3 degrees. The static "
                "finalizer applies yaw only, so the tilt survives into the "
                "published asset: a bookshelf speaker leaning 13.2 degrees loses "
                "9.6 percent of its height to its own bounding box and a soundbar "
                "leaning 14.4 loses 40.2. The pose guard asks for a view from "
                "slightly above eye level and Pixal reconstructs in a frame tied "
                "to the input view, which makes the camera pitch the likely "
                "cause. This revision asks for a level camera at the object's own "
                "mid-height instead, so that a fix at the cause can be measured "
                "against the same metric before any chain or contract change is "
                "considered."
            ),
            "this_is_not_a_seed_retry": True,
            "same_candidate_retry_allowed": False,
            "same_generation_seed_retry_allowed": False,
            "candidate_ranking_allowed": False,
            "next_execution_requires_new_profile_bound_request": True,
            "next_execution_requires_new_batch_seed": True,
            "measured_tilt_under_the_superseded_method_deg": {
                "audio_playback_smart_speaker_product_view_v1": [16.42, 17.10, 18.73],
                "audio_playback_soundbar_product_view_v1": [14.42, 18.10],
            },
        },
        "triggering_failure": {
            "measurement_report": (
                "/data/avengine_external/review/static_upright_20260826.json"
            ),
            "reviewer": (
                "first_author_claude_geometric_measurement_under_project_lead_"
                "delegation_20260826"
            ),
        },
    }
    out = REVISION_DIR / REVISION / f"{schema_id}.provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"revised {schema_id} -> {REVISION}")
