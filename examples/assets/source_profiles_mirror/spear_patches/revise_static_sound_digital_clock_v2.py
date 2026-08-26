#!/usr/bin/env python3
"""Revise the digital alarm-clock method after the first candidate lacked clock identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SCHEMA_ID = "household_clock_alarm_clock_digital_cube_product_view_v1"
OLD_REVISION = "2026_08_26_v1_static_sound_forms"
OLD_CANONICAL_SHA256 = "0765d0c61b7db1dfd742fb96ca3692d655fba724e3f567981d546a1f91373226"
NEW_REVISION = "2026_08_26_v2_digital_clock_identity_method_revision"
FAILED_INSTANCE = "household_clock_alarm_clock_digital_cube_product_view_1049b0ad0907"
FAILED_REQUEST_SHA256 = "1049b0ad09070cba7f4f5b0eada1ab0f246fe9a3f84b15d1bc6b6b1e48a85075"
FAILED_CANDIDATE_SHA256 = "d1c6222702c05e0b2623c72a1c810b5f6a29c96d9fd20990d3cf3b8a9b9fd9b8"


def canonical_sha256(payload: dict, contracts) -> str:
    return hashlib.sha256(contracts.canonical_json(payload).encode("utf-8")).hexdigest()


def file_record(path: Path, profile: dict, root: Path, contracts) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "profile_schema_id": profile["profile_schema_id"],
        "profile_revision": profile["profile_revision"],
        "canonical_sha256": canonical_sha256(profile, contracts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--avengine-root", required=True, type=Path)
    args = parser.parse_args()

    spear_profile = (
        args.spear_root
        / "data/controlled_source_attributes_v1/candidate_profiles/static_object"
        / f"{SCHEMA_ID}.json"
    )
    mirror_profile = (
        args.avengine_root
        / "examples/assets/source_profiles_mirror/candidate_profiles/static_object"
        / f"{SCHEMA_ID}.json"
    )
    if spear_profile.read_bytes() != mirror_profile.read_bytes():
        raise RuntimeError("SPEAR and AVEngine digital-clock profiles already diverged")

    sys.path.insert(0, str(args.spear_root))
    from tools import controlled_source_asset_schema as contracts

    old = json.loads(spear_profile.read_text(encoding="utf-8"))
    if old["profile_revision"] != OLD_REVISION:
        raise RuntimeError(f"unexpected old revision: {old['profile_revision']}")
    if canonical_sha256(old, contracts) != OLD_CANONICAL_SHA256:
        raise RuntimeError("old digital-clock canonical profile identity changed")

    new = json.loads(json.dumps(old))
    new["profile_revision"] = NEW_REVISION
    contract = new["generation_contract"]
    contract["prompt_template_id"] = "static_product_view_t2i_v2_digital_clock_segments_snooze"
    contract["positive_template"] = (
        "One {body_color} {material} {form_factor} {object_type}, a household "
        "{category}: a compact rectangular bedside digital alarm clock standing "
        "on two low feet. Its front has one wide smoked display window with four "
        "large unlit seven-segment numeral outlines reading 12:00 and one colon, "
        "so it unmistakably reads as a clock rather than a television. One long "
        "rectangular snooze bar sits on top. A separate field of small perforated "
        "buzzer holes is clearly visible on the upper front bezel beside the display."
    )
    old_guard = contract["pose_guard_prompt"]
    required = (
        "No humans, no hands, no text, no logos, no labels, no screens showing "
        "content,"
    )
    replacement = (
        "No humans, no hands, no letters, no words, no logos, no labels, no screen "
        "imagery beyond the specified unlit clock digits,"
    )
    if old_guard.count(required) != 1:
        raise RuntimeError("pose guard text did not match the reviewed v1 method")
    contract["pose_guard_prompt"] = old_guard.replace(required, replacement, 1)
    contract["negative_prompt"] = (
        f"{contract['negative_prompt']}, television, miniature television, monitor, "
        "computer screen, smart speaker, generic speaker box, blank featureless "
        "front, completely blank display, side-only grille, missing snooze bar, "
        "radio antenna, camera lens"
    )
    contracts.validate_attribute_profile(new)

    content = json.dumps(new, ensure_ascii=False, indent=1) + "\n"
    spear_profile.write_text(content, encoding="utf-8")
    mirror_profile.write_text(content, encoding="utf-8")

    provenance = {
        "schema": "avengine_controlled_profile_method_revision_provenance_v1",
        "profile": file_record(spear_profile, new, args.spear_root, contracts),
        "method_revision": {
            "kind": "digital_clock_category_identity_method_revision",
            "supersedes_profile_revision": OLD_REVISION,
            "superseded_profile_canonical_sha256": OLD_CANONICAL_SHA256,
            "reason": (
                "The v1 candidate was a black cube with a blank inset panel and a "
                "large side grille, so it read as a miniature television or generic "
                "speaker box rather than a digital alarm clock. This is a method "
                "failure, not an unlucky seed: the v1 method explicitly requested a "
                "blank display and supplied no clock-specific snooze control. The v2 "
                "method permits only functional unlit seven-segment clock digits, adds "
                "a snooze bar, and puts the buzzer holes on the visible front bezel."
            ),
            "this_is_not_a_seed_retry": True,
            "same_candidate_retry_allowed": False,
            "same_generation_seed_retry_allowed": False,
            "candidate_ranking_allowed": False,
            "next_execution_requires_new_profile_bound_request": True,
            "next_execution_requires_new_batch_seed": True,
            "forbidden_replay_request_sha256": [FAILED_REQUEST_SHA256],
            "forbidden_replay_candidate_sha256": [FAILED_CANDIDATE_SHA256],
        },
        "triggering_failure": {
            "review_batch": "tmp/static_sound_sources_first_pass_20260826_r1/review_2d/review_batch_manifest.json",
            "reviewer": "codex_full_resolution_visual_review_under_owner_instruction_20260826",
            "instances": [
                {
                    "instance_id": FAILED_INSTANCE,
                    "request_sha256": FAILED_REQUEST_SHA256,
                    "candidate_sha256": FAILED_CANDIDATE_SHA256,
                    "sampled_attributes": {"body_color": "black"},
                }
            ],
        },
    }

    spear_provenance = (
        args.spear_root
        / "data/controlled_source_attributes_v1/candidate_profile_revisions/static_object"
        / NEW_REVISION
        / "provenance.json"
    )
    mirror_provenance = (
        args.avengine_root
        / "examples/assets/source_profiles_mirror/candidate_profile_revisions/static_object"
        / NEW_REVISION
        / "provenance.json"
    )
    for path in (spear_provenance, mirror_provenance):
        if path.exists():
            raise RuntimeError(f"refusing to replace method provenance: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"STATIC_DIGITAL_CLOCK_METHOD_REVISION_OK {OLD_REVISION} -> {NEW_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
