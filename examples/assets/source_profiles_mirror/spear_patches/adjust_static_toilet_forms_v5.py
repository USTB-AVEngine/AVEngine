#!/usr/bin/env python3
"""Create two toilet forms with exposed, reviewable flush-water paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ID = "plumbing_fixture_toilet_floor_close_coupled_product_view_v1"
FLUSH_ID = "plumbing_fixture_toilet_floor_flushometer_product_view_v1"
HIGH_ID = "plumbing_fixture_toilet_high_tank_pull_chain_product_view_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--avengine-root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.spear_root))
    from tools import controlled_source_asset_schema as contracts

    spear_dir = args.spear_root / "data/controlled_source_attributes_v1/candidate_profiles/static_object"
    mirror_dir = args.avengine_root / "examples/assets/source_profiles_mirror/candidate_profiles/static_object"
    source_left = spear_dir / f"{SOURCE_ID}.json"
    source_right = mirror_dir / f"{SOURCE_ID}.json"
    if source_left.read_bytes() != source_right.read_bytes():
        raise RuntimeError("source toilet profile diverged")
    source = json.loads(source_left.read_text())

    def make(schema_id: str, revision: str, form: str, form_label: str, positive: str, negative: str, height: int, source_id: str) -> dict:
        payload = json.loads(json.dumps(source))
        payload["profile_schema_id"] = schema_id
        payload["profile_revision"] = revision
        payload["fixed_attributes"]["form_factor"] = form
        contract = payload["generation_contract"]
        contract["prompt_template_id"] = f"static_product_view_t2i_v1_{form}"
        contract["positive_template"] = positive
        contract["negative_prompt"] = contract["negative_prompt"] + ", " + negative
        contract["value_labels"]["form_factor"] = {form: form_label}
        physical = payload["target_physical_profiles"]
        physical["profile_id"] = f"{schema_id}_physical_candidate_v1"
        physical["reference_value_cm"] = height
        physical["reference_provenance"]["source_id"] = source_id
        physical["values"]["fixed"] = {"target_value_cm": height, "tolerance_cm": 12 if height < 100 else 20}
        payload["qa_contract"]["subject_label"] = form_label
        contracts.validate_attribute_profile(payload)
        return payload

    flush = make(
        FLUSH_ID,
        "2026_08_26_v1_floor_flushometer_form_adjustment",
        "floor_flushometer",
        "floor toilet with exposed flushometer",
        (
            "One {body_color} {material} {form_factor} {object_type}, a household "
            "{category}: one floor-standing commercial toilet bowl on one ceramic "
            "pedestal with no tank and no cistern. Directly above the rear of the bowl "
            "is one exposed chrome flushometer valve with a side lever and one thick "
            "vertical chrome flush pipe entering a clearly visible inlet at the rear "
            "of the bowl. The valve-to-pipe-to-bowl water path is unobstructed."
        ),
        "tank, cistern, close-coupled toilet, hidden plumbing, pipe behind bowl, missing flush valve, wall-hung bowl, closed lid, lowered seat",
        78,
        "floor_flushometer_toilet_typical_retail_dimension_v1",
    )
    high = make(
        HIGH_ID,
        "2026_08_26_v1_high_tank_pull_chain_form_adjustment",
        "high_tank_pull_chain",
        "high-tank pull-chain toilet",
        (
            "One {body_color} {material} {form_factor} {object_type}, a household "
            "{category}: one complete traditional high-level toilet assembly. A small "
            "rectangular cistern is elevated far above the floor-standing bowl and is "
            "physically connected to the rear bowl inlet by one long exposed vertical "
            "chrome flush pipe. One pull chain hangs from the high cistern. The entire "
            "tank, pipe, open seat and bowl fit in frame, and the pipe-to-bowl inlet is visible."
        ),
        "low tank, close-coupled tank, tank touching bowl, hidden flush pipe, missing pipe, disconnected tank, wall, bathroom scene, closed lid, lowered seat",
        175,
        "high_tank_pull_chain_toilet_typical_retail_dimension_v1",
    )

    for payload in (flush, high):
        name = f"{payload['profile_schema_id']}.json"
        left, right = spear_dir / name, mirror_dir / name
        if left.exists() or right.exists():
            raise RuntimeError(f"refusing to replace adjusted toilet profile: {name}")
    for payload in (flush, high):
        content = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
        name = f"{payload['profile_schema_id']}.json"
        (spear_dir / name).write_text(content)
        (mirror_dir / name).write_text(content)

    record = {
        "owner_scope": "static_sound_sources_first_pass_20260826",
        "supersedes_adjustment_record": "examples/assets/source_profiles_mirror/static_sound_form_adjustments_20260826.json",
        "adjustments": [
            {
                "replaced_profiles": [SOURCE_ID, "plumbing_fixture_toilet_floor_one_piece_product_view_v1"],
                "replacement_profiles": [FLUSH_ID, HIGH_ID],
                "reason": "Four close-coupled attempts and one one-piece attempt lacked a reviewable flush outlet. Exposed flushometer and high-tank forms keep two visibly distinct toilet assets while making the valve/pipe-to-bowl water path physically visible.",
                "old_profiles_retained_as_rejected_method_evidence": True,
            }
        ],
    }
    path = args.avengine_root / "examples/assets/source_profiles_mirror/static_sound_form_adjustments_20260826_v2.json"
    if path.exists():
        raise RuntimeError(f"refusing to replace {path}")
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(f"STATIC_TOILET_FORM_ADJUSTMENTS_OK profiles=2 record={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
