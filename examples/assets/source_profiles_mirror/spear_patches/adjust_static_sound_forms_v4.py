#!/usr/bin/env python3
"""Replace two persistently failing forms and strengthen the floor-toilet outlet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


FLOOR_ID = "plumbing_fixture_toilet_floor_close_coupled_product_view_v1"
PTRAP_ID = "plumbing_fixture_floor_drain_exposed_p_trap_product_view_v1"
WALL_ID = "plumbing_fixture_toilet_wall_hung_product_view_v1"
BOTTLE_ID = "plumbing_fixture_floor_drain_exposed_bottle_trap_product_view_v1"
ONE_PIECE_ID = "plumbing_fixture_toilet_floor_one_piece_product_view_v1"


def canonical(payload: dict, contracts) -> str:
    return hashlib.sha256(contracts.canonical_json(payload).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--avengine-root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.spear_root))
    from tools import controlled_source_asset_schema as contracts

    spear_dir = args.spear_root / "data/controlled_source_attributes_v1/candidate_profiles/static_object"
    mirror_dir = args.avengine_root / "examples/assets/source_profiles_mirror/candidate_profiles/static_object"

    def load_pair(schema_id: str) -> dict:
        left = spear_dir / f"{schema_id}.json"
        right = mirror_dir / f"{schema_id}.json"
        if left.read_bytes() != right.read_bytes():
            raise RuntimeError(f"{schema_id}: SPEAR and AVEngine diverged")
        return json.loads(left.read_text())

    floor_old = load_pair(FLOOR_ID)
    ptrap_old = load_pair(PTRAP_ID)
    wall_old = load_pair(WALL_ID)
    expected = {
        FLOOR_ID: ("2026_08_26_v2_visible_flush_jet_ring_method_revision", "d30d6065e5925439bc71975f9b40d7f63249a0c88f1377d6df6d009d4fec4d62"),
        PTRAP_ID: ("2026_08_26_v2_true_u_bend_water_seal_method_revision", "89c6c9decb5bd7037b2b69db6a10f14d2af58f54b5a0b703585abbf4685d3b5f"),
        WALL_ID: ("2026_08_26_v2_cisternless_wall_bowl_method_revision", "244b0fd39b7c0645d4c45025b2b88aeed5d3731623411af12c4ede30a5c21dd5"),
    }
    for schema_id, payload in ((FLOOR_ID, floor_old), (PTRAP_ID, ptrap_old), (WALL_ID, wall_old)):
        revision, digest = expected[schema_id]
        if payload["profile_revision"] != revision or canonical(payload, contracts) != digest:
            raise RuntimeError(f"{schema_id}: expected v2 profile changed")

    floor_new = json.loads(json.dumps(floor_old))
    floor_new["profile_revision"] = "2026_08_26_v3_large_rear_flush_outlet_method_revision"
    floor_contract = floor_new["generation_contract"]
    floor_contract["prompt_template_id"] = "static_product_view_t2i_v3_close_coupled_large_rear_flush_outlet"
    floor_contract["positive_template"] = (
        "One {body_color} {material} {form_factor} {object_type}, a household "
        "{category}: one floor-standing close-coupled toilet with one integrated "
        "rear cistern. The seat and lid are raised. Use a high three-quarter view "
        "into the empty bowl. On the vertical inner rear wall above the water line, "
        "one large dark horizontal flush-water outlet slot is plainly visible, at "
        "least one quarter of the bowl width and tall enough to remain obvious."
    )
    floor_contract["negative_prompt"] += (
        ", small invisible rim holes, smooth rear bowl wall, hidden flush outlet, "
        "flush outlet under an overhang, water covering outlet, closed lid"
    )
    contracts.validate_attribute_profile(floor_new)

    bottle = json.loads(json.dumps(ptrap_old))
    bottle["profile_schema_id"] = BOTTLE_ID
    bottle["profile_revision"] = "2026_08_26_v1_bottle_trap_form_adjustment"
    bottle["fixed_attributes"]["form_factor"] = "exposed_bottle_trap"
    bc = bottle["generation_contract"]
    bc["prompt_template_id"] = "static_product_view_t2i_v1_exposed_bottle_trap"
    bc["positive_template"] = (
        "One {body_color} {material} {form_factor} {object_type}, a household "
        "{category}: one exposed chrome bottle-trap assembly with a short vertical "
        "inlet entering the centre of one sealed cylindrical bottle chamber and one "
        "single horizontal wall outlet leaving the upper side. The cylinder has a "
        "flat removable bottom cap and exactly two pipe ends. The circular top inlet "
        "is clearly visible."
    )
    bc["negative_prompt"] += (
        ", P trap, U bend, tee fitting, T junction, three openings, straight-through "
        "pipe, multiple cylinders, disconnected pipes, sink, wall, room scene"
    )
    bc["value_labels"]["form_factor"] = {"exposed_bottle_trap": "exposed cylindrical bottle trap"}
    bottle["target_physical_profiles"]["profile_id"] = f"{BOTTLE_ID}_physical_candidate_v1"
    bottle["target_physical_profiles"]["reference_provenance"]["source_id"] = "exposed_bottle_trap_typical_retail_dimension_v1"
    bottle["qa_contract"]["subject_label"] = "exposed bottle trap"
    contracts.validate_attribute_profile(bottle)

    one_piece = json.loads(json.dumps(wall_old))
    one_piece["profile_schema_id"] = ONE_PIECE_ID
    one_piece["profile_revision"] = "2026_08_26_v1_one_piece_floor_form_adjustment"
    one_piece["fixed_attributes"]["form_factor"] = "floor_one_piece"
    oc = one_piece["generation_contract"]
    oc["prompt_template_id"] = "static_product_view_t2i_v1_floor_one_piece_visible_rear_outlet"
    oc["positive_template"] = (
        "One {body_color} {material} {form_factor} {object_type}, a household "
        "{category}: one smooth floor-standing one-piece toilet whose cistern, bowl "
        "and floor base form one continuous glazed ceramic shell with no tank-to-bowl "
        "seam. The seat and lid are raised. Use a high three-quarter view into the "
        "empty bowl. One large dark horizontal flush-water outlet slot is plainly "
        "visible on the vertical inner rear wall above the water line."
    )
    oc["pose_guard_prompt"] = floor_old["generation_contract"]["pose_guard_prompt"]
    oc["negative_prompt"] = floor_old["generation_contract"]["negative_prompt"] + (
        ", separate tank, visible tank-to-bowl seam, wall-hung bowl, hidden flush "
        "outlet, smooth rear bowl wall, water covering outlet, closed lid"
    )
    oc["value_labels"]["form_factor"] = {"floor_one_piece": "seamless floor-standing one-piece"}
    one_piece["target_physical_profiles"]["profile_id"] = f"{ONE_PIECE_ID}_physical_candidate_v1"
    one_piece["target_physical_profiles"]["reference_value_cm"] = 72
    one_piece["target_physical_profiles"]["reference_provenance"]["source_id"] = "floor_one_piece_toilet_typical_retail_dimension_v1"
    one_piece["target_physical_profiles"]["values"]["fixed"] = {"target_value_cm": 72, "tolerance_cm": 10}
    one_piece["qa_contract"]["subject_label"] = "one-piece floor toilet"
    contracts.validate_attribute_profile(one_piece)

    new_paths = [
        (spear_dir / f"{BOTTLE_ID}.json", mirror_dir / f"{BOTTLE_ID}.json", bottle),
        (spear_dir / f"{ONE_PIECE_ID}.json", mirror_dir / f"{ONE_PIECE_ID}.json", one_piece),
    ]
    for left, right, _payload in new_paths:
        if left.exists() or right.exists():
            raise RuntimeError(f"refusing to replace adjusted profile: {left} / {right}")
    provenance_revision = floor_new["profile_revision"]
    provenance_paths = [
        args.spear_root / "data/controlled_source_attributes_v1/candidate_profile_revisions/static_object" / provenance_revision / "provenance.json",
        args.avengine_root / "examples/assets/source_profiles_mirror/candidate_profile_revisions/static_object" / provenance_revision / "provenance.json",
    ]
    if any(path.exists() for path in provenance_paths):
        raise RuntimeError("refusing to replace floor-toilet v3 provenance")

    floor_content = json.dumps(floor_new, ensure_ascii=False, indent=1) + "\n"
    (spear_dir / f"{FLOOR_ID}.json").write_text(floor_content)
    (mirror_dir / f"{FLOOR_ID}.json").write_text(floor_content)
    for left, right, payload in new_paths:
        content = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
        left.write_text(content)
        right.write_text(content)

    provenance = {
        "schema": "avengine_controlled_profile_method_revision_provenance_v1",
        "profile": {
            "path": str((spear_dir / f"{FLOOR_ID}.json").relative_to(args.spear_root)),
            "sha256": hashlib.sha256(floor_content.encode()).hexdigest(),
            "size_bytes": len(floor_content.encode()),
            "profile_schema_id": FLOOR_ID,
            "profile_revision": floor_new["profile_revision"],
            "canonical_sha256": canonical(floor_new, contracts),
        },
        "method_revision": {
            "kind": "large_visible_rear_flush_outlet_method_revision",
            "supersedes_profile_revision": floor_old["profile_revision"],
            "superseded_profile_canonical_sha256": expected[FLOOR_ID][1],
            "reason": "Three requests across three batch seeds produced the correct close-coupled form but no reviewable rim jets. The v3 method replaces small rim details with one large rear-wall outlet that remains visible at product-view resolution.",
            "this_is_not_a_seed_retry": True,
            "same_candidate_retry_allowed": False,
            "same_generation_seed_retry_allowed": False,
            "candidate_ranking_allowed": False,
            "next_execution_requires_new_profile_bound_request": True,
            "next_execution_requires_new_batch_seed": True,
            "forbidden_replay_request_sha256": ["0d270aad9a0d16373325c9dabda070f267d179f0dbfc9654f110f18280301c36"],
            "forbidden_replay_candidate_sha256": ["b63d753daed10c23383eaed2b45fdcea15c74b8716dbb94d2226a3f269341c2c"],
        },
        "triggering_failure": {
            "review_batch": "tmp/static_sound_sources_method_retry_20260826_r3/review_2d/review_batch_manifest.json",
            "reviewer": "codex_full_resolution_visual_review_under_owner_instruction_20260826",
            "instances": [{"instance_id": "plumbing_fixture_toilet_floor_close_coupled_product_view_0d270aad9a0d", "request_sha256": "0d270aad9a0d16373325c9dabda070f267d179f0dbfc9654f110f18280301c36", "candidate_sha256": "b63d753daed10c23383eaed2b45fdcea15c74b8716dbb94d2226a3f269341c2c", "sampled_attributes": {"body_color": "white"}}],
        },
    }
    for path in provenance_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, ensure_ascii=False, indent=1) + "\n")

    adjustment = {
        "owner_scope": "static_sound_sources_first_pass_20260826",
        "adjustments": [
            {
                "replaced_profile": PTRAP_ID,
                "replacement_profile": BOTTLE_ID,
                "reason": "Three frozen candidates all became straight tee fittings; exposed bottle trap keeps an exposed water-seal form while remaining visually and generatively distinct.",
                "failed_instances": ["plumbing_fixture_floor_drain_exposed_p_trap_product_view_e4f7dfc371c3", "plumbing_fixture_floor_drain_exposed_p_trap_product_view_ba6c9a2c9253", "plumbing_fixture_floor_drain_exposed_p_trap_product_view_ec021022d584"],
            },
            {
                "replaced_profile": WALL_ID,
                "replacement_profile": ONE_PIECE_ID,
                "reason": "Three frozen wall-hung requests all became floor toilets with cisterns. A seamless one-piece floor form remains visibly distinct from the accepted close-coupled form without claiming a wall-hung mesh the method cannot produce.",
                "failed_instances": ["plumbing_fixture_toilet_wall_hung_product_view_dc3d760ab8b8", "plumbing_fixture_toilet_wall_hung_product_view_7b43d6aabbc5", "plumbing_fixture_toilet_wall_hung_product_view_7f28fa8a534b"],
            },
        ],
        "old_profiles_retained_as_rejected_method_evidence": True,
    }
    adjustment_path = args.avengine_root / "examples/assets/source_profiles_mirror/static_sound_form_adjustments_20260826.json"
    if adjustment_path.exists():
        raise RuntimeError(f"refusing to replace {adjustment_path}")
    adjustment_path.write_text(json.dumps(adjustment, ensure_ascii=False, indent=1) + "\n")
    print(f"STATIC_FORM_ADJUSTMENTS_OK new_profiles=2 revised_profiles=1 record={adjustment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
