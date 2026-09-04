from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/assets/publish_static_source_assets.py"
SPEC = importlib.util.spec_from_file_location("publish_static_source_assets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def candidate() -> dict:
    return {
        "profile_schema_id": "appliance_microwave_countertop_v1",
        "profile_sha256": "a" * 64,
        "fixed_attributes": {"form_factor": "countertop", "material": "steel"},
        "sampled_attributes": {"body_color": "black"},
    }


def test_static_variant_is_form_first_and_realized_attributes_are_complete() -> None:
    payload = candidate()

    assert publisher.publication_variant(payload) == "countertop_black"
    assert publisher.realized_attributes(payload) == {
        "form_factor": "countertop",
        "material": "steel",
        "body_color": "black",
    }


def test_profile_snapshot_must_match_candidate_identity() -> None:
    payload = candidate()
    profiles = publisher.profiles_by_id(
        {
            "profiles": [
                {
                    "profile_schema_id": payload["profile_schema_id"],
                    "profile_sha256": payload["profile_sha256"],
                    "profile": {"acoustic_profile": {"profile_id": "microwave_v1"}},
                }
            ]
        }
    )

    assert publisher.profile_for_candidate(payload, profiles)["acoustic_profile"] == {
        "profile_id": "microwave_v1"
    }
    payload["profile_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="hash does not match"):
        publisher.profile_for_candidate(payload, profiles)


def test_prompt_budget_selects_the_realized_sampled_combination() -> None:
    payload = candidate()
    measurement = {
        "profile_schema_id": payload["profile_schema_id"],
        "profile_revision": "2026_08_26_v1",
        "fits": True,
        "combinations": [
            {
                "sampled_values": {"body_color": "black"},
                "effective_tokens": 401,
                "fits": True,
            },
            {
                "sampled_values": {"body_color": "white"},
                "effective_tokens": 399,
                "fits": True,
            },
        ],
    }
    report = {
        "max_sequence_length": 512,
        "tokenizer_root": "/models/tokenizer",
        "effective_prompt_format": "{prompt} Avoid: {negative}.",
        "profiles": [measurement],
    }

    by_id = publisher.token_budget_by_id(report)
    selected = publisher.prompt_budget_for_candidate(
        payload, report, by_id, "2026_08_26_v1"
    )

    assert selected["measurement"]["effective_tokens"] == 401
    assert selected["profile_revision"] == "2026_08_26_v1"
    with pytest.raises(ValueError, match="profile revision does not match"):
        publisher.prompt_budget_for_candidate(
            payload, report, by_id, "2026_08_26_v2"
        )


def test_placement_carries_measured_bbox_and_rir_recompute_cost() -> None:
    placement = publisher.placement_record(
        {
            "attachment_surface": "wall",
            "facing": "+X faces the room interior",
        },
        {
            "bbox_minimum_xyz_m": [-0.1, 0.0, -0.2],
            "bbox_maximum_xyz_m": [0.1, 0.3, 0.2],
        },
    )

    assert placement["attachment_surface"] == "wall"
    assert placement["footprint_bbox"]["maximum_xyz_m"] == [0.1, 0.3, 0.2]
    assert placement["rir_cache_recompute_required"] is True
    assert "recompute" in placement["rir_cache_note"]


def test_index_axes_union_existing_and_new_static_records() -> None:
    index = {
        "assets": [
            {
                "asset_id": "existing_speaker",
                "entity_class": "rigid_static_object",
                "realized_attributes": {"finish": "walnut"},
            }
        ],
        "instance_axes": {"articulated_animal": ["coat_profile"]},
    }
    records = [
        {
            "asset_id": "new_microwave",
            "entity_class": "rigid_static_object",
            "realized_attributes": {
                "form_factor": "countertop",
                "material": "steel",
                "body_color": "black",
            },
        }
    ]

    merged = publisher.merge_assets_and_axes(index, records)

    assert merged["instance_axes"]["rigid_static_object"] == [
        "body_color",
        "finish",
        "form_factor",
        "material",
    ]
    assert [entry["asset_id"] for entry in merged["assets"]] == [
        "existing_speaker",
        "new_microwave",
    ]


def test_published_id_does_not_carry_the_admission_state():
    """owner 2026-09-03: the state is a field, not part of the name.

    These 44 assets were flipped from research to formal that day and every
    published id still read ..._research_v2, which then says the wrong thing
    about the current state.  New ids are generated_<type>_<variant>_v<N>.
    """
    import re
    source = (Path(__file__).resolve().parents[2]
              / "tools/assets/publish_static_source_assets.py").read_text(encoding="utf-8")
    formula = re.search(r'asset_id = f"generated_\{object_type\}_\{variant\}_\{([a-z_.]+)\}"',
                        source)
    assert formula, "the id formula moved; keep the state out of it"
    assert formula.group(1) == "args.revision"
    assert "args.admission_state}" not in source.split("asset_id =")[1][:200]
