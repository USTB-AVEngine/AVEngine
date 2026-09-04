"""Tests for explicit per-request QA-v3 asset policies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import design_qa_v3_scene_batch as batch  # noqa: E402
from qa_v3_asset_policy import (  # noqa: E402
    AssetPolicyError,
    load_asset_policy,
    resolve_asset_policy,
    slot_context,
)


REGISTRY_PATH = REPO / "examples/runtime/source_asset_runtime_profiles.json"
POLICY_PATH = REPO / "examples/qa/qa_v3_asset_policy_v1.json"


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _policy() -> dict:
    return load_asset_policy(POLICY_PATH)


def test_policy_resolves_dog_and_speaker_without_conflating_visual_and_sound_class():
    context = resolve_asset_policy(
        _policy(),
        registry=_registry(),
        pair_id="dog_and_black_ash_speaker",
        profiles=[
            {"id": "card8", "answer_kind": "time_band"},
            {"id": "card15b", "answer_kind": "event_count"},
        ],
    )
    assert context["pair_kind"] == "dog"
    batch.assert_assets_match_pair_kind(
        {"PAIR_KIND": "dog"},
        context["asset_ids"],
        context,
    )
    assert {
        context["asset_specs"][asset]["visual_family"]
        for asset in context["asset_ids"]
    } == {"dog", "speaker"}
    assert all(
        context["asset_specs"][asset]["allowed_sound_class_ids"]
        == ["animal_vocalization"]
        for asset in context["asset_ids"]
    )
    slots = slot_context(
        context,
        assets_by_slot={
            "source1": context["asset_ids"][0],
            "source2": context["asset_ids"][1],
        },
        target_slot="source2",
    )
    assert slots["motion_by_slot"]["source1"] == "must_move"
    assert slots["motion_by_slot"]["source2"] == "must_be_still"
    assert slots["referent_phrases_by_slot"]["source2"].endswith("speaker")
    assert context["facing_by_asset"][context["asset_ids"][1]] == "toward_camera"


def test_policy_resolves_two_static_speakers():
    context = resolve_asset_policy(
        _policy(),
        registry=_registry(),
        pair_id="two_speakers",
        profiles=[
            {"id": "card8", "answer_kind": "time_band"},
            {"id": "card15b", "answer_kind": "event_count"},
        ],
    )
    assert context["family_rule"] == "same_family_distinct_instances"
    assert context["asset_ids"][0] != context["asset_ids"][1]
    assert all(
        context["motion_by_asset"][asset] == "must_be_still"
        for asset in context["asset_ids"]
    )


def test_static_pair_rejects_displacement_dependent_profile():
    with pytest.raises(AssetPolicyError, match="does not support profile"):
        resolve_asset_policy(
            _policy(),
            registry=_registry(),
            pair_id="two_speakers",
            profiles=[{"id": "card6", "answer_kind": "motion_state"}],
        )


def test_generic_cell_plan_uses_policy_labels_and_asset_ids():
    policy = _policy()
    context = resolve_asset_policy(
        policy,
        registry=_registry(),
        pair_id="two_speakers",
        profiles=[{"id": "card15b", "answer_kind": "event_count"}],
    )
    profile = {
        "id": "card15b",
        "temporal": "instant",
        "answer_kind": "event_count",
        "binding_frames": [12, 40],
        "idle_choices": [0, 8],
        "answer_values": [3, 4],
        "anchor_binding": "none",
    }
    rows = batch.build_cell_plan(
        4,
        [profile],
        context["asset_ids"],
        {},
        "policy-test",
        asset_context=context,
    )
    assert len(rows) == 4
    assert {
        row["target_coat"]
        for row in rows
    } == {
        context["asset_specs"][asset]["label"]
        for asset in context["asset_ids"]
    }
    assert all(
        set(row["pair_assets"]) == set(context["asset_ids"])
        for row in rows
    )


def test_policy_rejects_unknown_asset():
    policy = _policy()
    policy["pairs"]["bad"] = {
        "asset_ids": ["missing_asset", policy["pairs"]["two_speakers"]["asset_ids"][0]],
        "allowed_answer_kinds": ["event_count"],
    }
    with pytest.raises(AssetPolicyError, match="absent from registry"):
        resolve_asset_policy(policy, registry=_registry(), pair_id="bad")


def test_endpoint_registry_join_uses_instance_identity_when_records_are_sorted():
    selection = {
        "actors": [
            {
                "source_slot_id": "source1",
                "legacy_timeline_actor_id": "dog_1",
            },
            {
                "source_slot_id": "source2",
                "entity_instance_id": "source2_actor",
            },
        ]
    }
    records = [
        {
            "source_endpoint_id": "ep_source2",
            "binding": {"entity_instance_id": "source2_actor"},
        },
        {
            "source_endpoint_id": "ep_dog1",
            "binding": {"entity_instance_id": "dog_1"},
        },
    ]
    assert batch.endpoint_ids_by_slot(selection, records) == {
        "source1": "ep_dog1",
        "source2": "ep_source2",
    }
