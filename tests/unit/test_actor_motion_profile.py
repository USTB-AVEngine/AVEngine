from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from avengine.qa.actor_motion_profile import (
    ActorMotionProfileError,
    build_actor_motion_profile,
    materialize_profile_frames,
    source_center_paths,
    validate_actor_motion_authorities,
    validate_actor_motion_profile,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["target", "distractor", "both"])
def real_authorities(request: pytest.FixtureRequest):
    paths = {
        "target": (
            ROOT
            / "examples/qa/native_strict_two_human_target_moves_native_rate_candidate_v1.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_target_moves_v2_cpu_candidate_v1/target_moves_v2_preflight.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_target_moves_v2_materialized_v1/suite_execution_plan.json",
        ),
        "distractor": (
            ROOT
            / "examples/qa/native_strict_two_human_distractor_moves_native_rate_candidate_v1.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_distractor_moves_v2_geometry_v1/distractor_moves_v2_preflight.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_distractor_moves_v2_materialized_v1/suite_execution_plan.json",
        ),
        "both": (
            ROOT
            / "examples/qa/native_strict_two_human_both_move_native_rate_candidate_v1.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_both_move_v1_adapter_v1/preflight.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_both_move_v1_materialized_v1/suite_execution_plan.json",
        ),
    }
    candidate_path, old_path, suite_path = paths[request.param]
    candidate = json.loads(candidate_path.read_text())
    old = json.loads(old_path.read_text())
    suite = json.loads(suite_path.read_text())
    return candidate_path, candidate, old_path, old["canaries"][0], suite_path, suite


def _profile(authorities):
    candidate_path, candidate, old_path, old_row, suite_path, suite = authorities
    return build_actor_motion_profile(
        candidate_path=candidate_path,
        candidate=candidate,
        old_preflight_path=old_path,
        selected_old_row=old_row,
        base_suite_path=suite_path,
        base_suite=suite,
    )


def test_three_real_profiles_bind_frames_sources_and_stride_one_rirs(real_authorities):
    profile = _profile(real_authorities)
    validate_actor_motion_profile(profile)
    assert profile["frames"] == materialize_profile_frames(profile)
    assert source_center_paths(profile)
    assert profile["rir_expectation"]["stride_frames"] == 1


def test_profile_hash_rejects_mutation(real_authorities):
    profile = _profile(real_authorities)
    forged = copy.deepcopy(profile)
    forged["authorities"]["candidate"]["value"]["mechanism"] = "forged"
    with pytest.raises(ActorMotionProfileError, match="content hash"):
        validate_actor_motion_profile(forged)


def test_three_real_authorities_pass_semantic_closure(real_authorities):
    _, candidate, _, old_row, _, suite = real_authorities
    validate_actor_motion_authorities(candidate, old_row, suite)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("active_boundary", "outside roots"),
        ("declaration", "actor declaration"),
        ("old_asset", "asset/revision"),
        ("audio", "audio role/timing"),
        ("camera", "camera cross-authority"),
        ("source_activation", "source activation"),
    ],
)
def test_semantic_validator_rejects_authority_drift(real_authorities, mutation, match):
    _, candidate_value, _, old_row_value, _, suite_value = real_authorities
    candidate = copy.deepcopy(candidate_value)
    old_row = copy.deepcopy(old_row_value)
    suite = copy.deepcopy(suite_value)
    actors = candidate["actors"]
    moving_slot = next(slot for slot, actor in actors.items() if actor["moving"])

    if mutation == "active_boundary":
        actor = actors[moving_slot]
        start = actor["native_rate_active_interval"]["output_frame_range_inclusive"][0]
        actor["root_path_m"][start - 1][0] += 0.01
        actor["translation_ue_cm_path"][start - 1][0] += 1.0
        state = candidate["frames"][start - 1]["actor_states"][
            list(actors).index(moving_slot)
        ]
        state["translation_m"][0] += 0.01
        state["translation_ue_cm"][0] += 1.0
    elif mutation == "declaration":
        actor_id = actors[moving_slot]["actor_id"]
        candidate["actor_declarations"][actor_id]["asset_id"] = "forged"
    elif mutation == "old_asset":
        role = "target" if candidate["target_slot"] == moving_slot else "distractor"
        old_row[role]["runtime_revision"] = "forged"
    elif mutation == "audio":
        candidate["audio_event_contract"]["speech_frame_window_inclusive"][0] += 1
    elif mutation == "camera":
        suite["scenarios"][0]["plan"]["camera"]["horizontal_fov_deg"] += 1
    elif mutation == "source_activation":
        candidate["source_activation_contract"]["source_logic"]["sources"][0][
            "activation"
        ] = "silent"

    with pytest.raises(ActorMotionProfileError, match=match):
        validate_actor_motion_authorities(candidate, old_row, suite)


def test_static_semantics_reject_root_motion(real_authorities):
    _, candidate_value, _, old_row, _, suite = real_authorities
    candidate = copy.deepcopy(candidate_value)
    static_slots = [
        slot for slot, actor in candidate["actors"].items() if not actor["moving"]
    ]
    if not static_slots:
        pytest.skip("real authority has no static actor")
    slot = static_slots[0]
    candidate["actors"][slot]["root_path_m"][-1][0] += 0.01
    candidate["frames"][-1]["actor_states"][list(candidate["actors"]).index(slot)][
        "translation_m"
    ][0] += 0.01
    with pytest.raises(ActorMotionProfileError, match="static actor"):
        validate_actor_motion_authorities(candidate, old_row, suite)
