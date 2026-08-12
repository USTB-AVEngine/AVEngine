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
