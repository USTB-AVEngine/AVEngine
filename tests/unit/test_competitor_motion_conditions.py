"""Behavior checks for the existing compiled companion-motion path.

C5 deliberately tests ``generation_conditions`` -> ``conditioned_motion`` ->
``conditioned_sampler``.  There is no second companion-motion implementation:
the existing solver owns role windows and ``_moving_flags`` consumes them.
"""
from __future__ import annotations

import numpy as np
import pytest

from avengine.qa import generation_conditions as gc
from avengine.rooms import conditioned_motion as cm
from avengine.rooms import conditioned_sampler as cs

pytestmark = pytest.mark.fast_unit

CLOCK = cm.EpisodeClock(frame_count=150, frame_rate_hz=15.0, sample_rate_hz=16000)
REGISTRY = {
    "assets": [
        {
            "asset_id": "arbitrary_target_asset",
            "entity_class": "articulated_human",
            "timeline": {
                "idle_action_id": "idle", "walking_action_id": "walk",
                "body_plan_id": "human", "walk_phase_period_frames": 30,
            },
        },
        {
            "asset_id": "arbitrary_competitor_asset",
            "entity_class": "articulated_animal",
            "timeline": {
                "idle_action_id": "idle", "walking_action_id": "walk",
                "body_plan_id": "animal", "walk_phase_period_frames": 24,
            },
        },
    ]
}
INSTANCES = [
    {"entity_instance_id": "target_instance", "asset_id": "arbitrary_target_asset",
     "source_class": "articulated_human", "role": "anchor"},
    {"entity_instance_id": "competitor_instance", "asset_id": "arbitrary_competitor_asset",
     "source_class": "articulated_animal"},
]


def _pool_row(name: str, seconds: float = 2.0) -> dict:
    rate = 16000
    count = int(seconds * rate)
    first, last = 1600, count - 1600
    return {
        "sound_asset_id": name,
        "sample_rate_hz": rate,
        "sample_count": count,
        "audible_start_sample": first,
        "audible_end_sample_exclusive": last,
        "source_activity_intervals_samples": [[first, last]],
        "source_origin": f"/prepared/{name}.wav",
    }


def _compile(qa_id: str, branch: str | None = None):
    return gc.compile_generation_conditions(
        {"qa_id": qa_id, "target_instance_ids": ["target_instance"],
         "event": {"kind": "target_audible_window"}},
        branch=branch, instances=INSTANCES, registry=REGISTRY, capabilities=cs,
    )


def _solve(qa_id: str, branch: str | None = None, *, budget=None):
    return cm.solve_motion_windows(
        _compile(qa_id, branch), clock=CLOCK, budget=budget,
        sounds={"target_instance": _pool_row("target_sound"),
                "competitor_instance": _pool_row("competitor_sound", 1.5)},
        registry=REGISTRY, camera={"motion": "static"},
    )


def test_compiler_emits_existing_opposite_companion_policy_for_qa06():
    moving = _compile("QA-06", "moving")
    still = _compile("QA-06", "still")

    assert moving.sampler_profile()["speech_motion"] == "speaker_moving"
    assert moving.sampler_profile()["competitor_motion"] == "still"
    assert still.sampler_profile()["speech_motion"] == "all_still"
    assert still.sampler_profile()["competitor_motion"] == "moving"


def test_existing_motion_solver_materializes_both_qa06_role_windows():
    moving = _solve("QA-06", "moving")
    still = _solve("QA-06", "still")

    assert moving.status == still.status == "solved"
    assert moving.requirement_for("target_instance").must_move is True
    assert moving.requirement_for("competitor_instance").still_frames
    assert still.requirement_for("target_instance").must_move is False
    assert still.requirement_for("competitor_instance").moving_frames is not None
    assert still.requirement_for("competitor_instance").moving_frames == (
        still.placement_for("target_instance").start_frame,
        still.placement_for("target_instance").end_frame,
    )


def test_qa15_uses_still_companion_and_existing_distance_solver_for_both_trends():
    for branch, endpoint in (("nearer", 2.5), ("farther", 5.5)):
        solution = _solve("QA-15", branch)
        target = solution.requirement_for("target_instance")
        competitor = solution.requirement_for("competitor_instance")
        assert solution.status == "solved"
        assert solution.semantics["distance_trend_sign"] == (
            "negative" if branch == "nearer" else "positive"
        )
        assert target.moving_frames is not None
        assert competitor.moving_frames is None
        assert competitor.still_frames == (target.moving_frames,)

        target_path = cm.build_motion_trajectory(
            polyline_m=np.array([[0.0, 0.0, 4.0], [0.0, 0.0, endpoint]]),
            moving_frames=target.moving_frames, clock=CLOCK, budget=cm.MotionBudget(),
        )["path_m"]
        report = cm.verify_motion_candidate(
            solution,
            positions_m={
                "target_instance": target_path,
                "competitor_instance": cm.static_trajectory([3.0, 0.0, 3.0], CLOCK.frame_count),
            },
            listener_position_m=[0.0, 0.0, 0.0],
        )
        assert report["status"] == "pass", report["failed_checks"]


def test_moving_flags_keep_solver_window_priority_over_profile_defaults():
    solution = _solve("QA-06", "moving")
    actors = [
        {"entity_instance_id": "target_instance", "entity_class": "articulated_human"},
        {"entity_instance_id": "competitor_instance", "entity_class": "articulated_animal"},
    ]
    # The profile says nothing useful here; solver-produced windows are authoritative.
    flags = cs._moving_flags(
        {"anchor_indices": [0], "speech_motion": "all_still", "competitor_motion": "any"},
        actors, np.random.default_rng(17),
        {item.entity_instance_id: item for item in solution.requirements},
    )
    assert flags == [True, False]


def test_qa16_compiles_a_still_companion_only_inside_the_question_window():
    compiled = _compile("QA-16")
    solution = _solve("QA-16", budget=cm.MotionBudget(minimum_motion_s=1.0))
    target = solution.requirement_for("target_instance")
    competitor = solution.requirement_for("competitor_instance")
    anchor_end = solution.placement_for("target_instance").end_frame
    query_end = solution.query_window["latest_query_frame_exclusive"]

    assert solution.status == "solved"
    distance_condition = next(
        item for item in compiled.conditions if item.kind == "distance_stable_after_sound"
    )
    assert distance_condition.planning == {
        "target_moved_after_sound": True,
        "competitor_motion": "still",
    }
    assert compiled.sampler_profile()["competitor_motion"] == "still"
    assert solution.semantics["competitor_motion"] == "still"
    assert target.must_move is True
    assert target.moving_frames is not None
    assert target.semantics == "post_sound_query_only"
    assert competitor.semantics == "post_sound_query_only"
    assert competitor.moving_frames is None
    assert competitor.must_move is False
    # The solver reuses its existing role requirement: still only from the
    # anchor event end through the legal query endpoint, with no whole-clip gate.
    assert competitor.still_frames == ((anchor_end, query_end),)
    assert competitor.still_frames[0][0] > 0
    assert competitor.still_frames[0][1] == query_end
