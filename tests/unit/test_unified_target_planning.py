"""Wave-1 CPU checks for unified target planning (C1/C4/C6/C7)."""
from __future__ import annotations

import numpy as np
import pytest

from avengine.qa import generation_conditions as gc
from avengine.qa.answerability import MeshHandle
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms.furniture_layout import clock_config
from avengine.rooms.walkable_space import RasterWalkableSpace
from avengine.routes.raster_pathfinder import RasterPathfinder


def _registry():
    return {"assets": [
        {
            "asset_id": f"human_{index}",
            "revision": "v1",
            "entity_class": "articulated_human",
            "identity": {"species_id": "human"},
            "display_label": f"person {index}",
            "realized_attributes": {
                "sex_or_gender_label": "male",
                "top_color": color,
            },
            "timeline": {
                "idle_action_id": "idle",
                "walking_action_id": "walk",
                "walk_phase_period_frames": 30,
                "local_anatomical_forward_axis": [1.0, 0.0, 0.0],
            },
            "default_emitter_anchor_id": "mouth",
            "emitter_anchors": [
                {"anchor_id": "mouth", "offset_m": [0.0, 1.6, 0.0],
                 "offset_space": "final_scaled_asset_root"}
            ],
        }
        for index, color in enumerate(("blue", "green"))
    ]}


def _sounds():
    return [
        {
            "sound_asset_id": f"speech_{index}",
            "sound_class": "speech",
            "gender": "M",
            "transcript": f"utterance {index}",
            "sample_count": 32000,
            "sample_rate_hz": 16000,
            "audible_start_sample": 800,
            "audible_end_sample_exclusive": 31200,
            "active_duration_s": 1.9,
            "source_activity_intervals_samples": [[800, 31200]],
            "path": f"/prepared/{index}.wav",
        }
        for index in range(2)
    ]


def _space():
    pathfinder = RasterPathfinder(
        np.ones((40, 40), dtype=bool),
        bounds_m=[[0.0, -1.0, 0.0], [10.0, 1.0, 10.0]],
        floor_height_m=0.0,
    )
    return RasterWalkableSpace(
        pathfinder,
        {"floor_height_m": 0.0, "resolution_m": 0.25, "authority": "fixture"},
    )


def _clock():
    return clock_config(frame_count=150, frame_rate_hz=15, sample_rate_hz=16000)


def _qa04_request():
    return {
        "episode_id": "wave1_qa04_fixture",
        "seed": 31,
        "sampling_policy": cs.POLICY,
        "source_asset_ids": ["human_0", "human_1"],
        "entities": {
            "total_count": 2,
            "silent_count": 0,
            "instances": [
                {"instance_id": "human_target", "asset_id": "human_0",
                 "source_class": "articulated_human", "role": "target",
                 "speaking": True},
                {"instance_id": "human_other", "asset_id": "human_1",
                 "source_class": "articulated_human", "role": "competitor",
                 "speaking": True},
            ],
        },
        "qa_targets": [{
            "qa_id": "QA-04",
            "target_instance_ids": ["human_target"],
            "event": {"kind": "target_audible_window"},
            "items": 1,
            "forms": ["mcq", "open"],
            "query_time_policy": "uniform_in_legal_window",
            "target_source": "config",
        }],
        "qa_sampling": {"acceptance_policy": {"question_mode": "ordinary_observation"}},
        "camera": {"fov_deg": 85, "height_above_floor_m": 1.55,
                   "resolution_hw": [720, 1280]},
        "profile": {
            "anchor_count": 1,
            "speech_motion": "all_still",
            "competitor_motion": "still",
            "anchor_median_plane_offset_deg": 5.0,
            "separation_bin_deg": [15.0, 180.0],
            "distance_range_m": [1.0, 6.0],
            "event_relation": "sequential",
            "reserve_tail_s": 1.0,
            "retry_budget_within_profile": 30,
        },
    }


def test_c4_declares_wave1_knobs_and_keeps_later_routes_explicit():
    declaration = cs.describe_generator_capabilities()
    declared = set(declaration["knobs"])
    assert {
        "visibility_transition",
        "pixel_occlusion_transition",
        "distance_trend_during_event",
        "anchor_median_plane_offset_deg",
    } <= declared
    assert "pixel_occlusion_partial_transition" in declared
    assert "registered_occluder_transition" in declared
    assert "visibility_transition" not in gc.KNOB_GAPS
    assert "pixel_occlusion_transition" not in gc.KNOB_GAPS
    assert "distance_trend_during_event" not in gc.KNOB_GAPS
    assert "anchor_median_plane_offset_deg" not in gc.KNOB_GAPS
    assert "pixel_occlusion_partial_transition" not in gc.KNOB_GAPS
    assert "registered_occluder_transition" not in gc.KNOB_GAPS
    assert "C2" in declaration["declared_at"]
    assert "C3" in declaration["declared_at"]
    resolved = cs.resolve_condition_profile(
        {
            "seed": 1,
            "camera": {"motion": "static"},
            "source_asset_ids": ["human_0", "human_1"],
            "profile": {
                "pixel_occlusion_partial_transition": "visible_occluded_to_visible_clear",
                "registered_occluder_transition": "registered_occluder_visible",
            },
        },
        _registry(),
    )
    assert resolved["pixel_occlusion_partial_transition"] == (
        "visible_occluded_to_visible_clear"
    )
    assert resolved["registered_occluder_transition"] == "registered_occluder_visible"


@pytest.mark.parametrize(
    ("qa_id", "branch"),
    [("QA-07", "left"), ("QA-09", "yes"), ("QA-10", None),
     ("QA-11", None), ("QA-15", "nearer")],
)
def test_wave1_target_compilation_has_no_interface_gap(qa_id, branch):
    instances = [
        {"instance_id": "human_target", "source_class": "articulated_human",
         "asset_id": "human_0", "role": "anchor"},
        {"instance_id": "human_other", "source_class": "articulated_human",
         "asset_id": "human_1", "role": "competitor"},
    ]
    target = {"qa_id": qa_id, "target_instance_ids": ["human_target"]}
    if qa_id in {"QA-10", "QA-11"}:
        target["target_instance_ids"] = ["human_target"]
    compiled = gc.compile_generation_conditions(
        target,
        branch=branch,
        instances=instances,
        capabilities=cs,
    )
    assert all(item.state != gc.STATE_NOT_IMPLEMENTED for item in compiled.conditions), (
        qa_id, branch, compiled.reason
    )


def test_c1_motion_templates_compile_their_event_and_silent_window_conditions():
    instances = [
        {"instance_id": "human_target", "source_class": "articulated_human",
         "asset_id": "human_0", "role": "anchor"},
        {"instance_id": "human_other", "source_class": "articulated_human",
         "asset_id": "human_1", "role": "competitor"},
    ]
    cases = [
        ("QA-06", "moving", "motion_during_event"),
        ("QA-16", None, "post_sound_silent_window"),
        ("QA-17", "yes", "motion_after_sound"),
    ]
    for qa_id, branch, expected_key in cases:
        target = {
            "qa_id": qa_id,
            "target_instance_ids": ["human_target"],
            "event": {"kind": "target_audible_window"},
        }
        compiled = gc.compile_generation_conditions(
            target,
            branch=branch,
            instances=instances,
            capabilities=cs,
        )
        assert compiled.state == gc.STATE_AVAILABLE, (qa_id, branch, compiled.reason)
        assert all(item.state != gc.STATE_NOT_IMPLEMENTED for item in compiled.conditions)
        assert any(item.key == expected_key for item in compiled.conditions)
        if qa_id in {"QA-16", "QA-17"}:
            public = next(
                item for item in compiled.conditions
                if item.key == "legal_integer_query_window"
            )
            assert public.detail["minimum_interval_s"] == pytest.approx(1.2)
            silent = next(
                item for item in compiled.conditions
                if item.key == "post_sound_silent_window"
            )
            assert silent.planning["min_gap_between_audible_windows_s"] == pytest.approx(0.5)
        if qa_id == "QA-06":
            motion = next(
                item for item in compiled.conditions
                if item.key == "motion_during_event"
            )
            assert motion.planning["speech_motion"] == "speaker_moving"


def test_c1_post_sound_schedule_keeps_anchor_last_and_queryable():
    sounds = {
        0: {
            "entity_instance_id": "human_target",
            "sound_asset_id": "speech_target",
            "sample_count": 32000,
            "audible_start_sample": 0,
            "audible_end_sample_exclusive": 32000,
        },
        1: {
            "entity_instance_id": "other1",
            "sound_asset_id": "speech_other",
            "sample_count": 32000,
            "audible_start_sample": 0,
            "audible_end_sample_exclusive": 32000,
        },
    }
    profile = {
        "anchor_indices": [0],
        "event_relation": "sequential",
        "min_gap_between_audible_windows_s": 0.5,
        "reserve_tail_s": 3.0,
        "target_moved_after_sound": True,
        "first_speaker_instance_id": "other1",
    }
    events, schedule, _event_start, _other_windows = cs._schedule_for_motion_solver(
        sounds, profile, _clock(), np.random.default_rng(9)
    )
    target_key, target = next(
        (key, event) for key, event in events.items()
        if event["entity_instance_id"] == "human_target"
    )
    other_key, other = next(
        (key, event) for key, event in events.items()
        if event["entity_instance_id"] == "other1"
    )
    target_start = int(schedule[target_key])
    other_start = int(schedule[other_key])
    target_end = target_start + int(target["sample_count"])
    assert other_start < target_start
    assert target_end + int(round(profile["reserve_tail_s"] * 16000)) < 160000 - 16000
    assert target_end < 80000


def test_c6_filters_and_records_the_anchor_median_plane_offset():
    plan = cs.build_conditioned_plan(
        room={"room_id": "wave1_fixture"},
        request=_qa04_request(),
        source_registry=_registry(),
        sounds=_sounds(),
        space=_space(),
        mesh=MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        clock=_clock(),
    )
    profile = plan["condition_profile"]
    conditions = plan["planned_conditions"]
    assert profile["anchor_median_plane_offset_deg"] == pytest.approx(5.0)
    assert conditions["anchor_median_plane_offset_deg"] == pytest.approx(5.0)
    assert conditions["planned_anchor_median_plane_offset_deg"] >= 5.0 - 1.0e-8
    assert conditions["anchor_median_plane_offset_basis"].startswith(
        "listener_relative_azimuth"
    )
    assert conditions["stages"].get(
        "poses_without_requested_anchor_median_plane_offset", 0
    ) > 0


def test_c7_uses_eighteen_frames_at_fifteen_fps_for_public_windows():
    assert int(np.ceil(cs.PUBLIC_QUERY_WINDOW_MIN_S * 15.0 - 1.0e-9)) == 18
    sound = {
        "sample_count": 48000,
        "audible_start_sample": 0,
        "audible_end_sample_exclusive": 48000,
        "source_activity_intervals_samples": [[0, 48000]],
    }
    clock = _clock()
    visible_then_hidden = np.r_[np.ones(20, dtype=bool), np.zeros(25, dtype=bool)]
    short_hidden = np.r_[np.ones(20, dtype=bool), np.zeros(10, dtype=bool)]
    assert cs._visible_then_hidden_ok(
        visible_then_hidden, sound, clock, 0
    )
    assert not cs._visible_then_hidden_ok(short_hidden, sound, clock, 0)


def test_c7_public_query_metadata_exposes_the_planning_margin():
    from avengine.qa.generation_conditions import _public_window_condition

    condition = _public_window_condition("QA-17", 0)
    assert condition.detail["minimum_interval_s"] == pytest.approx(1.2)
    assert condition.detail["planning_margin_s"] == pytest.approx(0.2)
