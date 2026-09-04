"""Clock propagation tests for ordinary QA generation paths."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import audio_profiles as AP  # noqa: E402
import scene_sampler as SS  # noqa: E402
from build_qa_v3_n_actor_canary import find_n_route_plan  # noqa: E402
from make_idle_then_walk_timeline import transform_idle_then_walk  # noqa: E402


def _params(frame_count, clip_seconds):
    return {
        "FRAME_COUNT": frame_count,
        "VIDEO_FPS": 15,
        "CLIP_SECONDS": clip_seconds,
        "SAMPLE_RATE_HZ": 16000,
        "TIME_BASE_HZ": 48000,
        "TICKS_PER_SAMPLE": 3,
        "TICKS_PER_FRAME": 3200,
        "MIN_CAMERA_DISTANCE_CM": 100.0,
        "MIN_AZIMUTH_SEP": 5.0,
        "THETA_FULL": 15.0,
        "THETA_HALF": 30.0,
        "VISUAL_FOV_MARGIN_DEG": 0.0,
    }


def _route(route_id, angle_deg, frame_count):
    radians = math.radians(angle_deg)
    return SS.Route(
        route_id,
        [(500.0 * math.cos(radians) + frame,
          500.0 * math.sin(radians) + frame)
         for frame in range(frame_count)],
        0.2,
        frame_rate_hz=15.0,
        duration_seconds=frame_count / 15.0,
    )


def _scene(frame_count):
    return SS.SceneInputs(
        scene_id="clock_room",
        backend="test",
        routes=[_route("r0", 0.0, frame_count),
                _route("r1", 45.0, frame_count),
                _route("r2", 90.0, frame_count),
                _route("r3", 135.0, frame_count)],
        stand_points=[(0.0, 0.0)],
        camera_points=[(0.0, 0.0)],
        camera_height_m=1.5,
        hfov_deg=179.0,
    )


def test_frame_clock_accepts_75_150_and_90_at_15fps():
    assert SS.validate_frame_clock(_params(75, 5.0))["sample_count"] == 80_000
    assert SS.validate_frame_clock(_params(150, 10.0))["sample_count"] == 160_000
    assert SS.validate_frame_clock(_params(90, 6.0))["sample_count"] == 96_000
    with pytest.raises(ValueError, match="requires CLIP_SECONDS"):
        SS.validate_frame_clock(_params(150, 5.0))


def test_route_transforms_use_declared_route_length():
    route = _route("r", 0.0, 150)
    shifted = route.shifted(10)
    paused = route.paused(20, 30)
    assert len(shifted.samples_xy) == 150
    assert len(paused.samples_xy) == 150
    assert shifted.at(149) == route.at(139)


def test_idle_then_walk_accepts_a_150_frame_timeline():
    frames = []
    for index in range(150):
        frames.append({
            "frame_index": index,
            "actor_states": [
                {"source_slot_id": "source1",
                 "translation_ue_cm": [float(index), 0.0, 0.0],
                 "yaw_ue_deg": 0.0, "action_id": "walk",
                 "action_phase": index / 150.0},
                {"source_slot_id": "source2",
                 "translation_ue_cm": [0.0, 0.0, 0.0],
                 "yaw_ue_deg": 0.0, "action_id": "idle",
                 "action_phase": 0.0},
            ],
        })
    timeline = {"render": {"frame_count": 150}, "frames": frames}
    result = transform_idle_then_walk(timeline, "source1", 10)
    assert len(result["frames"]) == 150
    assert result["frames"][10]["actor_states"][0]["translation_ue_cm"] == [0.0, 0.0, 0.0]
    assert result["frames"][-1]["actor_states"][0]["translation_ue_cm"] == [139.0, 0.0, 0.0]


def test_non_speech_schedule_maps_150_and_90_frame_times_to_samples():
    for frame_count, seconds in ((150, 10.0), (90, 6.0)):
        params = dict(_params(frame_count, seconds), EVENT_SECONDS=0.3,
                      GAP_MIN_S=0.3)
        schedule = AP.schedule_first_sound_at_frame(
            np.random.default_rng(3), params=params, query_frame=30)
        assert schedule.anchor.frame_span()[0] == 30
        assert schedule.anchor.start_sample == round(30 / frame_count * seconds * 16000)
        assert all(event.end_sample_exclusive <= seconds * 16000
                   for event in schedule.events)


def test_n_actor_solver_uses_150_frame_binding_window():
    params = _params(150, 10.0)
    plan = find_n_route_plan(
        _scene(75), params, actor_count=2, seed="clock-150",
        binding_frames=(30, 149), min_pairwise_sep_deg=5.0, max_attempts=20)
    assert all(len(route.samples_xy) == 150 for route in plan["routes"])
    assert plan["binding_azimuths_deg"]



def test_route_synthesis_uses_declared_frame_clock():
    from route_synthesis import PointSpec, RouteSynthesizer, SynthesisSettings, polyline_positions
    params = dict(_params(150, 10.0), ROUTE_SYNTHESIS_ENABLED=True,
                  ROUTE_SYNTHESIS_SPEED_MPS_RANGE=[0.6, 1.5],
                  ROUTE_SYNTHESIS_WALKABLE_MARGIN_M=0.2,
                  ROUTE_SYNTHESIS_MAX_CAMERA_DISTANCE_CM=600.0,
                  ROUTE_SYNTHESIS_ATTEMPTS=4)
    settings = SynthesisSettings.from_params(params)
    assert settings.frame_count == 150
    assert settings.frame_rate_hz == 15.0
    assert PointSpec(149, -10.0, 10.0, 100.0, 600.0).feasible(settings.frame_count)
    samples = polyline_positions(
        [(149, (100.0, 0.0))], 1.0, 0.0, 0.0, 0, frame_count=150)
    assert len(samples) == 150



def test_same_frame_count_with_new_rate_updates_speed_and_duration():
    scene = _scene(75)
    params = dict(_params(75, 2.5), VIDEO_FPS=30, TICKS_PER_FRAME=1600)
    adapted = SS.ensure_scene_clock(scene, params)
    assert all(route.frame_rate_hz == 30 for route in adapted.routes)
    assert all(route.duration_seconds == 2.5 for route in adapted.routes)
    assert all(route.implied_speed_mps == pytest.approx(0.4) for route in adapted.routes)
    assert scene.routes[0].frame_rate_hz == 15
    assert SS.ensure_scene_clock(adapted, params) is adapted


@pytest.mark.parametrize("key,value", [
    ("FRAME_COUNT", 75.1), ("SAMPLE_COUNT", 80000.1),
    ("SAMPLE_RATE_HZ", 16000.1), ("TICKS_PER_FRAME", 3200.1),
    ("TICKS_PER_SAMPLE", 3.1), ("TIME_BASE_HZ", 48000.1),
    ("SAMPLE_RATE_HZ", True),
])
def test_clock_rejects_fractional_and_boolean_integer_fields(key, value):
    with pytest.raises(ValueError, match="integer"):
        SS.validate_frame_clock(dict(_params(75, 5), **{key: value}))


def test_sample_clock_cannot_round_away_partial_sample():
    params = {"FRAME_COUNT": 2, "VIDEO_FPS": 3, "SAMPLE_RATE_HZ": 16000}
    with pytest.raises(ValueError, match="integer sample count"):
        SS.validate_frame_clock(params)


def test_frame_time_cannot_truncate_fractional_frame():
    with pytest.raises(ValueError, match="integer"):
        SS.frame_time_seconds(30.5, _params(75, 5))
