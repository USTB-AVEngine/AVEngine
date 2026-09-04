from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import design_qa_v3_offscreen_identity as identity  # noqa: E402


PROFILE = {
    "schema": "avengine_qa_v3_offscreen_identity_profile_v1",
    "id": "test_f2",
    "actor_count": 2,
    "appearance_colors": ["burgundy", "blue"],
    "early_window_seconds": [0.0, 0.4],
    "late_window_seconds": [0.6, 1.0],
    "gap_seconds": 0.05,
    "tail_seconds": 0.0,
    "min_pairwise_azimuth_deg": 10.0,
    "max_search_attempts": 8,
}


PARAMS = {
    "CLIP_SECONDS": 1.0,
    "SAMPLE_RATE_HZ": 100,
    "FRAME_COUNT": 10,
    "VIDEO_FPS": 10,
    "SOUND_SOURCE_MODE": "event_pool",
    "SOUND_EVENT_CLASS_BY_PAIR_KIND": {"human": "speech_playback"},
}


def _clip(asset_id: str, speaker: str, utterance: str, duration: int = 10):
    return SimpleNamespace(
        sound_asset_id=asset_id,
        speaker_id=speaker,
        utterance_id=utterance,
        transcript=f"line {utterance}",
        split="train",
        sample_rate_hz=100,
        duration_samples=duration,
        source_start_sample=3,
        source_end_sample_exclusive=3 + duration,
    )


def test_complete_pair_is_repeated_without_cropping_and_short_window_fails():
    clips = (_clip("a", "speaker-a", "utt-a"), _clip("b", "speaker-b", "utt-b", 12))
    early = identity._place_pair(
        clips,
        window_samples=(0, 40),
        gap_samples=3,
        rng=np.random.default_rng(2),
        phase="early_offscreen",
    )
    late = identity._place_pair(
        clips,
        window_samples=(50, 90),
        gap_samples=3,
        rng=np.random.default_rng(3),
        phase="late_onscreen_repeat",
    )
    for left, right in zip(early, late, strict=True):
        assert left["sound_asset_id"] == right["sound_asset_id"]
        assert left["speaker_id"] == right["speaker_id"]
        assert left["utterance_id"] == right["utterance_id"]
        assert left["source_start_sample"] == right["source_start_sample"]
        assert left["source_end_sample_exclusive"] == right["source_end_sample_exclusive"]
        assert left["duration_samples"] == right["duration_samples"]
    with pytest.raises(identity.OffscreenIdentityError, match="no clipping"):
        identity._place_pair(
            clips,
            window_samples=(0, 20),
            gap_samples=3,
            rng=np.random.default_rng(2),
            phase="early_offscreen",
        )


def test_speech_selection_and_placement_are_reproducible(monkeypatch):
    class _Source:
        def select_distinct_speech_clips(self, *_args, **kwargs):
            assert kwargs["max_total_duration_samples"] == 35
            return [_clip("a", "speaker-a", "utt-a"), _clip("b", "speaker-b", "utt-b")]

    monkeypatch.setattr(identity, "clip_source_from_params", lambda *_args, **_kwargs: _Source())
    first, first_meta = identity._speech_occurrences(PARAMS, PROFILE, seed="same")
    second, second_meta = identity._speech_occurrences(PARAMS, PROFILE, seed="same")
    assert first == second
    assert first_meta["selection"] == second_meta["selection"]
    assert [row["phase"] for row in first] == [
        "early_offscreen",
        "early_offscreen",
        "late_onscreen_repeat",
        "late_onscreen_repeat",
    ]


def test_route_plan_uses_each_actor_own_early_and_late_windows(monkeypatch):
    class _Route:
        def __init__(self, route_id, points):
            self.route_id = route_id
            self.samples_xy = points
            self.implied_speed_mps = 1.0
            self.displacement_cm = float(np.linalg.norm(np.asarray(points[-1]) - points[0]))

        @property
        def source(self):
            return "bank"

        @property
        def source_record(self):
            return {"source": "bank", "route_id": self.route_id}

        def at(self, frame):
            return self.samples_xy[int(frame)]

    # Actor 1 only has an early event at frames 0..1 and an on-screen repeat at
    # frames 6..7. Actor 2 has the inverse angles in its own windows; neither
    # route is required to satisfy the other actor's speech window.
    route_a = _Route("a", [(0.0, 100.0)] * 6 + [(100.0, 100.0)] * 4)
    route_b = _Route(
        "b",
        [(100.0, 100.0)] * 2
        + [(0.0, -100.0)] * 2
        + [(100.0, 100.0)] * 2
        + [(200.0, 50.0)] * 4,
    )
    scene = SimpleNamespace(
        routes=[route_a, route_b],
        camera_points=[(0.0, 0.0)],
        hfov_deg=105.0,
        line_of_sight_screened=False,
    )
    monkeypatch.setattr(
        identity.SS,
        "sample_clear_yaw",
        lambda *_args, **_kwargs: (0.0, {"screened": False}),
    )
    events = [
        {"phase": "early_offscreen", "slot": "source1", "start_sample": 0, "duration_samples": 20},
        {"phase": "early_offscreen", "slot": "source2", "start_sample": 20, "duration_samples": 20},
        {"phase": "late_onscreen_repeat", "slot": "source1", "start_sample": 60, "duration_samples": 20},
        {"phase": "late_onscreen_repeat", "slot": "source2", "start_sample": 80, "duration_samples": 20},
    ]
    plan = identity._find_offscreen_entry_plan(
        scene,
        {"VISUAL_FOV_MARGIN_DEG": 5.0, "MIN_CAMERA_DISTANCE_CM": 0.0},
        PROFILE,
        events,
        {"sample_rate_hz": 100, "frame_rate_hz": 10.0, "frame_count": 10},
        seed="route-test",
    )
    assert [route.route_id for route in plan["routes"]] == ["a", "b"]
    assert all(report["early"]["visibility_ok"] for report in plan["route_reports"])
    assert all(report["late"]["visibility_ok"] for report in plan["route_reports"])


def test_gatea_fact_replaces_gold_while_preserving_question():
    main = {
        "point_id": "p",
        "target": {"first_speaker_appearance": "burgundy", "speaker_id": "s", "utterance_id": "u"},
        "mcq": {"stem": "same", "options_space": ["burgundy", "blue"], "truth_option": "burgundy"},
        "open": {"stem": "same", "truth_value": "burgundy", "scoring": "closed_set"},
        "audio": {"program": "audio_program.json"},
    }
    gatea = identity._gatea_fact(
        main,
        question={"mcq": {"stem": "same", "options_space": ["burgundy", "blue"], "truth_option": "blue"}, "open": {"stem": "same", "truth_value": "blue", "scoring": "closed_set"}},
        point_id="p",
        gold="blue",
        appearance_by_slot={"source1": "blue", "source2": "burgundy"},
        speech_bindings=[],
        schedule={},
        target_slot="source2",
    )
    assert gatea["mcq"]["stem"] == main["mcq"]["stem"]
    assert gatea["mcq"]["options_space"] == main["mcq"]["options_space"]
    assert gatea["mcq"]["truth_option"] == "blue"
    assert gatea["open"]["truth_value"] == "blue"
    assert main["mcq"]["truth_option"] == "burgundy"


def test_gatea_swaps_every_audio_occurrence_without_changing_visual_slot_files():
    events = [
        {
            "phase": "early",
            "slot": "source1",
            "sound_asset_id": "voice-a",
            "start_sample": 10,
            "duration_samples": 20,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 20,
        },
        {
            "phase": "early",
            "slot": "source2",
            "sound_asset_id": "voice-b",
            "start_sample": 40,
            "duration_samples": 20,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 20,
        },
        {
            "phase": "late",
            "slot": "source1",
            "sound_asset_id": "voice-a",
            "start_sample": 70,
            "duration_samples": 20,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 20,
        },
        {
            "phase": "late",
            "slot": "source2",
            "sound_asset_id": "voice-b",
            "start_sample": 100,
            "duration_samples": 20,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 20,
        },
    ]
    for index, event in enumerate(events):
        event["event_id"] = f"voice_occurrence_{index + 1}"
    gatea = identity._swap_audio_slots(events)
    assert [event["event_id"] for event in gatea] == [
        "voice_occurrence_1", "voice_occurrence_2",
        "voice_occurrence_3", "voice_occurrence_4",
    ]
    assert [event["slot"] for event in gatea] == [
        "source2", "source1", "source2", "source1"
    ]
    assert [
        (event["sound_asset_id"], event["start_sample"], event["duration_samples"])
        for event in gatea
    ] == [
        (event["sound_asset_id"], event["start_sample"], event["duration_samples"])
        for event in events
    ]
    main_program = {
        "events": [
            {
                "source_endpoint_id": event["slot"],
                "sound_asset_id": event["sound_asset_id"],
                "start_sample": event["start_sample"],
                "end_sample_exclusive": event["start_sample"] + event["duration_samples"],
                "source_start_sample": event["source_start_sample"],
                "source_end_sample_exclusive": event["source_end_sample_exclusive"],
            }
            for event in events
        ]
    }
    gatea_program = {
        "events": [
            {
                **event,
                "source_endpoint_id": audio["slot"],
            }
            for event, audio in zip(main_program["events"], gatea, strict=True)
        ]
    }
    assert identity._audio_signature(main_program, include_endpoint=False) == identity._audio_signature(gatea_program, include_endpoint=False)
    assert identity._audio_signature(main_program, include_endpoint=True) != identity._audio_signature(gatea_program, include_endpoint=True)


def test_gatea_visual_artifact_names_are_gateb_only():
    descriptor = identity._gateb_descriptor()
    assert all("gateA" not in value for value in descriptor.values())
    assert descriptor["audio_program"] == "audio_program.json"
