"""Side-path assemblers honour SOUND_SOURCE_MODE; they do not silently dry-canvas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

import build_qa_v3_n_actor_canary as n_actor  # noqa: E402
import design_qa_v3_extended_profile as extended  # noqa: E402
import design_qa_v3_pilot_batch as pilot  # noqa: E402
from build_qa_v3_programs import (  # noqa: E402
    program_request_fields,
    require_dry_canvas_source_mode,
)


POLICY = {
    "SAMPLE_RATE_HZ": 16000,
    "CLIP_SECONDS": 5.0,
    "PROGRAM_LINEAR_GAIN": 0.18,
    "PROGRAM_FADE_SAMPLES": 80,
    "PROGRAM_MODE": "sequential_sources",
    "TIME_BASE_HZ": 48000,
    "TICKS_PER_FRAME": 3200,
    "VIDEO_FPS": 15,
    "FRAME_COUNT": 75,
    "TICKS_PER_SAMPLE": 3,
    "PROGRAM_NORMALIZATION_POLICY": "use_sound_asset_policy",
    "PROGRAM_RENDER_SOURCE_STEM": True,
    "PROGRAM_SOURCE_SPECIFIC_STEMS": True,
    "PROGRAM_ADMISSION_STATE": "research",
}


def test_require_dry_canvas_source_mode_fail_closed():
    require_dry_canvas_source_mode(
        {"SOUND_SOURCE_MODE": "dry_canvas_window"}, owner="t")
    with pytest.raises(ValueError, match="SOUND_SOURCE_MODE"):
        require_dry_canvas_source_mode({}, owner="t")
    with pytest.raises(ValueError, match="dry_canvas_window"):
        require_dry_canvas_source_mode(
            {"SOUND_SOURCE_MODE": "event_pool"}, owner="t")


def test_pilot_batch_refuses_event_pool(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"SOUND_SOURCE_MODE": "event_pool"}))
    with pytest.raises(ValueError, match="dry_canvas_window"):
        pilot.main([
            "--output-root", str(tmp_path / "out"),
            "--seed", "s",
            "--params", str(params),
            "--historical-reproduction",
        ])


def test_n_actor_canary_refuses_event_pool(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"SOUND_SOURCE_MODE": "event_pool"}))
    scene = tmp_path / "scene.json"
    scene.write_text("{}")
    with pytest.raises(ValueError, match="dry_canvas_window"):
        n_actor.main([
            "--scene-config", str(scene),
            "--params", str(params),
            "--seed", "s",
            "--out-root", str(tmp_path / "out"),
            "--snapshot-content", "/unused",
        ])


def test_extended_profile_refuses_event_pool(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"SOUND_SOURCE_MODE": "event_pool"}))
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps([{"id": "card11"}]))
    scene = tmp_path / "scene.json"
    scene.write_text("{}")
    with pytest.raises(ValueError, match="dry_canvas_window"):
        extended.main([
            "--scene-config", str(scene),
            "--profiles", str(profiles),
            "--params", str(params),
            "--out-root", str(tmp_path / "out"),
            "--seed", "s",
            "--snapshot-content", "/unused",
        ])


def test_extended_does_not_require_program_mode():
    fields = program_request_fields(POLICY, include_mode=True)
    assert fields["mode"] == "sequential_sources"
    without_mode = {key: value for key, value in POLICY.items()
                    if key != "PROGRAM_MODE"}
    with pytest.raises(ValueError, match="PROGRAM_MODE"):
        program_request_fields(without_mode)
    fields = program_request_fields(without_mode, include_mode=False)
    assert "mode" not in fields
    assert extended.audio_program_mode(
        [("source1", 8000, "bark")]) == "one_active_of_n"
    assert extended.audio_program_mode(
        [("source1", 8000, "bark"), ("source2", 24000, "bark")]
    ) == "sequential_sources"
