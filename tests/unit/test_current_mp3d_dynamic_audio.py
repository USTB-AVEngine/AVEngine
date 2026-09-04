from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import avengine.cli as cli
from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    _program_clock_binding,
    _program_for_visual_clock,
    load_captured_render_clock,
    listener_pose_from_m1_request,
    load_captured_source_paths,
    render_dynamic_research_audio,
)
from avengine.timeline.audio_program import (
    bind_audio_program_hash,
    validate_audio_program,
)
from avengine.registry.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PROGRAM_PATH = (
    REPOSITORY
    / "examples/timeline/current_mp3d/audio_programs"
    / "current_mp3d_two_beagle_turn_taking_v1.json"
)


def test_turn_taking_program_validates_against_repository_registries() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/registry/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/registry/registries/sound_assets_v1.json"
    )
    errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert errors == []
    assert program["mode"] == "sequential_sources"
    assert program["candidate_source_endpoint_ids"] == [
        "beagle_0_muzzle",
        "beagle_1_muzzle",
    ]


def test_turn_taking_program_alternates_without_overlap() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    events = program["events"]
    spans = [(event["start_sample"], event["end_sample_exclusive"]) for event in events]
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    sources = [event["source_endpoint_id"] for event in events]
    assert all(sources[i] != sources[i + 1] for i in range(len(sources) - 1))
    assert set(sources) == {"beagle_0_muzzle", "beagle_1_muzzle"}


def test_load_captured_source_paths_reads_all_frames(tmp_path: Path) -> None:
    frames = [
        {
            "frame_index": index,
            "source_positions_m": [
                [float(index), 0.5, 0.0],
                [float(index), 0.5, 1.0],
            ],
        }
        for index in range(75)
    ]
    (tmp_path / "frame_records.json").write_text(json.dumps({"frames": frames}))
    trajectories = load_captured_source_paths(tmp_path, ("a", "b"))
    assert len(trajectories["a"]) == 75
    assert trajectories["b"][74] == [74.0, 0.5, 1.0]


def test_load_captured_source_paths_rejects_wrong_slot_count(tmp_path: Path) -> None:
    frames = [
        {"frame_index": index, "source_positions_m": [[0.0, 0.0, 0.0]]}
        for index in range(75)
    ]
    (tmp_path / "frame_records.json").write_text(json.dumps({"frames": frames}))
    with pytest.raises(CurrentMP3DDynamicAudioError):
        load_captured_source_paths(tmp_path, ("a", "b"))


def test_load_captured_source_paths_infers_variable_length_without_receipt(
    tmp_path: Path,
) -> None:
    frames = [
        {"frame_index": index, "source_positions_m": [[0.0, 0.0, 0.0]] * 2}
        for index in range(10)
    ]
    (tmp_path / "frame_records.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    trajectories = load_captured_source_paths(tmp_path, ("a", "b"))
    assert len(trajectories["a"]) == 10
    assert load_captured_render_clock(tmp_path)["sample_count"] == 10_667


def test_listener_pose_composes_camera_colocated_listener() -> None:
    request = json.loads(
        (REPOSITORY / "examples/rooms/requests/habitat_mp3d_example.json").read_text(
            encoding="utf-8"
        )
    )
    position, orientation_wxyz = listener_pose_from_m1_request(request)
    assert len(position) == 3
    assert len(orientation_wxyz) == 4
    assert abs(float(np.linalg.norm(np.asarray(orientation_wxyz))) - 1.0) < 1.0e-6


def _write_dynamic_capture(
    directory: Path,
    *,
    frame_count: int,
    include_pts: bool = True,
) -> None:
    frames = []
    for index in range(frame_count):
        frame = {
            "frame_index": index,
            "source_positions_m": [
                [float(index), 0.5, 0.0],
                [float(index), 0.5, 1.0],
            ],
        }
        if include_pts:
            frame["pts_ticks"] = index * 3200
        frames.append(frame)
    (directory / "frame_records.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    (directory / "research_receipt.json").write_text(
        json.dumps({
            "capture": {
                "frame_count": frame_count,
                "frame_rate_hz": 15,
                "ticks_per_frame": 3200,
                "time_base_hz": 48000,
            }
        }),
        encoding="utf-8",
    )


def test_150_frame_capture_clock_drives_trajectory_and_sample_duration(
    tmp_path: Path,
) -> None:
    _write_dynamic_capture(tmp_path, frame_count=150)
    clock = load_captured_render_clock(tmp_path)
    assert clock == {
        "frame_count": 150,
        "frame_rate_hz": 15,
        "ticks_per_frame": 3200,
        "time_base_hz": 48000,
        "sample_rate_hz": 16000,
        "sample_count": 160000,
    }
    trajectories = load_captured_source_paths(tmp_path, ("a", "b"))
    assert len(trajectories["a"]) == 150
    assert trajectories["b"][-1] == [149.0, 0.5, 1.0]


def test_capture_clock_rejects_a_pts_mismatch(tmp_path: Path) -> None:
    _write_dynamic_capture(tmp_path, frame_count=150)
    payload = json.loads(
        (tmp_path / "frame_records.json").read_text(encoding="utf-8")
    )
    payload["frames"][17]["pts_ticks"] += 1
    (tmp_path / "frame_records.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(
        CurrentMP3DDynamicAudioError,
        match="PTS differs",
    ):
        load_captured_render_clock(tmp_path)


def _ten_second_clock() -> dict[str, int]:
    return {
        "frame_count": 150,
        "frame_rate_hz": 15,
        "ticks_per_frame": 3200,
        "time_base_hz": 48000,
        "sample_count": 160000,
    }


def test_audio_program_clock_binding_rejects_explicit_duration_mismatch() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    with pytest.raises(
        CurrentMP3DDynamicAudioError,
        match="frame_count declares 75",
    ):
        _program_for_visual_clock(program, _ten_second_clock())


def test_dynamic_renderer_rejects_visual_program_duration_mismatch_before_runtime(
    tmp_path: Path,
) -> None:
    trajectories = {
        "beagle_0_muzzle": [[0.0, 0.0, 0.0] for _ in range(150)],
        "beagle_1_muzzle": [[1.0, 0.0, 0.0] for _ in range(150)],
    }
    with pytest.raises(
        CurrentMP3DDynamicAudioError,
        match="frame_count declares 75",
    ):
        render_dynamic_research_audio(
            source_trajectories_m=trajectories,
            listener_position_m=[0.0, 0.0, 0.0],
            listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            simulation_request_path=tmp_path / "unused-simulation.json",
            package_manifest_path=tmp_path / "unused-package.json",
            audio_program_path=PROGRAM_PATH,
            source_endpoint_registry_path=(
                REPOSITORY / "examples/registry/registries/source_endpoints_v1.json"
            ),
            sound_asset_registry_path=(
                REPOSITORY / "examples/registry/registries/sound_assets_v1.json"
            ),
            external_sound_asset_paths={},
            hrtf_file_path=tmp_path / "unused.sofa",
            output_path=tmp_path / "out",
            position_authority="test",
            listener_authority="test",
        )


def test_missing_program_clock_metadata_uses_an_explicit_legacy_fill() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    program = deepcopy(program)
    program["timeline"] = {}
    clock = _ten_second_clock()
    binding = _program_clock_binding(program, clock)
    assert binding == {
        "mode": "legacy_default_fill",
        "filled_fields": [
            "time_base_hz",
            "ticks_per_frame",
            "video_fps",
            "frame_count",
            "sample_rate_hz",
            "ticks_per_sample",
            "sample_count",
        ],
    }
    bound = _program_for_visual_clock(program, clock)
    assert validate_audio_program(bound) == []
    assert bound["timeline"]["sample_count"] == 160000


def test_audio_program_clock_binding_accepts_a_new_explicit_duration() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    program = deepcopy(program)
    program["timeline"]["frame_count"] = 150
    program["timeline"]["sample_count"] = 160000
    program = bind_audio_program_hash(program)
    bound = _program_for_visual_clock(program, _ten_second_clock())
    assert bound == program
    assert validate_audio_program(bound) == []


def test_cli_propagates_explicit_dynamic_clock_options() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([
        "m5",
        "render-current-mp3d-dynamic-audio",
        "--visual-capture-dir", "capture",
        "--m1-request", "m1.json",
        "--simulation-request", "simulation.json",
        "--package-manifest", "package.json",
        "--audio-program", "program.json",
        "--source-endpoint-registry", "endpoints.json",
        "--sound-asset-registry", "sounds.json",
        "--beagle-audio", "beagle.wav",
        "--hrtf", "hrtf.sofa",
        "--runtime-prefix", "runtime",
        "--rlr-sdk-root", "rlr",
        "--frame-count", "150",
        "--frame-rate-hz", "15",
        "--ticks-per-frame", "3200",
        "--output", "out",
    ])
    assert args.frame_count == 150
    assert args.frame_rate_hz == 15.0
    assert args.ticks_per_frame == 3200

    author = parser.parse_args([
        "m5",
        "author-current-apartment-visual-timeline",
        "--actor-selection", "selection.json",
        "--source-asset-registry", "registry.json",
        "--camera-position-ue-cm", "0", "0", "100",
        "--camera-yaw-deg", "0",
        "--human-start-ue-cm", "0", "0", "0",
        "--human-end-ue-cm", "100", "0", "0",
        "--beagle-start-ue-cm", "0", "100", "0",
        "--beagle-end-ue-cm", "100", "100", "0",
        "--frame-count", "150",
        "--frame-rate-hz", "15",
        "--ticks-per-frame", "3200",
        "--output", "timeline.json",
    ])
    assert author.frame_count == 150
    assert author.frame_rate_hz == 15.0
    assert author.ticks_per_frame == 3200
