from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    listener_pose_from_m1_request,
    load_captured_source_paths,
)
from avengine.timeline.audio_program import validate_audio_program
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


def test_load_captured_source_paths_rejects_short_captures(tmp_path: Path) -> None:
    frames = [
        {"frame_index": index, "source_positions_m": [[0.0, 0.0, 0.0]] * 2}
        for index in range(10)
    ]
    (tmp_path / "frame_records.json").write_text(json.dumps({"frames": frames}))
    with pytest.raises(CurrentMP3DDynamicAudioError):
        load_captured_source_paths(tmp_path, ("a", "b"))


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
