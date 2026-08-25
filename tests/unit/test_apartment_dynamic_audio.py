from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.timeline.current_mp3d_dynamic_audio import CurrentMP3DDynamicAudioError
from avengine.timeline.audio_program import validate_audio_program
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)
from avengine.m7.apartment_dynamic_audio import (
    apartment_ue_point_to_world_m,
    captured_static_camera_world_m,
    load_ue_anchor_trajectories,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PROGRAM_PATH = (
    REPOSITORY
    / "examples/m7/current_apartment/audio_programs"
    / "current_apartment_human_beagle_turn_taking_v1.json"
)


def test_apartment_program_validates_against_repository_registries() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert errors == []
    assert program["candidate_source_endpoint_ids"] == [
        "m6x_dog0_muzzle",
        "m6x_human0_mouth",
    ]
    events = program["events"]
    spans = [(event["start_sample"], event["end_sample_exclusive"]) for event in events]
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    sources = [event["source_endpoint_id"] for event in events]
    assert all(sources[i] != sources[i + 1] for i in range(len(sources) - 1))


def test_ue_point_transform_matches_the_camera_authority() -> None:
    # The fixed-apartment M1 review camera sits at [-0.7, 1.471, 0.65] m and
    # the UE capture records it at [-70, 65, 147.1] cm.
    assert apartment_ue_point_to_world_m([-70.0, 65.0, 147.1]) == pytest.approx(
        [-0.7, 1.471, 0.65], abs=1.0e-12
    )


def _synthetic_capture(tmp_path: Path, *, move_camera: bool = False) -> Path:
    frames = []
    for index in range(75):
        camera = {
            "location_cm": [-70.0, 65.0 + (index if move_camera else 0), 147.1],
            "rotation_deg": [0.0, 0.0, -145.0],
        }
        frames.append(
            {
                "frame_index": index,
                "camera_pose": camera,
                "actor_anchor_poses": {
                    "source1": {"location_cm": [float(index), 0.0, 27.1]},
                    "source2": {"location_cm": [float(index), 100.0, 27.1]},
                },
            }
        )
    (tmp_path / "frame_records.json").write_text(json.dumps({"frames": frames}))
    return tmp_path


def test_ue_anchor_trajectories_apply_emitter_heights(tmp_path: Path) -> None:
    capture = _synthetic_capture(tmp_path)
    trajectories = load_ue_anchor_trajectories(capture)
    human = trajectories["m6x_human0_mouth"]
    dog = trajectories["m6x_dog0_muzzle"]
    assert len(human) == 75 and len(dog) == 75
    assert human[10] == [0.1, 1.63, 0.0]
    assert dog[10] == [0.1, 0.45, 1.0]


def test_static_camera_extraction_and_motion_rejection(tmp_path: Path) -> None:
    capture = _synthetic_capture(tmp_path)
    world, ue_yaw = captured_static_camera_world_m(capture)
    assert world == pytest.approx([-0.7, 1.471, 0.65], abs=1.0e-12)
    assert ue_yaw == -145.0
    moving_dir = tmp_path / "moving"
    moving_dir.mkdir()
    with pytest.raises(CurrentMP3DDynamicAudioError):
        captured_static_camera_world_m(
            _synthetic_capture(moving_dir, move_camera=True)
        )
