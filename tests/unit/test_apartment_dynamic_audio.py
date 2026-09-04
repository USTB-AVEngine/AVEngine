from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.timeline.current_mp3d_dynamic_audio import CurrentMP3DDynamicAudioError
from avengine.timeline.audio_program import validate_audio_program
from avengine.registry.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)
from avengine.dataset.apartment_dynamic_audio import (
    apartment_ue_point_to_world_m,
    captured_static_camera_world_m,
    load_ue_anchor_trajectories,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PROGRAM_PATH = (
    REPOSITORY
    / "examples/dataset/current_apartment/audio_programs"
    / "current_apartment_human_beagle_turn_taking_v1.json"
)


def test_apartment_program_validates_against_repository_registries() -> None:
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


def test_observed_emitter_retains_horizontal_offset_and_height(tmp_path):
    capture = _synthetic_capture(tmp_path)
    path = capture / "frame_records.json"
    payload = json.loads(path.read_text())
    for frame in payload["frames"]:
        # Rotating an off-center source moves its emitter on both horizontal
        # axes. These are UE readbacks, not the visual actor's root position.
        frame["source_emitter_poses"] = {
            "source1": {"location_cm": [125.0, 240.0, 85.0]},
        }
    path.write_text(json.dumps(payload))
    result = load_ue_anchor_trajectories(
        capture, slot_endpoints={"source1": "speaker", "source2": "dog"},
        emitter_heights_m={"source2": 0.45},
    )
    assert result["speaker"][10] == pytest.approx([1.25, 0.85, 2.4])
    assert result["dog"][10] == pytest.approx([0.1, 0.45, 1.0])
    # Losing a required rigid readback must not silently return to root height.
    del payload["frames"][12]["source_emitter_poses"]["source1"]
    path.write_text(json.dumps(payload))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="frame 12.*required source1"):
        load_ue_anchor_trajectories(
            capture, slot_endpoints={"source1": "speaker", "source2": "dog"},
            emitter_heights_m={"source2": 0.45},
        )

def test_150_frame_capture_clock_drives_apartment_audio_trajectories(
    tmp_path: Path,
) -> None:
    frames = []
    for index in range(150):
        frames.append({
            "frame_index": index,
            "pts_ticks": index * 3200,
            "camera_pose": {
                "location_cm": [-70.0, 65.0, 147.1],
                "rotation_deg": [0.0, 0.0, -145.0],
            },
            "actor_anchor_poses": {
                "source1": {"location_cm": [float(index), 0.0, 27.1]},
                "source2": {"location_cm": [float(index), 100.0, 27.1]},
            },
        })
    (tmp_path / "frame_records.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    (tmp_path / "research_receipt.json").write_text(
        json.dumps({
            "capture": {
                "frame_count": 150,
                "frame_rate_hz": 15,
                "ticks_per_frame": 3200,
                "time_base_hz": 48000,
            }
        }),
        encoding="utf-8",
    )
    trajectories = load_ue_anchor_trajectories(tmp_path)
    assert len(trajectories["m6x_human0_mouth"]) == 150
    assert trajectories["m6x_dog0_muzzle"][-1] == pytest.approx(
        [1.49, 0.45, 1.0]
    )
    world, yaw = captured_static_camera_world_m(tmp_path)
    assert world == pytest.approx([-0.7, 1.471, 0.65])
    assert yaw == -145.0


def test_explicit_canonical_height_overrides_existing_emitter_readback(
    tmp_path: Path,
) -> None:
    capture = _synthetic_capture(tmp_path)
    payload = json.loads(
        (capture / "frame_records.json").read_text(encoding="utf-8")
    )
    for frame in payload["frames"]:
        frame["source_emitter_poses"] = {
            "source1": {"location_cm": [125.0, 240.0, 85.0]},
        }
    (capture / "frame_records.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = load_ue_anchor_trajectories(
        capture,
        slot_endpoints={"source1": "speaker", "source2": "dog"},
        emitter_heights_m={"source2": 0.45},
        canonical_emitter_height_m=0.77,
    )
    assert result["speaker"][10] == pytest.approx([1.25, 0.77, 2.4])
