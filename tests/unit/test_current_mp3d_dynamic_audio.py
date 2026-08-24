from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.m5.current_mp3d_dynamic_audio import (
    CURRENT_MP3D_SOURCE_AXIS_BINDINGS,
    CurrentMP3DDynamicAudioError,
    _listener_view_identity,
    build_audio_timing_evidence,
    build_source_identity_evidence,
    listener_pose_from_m1_request,
    load_captured_source_paths,
)
from avengine.m5.timeline import frame_sample_interval
from avengine.m6.audio_program import validate_audio_program
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PROGRAM_PATH = (
    REPOSITORY
    / "examples/m5/current_mp3d/audio_programs"
    / "current_mp3d_two_beagle_turn_taking_v1.json"
)


def _axis_bindings() -> list[dict[str, object]]:
    return [dict(item) for item in CURRENT_MP3D_SOURCE_AXIS_BINDINGS]


def _write_visual_capture(
    root: Path,
    *,
    receipt_source_ids: list[str] | None = None,
    stored_positions: np.ndarray | None = None,
) -> tuple[list[dict[str, object]], np.ndarray]:
    bindings = _axis_bindings()
    positions = np.asarray(
        [
            [
                [float(index), 0.5, 0.0],
                [float(index), 0.5, 1.0],
            ]
            for index in range(75)
        ],
        dtype=np.float64,
    )
    frames = [
        {
            "frame_index": index,
            "source_positions_m": positions[index].tolist(),
        }
        for index in range(75)
    ]
    arrays = root / "arrays"
    arrays.mkdir()
    np.save(
        arrays / "source_positions_m.npy",
        positions if stored_positions is None else stored_positions,
    )
    (root / "frame_records.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    (root / "research_receipt.json").write_text(
        json.dumps(
            {
                "status": "research_only",
                "research_only": True,
                "episode_counted": False,
                "capture": {
                    "native_habitat_started": True,
                    "frame_count": 75,
                },
                "actors": {
                    "source_ids": receipt_source_ids
                    or [str(item["source_id"]) for item in bindings],
                    "actor_ids": [str(item["actor_id"]) for item in bindings],
                    "emitter_link_names": [
                        str(item["emitter_link_name"]) for item in bindings
                    ],
                },
                "artifacts": {
                    "frame_records": "frame_records.json",
                    "source_positions_m": "arrays/source_positions_m.npy",
                },
            }
        ),
        encoding="utf-8",
    )
    return bindings, positions


def _resolved_dry_assets() -> dict[str, dict[str, str]]:
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    sound = next(
        item
        for item in sounds["sound_assets"]
        if item["sound_asset_id"] == "dog_beagle_v2_scheduled_dry"
    )
    return {
        sound["sound_asset_id"]: {
            "path": "/external/research/beagle.wav",
            "sha256": sound["dry_audio"]["sha256"],
        }
    }


def _timing_program(events: list[dict[str, object]]) -> dict[str, object]:
    return {
        "timeline": {
            "time_base_hz": 48_000,
            "ticks_per_frame": 3_200,
            "video_fps": 15,
            "frame_count": 75,
            "sample_rate_hz": 16_000,
            "ticks_per_sample": 3,
            "sample_count": 80_000,
        },
        "candidate_source_endpoint_ids": ["endpoint0", "endpoint1"],
        "events": events,
    }


def _event(
    event_id: str, endpoint_id: str, start_sample: int, end_sample: int
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "source_endpoint_id": endpoint_id,
        "sound_asset_id": "dry0",
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "start_tick": start_sample * 3,
        "end_tick_exclusive": end_sample * 3,
    }


def test_turn_taking_program_validates_against_repository_registries() -> None:
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
    bindings, _ = _write_visual_capture(tmp_path)
    trajectories = load_captured_source_paths(tmp_path, bindings)
    assert len(trajectories["beagle_0_muzzle"]) == 75
    assert trajectories["beagle_1_muzzle"][74] == [74.0, 0.5, 1.0]


def test_load_captured_source_paths_rejects_wrong_slot_count(tmp_path: Path) -> None:
    bindings, _ = _write_visual_capture(tmp_path)
    value = json.loads((tmp_path / "frame_records.json").read_text())
    value["frames"][0]["source_positions_m"] = [[0.0, 0.0, 0.0]]
    (tmp_path / "frame_records.json").write_text(json.dumps(value))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="one visual position"):
        load_captured_source_paths(tmp_path, bindings)


def test_load_captured_source_paths_rejects_short_captures(tmp_path: Path) -> None:
    bindings, _ = _write_visual_capture(tmp_path)
    value = json.loads((tmp_path / "frame_records.json").read_text())
    value["frames"] = value["frames"][:10]
    (tmp_path / "frame_records.json").write_text(json.dumps(value))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="75 episode frames"):
        load_captured_source_paths(tmp_path, bindings)


def test_visual_receipt_axis_swap_is_identity_corruption(tmp_path: Path) -> None:
    bindings, _ = _write_visual_capture(
        tmp_path, receipt_source_ids=["source1", "source0"]
    )
    with pytest.raises(CurrentMP3DDynamicAudioError, match="axis order differs"):
        load_captured_source_paths(tmp_path, bindings)


def test_visual_npy_axis_swap_against_frame_records_is_rejected(
    tmp_path: Path,
) -> None:
    positions = np.asarray(
        [[[float(index), 0.5, 0.0], [float(index), 0.5, 1.0]] for index in range(75)],
        dtype=np.float64,
    )
    bindings, _ = _write_visual_capture(
        tmp_path, stored_positions=positions[:, ::-1, :]
    )
    with pytest.raises(CurrentMP3DDynamicAudioError, match="NPY and frame-record"):
        load_captured_source_paths(tmp_path, bindings)


def test_spatial_source_exchange_preserves_identity_axes(tmp_path: Path) -> None:
    bindings, positions = _write_visual_capture(tmp_path)
    exchanged = positions.copy()
    exchanged[:, 0, 0] = np.arange(75, dtype=np.float64)
    exchanged[:, 1, 0] = np.arange(74, -1, -1, dtype=np.float64)
    np.save(tmp_path / "arrays/source_positions_m.npy", exchanged)
    frames = [
        {
            "frame_index": index,
            "source_positions_m": exchanged[index].tolist(),
        }
        for index in range(75)
    ]
    (tmp_path / "frame_records.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    trajectories = load_captured_source_paths(tmp_path, bindings)
    assert trajectories["beagle_0_muzzle"][74][0] == 74.0
    assert trajectories["beagle_1_muzzle"][74][0] == 0.0


def test_source_identity_is_complete_and_carries_per_entity_audio_ids() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    records = build_source_identity_evidence(
        _axis_bindings(),
        candidate_source_endpoint_ids=program["candidate_source_endpoint_ids"],
        endpoint_registry=endpoints,
        sound_registry=sounds,
        materialized_program=program,
        dry_asset_bindings=_resolved_dry_assets(),
    )
    assert [item["axis_index"] for item in records] == [0, 1]
    assert [item["source_id"] for item in records] == ["source0", "source1"]
    assert [item["actor_id"] for item in records] == ["actor0", "actor1"]
    assert [item["audio_endpoint_id"] for item in records] == [
        "beagle_0_muzzle",
        "beagle_1_muzzle",
    ]
    assert all(item["emitter_link_name"] == "beagle Xtra Mouth" for item in records)
    assert records[0]["audio_event_ids"] == [
        "beagle0_bark_1",
        "beagle0_bark_2",
        "beagle0_bark_3",
    ]
    assert records[1]["dry_audio_asset_ids"] == ["dog_beagle_v2_scheduled_dry"]
    assert records[0]["event_bindings"][0]["start_sample"] == 6000
    assert records[0]["event_bindings"][0]["start_tick"] == 18000


def test_source_identity_rejects_swapped_audio_endpoints() -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    bindings = _axis_bindings()
    bindings[0]["audio_endpoint_id"], bindings[1]["audio_endpoint_id"] = (
        bindings[1]["audio_endpoint_id"],
        bindings[0]["audio_endpoint_id"],
    )
    with pytest.raises(CurrentMP3DDynamicAudioError, match="ordered unique bijection"):
        build_source_identity_evidence(
            bindings,
            candidate_source_endpoint_ids=program["candidate_source_endpoint_ids"],
            endpoint_registry=endpoints,
            sound_registry=sounds,
            materialized_program=program,
            dry_asset_bindings=_resolved_dry_assets(),
        )


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_source_identity_rejects_duplicate_or_missing_endpoint(mutation: str) -> None:
    program = json.loads(PROGRAM_PATH.read_text(encoding="utf-8"))
    endpoints = load_source_endpoint_registry(
        REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
    )
    sounds = load_sound_asset_registry(
        REPOSITORY / "examples/m6/registries/sound_assets_v1.json"
    )
    bindings = _axis_bindings()
    if mutation == "duplicate":
        bindings[1]["audio_endpoint_id"] = bindings[0]["audio_endpoint_id"]
    else:
        bindings.pop()
    with pytest.raises(CurrentMP3DDynamicAudioError):
        build_source_identity_evidence(
            bindings,
            candidate_source_endpoint_ids=program["candidate_source_endpoint_ids"],
            endpoint_registry=endpoints,
            sound_registry=sounds,
            materialized_program=program,
            dry_asset_bindings=_resolved_dry_assets(),
        )


def test_audio_timing_handles_no_events_without_audible_onset_claim() -> None:
    evidence = build_audio_timing_evidence(_timing_program([]))
    assert evidence["event_count"] == 0
    assert len(evidence["frames"]) == 75
    assert all(not item["active_event_ids"] for item in evidence["frames"])
    assert evidence["frames"][-1]["end_sample_exclusive"] == 80_000
    assert evidence["frames"][-1]["end_tick_exclusive"] == 240_000
    waveform = evidence["waveform_activity_verification"]
    assert waveform["status"] == "pending"
    assert waveform["actual_waveform_activity_verified"] is None
    assert waveform["audible_onset_claim"] is False


def test_audio_timing_preserves_cross_source_overlap_and_exact_boundaries() -> None:
    boundary = frame_sample_interval(1)[0]
    evidence = build_audio_timing_evidence(
        _timing_program(
            [
                _event("event0", "endpoint0", boundary - 100, boundary + 100),
                _event("event1", "endpoint1", boundary, boundary + 200),
            ]
        )
    )
    assert evidence["frames"][0]["active_event_ids"] == ["event0"]
    assert evidence["frames"][1]["active_event_ids"] == ["event0", "event1"]
    assert evidence["frames"][1]["active_audio_endpoint_ids"] == [
        "endpoint0",
        "endpoint1",
    ]
    assert evidence["events"][1]["start_sample"] == boundary
    assert evidence["events"][1]["start_tick"] == boundary * 3


def test_listener_view_identity_rejects_old_capture_m1_reuse(tmp_path: Path) -> None:
    current_m1 = tmp_path / "current.json"
    old_m1 = tmp_path / "old.json"
    current_m1.write_text("{}", encoding="utf-8")
    old_m1.write_text("{}", encoding="utf-8")
    request = json.loads(
        (REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json").read_text(
            encoding="utf-8"
        )
    )
    request["request_id"] = "current_view"
    request["sources"] = [
        {"source_id": "source0"},
        {"source_id": "source1"},
    ]
    with pytest.raises(CurrentMP3DDynamicAudioError, match="same M1 listener/view"):
        _listener_view_identity(
            m1_request=request,
            m1_request_path=current_m1,
            visual_evidence={
                "receipt_path": tmp_path / "research_receipt.json",
                "receipt": {
                    "inputs": {"m1_request": str(old_m1)},
                    "selected_room": {"room_id": request["room_id"]},
                },
            },
            source_axis_bindings=_axis_bindings(),
        )


def test_listener_pose_composes_camera_colocated_listener() -> None:
    request = json.loads(
        (REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json").read_text(
            encoding="utf-8"
        )
    )
    position, orientation_wxyz = listener_pose_from_m1_request(request)
    assert len(position) == 3
    assert len(orientation_wxyz) == 4
    assert abs(float(np.linalg.norm(np.asarray(orientation_wxyz))) - 1.0) < 1.0e-6
