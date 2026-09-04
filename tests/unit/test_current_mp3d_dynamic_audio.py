from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import avengine.cli as cli
import avengine.timeline.current_mp3d_dynamic_audio as dynamic_audio
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
from avengine.spatial_audio.audio import read_float32_wav

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


@pytest.mark.parametrize(
    ("frame_count", "sample_count"),
    [(75, 80000), (90, 96000), (150, 160000)],
)
def test_capture_clock_drives_trajectory_and_sample_duration(
    tmp_path: Path, frame_count: int, sample_count: int
) -> None:
    _write_dynamic_capture(tmp_path, frame_count=frame_count)
    clock = load_captured_render_clock(tmp_path)
    assert clock == {
        "frame_count": frame_count,
        "frame_rate_hz": 15,
        "ticks_per_frame": 3200,
        "time_base_hz": 48000,
        "sample_rate_hz": 16000,
        "sample_count": sample_count,
    }
    trajectories = load_captured_source_paths(tmp_path, ("a", "b"))
    assert len(trajectories["a"]) == frame_count
    assert trajectories["b"][-1] == [float(frame_count - 1), 0.5, 1.0]


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


@pytest.mark.parametrize(
    ("frame_count", "sample_count"),
    [(75, 80000), (90, 96000), (150, 160000)],
)
def test_dynamic_runtime_serializes_exact_clock_length_waves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
    sample_count: int,
) -> None:
    """Exercise the runtime's clock and real WAVE writer with tiny downstream fakes."""

    program = deepcopy(json.loads(PROGRAM_PATH.read_text(encoding="utf-8")))
    program["timeline"].update({
        "frame_count": frame_count,
        "sample_count": sample_count,
    })
    program = bind_audio_program_hash(program)
    program_path = tmp_path / "program.json"
    program_path.write_text(json.dumps(program), encoding="utf-8")
    simulation_path = tmp_path / "simulation.json"
    package_path = tmp_path / "package.json"
    hrtf_path = tmp_path / "hrtf.sofa"
    simulation_path.write_text("{}", encoding="utf-8")
    package_path.write_text("{}", encoding="utf-8")
    hrtf_path.write_bytes(b"tiny test hrtf")

    trajectories = {
        "beagle_0_muzzle": [
            [float(index), 0.0, 0.0] for index in range(frame_count)
        ],
        "beagle_1_muzzle": [
            [float(index), 0.0, 1.0] for index in range(frame_count)
        ],
    }
    monkeypatch.setattr(
        dynamic_audio,
        "_load_simulation_request",
        lambda _path: (None, None),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "load_compiled_acoustic_scene",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_binaural_rir_sequence",
        lambda *_args, **_kwargs: SimpleNamespace(
            keyframe_samples=(0,), trajectory_sha256="test-trajectory"
        ),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "_asset_bindings",
        lambda *_args, **_kwargs: {},
    )

    def fake_assembly(materialized_program, _variant_id, **_kwargs):
        return SimpleNamespace(
            materialized_program=materialized_program,
            dry_audio=SimpleNamespace(
                buses={
                    source_id: np.zeros(sample_count, dtype=np.float64)
                    for source_id in trajectories
                },
                placement_receipts=(),
            ),
        )

    monkeypatch.setattr(
        dynamic_audio,
        "assemble_audio_program_dry_buses",
        fake_assembly,
    )

    def fake_binaural_audio(dry_buses, _sequence, *, grid):
        expected = int(grid.episode_sample_count)
        stems = {
            source_id: SimpleNamespace(
                episode=np.zeros((2, expected), dtype=np.float32)
            )
            for source_id in dry_buses
        }
        return stems, np.zeros((2, expected), dtype=np.float32)

    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_binaural_audio",
        fake_binaural_audio,
    )
    output = tmp_path / "rendered"
    receipt = render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=[0.0, 0.0, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        simulation_request_path=simulation_path,
        package_manifest_path=package_path,
        audio_program_path=program_path,
        source_endpoint_registry_path=(
            REPOSITORY / "examples/registry/registries/source_endpoints_v1.json"
        ),
        sound_asset_registry_path=(
            REPOSITORY / "examples/registry/registries/sound_assets_v1.json"
        ),
        external_sound_asset_paths={},
        hrtf_file_path=hrtf_path,
        output_path=output,
        position_authority="test",
        listener_authority="test",
    )

    assert receipt["audio"]["sample_count"] == sample_count
    wave_paths = sorted(output.rglob("*.wav"))
    assert len(wave_paths) == 5
    for wave_path in wave_paths:
        wave = read_float32_wav(wave_path)
        assert wave.sample_rate_hz == 16000
        assert wave.frame_count == sample_count


def test_dynamic_runtime_rejects_cropped_complete_utterance() -> None:
    assembly = SimpleNamespace(
        dry_audio=SimpleNamespace(
            placement_receipts=(
                {
                    "event_id": "long-speech",
                    "fit": {"cropped_tail_sample_count": 3},
                },
            )
        )
    )
    with pytest.raises(
        CurrentMP3DDynamicAudioError,
        match="long-speech.*refusing to crop",
    ):
        dynamic_audio._assert_no_cropped_dry_audio(assembly)


def test_dynamic_runtime_rejects_wrong_episode_sample_shape() -> None:
    with pytest.raises(
        CurrentMP3DDynamicAudioError,
        match="requires exactly 96000",
    ):
        dynamic_audio._require_exact_episode_samples(
            np.zeros(95999),
            expected=96000,
            owner="dry bus",
            channel_major=False,
        )


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


def test_explicit_excerpt_from_long_asset_is_not_an_implicit_tail_crop():
    from types import SimpleNamespace
    from avengine.timeline.current_mp3d_dynamic_audio import _assert_no_cropped_dry_audio
    # The selected 0.3 second excerpt is complete even though its parent asset is 5 seconds.
    assembly = SimpleNamespace(dry_audio=SimpleNamespace(placement_receipts=[{
        "event_id": "ordinary_excerpt", "dry_asset": {"frame_count": 80000},
        "dry_clip_source_native_interval": {"start_sample": 3200,
            "end_sample_exclusive": 8000, "sample_count": 4800},
        "fit": {"cropped_tail_sample_count": 0, "copied_sample_count": 4800}}]))
    _assert_no_cropped_dry_audio(assembly)
