from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import avengine.cli as cli
import avengine.timeline.current_mp3d_dynamic_audio as dynamic_audio
from avengine.contracts.json_io import sha256_file
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
            hrtf_file_path=None,
            layouts=("ambisonics",),
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
    rir_calls = []

    def fake_rir(
        _scene, _simulation, *, grid, layout_type, hrtf_file_path=None
    ):
        labels = (
            ("left", "right")
            if layout_type == "binaural"
            else ("W", "Y", "Z", "X")
        )
        rir_calls.append((layout_type, hrtf_file_path))
        return SimpleNamespace(
            layout_type=layout_type,
            layout_id=f"test-{layout_type}",
            channel_labels=labels,
            keyframe_samples=(0,),
            trajectory_sha256="test-trajectory",
        )

    monkeypatch.setattr(
        dynamic_audio, "render_research_review_rir_sequence", fake_rir
    )

    monkeypatch.setattr(
        dynamic_audio,
        "_asset_bindings",
        lambda *_args, **_kwargs: {},
    )

    def fake_assembly(materialized_program, _variant_id, **_kwargs):
        if _variant_id != "A":
            raise RuntimeError("invalid variant discovered during program assembly")
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

    def fake_layout_audio(dry_buses, sequence, *, grid):
        expected = int(grid.episode_sample_count)
        channels = 2 if sequence.layout_type == "binaural" else 4
        stems = {
            source_id: SimpleNamespace(
                episode=np.zeros((channels, expected), dtype=np.float32)
            )
            for source_id in dry_buses
        }
        return stems, np.zeros((channels, expected), dtype=np.float32)

    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_audio",
        fake_layout_audio,
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
        execution_variant="gateA",
    )

    assert receipt["audio"]["sample_count"] == sample_count
    assert receipt["audio"]["layouts"] == ["binaural"]
    assert receipt["audio"]["layout_type"] == "binaural"
    assert receipt["audio"]["channel_labels"] == ["left", "right"]
    assert receipt["rir"]["by_layout"]["binaural"]["channel_labels"] == [
        "left", "right"
    ]
    assert receipt["audio_program"]["variant_id"] == "A"
    assert receipt["execution_variant"] == "gateA"
    wave_paths = sorted(output.rglob("*.wav"))
    assert len(wave_paths) == 5
    for wave_path in wave_paths:
        wave = read_float32_wav(wave_path)
        assert wave.sample_rate_hz == 16000
        assert wave.frame_count == sample_count
    assert len(rir_calls) == 1
    assert rir_calls[0][0] == "binaural"

    multi_output = tmp_path / "rendered_multi"
    multi = render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=[0.0, 0.0, 0.0],
        listener_orientation_wxyz=[0.0, 0.0, 0.0, 1.0],
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
        output_path=multi_output,
        position_authority="test",
        listener_authority="test",
        layouts=("binaural", "ambisonics"),
    )
    assert multi["audio"]["layouts"] == ["binaural", "ambisonics"]
    assert set(multi["audio"]["by_layout"]) == {"binaural", "ambisonics"}
    assert multi["audio"]["by_layout"]["ambisonics"] == {
        "layout_type": "ambisonics",
        "output_directory": "foa",
        "channel_count": 4,
        "channel_labels": ["W", "Y", "Z", "X"],
        "sample_rate_hz": 16000,
        "sample_count": sample_count,
    }
    assert set(multi["rir"]["by_layout"]) == {"binaural", "ambisonics"}
    assert multi["rir"]["layout_type"] == "binaural"
    assert multi["rir"]["channel_labels"] == ["left", "right"]
    assert len(list(multi_output.rglob("*.wav"))) == 8
    assert (multi_output / "audio" / "binaural" / "mixture.wav").is_file()
    assert (multi_output / "audio" / "foa" / "mixture.wav").is_file()
    assert [item[0] for item in rir_calls] == [
        "binaural", "binaural", "ambisonics"
    ]

    foa_output = tmp_path / "rendered_foa"
    foa_only = render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=[0.0, 0.0, 0.0],
        listener_orientation_wxyz=[0.0, 0.0, 0.0, 1.0],
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
        hrtf_file_path=None,
        output_path=foa_output,
        position_authority="test",
        listener_authority="test",
        layouts=("ambisonics",),
    )
    assert foa_only["audio"]["layouts"] == ["ambisonics"]
    assert foa_only["audio"]["layout_type"] == "ambisonics"
    assert foa_only["audio"]["channel_labels"] == ["W", "Y", "Z", "X"]
    assert foa_only["inputs"]["hrtf"] is None
    assert len(list(foa_output.rglob("*.wav"))) == 5
    assert (foa_output / "audio" / "foa" / "mixture.wav").is_file()
    assert rir_calls[-1][0] == "ambisonics"
    assert rir_calls[-1][1] is None

    with pytest.raises(RuntimeError, match="invalid variant"):
        render_dynamic_research_audio(
            source_trajectories_m=trajectories,
            listener_position_m=[0, 0, 0],
            listener_orientation_wxyz=[1, 0, 0, 0],
            simulation_request_path=simulation_path,
            package_manifest_path=package_path,
            audio_program_path=program_path,
            source_endpoint_registry_path=REPOSITORY / "examples/registry/registries/source_endpoints_v1.json",
            sound_asset_registry_path=REPOSITORY / "examples/registry/registries/sound_assets_v1.json",
            external_sound_asset_paths={}, hrtf_file_path=hrtf_path,
            output_path=tmp_path / "invalid_variant", position_authority="test",
            listener_authority="test", variant_id="invalid")
    assert len(rir_calls) == 4


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
        "--layouts", "binaural,ambisonics",
        "--execution-variant", "main",
        "--output", "out",
    ])
    assert args.frame_count == 150
    assert args.frame_rate_hz == 15.0
    assert args.ticks_per_frame == 3200
    assert args.layouts == ("binaural", "ambisonics")
    assert args.execution_variant == "main"

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



def test_capture_endpoint_order_is_authoritative_over_program_order(tmp_path):
    frames = [{"frame_index": i, "source_positions_m": [[i, 1, 0], [i, 2, 0]]}
              for i in range(75)]
    (tmp_path / "frame_records.json").write_text(json.dumps({
        "source_endpoint_ids": ["b", "a"], "frames": frames}))
    result = load_captured_source_paths(tmp_path, ("a", "b"))
    assert result["a"][10] == [10.0, 2.0, 0.0]
    assert result["b"][10] == [10.0, 1.0, 0.0]


def test_capture_endpoint_ids_cannot_silently_disagree(tmp_path):
    (tmp_path / "frame_records.json").write_text(json.dumps({
        "source_endpoint_ids": ["a", "wrong"],
        "frames": [{"frame_index": i, "source_positions_m": [[0, 0, 0], [1, 0, 0]]}
                   for i in range(75)]}))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="uniquely match"):
        load_captured_source_paths(tmp_path, ("a", "b"))


@pytest.fixture(autouse=True)
def _use_minimal_sound_index_for_binding_unit_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dynamic_audio,
        "sound_index",
        lambda registry: {
            row["sound_asset_id"]: row
            for row in registry.get("sound_assets", [])
        },
    )


def _sound_registry_record(sound_id: str, uri: str, digest: str) -> dict:
    return {
        "sound_assets": [
            {
                "sound_asset_id": sound_id,
                "dry_audio": {"uri": uri, "sha256": digest},
            }
        ]
    }


def test_asset_bindings_resolve_registry_declared_file_uri_and_percent_escape(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "speech sample.wav"
    audio.write_bytes(b"file-uri-audio")
    uri = audio.as_uri()
    sounds = _sound_registry_record("speech", uri, sha256_file(audio))
    bindings = dynamic_audio._asset_bindings(
        sounds,
        repository_root=tmp_path / "repo",
        external_sound_asset_paths={},
        required_sound_ids={"speech"},
    )
    assert bindings["speech"] == {
        "path": str(audio.resolve()),
        "sha256": sha256_file(audio),
    }
    assert "%20" in uri


@pytest.mark.parametrize("uri", ["file:relative.wav", "file://host/tmp/speech.wav", "file://localhost/tmp/speech.wav"])
def test_asset_bindings_reject_relative_or_host_file_uris(
    tmp_path: Path, uri: str,
) -> None:
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"file-uri-audio")
    if uri.endswith("speech.wav") and "host" in uri:
        uri = f"file://host{audio.resolve()}"
    sounds = _sound_registry_record("speech", uri, sha256_file(audio))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="file URI.*(host|absolute path)"):
        dynamic_audio._asset_bindings(
            sounds,
            repository_root=tmp_path / "repo",
            external_sound_asset_paths={},
            required_sound_ids={"speech"},
        )


def test_asset_bindings_keep_explicit_mapping_for_legacy_artifact_uri(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "mapped.wav"
    audio.write_bytes(b"mapped-audio")
    sounds = _sound_registry_record(
        "legacy", "artifact://legacy/event.wav", sha256_file(audio)
    )
    bindings = dynamic_audio._asset_bindings(
        sounds,
        repository_root=tmp_path / "repo",
        external_sound_asset_paths={"legacy": audio},
        required_sound_ids={"legacy"},
    )
    assert bindings["legacy"]["path"] == str(audio.resolve())


def test_asset_bindings_resolve_repo_uri_and_retain_digest_check(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    audio = repository / "examples" / "speech.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"repo-audio")
    sounds = _sound_registry_record("repo", "repo://examples/speech.wav", sha256_file(audio))
    bindings = dynamic_audio._asset_bindings(
        sounds,
        repository_root=repository,
        external_sound_asset_paths={},
        required_sound_ids={"repo"},
    )
    assert bindings["repo"]["path"] == str(audio.resolve())
    bad = _sound_registry_record("repo", "repo://examples/speech.wav", "0" * 64)
    with pytest.raises(CurrentMP3DDynamicAudioError, match="registry digest"):
        dynamic_audio._asset_bindings(
            bad,
            repository_root=repository,
            external_sound_asset_paths={},
            required_sound_ids={"repo"},
        )


@pytest.mark.parametrize("uri", ["relative.wav", "https://example.invalid/speech.wav", "artifact://legacy/event.wav"])
def test_asset_bindings_require_mapping_for_non_file_repo_uri(
    tmp_path: Path, uri: str,
) -> None:
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"mapped-audio")
    sounds = _sound_registry_record("unmapped", uri, sha256_file(audio))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="explicit external dry path"):
        dynamic_audio._asset_bindings(
            sounds,
            repository_root=tmp_path / "repo",
            external_sound_asset_paths={},
            required_sound_ids={"unmapped"},
        )


def test_layout_validation_uses_only_canonical_unique_values():
    assert dynamic_audio._normalize_layouts(None) == ("binaural",)
    assert dynamic_audio._normalize_layouts(
        ("ambisonics", "binaural")
    ) == ("ambisonics", "binaural")
    with pytest.raises(CurrentMP3DDynamicAudioError, match="only binaural, ambisonics"):
        dynamic_audio._normalize_layouts(("foa",))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="must not contain duplicates"):
        dynamic_audio._normalize_layouts(("binaural", "binaural"))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="sequence"):
        dynamic_audio._normalize_layouts("binaural")


def test_dataset_renderer_cli_parses_layouts_without_defaulting_to_foa():
    import importlib.util

    tool_path = REPOSITORY / "tools" / "dataset" / (
        "render_current_apartment_dynamic_audio.py"
    )
    spec = importlib.util.spec_from_file_location("dataset_dynamic_audio", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_layouts("binaural, ambisonics") == (
        "binaural", "ambisonics"
    )
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_layouts("binaural,foa")
