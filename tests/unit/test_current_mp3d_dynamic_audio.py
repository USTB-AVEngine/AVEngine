from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import avengine.m5.current_mp3d_dynamic_audio as dynamic_audio
from avengine.contracts.json_io import canonical_json_sha256
from avengine.m4.audio import read_float32_wav
from avengine.m4.spatial import rlr_foa_wav_metadata
from avengine.m5.acoustics import DynamicRIRSequence
from avengine.m5.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    listener_pose_from_m1_request,
    load_captured_source_paths,
)
from avengine.m5_1.acoustics import research_review_trajectory_record
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


def test_apartment_dynamic_audio_tool_description_declares_dual_output() -> None:
    tool_path = (
        REPOSITORY
        / "tools/m7/render_current_apartment_dynamic_audio.py"
    )
    description = ast.get_docstring(
        ast.parse(tool_path.read_text(encoding="utf-8"))
    )
    assert description is not None
    assert description.startswith(
        "Render motion-following binaural plus native FOA audio"
    )


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
        (REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json").read_text(
            encoding="utf-8"
        )
    )
    position, orientation_wxyz = listener_pose_from_m1_request(request)
    assert len(position) == 3
    assert len(orientation_wxyz) == 4
    assert abs(float(np.linalg.norm(np.asarray(orientation_wxyz))) - 1.0) < 1.0e-6


def _fake_dynamic_sequence(grid, *, layout: str) -> DynamicRIRSequence:
    if layout == "binaural":
        channel_count = 2
        layout_type = "binaural"
        layout_id = "rlr_binaural_lr_v1"
        labels = ("left", "right")
        metadata = {
            "normalization": "not_applicable",
            "coordinate_frame": "listener_local",
        }
    else:
        channel_count = 4
        layout_type = "ambisonics"
        layout_id = "rlr_foa_acn_n3d_world_v1"
        labels = ("W", "Y", "Z", "X")
        metadata = {
            "normalization": "N3D",
            "coordinate_frame": "avengine_world",
            "ambisonics": {
                "order": 1,
                "channel_ordering": "ACN",
                "acn_indices": [0, 1, 2, 3],
            },
        }
    samples = np.ones(
        (len(grid.keyframes), len(grid.source_ids), channel_count, 1),
        dtype=np.float32,
    )
    trajectory = research_review_trajectory_record(grid)
    return DynamicRIRSequence(
        samples=samples,
        lengths=np.ones(
            (len(grid.keyframes), len(grid.source_ids)), dtype=np.uint32
        ),
        source_ids=grid.source_ids,
        keyframe_ticks=tuple(frame.tick for frame in grid.keyframes),
        keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
        sample_rate_hz=grid.sample_rate_hz,
        layout_type=layout_type,
        layout_id=layout_id,
        channel_labels=labels,
        trajectory_sha256=canonical_json_sha256(trajectory),
        metadata=metadata,
    )


def _install_dual_layout_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    drift_foa_identity: bool = False,
) -> list[tuple[str, object, str | None]]:
    calls: list[tuple[str, object, str | None]] = []

    def binaural(scene, simulation, *, grid, hrtf_file_path):
        calls.append(("binaural", grid, hrtf_file_path))
        return _fake_dynamic_sequence(grid, layout="binaural")

    def foa(scene, simulation, *, grid):
        calls.append(("foa", grid, None))
        sequence = _fake_dynamic_sequence(grid, layout="foa")
        if drift_foa_identity:
            sequence = replace(sequence, trajectory_sha256="0" * 64)
        return sequence

    monkeypatch.setattr(
        dynamic_audio,
        "_load_simulation_request",
        lambda path: ({}, object()),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "load_compiled_acoustic_scene",
        lambda path, **kwargs: object(),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "load_source_endpoint_registry",
        lambda path: {},
    )
    monkeypatch.setattr(
        dynamic_audio,
        "load_sound_asset_registry",
        lambda path: {},
    )
    monkeypatch.setattr(
        dynamic_audio,
        "_asset_bindings",
        lambda *args, **kwargs: {
            "sound0": {"path": "/fake/sound0.wav", "sha256": "1" * 64}
        },
    )
    monkeypatch.setattr(
        dynamic_audio,
        "assemble_audio_program_dry_buses",
        lambda *args, **kwargs: SimpleNamespace(
            dry_audio=SimpleNamespace(
                buses={
                    "a": np.ones(80_000, dtype=np.float64),
                    "b": np.full(80_000, 0.25, dtype=np.float64),
                }
            ),
            materialized_program={"program_content_sha256": "2" * 64},
        ),
    )
    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_binaural_rir_sequence",
        binaural,
    )
    monkeypatch.setattr(
        dynamic_audio,
        "render_research_review_foa_rir_sequence",
        foa,
    )
    return calls


def _dynamic_render_kwargs(tmp_path: Path) -> dict:
    program_path = tmp_path / "program.json"
    program_path.write_text(
        json.dumps(
            {
                "program_id": "dual_layout_test",
                "revision": "v1",
                "mode": "sequential_sources",
                "candidate_source_endpoint_ids": ["a", "b"],
                "timeline": {
                    "frame_count": 75,
                    "sample_rate_hz": 16_000,
                    "sample_count": 80_000,
                },
                "events": [{"sound_asset_id": "sound0"}],
            }
        ),
        encoding="utf-8",
    )
    placeholders = {}
    for name in ("simulation", "package", "endpoints", "sounds"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        placeholders[name] = path
    hrtf = tmp_path / "listener.sofa"
    hrtf.write_bytes(b"fixture-hrtf")
    trajectories = {
        source_id: [[float(index), 0.5, slot] for index in range(75)]
        for slot, source_id in enumerate(("a", "b"))
    }
    return {
        "source_trajectories_m": trajectories,
        "listener_position_m": [0.0, 1.5, 0.0],
        "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "simulation_request_path": placeholders["simulation"],
        "package_manifest_path": placeholders["package"],
        "audio_program_path": program_path,
        "source_endpoint_registry_path": placeholders["endpoints"],
        "sound_asset_registry_path": placeholders["sounds"],
        "external_sound_asset_paths": {},
        "hrtf_file_path": hrtf,
        "output_path": tmp_path / "output",
        "position_authority": "fixture trajectories",
        "listener_authority": "fixture listener",
    }


def test_dynamic_core_dual_writes_same_trajectory_foa_and_binaural(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_dual_layout_fakes(monkeypatch)
    kwargs = _dynamic_render_kwargs(tmp_path)

    receipt = dynamic_audio.render_dynamic_research_audio(**kwargs)

    assert [call[0] for call in calls] == ["binaural", "foa"]
    assert calls[0][1] is calls[1][1]
    assert calls[0][2] == str(Path(kwargs["hrtf_file_path"]).resolve())
    assert calls[1][2] is None
    assert receipt["audio"] == {
        "sample_rate_hz": 16_000,
        "sample_count": 80_000,
        "layout_type": "binaural",
        "channel_labels": ["left", "right"],
    }
    assert "native FOA" in receipt["claim_boundary"]
    assert receipt["audio_layouts"]["foa"]["channel_labels"] == [
        "W",
        "Y",
        "Z",
        "X",
    ]
    assert receipt["audio_layouts"]["foa"]["channel_ordering"] == "ACN"
    assert receipt["audio_layouts"]["foa"]["normalization"] == "N3D"
    assert receipt["audio_layouts"]["foa"]["hrtf_used"] is False
    assert receipt["dual_layout_identity"]["same_source_and_trajectory_identity"] is True
    assert (
        receipt["rir_layouts"]["foa"]["trajectory_sha256"]
        == receipt["rir_layouts"]["binaural"]["trajectory_sha256"]
    )
    output = Path(kwargs["output_path"])
    assert len(list(output.rglob("*.wav"))) == 8
    foa_mix = read_float32_wav(output / "audio/foa/mixture.wav")
    assert foa_mix.samples.shape == (
        4,
        80_000,
    )
    assert read_float32_wav(
        output / "audio/binaural/mixture.wav"
    ).samples.shape == (2, 80_000)
    assert set(receipt["outputs"]) == {
        "audio/dry/a.wav",
        "audio/dry/b.wav",
        "audio/binaural/a_stem.wav",
        "audio/binaural/b_stem.wav",
        "audio/binaural/mixture.wav",
        "audio/foa/a_stem.wav",
        "audio/foa/b_stem.wav",
        "audio/foa/mixture.wav",
    }
    expected_spatial = rlr_foa_wav_metadata()["spatial_format"]
    assert expected_spatial["raw_channel_order"] == ["W", "Y", "Z", "X"]
    assert expected_spatial["acn_indices"] == [0, 1, 2, 3]
    assert expected_spatial["normalization"] == "N3D"
    assert expected_spatial["coordinate_frame"] == "avengine_world"
    for source_id in ("a", "b"):
        stem = read_float32_wav(output / f"audio/foa/{source_id}_stem.wav")
        assert stem.sidecar is not None
        metadata = stem.sidecar["metadata"]
        assert metadata["spatial_format"] == expected_spatial
        assert metadata["audio_role"] == "dynamic_wet_stem"
        assert metadata["source_id"] == source_id
        assert metadata["hrtf_used"] is False
        assert metadata["resampling"] == "not_applied"
        assert metadata["amplitude_normalization"] == "not_applied"
        assert metadata["limiter"] == "not_applied"
    assert foa_mix.sidecar is not None
    mix_metadata = foa_mix.sidecar["metadata"]
    assert mix_metadata["spatial_format"] == expected_spatial
    assert mix_metadata["audio_role"] == "dynamic_wet_mixture"
    assert mix_metadata["source_ids"] == ["a", "b"]
    assert mix_metadata["summation_order"] == ["a", "b"]
    assert mix_metadata["hrtf_used"] is False
    assert mix_metadata["resampling"] == "not_applied"
    assert mix_metadata["amplitude_normalization"] == "not_applied"
    assert mix_metadata["limiter"] == "not_applied"
    for relative in (
        "audio/dry/a.wav",
        "audio/binaural/a_stem.wav",
        "audio/binaural/mixture.wav",
    ):
        legacy = read_float32_wav(output / relative)
        assert legacy.sidecar is not None
        assert legacy.sidecar["metadata"] == {}


def test_dynamic_core_rejects_dual_layout_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dual_layout_fakes(monkeypatch, drift_foa_identity=True)
    kwargs = _dynamic_render_kwargs(tmp_path)

    with pytest.raises(
        CurrentMP3DDynamicAudioError,
        match="share source and trajectory identity",
    ):
        dynamic_audio.render_dynamic_research_audio(**kwargs)


def test_dynamic_core_preserves_fresh_no_clobber(
    tmp_path: Path,
) -> None:
    kwargs = _dynamic_render_kwargs(tmp_path)
    Path(kwargs["output_path"]).mkdir()

    with pytest.raises(CurrentMP3DDynamicAudioError, match="refusing to replace"):
        dynamic_audio.render_dynamic_research_audio(**kwargs)
