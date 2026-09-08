
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

import avengine.timeline.current_mp3d_dynamic_audio as dynamic_audio
from avengine.contracts.json_io import sha256_file
from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav
from avengine.timeline.audio_program import bind_audio_program_hash
from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    _apply_post_assembly_convolution_gain,
    render_dynamic_research_audio,
    validate_post_assembly_convolution_gain,
)
from avengine.timeline.unified_audio_receipt import (
    UnifiedAudioReceiptError,
    validate_unified_audio_receipt,
)
from tools.acoustics.render_frame_readback_sequential_speech import _simulation


def _program(clip: Path) -> dict:
    return bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": "gain_fixture",
            "revision": "v1",
            "mode": "one_active_of_n",
            "timeline": {
                "time_base_hz": 48_000,
                "ticks_per_frame": 3_200,
                "video_fps": 15,
                "frame_count": 15,
                "sample_rate_hz": 16_000,
                "ticks_per_sample": 3,
                "sample_count": 16_000,
            },
            "candidate_source_endpoint_ids": ["source1", "source2"],
            "events": [
                {
                    "event_id": "event1",
                    "source_endpoint_id": "source1",
                    "sound_asset_id": "tone",
                    "start_tick": 3_000,
                    "end_tick_exclusive": 6_000,
                    "start_sample": 1_000,
                    "end_sample_exclusive": 2_000,
                    "source_start_sample": 0,
                    "source_end_sample_exclusive": 1_000,
                    "linear_gain": 1.0,
                    "fade_samples": 20,
                    "render_source_stem": True,
                    "normalization_policy": "use_sound_asset_policy",
                }
            ],
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )


def _write_pcm16(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(
            np.rint(np.asarray(samples) * 32767.0).astype("<i2").tobytes()
        )


def test_post_assembly_gain_rejects_invalid_values() -> None:
    assert validate_post_assembly_convolution_gain(None) == 1.0
    assert validate_post_assembly_convolution_gain(0.5) == 0.5
    for value in (True, "0.5", -0.1, float("nan"), float("inf")):
        with pytest.raises(CurrentMP3DDynamicAudioError, match="finite non-negative"):
            validate_post_assembly_convolution_gain(value)


def test_post_assembly_gain_rejects_nonfinite_pcm_after_scaling() -> None:
    with pytest.raises(CurrentMP3DDynamicAudioError, match="contains non-finite"):
        _apply_post_assembly_convolution_gain(
            np.asarray([np.nan]), gain=0.5, owner="fixture"
        )
    with pytest.raises(CurrentMP3DDynamicAudioError, match="becomes non-finite"):
        _apply_post_assembly_convolution_gain(
            np.asarray([1.0e308]), gain=1.0e308, owner="fixture"
        )


def test_dynamic_renderer_scales_wet_pcm_and_receipt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "tone.wav"
    _write_pcm16(clip, np.full(1_000, 0.25, dtype=np.float64))
    hrtf = tmp_path / "fixture.sofa"
    hrtf.write_bytes(b"fixture")
    package = tmp_path / "package.json"
    package.write_text("{}", encoding="utf-8")
    program = _program(clip)
    trajectories = {
        source_id: [[float(index), 0.0, -2.0] for index in range(15)]
        for source_id in ("source1", "source2")
    }

    monkeypatch.setattr(
        dynamic_audio, "load_compiled_acoustic_scene", lambda *args, **kwargs: object()
    )

    def fake_rir(_scene, _simulation, *, grid, layout_type, hrtf_file_path=None):
        return SimpleNamespace(
            layout_type=layout_type,
            layout_id="fixture-binaural",
            channel_labels=("left", "right"),
            keyframe_samples=tuple(item.sample_index for item in grid.keyframes),
            trajectory_sha256="fixture-trajectory",
            samples=np.ones((len(grid.keyframes), 2, 2, 2), dtype="<f4"),
            lengths=np.full((len(grid.keyframes), 2), 2, dtype="<u4"),
        )

    def fake_audio(dry_buses, sequence, *, grid):
        count = int(grid.episode_sample_count)
        stems = {
            source_id: SimpleNamespace(
                episode=np.full((2, count), 0.4, dtype=np.float64)
            )
            for source_id in dry_buses
        }
        return stems, np.full((2, count), 0.8, dtype=np.float64)

    monkeypatch.setattr(
        dynamic_audio, "render_research_review_rir_sequence", fake_rir
    )
    monkeypatch.setattr(dynamic_audio, "render_research_review_audio", fake_audio)

    receipt = render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        simulation_mapping=_simulation(
            direct_ray_count=2,
            indirect_ray_count=2,
            source_ray_count=2,
            indirect_ray_depth=2,
            source_ray_depth=2,
        ).to_dict(),
        package_manifest_path=package,
        audio_program=program,
        event_asset_bindings={"tone": clip},
        hrtf_file_path=hrtf,
        output_path=tmp_path / "rendered",
        position_authority="fixture",
        listener_authority="fixture",
        post_assembly_convolution_gain=0.5,
    )

    wet_stem = read_float32_wav(
        tmp_path / "rendered/audio/binaural/source1_stem.wav"
    )
    wet_mix = read_float32_wav(tmp_path / "rendered/audio/binaural/mixture.wav")
    dry = read_float32_wav(tmp_path / "rendered/audio/dry/source1.wav")
    assert float(np.max(np.abs(wet_stem.samples))) == pytest.approx(0.2, abs=2e-5)
    assert float(np.max(np.abs(wet_mix.samples))) == pytest.approx(0.4, abs=2e-5)
    assert float(np.max(np.abs(dry.samples))) == pytest.approx(0.25, abs=2e-5)
    assert receipt["gain_application"]["post_assembly_convolution_gain"] == 0.5
    assert receipt["events"][0]["gain_application"]["post_assembly_convolution_gain"] == 0.5
    assert receipt["qa"]["event_clock_and_gain"]["post_assembly_convolution_gain"] == 0.5
    assert validate_unified_audio_receipt(receipt)["status"] == "pass"


def _minimal_receipt(gain: float, *, event_gain: float | None = None) -> dict:
    event_gain = gain if event_gain is None else event_gain
    return {
        "schema": "avengine_unified_audio_receipt_v1",
        "clock": {
            "time_base_hz": 48_000,
            "ticks_per_frame": 3_200,
            "frame_rate_hz": 15,
            "frame_count": 15,
            "sample_rate_hz": 16_000,
            "sample_count": 16_000,
            "duration_seconds": 1.0,
            "ticks_per_sample": 3,
        },
        "audio": {
            "sample_rate_hz": 16_000,
            "sample_count": 16_000,
            "layouts": ["binaural"],
            "layout_type": "binaural",
            "channel_labels": ["left", "right"],
            "mixture_path": "mixture.wav",
            "stems": {"source1": "stem.wav"},
            "by_layout": {
                "binaural": {
                    "channel_count": 2,
                    "channel_labels": ["left", "right"],
                    "sample_rate_hz": 16_000,
                    "sample_count": 16_000,
                }
            },
        },
        "outputs_by_layout": {
            "binaural": {
                "mixture": "mixture.wav",
                "stems": {"source1": "stem.wav"},
            }
        },
        "events": [
            {
                "event_id": "event1",
                "source_activity_intervals_samples": [
                    {"start_sample": 1_000, "end_sample_exclusive": 1_500}
                ],
                "wet_tail_intervals": [
                    {"start_sample": 1_000, "end_sample_exclusive": 2_000}
                ],
                "gain_application": {
                    "application_count": 1,
                    "post_assembly_convolution_gain": event_gain,
                    "post_assembly_convolution_gain_application_count": 1,
                    "normalization": False,
                },
            }
        ],
        "wet_tail_intervals": [
            {
                "event_id": "event1",
                "start_sample": 1_000,
                "end_sample_exclusive": 2_000,
            }
        ],
        "peak_dbfs": {"mixture": -12.0},
        "gain_application": {
            "applied_once_per_event": True,
            "post_assembly_convolution_gain": gain,
            "post_assembly_convolution_gain_application_count": 1,
            "normalization": False,
            "limiting": False,
        },
        "propagation": {"diffraction": False, "max_diffraction_order": 0},
        "hrtf": {"id": "fixture"},
        "input_neutral_readback": {"path": "neutral.json"},
    }


def test_receipt_accepts_declared_nonunity_gain_and_rejects_mismatch() -> None:
    assert validate_unified_audio_receipt(
        _minimal_receipt(0.5), require_files=False
    )["status"] == "pass"
    with pytest.raises(UnifiedAudioReceiptError, match="gains differ"):
        validate_unified_audio_receipt(
            _minimal_receipt(0.5, event_gain=1.0), require_files=False
        )


def test_receipt_does_not_infer_missing_nonunity_gain_proof():
    receipt = _minimal_receipt(0.5)
    receipt["events"][0]["gain_application"].pop("post_assembly_convolution_gain")
    with pytest.raises(UnifiedAudioReceiptError, match="finite non-negative"):
        validate_unified_audio_receipt(receipt, require_files=False)
    receipt = _minimal_receipt(0.5)
    receipt["gain_application"].pop("post_assembly_convolution_gain_application_count")
    with pytest.raises(UnifiedAudioReceiptError, match="one-time"):
        validate_unified_audio_receipt(receipt, require_files=False)
