"""Preparing collected clips: the filter is real, and the gate holds.

The anti-aliasing test is the one that earns its keep. Dropping every
third sample would also turn 44.1 kHz into 16 kHz, and the result would
sound plausible while a 12 kHz component silently reappeared at 4 kHz,
right where the spatial cues are. The test feeds both tones at once and
insists the impostor stays far below the real one.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from avengine.assets.sound_prepare import (
    PrepareError,
    activity_profile_for_class,
    bridge_speech_metadata,
    build_nonverbal_source_inventory,
    measure_speech_band,
    prepare_speech_clip,
    prepare_speech_registry,
    prepare_library,
    prepare_samples,
)


def _write(path: Path, samples: np.ndarray, rate: int) -> Path:
    ints = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())
    return path


def _tone(freq: float, rate: int, seconds: float, amplitude: float = 0.4):
    time = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * freq * time)


def _magnitude_at(samples: np.ndarray, rate: int, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    bins = np.fft.rfftfreq(len(samples), 1 / rate)
    index = int(np.argmin(np.abs(bins - freq)))
    return float(spectrum[max(0, index - 2) : index + 3].max())


def test_downsampling_filters_instead_of_folding(tmp_path: Path) -> None:
    """12 kHz must be removed, not folded down to 4 kHz."""

    source_rate = 44100
    mixed = _tone(1000, source_rate, 1.0) + _tone(12000, source_rate, 1.0)
    prepared, facts = prepare_samples(mixed, source_rate)

    assert facts["resample_ratio"] == "160/441"
    assert facts["antialiased"] is True
    kept = _magnitude_at(prepared, 16000, 1000)
    alias = _magnitude_at(prepared, 16000, 4000)
    assert kept > 20 * alias, f"混叠没被压住: 1 kHz {kept:.1f} vs 4 kHz {alias:.1f}"


def test_dc_offset_and_peak_are_normalised(tmp_path: Path) -> None:
    biased = _tone(500, 16000, 1.0, amplitude=0.2) + 0.36
    prepared, facts = prepare_samples(biased, 16000)
    assert abs(facts["removed_dc_offset"] - 0.36) < 0.01
    assert abs(float(prepared.mean())) < 1e-3
    assert abs(facts["prepared_peak_dbfs"] - (-3.0)) < 0.1
    assert facts["applied_gain_db"] > 0


def test_leading_silence_is_trimmed_with_a_guard(tmp_path: Path) -> None:
    rate = 16000
    padded = np.concatenate(
        [np.zeros(rate), _tone(800, rate, 0.5), np.zeros(rate)]
    )
    prepared, facts = prepare_samples(padded, rate)
    # a second of silence goes, but not the guard before the first sound
    assert 0.9 < facts["trimmed_head_s"] <= 1.0
    assert 0.9 < facts["trimmed_tail_s"] <= 1.0
    assert 0.5 < facts["prepared_duration_s"] < 0.7


def test_digital_silence_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PrepareError):
        prepare_samples(np.zeros(16000), 16000)


def _library(tmp_path: Path) -> Path:
    """A library holding: one good clip, one QC-failed, one unchecked, and
    a byte-identical copy of the good one under another class."""

    root = tmp_path / "library"
    good = _write(root / "dog_bark/one/clip.wav", _tone(700, 44100, 1.0), 44100)
    (root / "dog_bark/one/clip.qc.json").write_text(
        json.dumps({"verdict": "pass", "findings": []})
    )
    _write(root / "fire/bad/clip.wav", _tone(700, 44100, 1.0), 44100)
    (root / "fire/bad/clip.qc.json").write_text(
        json.dumps(
            {
                "verdict": "fail",
                "findings": [
                    {"severity": "fail", "reason_zh": "削波严重:3.2% 的采样点顶格"}
                ],
            }
        )
    )
    _write(root / "cat_meow/nocheck/clip.wav", _tone(900, 44100, 1.0), 44100)
    copy = root / "animal/one/clip.wav"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(good.read_bytes())
    (root / "animal/one/clip.qc.json").write_text(
        json.dumps({"verdict": "pass", "findings": []})
    )
    return root


def test_the_gate_skips_unusable_and_unchecked_and_aliases_copies(
    tmp_path: Path,
) -> None:
    root = _library(tmp_path)
    out = tmp_path / "prepared"
    report = prepare_library(root, out)
    by_source = {clip["source"]: clip for clip in report["clips"]}

    assert by_source["animal/one/clip.wav"]["status"] in ("prepared", "alias")
    assert by_source["dog_bark/one/clip.wav"]["status"] in ("prepared", "alias")
    prepared = [c for c in report["clips"] if c["status"] == "prepared"]
    aliases = [c for c in report["clips"] if c["status"] == "alias"]
    # the identical pair is prepared once and aliased once
    assert len(prepared) == 1 and len(aliases) == 1

    failed = by_source["fire/bad/clip.wav"]
    assert failed["status"] == "skipped" and "削波" in failed["reason_zh"]
    unchecked = by_source["cat_meow/nocheck/clip.wav"]
    assert unchecked["status"] == "skipped" and "质检" in unchecked["reason_zh"]

    # originals untouched, output is 16 kHz mono
    assert (root / "dog_bark/one/clip.wav").is_file()
    written = list(out.rglob("*.wav"))
    assert len(written) == 1
    with wave.open(str(written[0]), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1

def test_prepare_manifest_carries_explicit_speech_metadata(tmp_path: Path) -> None:
    root = tmp_path / "library"
    source = _write(
        root / "speech_playback" / "vctk_p225_001" / "clip.wav",
        _tone(700, 16000, 1.0),
        16000,
    )
    source.with_suffix(".qc.json").write_text(
        json.dumps({"verdict": "pass", "findings": []})
    )
    source.with_suffix(".json").write_text(
        json.dumps(
            {
                "event_classes": ["speech_playback"],
                "speaker_id": "p225",
                "utterance_id": "001",
                "transcript": "Please call Stella.",
                "split": "eval",
                "dry": True,
            }
        )
    )

    row = prepare_library(root, tmp_path / "prepared")["clips"][0]
    assert row["speaker_id"] == "p225"
    assert row["utterance_id"] == "001"
    assert row["transcript"] == "Please call Stella."
    assert row["split"] == "eval"
    assert "dry" not in row


def _speech_fixture(rate: int = 16000) -> np.ndarray:
    time = np.arange(int(rate * 4.6)) / rate
    samples = np.zeros_like(time)
    low_end = int(0.7 * rate)
    samples[:low_end] = 0.4 * np.sin(2 * np.pi * 45 * time[:low_end])
    first_start, first_end = int(0.8 * rate), int(2.2 * rate)
    second_start, second_end = int(2.45 * rate), int(3.35 * rate)
    samples[first_start:first_end] = 0.35 * np.sin(
        2 * np.pi * 700 * time[first_start:first_end]
    )
    samples[second_start:second_end] = 0.35 * np.sin(
        2 * np.pi * 700 * time[second_start:second_end]
    )
    tail_start, tail_end = int(3.4 * rate), int(3.48 * rate)
    samples[tail_start:tail_end] = 0.12 * np.sin(
        2 * np.pi * 2800 * time[tail_start:tail_end]
    )
    return samples


def test_speech_band_crop_removes_low_frequency_preroll_and_keeps_one_clip() -> None:
    samples = _speech_fixture()
    measurement = measure_speech_band(samples, 16000)
    prepared, facts = prepare_speech_clip(samples, 16000)

    assert measurement["source_activity_interval_count"] >= 2
    assert facts["source_crop_start_sample"] > int(0.6 * 16000)
    assert facts["source_crop_end_sample_exclusive"] >= int(3.45 * 16000)
    assert facts["source_offset_s"] > 0.6
    assert facts["prepared_duration_s"] <= 5.0
    assert len(facts["source_activity_intervals_s"]) >= 2
    assert prepared.ndim == 1


def test_speech_short_and_overlong_candidates_are_rejected() -> None:
    rate = 16000
    with pytest.raises(PrepareError, match="below minimum"):
        prepare_speech_clip(_tone(700, rate, 1.0), rate)
    with pytest.raises(PrepareError, match="exceeds maximum"):
        prepare_speech_clip(_tone(700, rate, 5.5), rate)


def test_speech_metadata_bridge_uses_sidecar_not_registry_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "speech_playback" / "vctk_p001_001" / "clip.wav"
    metadata = source.with_suffix(".json")
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "gender": "F",
                "speaker_id": "p001",
                "transcript": "A controlled sentence.",
                "split": "eval",
            }
        )
    )
    row = {
        "sound_asset_id": "source-1",
        "gender": "M",
        "transcript": "invented",
        "provenance": {
            "origin": (
                "source=speech_playback/vctk_p001_001/clip.wav; "
                "source_sha256=ignored"
            )
        },
    }

    bridged = bridge_speech_metadata(row, tmp_path)
    assert bridged["gender"] == "F"
    assert bridged["transcript"] == "A controlled sentence."
    assert bridged["speaker_id"] == "p001"
    assert bridged["gender_source"].endswith("clip.json#/gender")
    assert bridged["is_vctk"] is True


def test_activity_profiles_keep_non_speech_rules_separate() -> None:
    animal = activity_profile_for_class("dog_bark")
    device = activity_profile_for_class("air_conditioning")
    prompt = activity_profile_for_class("doorbell")
    unknown = activity_profile_for_class("unclassified_source")

    assert animal["activity_family"] == "animal_call"
    assert animal["filter"]["highpass_hz"] is None
    assert animal["minimum_audible_duration_s"] == 0.5
    assert device["activity_family"] == "device_continuous"
    assert device["minimum_audible_duration_s"] is None
    assert device["coverage_requirement"] == "source_activity_covers_query_window"
    assert prompt["repeat_policy"] == "allowed_and_counted_per_event"
    assert unknown["activity_family"] == "unknown"
    assert unknown["filter"] is None


def test_nonverbal_inventory_counts_independent_sources(tmp_path: Path) -> None:
    path = tmp_path / "single_label_output.csv"
    path.write_text(
        "fname,labels,ys\n"
        "1,Laughter,\n"
        "1,Laughter,\n"
        "2,Cough,\n"
        "3,Sneeze,\n"
        "3,Sneeze,\n"
    )
    inventory = build_nonverbal_source_inventory(path)
    assert inventory["label_exact_match_counts"] == {
        "Sneeze": 2,
        "Laughter": 2,
        "Cough": 1,
    }
    assert inventory["independent_source_counts"] == {
        "Sneeze": 1,
        "Laughter": 1,
        "Cough": 1,
    }


def test_speech_registry_is_no_clobber_and_bridges_gender(tmp_path: Path) -> None:
    source_root = tmp_path / "library"
    source = _write(
        source_root / "speech_playback" / "vctk_p001_001" / "clip.wav",
        _tone(700, 16000, 2.0),
        16000,
    )
    source.with_suffix(".json").write_text(
        json.dumps(
            {
                "gender": "M",
                "speaker_id": "p001",
                "transcript": "A sentence.",
                "split": "train",
            }
        )
    )
    registry = {
        "schema": "test_registry_v1",
        "sound_assets": [
            {
                "sound_asset_id": "source-1",
                "revision": "v1",
                "semantic_sound_class": "speech_playback",
                "provenance": {
                    "origin": "source=speech_playback/vctk_p001_001/clip.wav"
                },
            }
        ],
    }
    output_root = tmp_path / "prepared"
    manifest = prepare_speech_registry(
        registry,
        output_root,
        source_library_root=source_root,
        min_audible_s=1.0,
    )
    assert manifest["counts"]["prepared"] == 1
    row = next(row for row in manifest["clips"] if row["status"] == "prepared")
    assert row["gender"] == "M"
    assert row["transcript"] == "A sentence."
    assert row["human_review"]["status"] == "pending_human"
    assert (output_root / row["prepared"]).is_file()
    with pytest.raises(FileExistsError):
        prepare_speech_registry(
            registry,
            output_root,
            source_library_root=source_root,
            min_audible_s=1.0,
        )


def test_speech_output_keeps_high_frequency_content_used_outside_detector_band():
    rate = 16000
    time = np.arange(rate * 2) / rate
    samples = 0.2 * np.sin(2 * np.pi * 700 * time) + 0.1 * np.sin(2 * np.pi * 5500 * time)
    prepared, facts = prepare_speech_clip(samples, rate)
    # Interior projection avoids filter/crop edge transients. Detector-only
    # bandpass must not remove consonant-frequency content from delivered PCM.
    offset = facts["source_crop_start_sample"]
    interior = prepared[1000:-1000]
    reference = np.sin(2 * np.pi * 5500 * (np.arange(len(interior)) + offset + 1000) / rate)
    amplitude = 2 * float(np.dot(interior, reference)) / len(interior)
    assert amplitude > 0.09
    assert "bandpass_high_hz" not in facts["filter"]
    assert facts["activity_filter"]["bandpass_high_hz"] == 3400


def test_speech_preparation_refuses_silent_pcm_clipping():
    rate = 16000
    signal = 1.1 * np.sin(2 * np.pi * 700 * np.arange(rate * 2) / rate)
    with pytest.raises(PrepareError, match="PCM full scale"):
        prepare_speech_clip(signal, rate)


def test_prepared_id_distinguishes_an_explicit_normalization_change():
    from avengine.assets.sound_prepare import make_prepared_audio_id
    plain = {"operation": "speech", "normalization_applied": False, "applied_gain_db": 0.0}
    normalized = {**plain, "normalization_applied": True, "applied_gain_db": -3.0}
    assert make_prepared_audio_id("voice", source_sha256="source", facts=plain) != make_prepared_audio_id("voice", source_sha256="source", facts=normalized)
