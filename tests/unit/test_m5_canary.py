from __future__ import annotations

import hashlib
from pathlib import Path
import wave

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file, write_json
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m5.audio import (
    M5_AUDIO_SAMPLE_RATE_HZ,
    extract_faded_clip,
    place_simultaneous_events,
    read_pcm16_mono_wav,
    render_dynamic_stems_and_mix,
)
import avengine.m5.canary as canary


def _write_pcm16(path: Path, frequency_hz: float) -> None:
    indices = np.arange(canary.M5_DRY_CLIP_END, dtype=np.float64)
    samples = np.asarray(
        np.sin(2.0 * np.pi * frequency_hz * indices / M5_AUDIO_SAMPLE_RATE_HZ)
        * 12_000,
        dtype="<i2",
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(M5_AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(samples.tobytes())


def _audio_record(path: Path, samples: np.ndarray) -> dict[str, object]:
    artifact = write_float32_wav(
        path,
        samples,
        M5_AUDIO_SAMPLE_RATE_HZ,
        channel_axis=0,
        metadata={"test_authority": True},
    )
    return {
        "audio_path": path.as_posix(),
        "sidecar_path": artifact.sidecar_path.as_posix(),
        "audio_sha256": artifact.audio_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
    }


def _reconstruction_fixture(
    root: Path,
) -> tuple[
    dict[str, object],
    dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]],
    dict[str, object],
]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    beagle = inputs / "beagle_dry.wav"
    golden = inputs / "golden_dry.wav"
    _write_pcm16(beagle, 320.0)
    _write_pcm16(golden, 510.0)

    raw_assets: dict[str, tuple[str, np.ndarray]] = {}
    for asset_id, path in (("beagle_call", beagle), ("golden_call", golden)):
        samples, _ = read_pcm16_mono_wav(path)
        raw_assets[sha256_file(path)] = (
            asset_id,
            extract_faded_clip(
                samples,
                start_sample=canary.M5_DRY_CLIP_START,
                end_sample=canary.M5_DRY_CLIP_END,
                fade_samples=80,
            ),
        )
    beagle_hash = sha256_file(beagle)
    golden_hash = sha256_file(golden)
    requests = {
        "A": {
            "events": [
                {"source_id": "source0", "dry_audio_asset_sha256": beagle_hash},
                {"source_id": "source1", "dry_audio_asset_sha256": golden_hash},
            ]
        },
        "B": {
            "events": [
                {"source_id": "source0", "dry_audio_asset_sha256": golden_hash},
                {"source_id": "source1", "dry_audio_asset_sha256": beagle_hash},
            ]
        },
    }
    write_json(
        root / "episodes" / "counterfactual_pair.json",
        {"episodes": {variant: {"request": request} for variant, request in requests.items()}},
    )

    keyframe_samples = (0, canary.M5_AUDIO_SAMPLE_COUNT // 2)
    trajectory: dict[str, object] = {
        "keyframes": [{"sample_index": value} for value in keyframe_samples]
    }
    rir: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    for layout, channels in (("foa", 4), ("binaural", 2)):
        samples = np.zeros((len(keyframe_samples), 2, channels, 1), dtype="<f4")
        samples[:, 0, :, 0] = np.arange(1, channels + 1, dtype=np.float32) / 20.0
        samples[:, 1, :, 0] = np.arange(channels, 0, -1, dtype=np.float32) / 24.0
        lengths = np.ones((len(keyframe_samples), 2), dtype="<u4")
        rir[layout] = (samples, lengths, {})

    audio: dict[str, object] = {}
    for variant, request in requests.items():
        route = {
            event["source_id"]: raw_assets[event["dry_audio_asset_sha256"]][0]
            for event in request["events"]
        }
        buses, _ = place_simultaneous_events(
            {asset_id: clip for asset_id, clip in raw_assets.values()},
            route,
            start_samples=canary.M5_EVENT_STARTS,
            output_sample_count=canary.M5_AUDIO_SAMPLE_COUNT,
            linear_gain=canary.M5_DRY_LINEAR_GAIN,
        )
        records: dict[str, object] = {"dry_buses": {}, "foa": {}, "binaural": {}}
        for source_id in ("source0", "source1"):
            relative = Path("episodes") / variant / "audio" / "dry" / f"{source_id}.wav"
            record = _audio_record(root / relative, buses[source_id][None, :])
            record["audio_path"] = relative.as_posix()
            record["sidecar_path"] = relative.with_suffix(".wav.json").as_posix()
            records["dry_buses"][source_id] = record  # type: ignore[index]
        for layout in ("foa", "binaural"):
            rir_samples, rir_lengths, _ = rir[layout]
            stems, mixture = render_dynamic_stems_and_mix(
                buses,
                rir_samples,
                rir_lengths,
                source_ids=("source0", "source1"),
                keyframe_samples=keyframe_samples,
                output_sample_count=canary.M5_AUDIO_SAMPLE_COUNT,
            )
            for source_id in ("source0", "source1"):
                relative = (
                    Path("episodes")
                    / variant
                    / "audio"
                    / layout
                    / f"{source_id}_stem.wav"
                )
                record = _audio_record(root / relative, stems[source_id].episode)
                record["audio_path"] = relative.as_posix()
                record["sidecar_path"] = relative.with_suffix(".wav.json").as_posix()
                records[layout][source_id] = record  # type: ignore[index]
            relative = Path("episodes") / variant / "audio" / layout / "mixture.wav"
            record = _audio_record(root / relative, mixture)
            record["audio_path"] = relative.as_posix()
            record["sidecar_path"] = relative.with_suffix(".wav.json").as_posix()
            records[layout]["mixture"] = record  # type: ignore[index]
        audio[variant] = records

    evidence: dict[str, object] = {
        "inputs": {
            "beagle_dry": {"path": "inputs/beagle_dry.wav", "sha256": beagle_hash},
            "golden_dry": {"path": "inputs/golden_dry.wav", "sha256": golden_hash},
        },
        "audio": audio,
    }
    return evidence, rir, trajectory


def test_rehashed_stem_tamper_fails_independent_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(canary, "M5_DRY_CLIP_START", 0)
    monkeypatch.setattr(canary, "M5_DRY_CLIP_END", 256)
    monkeypatch.setattr(canary, "M5_EVENT_STARTS", (256, 768, 1_280))
    monkeypatch.setattr(canary, "M5_AUDIO_SAMPLE_COUNT", 2_048)
    evidence, rir, trajectory = _reconstruction_fixture(tmp_path)
    assert canary._audio_reconstruction_errors(tmp_path, evidence, rir, trajectory) == []

    record = evidence["audio"]["A"]["binaural"]["source0"]  # type: ignore[index]
    audio_path = tmp_path / record["audio_path"]  # type: ignore[index]
    sidecar_path = tmp_path / record["sidecar_path"]  # type: ignore[index]
    decoded = read_float32_wav(audio_path, verify_sidecar=True)
    tampered = decoded.samples.copy()
    tampered[0, 100] += np.float32(0.25)
    audio_path.unlink()
    sidecar_path.unlink()
    artifact = write_float32_wav(
        audio_path,
        tampered,
        M5_AUDIO_SAMPLE_RATE_HZ,
        channel_axis=0,
        metadata={"test_authority": True},
    )
    record["audio_sha256"] = artifact.audio_sha256  # type: ignore[index]
    record["sidecar_sha256"] = artifact.sidecar_sha256  # type: ignore[index]
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)

    errors = canary._audio_reconstruction_errors(tmp_path, evidence, rir, trajectory)
    assert any("A/binaural/source0 stem cannot be rebuilt" in item for item in errors)
    assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == record["audio_sha256"]  # type: ignore[index]
