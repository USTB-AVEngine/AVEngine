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
