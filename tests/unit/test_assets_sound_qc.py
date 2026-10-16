"""Dry-clip QC: the defects it must catch, and the two it must not invent.

Both negative tests come from the first real delivery. A continuous sound
has no quiet passage, and a clip peak-normalised to full scale is not
distorted; earlier drafts of these thresholds flagged nineteen and eleven
healthy clips respectively, and a checker that cries wolf on healthy
material teaches its user to ignore it.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

from avengine.assets.sound_qc import audit_clip, write_clip_qc


def _write(path: Path, samples: np.ndarray, rate: int = 16000) -> Path:
    clipped = np.clip(samples, -1.0, 1.0)
    ints = np.round(clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())
    return path


def _burst(rate: int = 16000, seconds: float = 2.0, decay: float = 30.0) -> np.ndarray:
    """One impulsive event: a tone that starts, decays, then silence."""

    time = np.arange(int(rate * seconds)) / rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * time) * np.exp(-decay * time)
    return tone


def _continuous(rate: int = 16000, seconds: float = 2.0) -> np.ndarray:
    """A sound with no quiet passage at all - a fan, an air conditioner."""

    generator = np.random.default_rng(20260830)
    return 0.2 * generator.standard_normal(int(rate * seconds)).astype(np.float32)


def _names(report: dict) -> set[str]:
    return {finding["name"] for finding in report["findings"]}


def test_a_clean_dry_clip_passes(tmp_path: Path) -> None:
    report = audit_clip(_write(tmp_path / "clean.wav", _burst()))
    assert report["verdict"] == "pass", report["findings"]
    assert report["measured"]["continuous"] is False
    assert report["measured"]["decay_to_minus20db_s"] is not None


def test_continuous_sound_is_not_called_noisy(tmp_path: Path) -> None:
    """The false alarm that flagged air conditioning, blenders, busy signals."""

    report = audit_clip(_write(tmp_path / "fan.wav", _continuous()))
    assert report["measured"]["continuous"] is True
    assert "noise_floor" not in _names(report)
    assert "reverb_tail" not in _names(report)
    assert report["verdict"] == "pass", report["findings"]


def test_peak_normalised_clip_is_not_called_clipped(tmp_path: Path) -> None:
    """One sample touching full scale is normalisation, not distortion."""

    samples = _burst()
    samples[100] = 1.0
    report = audit_clip(_write(tmp_path / "normalised.wav", samples))
    assert "clipping" not in _names(report)


def test_real_clipping_fails(tmp_path: Path) -> None:
    time = np.arange(16000 * 2) / 16000
    hammered = np.clip(3.0 * np.sin(2 * np.pi * 300 * time), -1.0, 1.0)
    report = audit_clip(_write(tmp_path / "clipped.wav", hammered))
    assert report["verdict"] == "fail"
    assert "clipping" in _names(report)


def test_dc_offset_fails(tmp_path: Path) -> None:
    report = audit_clip(_write(tmp_path / "dc.wav", _burst() * 0.4 + 0.36))
    assert report["verdict"] == "fail"
    assert "dc_offset" in _names(report)


def test_silence_and_unreadable_are_reported_not_raised(tmp_path: Path) -> None:
    silent = audit_clip(_write(tmp_path / "silent.wav", np.zeros(16000)))
    assert silent["verdict"] == "fail" and "silent" in _names(silent)

    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF not really a wav")
    report = audit_clip(broken)
    assert report["verdict"] == "fail" and "unreadable" in _names(report)


def test_long_tail_on_an_impulsive_clip_asks_for_a_listen(tmp_path: Path) -> None:
    """A clip still ringing a second after its peak may have been recorded
    in a room - advisory, because a bell does this by itself."""

    report = audit_clip(
        _write(tmp_path / "reverb.wav", _burst(seconds=4.0, decay=1.2))
    )
    assert "reverb_tail" in _names(report)
    assert report["verdict"] == "warn"


def test_write_clip_qc_lands_beside_the_clip(tmp_path: Path) -> None:
    clip = _write(tmp_path / "clip.wav", _burst())
    write_clip_qc(clip)
    assert (tmp_path / "clip.qc.json").is_file()
