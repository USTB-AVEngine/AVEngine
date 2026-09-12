"""Contracts for the prepared-clip helpers the segment cutter builds on.

The existing suite for this module is ``test_assets_sound_prepare.py``; this
one covers what the 2026-09-10 segment-cache defect made necessary: an
identity that can bind a producer's own processing settings, and a writer that
actually refuses to replace a file instead of merely checking first.
"""

from __future__ import annotations

import hashlib
import threading
import wave

import numpy as np
import pytest

from avengine.assets import sound_prepare
from avengine.assets.sound_prepare import (
    PrepareError,
    as_mono_float,
    make_prepared_audio_id,
    wav_mono_bytes,
    write_wav_mono,
    write_wav_mono_no_clobber,
    zero_phase_filter,
)

# One realistic facts dict, and the id it produced before this module learned
# about processing settings.  Pinned so a later edit cannot quietly renumber
# every prepared speech clip already on disk.
LEGACY_FACTS = {
    "operation": "speech_highpass_with_band_detected_crop_v1",
    "source_crop_start_sample": 1234,
    "source_crop_end_sample_exclusive": 56789,
    "filter": {"highpass_hz": 80.0, "order": 4, "phase": "zero_phase"},
    "activity_filter": {
        "highpass_hz": 80.0, "bandpass_low_hz": 300.0,
        "bandpass_high_hz": 3400.0, "order": 4, "phase": "zero_phase",
    },
    "detector": {"window_s": 0.02, "hop_s": 0.01},
    "target_rate_hz": 16000,
}
LEGACY_SHA = "de" * 32
LEGACY_ID = "prepared_speech_band_4f289272b79b_v1"


def test_an_existing_prepared_id_is_unchanged():
    assert make_prepared_audio_id(
        "asset_x", source_sha256=LEGACY_SHA, facts=LEGACY_FACTS
    ) == LEGACY_ID


def test_no_processing_and_empty_processing_are_the_same_call():
    assert make_prepared_audio_id(
        "asset_x", source_sha256=LEGACY_SHA, facts=LEGACY_FACTS, processing={}
    ) == LEGACY_ID


def test_processing_settings_reach_the_identity():
    """The defect: two different fades resolved to one id and one file."""

    quiet = make_prepared_audio_id(
        "a", source_sha256=LEGACY_SHA, facts=LEGACY_FACTS,
        processing={"edge_fade_s": 0.0},
    )
    faded = make_prepared_audio_id(
        "a", source_sha256=LEGACY_SHA, facts=LEGACY_FACTS,
        processing={"edge_fade_s": 0.005},
    )
    assert quiet != faded
    assert quiet != LEGACY_ID


def test_a_peak_target_reaches_the_identity_even_though_the_gain_is_not_known_yet():
    """``normalization_applied`` alone never distinguished two peak targets."""

    facts = {**LEGACY_FACTS, "normalization_applied": True,
             "applied_gain_db": None}
    loud = make_prepared_audio_id("a", source_sha256=LEGACY_SHA, facts=facts,
                                  processing={"target_peak_dbfs": -3.0})
    soft = make_prepared_audio_id("a", source_sha256=LEGACY_SHA, facts=facts,
                                  processing={"target_peak_dbfs": -12.0})
    assert loud != soft
    # and without the processing block they would have been the same id
    assert make_prepared_audio_id(
        "a", source_sha256=LEGACY_SHA, facts=facts
    ) == make_prepared_audio_id("a", source_sha256=LEGACY_SHA, facts=facts)


def test_the_prefix_is_the_callers_to_choose():
    name = make_prepared_audio_id(
        "a", source_sha256=LEGACY_SHA, facts=LEGACY_FACTS,
        processing={"edge_fade_s": 0.0}, prefix="sound_segment",
    )
    assert name.startswith("sound_segment_")
    assert name.endswith("_v1")


def test_a_non_finite_setting_is_refused_rather_than_named_nan():
    with pytest.raises(PrepareError, match="not serialisable"):
        make_prepared_audio_id(
            "a", source_sha256=LEGACY_SHA, facts=LEGACY_FACTS,
            processing={"edge_fade_s": float("nan")},
        )


def test_the_private_helper_names_still_resolve():
    """Existing internal callers keep working; the public names are aliases."""

    assert sound_prepare._as_mono_float is as_mono_float
    assert sound_prepare._zero_phase_filter is zero_phase_filter
    assert sound_prepare._write_wav_mono is write_wav_mono
    assert sound_prepare._write_wav_no_clobber is write_wav_mono_no_clobber


def test_as_mono_float_and_zero_phase_filter_are_usable_as_public_names():
    stereo = np.stack([np.ones(64), np.zeros(64)], axis=1)
    assert as_mono_float(stereo).shape == (64,)
    t = np.arange(1600) / 16000.0
    filtered = zero_phase_filter(
        np.sin(2 * np.pi * 20.0 * t), 16000, kind="highpass",
        cutoff=300.0, order=4,
    )
    assert float(np.abs(filtered).max()) < 0.3


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def test_wav_bytes_and_file_agree(tmp_path):
    samples = np.sin(2 * np.pi * 440.0 * np.arange(1600) / 16000.0) * 0.5
    payload = wav_mono_bytes(samples, 16000)
    digest = write_wav_mono(tmp_path / "clip.wav", samples, 16000)
    assert (tmp_path / "clip.wav").read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    with wave.open(str(tmp_path / "clip.wav"), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 1600


def test_the_no_clobber_writer_refuses_and_leaves_the_original_alone(tmp_path):
    path = tmp_path / "clip.wav"
    first = np.full(800, 0.25)
    write_wav_mono_no_clobber(path, first, 16000)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_wav_mono_no_clobber(path, np.full(800, -0.5), 16000)
    assert path.read_bytes() == before


def test_only_one_of_many_concurrent_writers_creates_the_file(tmp_path):
    """Two workers preparing the same clip must not both open it for writing."""

    path = tmp_path / "shared" / "clip.wav"
    path.parent.mkdir(parents=True)
    samples = np.full(4000, 0.3)
    start = threading.Barrier(6)
    winners: list[str] = []
    losers: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        start.wait()
        try:
            digest = write_wav_mono_no_clobber(path, samples, 16000)
        except FileExistsError:
            with lock:
                losers.append("lost")
        else:
            with lock:
                winners.append(digest)

    threads = [threading.Thread(target=attempt) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(winners) == 1
    assert len(losers) == 5
    assert hashlib.sha256(path.read_bytes()).hexdigest() == winners[0]


def test_a_failed_write_removes_only_what_it_created(tmp_path):
    path = tmp_path / "clip.wav"
    with pytest.raises((TypeError, ValueError)):
        write_wav_mono_no_clobber(path, np.array(["a", "b"]), 16000)
    assert not path.exists()
