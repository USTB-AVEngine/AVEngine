"""Regression for arbitrary STFT crop phase and sparse tick fingerprints."""
import numpy as np
import pytest
from scipy import signal
from avengine.qa.v7_sao_fingerprint import waveform_fingerprint, robust_alignment, possible_references
pytestmark = pytest.mark.fast_unit


def test_arbitrary_crop_and_gain_resampling_are_verified_as_included_waveform():
    rng = np.random.default_rng(2416)
    t = np.arange(80000) / 16000
    original = .08 * signal.chirp(t, f0=120, f1=2900, t1=5) + rng.normal(0, .03, len(t))
    reference = waveform_fingerprint(original, 16000)
    # 128/8000 is the old STFT hop; this starts between its frame positions.
    crop = original[17123:63321] * .137
    crop = signal.resample_poly(signal.resample_poly(crop, 3, 2), 2, 3)[:len(crop)]
    aligned = robust_alignment(waveform_fingerprint(crop, 16000), reference)
    assert aligned['matched'] and aligned['normalized_correlation'] > .999
    assert aligned['matched_seconds'] > 2


def test_same_spectrum_with_independent_phases_is_not_a_duplicate():
    rng = np.random.default_rng(1722)
    first = rng.normal(0, .03, 48000)
    spectrum = np.fft.rfft(first)
    second = np.fft.irfft(np.abs(spectrum) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(spectrum))), n=len(first))
    assert not robust_alignment(waveform_fingerprint(first, 16000), waveform_fingerprint(second, 16000))['matched']


def test_sparse_ticks_are_not_pruned_for_lacking_twelve_landmark_hashes():
    assert possible_references({'one': 0}, 4, {'one': 200, 'two': 3}) == {'one', 'two'}
    assert possible_references({'one': 4}, 200, {'one': 150, 'two': 3}) == {'one', 'two'}


def test_silence_does_not_produce_a_waveform_match():
    silent = waveform_fingerprint(np.zeros(32000), 16000)
    assert not robust_alignment(silent, silent)['matched']
