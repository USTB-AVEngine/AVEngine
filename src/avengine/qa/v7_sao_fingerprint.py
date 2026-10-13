"""Verify spectral fingerprint candidates with gain-invariant aligned waveforms.

An arbitrary crop shifts the STFT grid and can destroy exact landmark hashes.
Normalized cross-correlation verifies an included waveform independently of
that grid; it never treats a failed positive control as an audio duplicate.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import signal
from avengine.qa.v7_auto_admission import audio_fingerprint, fingerprint_alignment

MIN_CORRELATION = .995
MIN_MATCH_SECONDS = 1.0


def waveform_fingerprint(wave, rate):
    base = audio_fingerprint(wave, rate)
    divisor = math.gcd(int(rate), 8000)
    x = signal.resample_poly(np.asarray(wave, dtype=np.float64), 8000 // divisor, int(rate) // divisor)
    peak = float(np.max(np.abs(x)))
    base['waveform_8k'] = (x / peak if peak > 1e-12 else x).astype(np.float32)
    if int(rate) == 16000 and len(wave) > 1:
        shifted = signal.resample_poly(np.asarray(wave, dtype=np.float64)[1:], 1, 2)
        shifted_peak = float(np.max(np.abs(shifted)))
        base['waveform_8k_shifted'] = (shifted / shifted_peak if shifted_peak > 1e-12 else shifted).astype(np.float32)
    return base


def _waveform_alignment(query, reference):
    q = np.asarray(query['waveform_8k'], dtype=np.float64)
    r = np.asarray(reference['waveform_8k'], dtype=np.float64)
    reverse = len(q) > len(r)
    short, long = (r, q) if reverse else (q, r)
    count = len(short)
    result = dict(matched=False, method='full_shorter_waveform_normalized_cross_correlation',
                  normalized_correlation=0., matched_seconds=count / 8000,
                  offset_samples_8k=None, shorter_is_reference=reverse,
                  minimum_correlation=MIN_CORRELATION, minimum_match_seconds=MIN_MATCH_SECONDS)
    if count < round(MIN_MATCH_SECONDS * 8000):
        return result
    centered = short - short.mean()
    energy = float(np.dot(centered, centered))
    if energy < 1e-12:
        return result
    # Query is centered. Center each aligned reference window in the denominator.
    sums = np.concatenate(([0.], np.cumsum(long)))
    squares = np.concatenate(([0.], np.cumsum(long * long)))
    local_sum = sums[count:] - sums[:-count]
    local_energy = np.maximum(0., squares[count:] - squares[:-count] - local_sum * local_sum / count)
    correlation = signal.correlate(long, centered, mode='valid', method='fft')
    denom = np.sqrt(local_energy * energy)
    valid = denom > 1e-12
    values = np.zeros_like(correlation)
    values[valid] = np.minimum(1., np.abs(correlation[valid]) / denom[valid])
    winner = int(np.argmax(values))
    value = float(values[winner])
    result.update(normalized_correlation=value, offset_samples_8k=winner,
                  relative_rms_error=math.sqrt(max(0., 1. - value * value)),
                  matched=value >= MIN_CORRELATION)
    return result


def spectral_duplicate(alignment):
    return bool(alignment['exact_normalized_pcm'] or
                (alignment['matched_hashes'] >= 12 and alignment['aligned_fraction'] >= .25))


def robust_alignment(query, reference):
    if query['pcm_sha256'] and query['pcm_sha256'] == reference['pcm_sha256']:
        return dict(matched=True, method='exact_normalized_pcm', exact_normalized_pcm=True)
    best = _waveform_alignment(query, reference)
    best['longer_waveform_phase_16k_samples'] = 0
    if best['matched']:
        return best
    # At 16k, an odd crop start is a half-sample displacement after 8k
    # downsampling. Verify the second original-sample phase as well.
    longer = query if len(query['waveform_8k']) > len(reference['waveform_8k']) else reference
    shifted = longer.get('waveform_8k_shifted')
    if shifted is not None:
        other = {**longer, 'waveform_8k': shifted}
        alternative = _waveform_alignment(other, reference) if longer is query else _waveform_alignment(query, other)
        alternative['longer_waveform_phase_16k_samples'] = 1
        if alternative['normalized_correlation'] > best['normalized_correlation']:
            best = alternative
    return best


def possible_references(counts, query_hash_count, reference_hash_counts):
    # A strict landmark-fraction prefilter loses arbitrary crops. A weak shared
    # hash screen is safe because the subsequent waveform test is stringent.
    if query_hash_count < 12:
        return set(reference_hash_counts)
    selected = {key for key, count in counts.items() if count >= 4}
    selected.update(key for key, count in reference_hash_counts.items() if count < 12)
    return selected
