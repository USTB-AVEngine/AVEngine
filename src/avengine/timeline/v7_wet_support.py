"""Certify conservative wet support from the actual dry samples and native FIRs.

This is finite convolution support, not an audibility threshold or a guessed
RT60. An upper bound may defer a question; it can never certify early silence.
No source waveform, RIR, audio rendering or answer predicate is changed.
"""
from __future__ import annotations
import numpy as np

WET_SUPPORT_REVISION = 'native-dry-FIR-finite-support-20261015-v2-tailcap0.5s'


def event_wet_support(dry, bindings, filters, *, delay_spread_samples=0, sample_rate=16000):
    x = np.asarray(dry)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('wet support needs actual finite mono dry samples')
    delay = int(delay_spread_samples)
    if delay < 0 or delay != delay_spread_samples:
        raise ValueError('invalid measured propagation delay spread')
    firsts = []; lasts = []
    for kernel in filters:
        h = np.asarray(kernel)
        if h.ndim != 2 or h.shape[0] != 4 or not np.isfinite(h).all():
            raise ValueError('wet support needs actual finite four-channel native FIRs')
        nz = np.flatnonzero(np.any(h != 0, axis=0))
        if len(nz):
            firsts.append(int(nz[0])); lasts.append(int(nz[-1]))
    if not firsts:
        raise ValueError('native wet support has no nonzero kernel')
    # Endpoint events must be disjoint to isolate their actual assembled samples.
    # Overlapping events are deliberately left without per-event certification.
    spans = [(int(b['output_start_sample']), int(b['output_end_sample_exclusive'])) for b in bindings]
    result = {}
    for index, (binding, (lo, hi)) in enumerate(zip(bindings, spans)):
        if not 0 <= lo < hi <= len(x):
            raise ValueError('event support escaped the actual dry clock')
        if any(index != j and lo < stop and start < hi for j, (start, stop) in enumerate(spans)):
            continue
        nz = np.flatnonzero(x[lo:hi] != 0)
        if not len(nz):
            continue
        lower = max(0, lo + int(nz[0]) + min(firsts) - delay)
        upper = lo + int(nz[-1]) + max(lasts) + delay + 1
        result[binding['event_id']] = {
            'wet_intervals_global_s': [[lower / sample_rate, upper / sample_rate]],
            'support_samples_global': [lower, upper],
            'source_nonzero_samples_global': [lo + int(nz[0]), lo + int(nz[-1]) + 1],
            'native_FIR_support_samples': [min(firsts), max(lasts) + 1],
            'propagation_delay_spread_samples': delay,
            'method': WET_SUPPORT_REVISION,
            'interval_kind': 'conservative_finite_convolution_support',
            'silence_threshold_added': False,
        }
    return result


WET_TAIL_CAP_S = .5     # owner 10-15: "拖尾，我不是说如果太长就强行静音截断吗？"
WET_TAIL_FADE_S = .1


def cap_wet_tail(stem, dry, *, cap_s=WET_TAIL_CAP_S, fade_s=WET_TAIL_FADE_S, sample_rate=16000, block=160):
    """Mute a rendered wet stem beyond `cap_s` after each dry emission ends, with a raised-cosine fade.

    The room reverberation of each sound lasts at most cap_s after the sound stops, so the post-sound question
    windows planned with a 0.5 s tail are exactly what is rendered.  Returns (capped_stem, emission_ends_samples).
    """
    stem = np.asarray(stem, float); x = np.asarray(dry)
    flags = [bool(np.any(x[i:i + block] != 0)) for i in range(0, len(x), block)]
    runs = []; start = None
    for i, value in enumerate(flags + [False]):
        if value and start is None: start = i
        if not value and start is not None:
            lo = start * block; nz = np.flatnonzero(x[lo:min(len(x), i * block)] != 0)
            runs.append((lo + int(nz[0]), lo + int(nz[-1]) + 1)); start = None   # exact first/last sounding sample
    n = stem.shape[-1]; cap = int(round(cap_s * sample_rate)); fade = int(round(fade_s * sample_rate))
    envelope = np.zeros(n); ends = [hi for _, hi in runs]
    for lo, hi in runs:
        keep = min(n, hi + cap - fade)
        envelope[max(0, lo):keep] = 1.
        stop = min(n, hi + cap)
        if stop > keep:
            t = (np.arange(keep, stop) - keep) / max(1, fade)
            envelope[keep:stop] = np.maximum(envelope[keep:stop], .5 * (1 + np.cos(np.pi * t)))
    return stem * envelope, ends


def cap_wet_support(support, *, cap_s=WET_TAIL_CAP_S, sample_rate=16000):
    """Wet-support certificates after the tail cap: support ends at most cap_s after the source's last sample."""
    out = {}
    for event, record in support.items():
        record = dict(record)
        lower, upper = record['support_samples_global']; source_end = record['source_nonzero_samples_global'][1]
        upper = min(int(upper), int(source_end) + int(round(cap_s * sample_rate)))
        record.update(support_samples_global=[int(lower), upper], wet_intervals_global_s=[[lower / sample_rate, upper / sample_rate]],
                      wet_tail_cap_s=cap_s, wet_tail_policy='owner 10-15: reverberation muted after the cap with a fade')
        out[event] = record
    return out
