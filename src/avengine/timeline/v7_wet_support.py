"""Certify conservative wet support from the actual dry samples and native FIRs.

This is finite convolution support, not an audibility threshold or a guessed
RT60. An upper bound may defer a question; it can never certify early silence.
No source waveform, RIR, audio rendering or answer predicate is changed.
"""
from __future__ import annotations
import numpy as np

WET_SUPPORT_REVISION = 'native-dry-FIR-finite-support-20261015-v1'


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
