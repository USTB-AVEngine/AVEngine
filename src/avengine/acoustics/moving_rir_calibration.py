"""Explicit native-noise calibration; preserve the historical motion budgets."""
from __future__ import annotations
import numpy as np
from avengine.acoustics.moving_rir_acceptance import qualify_energy

ENERGY_FIELDS=('direct_energy_difference_db','total_energy_difference_db','drr_difference_db')


def errors(check):
    values={k:abs(float(check[k])) for k in (*ENERGY_FIELDS,'direct_delay_difference_ms')}
    values.update({f'octave_{b["center_hz"]}_energy_db':abs(float(b['energy_difference_db'])) for b in check['octave_energy']})
    return values


def calibrate(checks,config):
    selected=[c for c in checks if c['split']=='calibration']
    if any(c['status']!='measured' for c in selected):raise ValueError('invalid native baseline metric')
    grouped={}
    for c in selected:grouped.setdefault((c['house'],c['pose_index']),[]).append(errors(c))
    if len(grouped)<config['calibration_poses'] or any(len(v)<3 for v in grouped.values()):
        raise ValueError('need >=32 distinct baseline poses, >=3 independent repeat comparisons each')
    metrics={}
    for name in errors(selected[0]):
        worst=np.asarray([max(v[name] for v in group) for group in grouped.values()])
        rng=np.random.default_rng(config['bootstrap_seed'])
        draws=rng.choice(worst,size=(1000,len(worst)),replace=True)
        upper=float(np.percentile(np.percentile(draws,95,axis=1),95))
        margin=config['arrival_extra_margin_ms'] if name=='direct_delay_difference_ms' else config['energy_and_DRR_extra_margin_db']
        metrics[name]={'native_pose_worst_p95':float(np.percentile(worst,95)),
            'baseline_p95_upper':upper,'extra_margin':margin,'maximum_allowed_abs_error':upper+margin}
    return {'revision':config['revision'],'calibration_pose_count':len(grouped),'metrics':metrics,
            'target_waveforms_used':False,'fixed80_used_to_fit':False}


def qualify_calibrated(checks,budget):
    original=qualify_energy(checks)
    # Direction, validity, decay and tail criteria retain their original code.
    retained=[f for f in original['failures'] if f.rsplit(':',1)[-1] not in ('energy','DRR','arrival_delay')]
    failures=list(retained);rows=[]
    for index,c in enumerate(checks):
        if c['status']!='measured':continue
        values=errors(c);residuals={}
        for name,v in values.items():
            metric=budget['metrics'][name]
            residual=max(0.,v-metric['baseline_p95_upper']);residuals[name]=residual
            if residual>metric['extra_margin']+1e-12:failures.append(f'{index}:{name}')
        rows.append({'index':index,'noise_subtracted_absolute_errors':residuals})
    bad_indices={int(f.split(':',1)[0]) for f in failures if f.split(':',1)[0].isdigit()}
    return {'status':'fail' if failures else 'pass','sample_count':len(checks),'pass_count':len(checks)-len(bad_indices),
        'failures':failures,'historical_engineering_result':original,'calibrated_budget':budget,
        'unchanged_nonenergy_failures':retained,'residuals':rows}


CROSSFADE_REVISION = 'integer-arrival-local-correlation-native-key-emission-grid-v2'


class CorrelatedRIRInterval:
    """Endpoint-only weights; no midpoint fit or per-channel normalization.

    Quantized first arrivals are aligned. Coherent reflection fields receive
    linear weights; decorrelated fields receive constant-power weights. The
    shared scalar weight preserves the four-channel FOA relationship. Keep
    only the current delay state, independent of continuous-scene duration.
    """
    def __init__(self, a, b, sample_rate=16000):
        from avengine.acoustics.house_acceptance_metrics import first_arrival_samples
        a, b = np.asarray(a, float), np.asarray(b, float)
        if any(w.ndim != 2 or w.shape[0] != 4 or not w.shape[1] or not np.isfinite(w).all() for w in (a, b)):
            raise ValueError('finite four-channel native RIRs required')
        self.sample_rate = sample_rate
        self.length = max(a.shape[1], b.shape[1])
        self.a = np.pad(a, ((0, 0), (0, self.length-a.shape[1])))
        self.b = np.pad(b, ((0, 0), (0, self.length-b.shape[1])))
        self.stationary = np.array_equal(self.a, self.b)
        self.first = [first_arrival_samples(w[0])[1] for w in (self.a, self.b)]
        self._target = None
        self._state = None

    def sample(self, u):
        from scipy.ndimage import uniform_filter1d
        if not 0 <= u <= 1:
            raise ValueError('crossfade fraction outside [0,1]')
        if self.stationary:
            return self.a
        target = round((1-u)*self.first[0]+u*self.first[1])
        if target != self._target:
            extra = max(0, *(target-onset for onset in self.first))
            length = self.length+extra
            shifted = []
            for wave, onset in zip((self.a, self.b), self.first, strict=True):
                padded = np.pad(wave, ((0, 0), (0, extra)))
                delta = target-onset
                out = np.zeros_like(padded)
                if delta >= 0:
                    out[:, delta:] = padded[:, :length-delta] if delta else padded
                else:
                    out[:, :length+delta] = padded[:, -delta:]
                shifted.append(out)
            window = max(1, round(.020*self.sample_rate))
            pa = uniform_filter1d(np.square(shifted[0][0]), window, mode='constant')
            pb = uniform_filter1d(np.square(shifted[1][0]), window, mode='constant')
            cross = uniform_filter1d(shifted[0][0]*shifted[1][0], window, mode='constant')
            # Running-sum roundoff can make vanishing tail powers negative.
            denominator = np.sqrt(np.maximum(pa, 0.)*np.maximum(pb, 0.))
            rho = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 1e-30)
            # Cap mixing amplification at sqrt(2); do not boost a cancellation.
            rho = np.clip(rho, 0., 1.)
            x = np.clip((np.arange(length)/self.sample_rate-target/self.sample_rate-.0005)/.0015, 0, 1)
            late = .5-.5*np.cos(np.pi*x)
            self._state = shifted, rho, late
            self._target = target
        shifted, rho, late = self._state
        early = (1-u)*shifted[0]+u*shifted[1]
        weights = np.sqrt((1-u)**2+u**2+2*u*(1-u)*rho)
        return (1-late)[None, :]*early+late[None, :]*early/weights[None, :]


def integer_arrival_crossfade(a, b, u, sample_rate=16000):
    return CorrelatedRIRInterval(a, b, sample_rate).sample(u)


class CorrelatedRIRTimeline:
    def __init__(self, filters, key_samples, sample_rate=16000):
        self.filters = [np.asarray(h, float) for h in filters]
        self.keys = np.asarray(key_samples, dtype=np.int64)
        if not len(self.filters) or len(self.keys) != len(self.filters) or self.keys[0] != 0 or np.any(np.diff(self.keys) <= 0):
            raise ValueError('native RIR key samples must increase from zero')
        if any(h.ndim != 2 or h.shape[0] != 4 or not h.shape[1] or not np.isfinite(h).all() for h in self.filters):
            raise ValueError('native RIRs must be finite four-channel arrays')
        from avengine.acoustics.house_acceptance_metrics import first_arrival_samples
        self.arrivals = [first_arrival_samples(h[0])[1] for h in self.filters]
        self.sample_rate = sample_rate
        self._interval_index = None
        self._interval = None

    def delays_at(self, samples):
        return np.rint(np.interp(samples, self.keys, self.arrivals)).astype(np.int64)

    def at(self, sample):
        if len(self.filters) == 1 or sample <= self.keys[0]:
            return self.filters[0]
        if sample >= self.keys[-1]:
            return self.filters[-1]
        i = int(np.searchsorted(self.keys, sample, side='right'))-1
        if i != self._interval_index:
            self._interval = CorrelatedRIRInterval(self.filters[i], self.filters[i+1], self.sample_rate)
            self._interval_index = i
        u = (sample-self.keys[i])/(self.keys[i+1]-self.keys[i])
        return self._interval.sample(u)


def _virtual_keys(length, block_samples, native_keys=()):
    if length < 1 or block_samples < 1:
        raise ValueError('positive dry length and convolution step required')
    keys = list(range(0, length, block_samples))
    if keys[-1] != length-1:
        keys.append(length-1)
    return sorted(set(keys)|{int(k)for k in native_keys if 0<=int(k)<length})


def _align_first_peak(kernel, target):
    from avengine.acoustics.house_acceptance_metrics import first_arrival_samples
    first=first_arrival_samples(kernel[0])[1]
    delta=int(target)-first
    if delta==0:return kernel
    if delta>0:return np.pad(kernel,((0,0),(delta,0)))
    return kernel[:,-delta:]


def effective_emission_filter(timeline, sample, dry_length, block_samples=160):
    """The exact source-time impulse response of correlated_convolve below."""
    keys = _virtual_keys(dry_length, block_samples,timeline.keys)
    if not 0 <= sample < dry_length:
        raise ValueError('emission outside continuous dry world')
    if len(keys) == 1 or sample == keys[-1]:
        return timeline.at(sample)
    i = max(0, int(np.searchsorted(keys, sample, side='right'))-1)
    lo, hi = keys[i:i+2]
    u = (sample-lo)/(hi-lo)
    target=int(timeline.delays_at(sample))
    a, b = [_align_first_peak(timeline.at(k),target) for k in (lo,hi)]
    length = max(a.shape[1], b.shape[1])
    a = np.pad(a, ((0, 0), (0, length-a.shape[1])))
    b = np.pad(b, ((0, 0), (0, length-b.shape[1])))
    return (1-u)*a+u*b


def correlated_convolve(dry, filters, key_samples, *, sample_rate=16000, block_samples=160):
    """Stream full-tail source-time convolution with 10 ms virtual kernels.

    Native RIR count stays unchanged. Adjacent triangular dry partitions sum
    to one, preserving emissions before a crop and reverberation after stops.
    Virtual kernels are computed then released; no duration-by-RIR allocation.
    Every dry partition is split by its emission-time quantized propagation
    delay, so adjacent virtual kernels share a direct arrival before addition.
    """
    from scipy import signal
    from avengine.acoustics.house_acceptance_metrics import first_arrival_samples
    x = np.asarray(dry, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite mono dry bus required')
    timeline = CorrelatedRIRTimeline(filters, key_samples, sample_rate)
    if timeline.keys[-1] >= len(x):
        raise ValueError('RIR key outside continuous dry world')
    if len(filters) == 1:
        return np.stack([signal.fftconvolve(x, h, mode='full') for h in timeline.filters[0]])
    arrivals = [first_arrival_samples(h[0])[1] for h in timeline.filters]
    maximum_tail = max(h.shape[1] for h in timeline.filters)+max(arrivals)-min(arrivals)
    output = np.zeros((4, len(x)+maximum_tail-1))
    keys = _virtual_keys(len(x), block_samples,timeline.keys)
    for i, center in enumerate(keys):
        lo = keys[i-1] if i else 0
        hi = keys[i+1]+1 if i+1 < len(keys) else len(x)
        indices = np.arange(lo, hi)
        weight = np.ones(hi-lo)
        if i:
            mask = indices <= center
            weight[mask] = (indices[mask]-lo)/(center-lo)
        if i+1 < len(keys):
            mask = indices >= center
            weight[mask] = (hi-1-indices[mask])/(hi-1-center)
        weighted = x[lo:hi]*weight
        nonzero = np.flatnonzero(weighted != 0)
        if not len(nonzero):
            continue
        begin, end = int(nonzero[0]), int(nonzero[-1])+1
        kernel = timeline.at(center)
        delays=timeline.delays_at(indices)
        # Split the weighted dry support by its actual emission-time delay.
        # Both neighboring virtual kernels then share the same first arrival
        # for every source sample. No arrival is linearly double-pulsed.
        for target in np.unique(delays[nonzero]):
            grouped=weighted*(delays==target)
            support=np.flatnonzero(grouped!=0)
            begin,end=int(support[0]),int(support[-1])+1
            aligned=_align_first_peak(kernel,int(target))
            for channel in range(4):
                wet=signal.fftconvolve(grouped[begin:end],aligned[channel],mode='full')
                stop=lo+begin+len(wet)
                if stop>output.shape[1]:output=np.pad(output,((0,0),(0,stop-output.shape[1])))
                output[channel,lo+begin:stop]+=wet
    return output
