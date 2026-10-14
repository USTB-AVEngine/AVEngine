"""Endpoint-only frequency-dependent coherence for moving native RIRs.

Broadband coherence overestimates the coherence of high-frequency reflections,
attenuating their crossfade. The original sqrt(2) weight cap, early linear path,
arrival alignment and all external acceptance thresholds remain unchanged.
No independent reference, fitted gain or target normalization enters this law.
"""
import numpy as np
from scipy.fft import rfft,irfft,next_fast_len,rfftfreq
from scipy.ndimage import uniform_filter1d
from avengine.acoustics.moving_rir_calibration import CorrelatedRIRInterval,CorrelatedRIRTimeline

REVISION='endpoint-only-octave-coherence-emission-aligned-v1'

class FrequencyCoherentInterval(CorrelatedRIRInterval):
    def sample(self,u):
        standard=super().sample(u)
        if self.stationary or u==0 or u==1:return standard
        shifted,_,late=self._state
        if getattr(self,'band_target',None)!=self._target:
            length=shifted[0].shape[1];nfft=next_fast_len(2*length);freq=rfftfreq(nfft,1/self.sample_rate)
            centers=np.array([125.,250.,500.,1000.,2000.,4000.,8000.])
            x=np.clip(np.log2(np.maximum(freq,125.)/125.),0,len(centers)-1)
            low=np.minimum(np.floor(x).astype(int),len(centers)-2);fraction=x-low
            masks=np.zeros((len(centers),len(freq)));idx=np.arange(len(freq))
            masks[low,idx]=np.cos(fraction*np.pi/2)**2;masks[low+1,idx]=np.sin(fraction*np.pi/2)**2
            a,b=[irfft(rfft(w,nfft,axis=1)[None,:,:]*masks[:,None,:],nfft,axis=2)[:,:,:length] for w in shifted]
            window=max(1,round(.02*self.sample_rate));pa=uniform_filter1d(a[:,0]**2,window,axis=1,mode='constant');pb=uniform_filter1d(b[:,0]**2,window,axis=1,mode='constant');cross=uniform_filter1d(a[:,0]*b[:,0],window,axis=1,mode='constant')
            denom=np.sqrt(np.maximum(pa,0)*np.maximum(pb,0));rho=np.divide(cross,denom,out=np.zeros_like(cross),where=denom>1e-30)
            self.band_state=a,b,np.clip(rho,0,1);self.band_target=self._target
        a,b,rho=self.band_state
        denominator=np.sqrt((1-u)**2+u*u+2*u*(1-u)*rho)
        late_kernel=np.sum(((1-u)*a+u*b)/denominator[:,None,:],axis=0)
        early=(1-u)*shifted[0]+u*shifted[1]
        return early*(1-late)[None,:]+late_kernel*late[None,:]

class FrequencyCoherentTimeline(CorrelatedRIRTimeline):
    def at(self,sample):
        if len(self.filters)==1 or sample<=self.keys[0]:return self.filters[0]
        if sample>=self.keys[-1]:return self.filters[-1]
        i=int(np.searchsorted(self.keys,sample,side='right'))-1
        if i!=self._interval_index:self._interval=FrequencyCoherentInterval(self.filters[i],self.filters[i+1],self.sample_rate);self._interval_index=i
        return self._interval.sample((sample-self.keys[i])/(self.keys[i+1]-self.keys[i]))


from avengine.acoustics.moving_rir_calibration import _virtual_keys,_align_first_peak

def frequency_correlated_convolve(dry, filters, key_samples, *, sample_rate=16000, block_samples=160):
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
    timeline = FrequencyCoherentTimeline(filters, key_samples, sample_rate)
    if timeline.keys[-1] >= len(x):
        raise ValueError('RIR key outside continuous dry world')
    if len(filters) == 1:
        return np.stack([signal.fftconvolve(x, h, mode='full') for h in timeline.filters[0]])
    arrivals = [first_arrival_samples(h[0])[1] for h in timeline.filters]
    maximum_tail = max(h.shape[1] for h in timeline.filters)+max(arrivals)-min(arrivals)
    output = np.zeros((4, len(x)+maximum_tail-1))
    keys = _virtual_keys(len(x), block_samples)
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
