"""Owner-authorized dry tail muting; evidence measures the processed waveform."""
import numpy as np

REASON='owner 10-15 allows muting long tails'
SPEECH_CLASSES={'speech_playback','human_speech_male','human_speech_female'}


def measure_processed_activity(samples,sample_rate_hz=16000):
    samples=np.asarray(samples,float)
    if samples.ndim!=1 or not np.isfinite(samples).all():raise ValueError('processed sound must be finite mono')
    active=samples!=0.
    changes=np.diff(np.r_[False,active,False].astype(np.int8))
    intervals=[[int(lo),int(hi)]for lo,hi in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
    return {'sample_rate_hz':int(sample_rate_hz),'sample_count':len(samples),
        'activity_intervals_samples':intervals,'active_sample_count':int(active.sum()),
        'method':'exact nonzero runs of actual processed waveform; original activity labels not reused',
        'wet_tail_status':'pending actual native rendered waveform measurement'}


def tail_muted_segment(clip,profile,*,output_sample_count=None,sample_rate_hz=16000):
    """Render an admitted crop, fade its final 60ms, and zero all later samples."""
    if profile.get('tail_muted')is not True or sample_rate_hz!=16000:raise ValueError('explicit 16kHz tail-muteful program required')
    count=int(profile['sample_count']);stop=int(profile.get('mute_after_sample',count))
    total=count if output_sample_count is None else int(output_sample_count)
    samples=np.asarray(clip,float)
    if not 0<stop<=count<=38400 or total<count or samples.ndim!=1 or len(samples)<count or not np.isfinite(samples[:count]).all():
        raise ValueError('tail mute crop must fit the actual source and <=2.4s motion program')
    fade=int(profile.get('fade_out_samples',960))
    if fade<=0 or fade>stop:raise ValueError('tail fade must fit the retained source')
    segment=np.zeros(total);segment[:stop]=samples[:stop]
    onset=min(80,stop//2)
    if onset:segment[:onset]*=np.linspace(0.,1.,onset)
    segment[stop-fade:stop]*=np.linspace(1.,0.,fade)
    return segment


def read_tail_muted_program(record,profile):
    import soundfile as sf
    if record.get('source_type')=='author_tts' or record.get('sound_class')in SPEECH_CLASSES:
        raise ValueError('human utterances must remain whole; tail mute is not a speech crop')
    path=record.get('normalized_path') or record.get('path')
    if not path:raise ValueError('tail muting needs an actual admitted waveform path')
    samples,rate=sf.read(path,dtype='float64',always_2d=True)
    begin=int(profile.get('crop_start_sample',0));count=int(profile['sample_count'])
    if rate!=16000 or samples.shape[1]!=1 or begin<0 or begin+count>len(samples):
        raise ValueError('tail mute crop differs from the actual 16kHz mono recording')
    rendered=tail_muted_segment(samples[begin:,0],profile,sample_rate_hz=rate)
    return rendered,measure_processed_activity(rendered,rate)


def select_muted_program(records,allowed_classes,minimum_activity_s,maximum_program_s=2.4):
    """Fallback only: a shorter whole call remains the preferred source."""
    from .v7_motion_source_roles import activity
    options=[]
    maximum_samples=int(min(2.4,float(maximum_program_s))*16000)//1600*1600
    for original in records:
        if original.get('sound_class')not in allowed_classes or original.get('source_type')=='author_tts' or original.get('sound_class')in SPEECH_CLASSES:continue
        header=original.get('header',{});rate=int(header.get('sample_rate_hz',16000));n=int(header.get('sample_count',round(float(original.get('duration_s',0))*rate)))
        if rate!=16000:continue
        for lo,hi in activity(original):
            begin=max(0,int(lo)-1600);count=min(maximum_samples,(n-begin)//1600*1600)
            if count<960:continue
            profile={'crop_start_sample':begin,'sample_count':count,'tail_muted':True,
                'fade_out_samples':960,'mute_after_sample':count,'reason':REASON,
                'source_activity_start_sample':int(lo)}
            try:_,measured=read_tail_muted_program(original,profile)
            except (ValueError,OSError,RuntimeError):continue
            intervals=measured['activity_intervals_samples']
            if not intervals:continue
            span=(intervals[-1][1]-intervals[0][0])/16000.
            if span<float(minimum_activity_s):continue
            profile['processed_activity']=measured
            options.append((span,original['sound_asset_id'],original,profile))
    if not options:return None
    options.sort(key=lambda x:(-x[0],x[1]))
    return options[0]
