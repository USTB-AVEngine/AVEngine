"""CPU media-only boundary for frozen panel model requests.

This module never opens scoring_private or construction metadata. References,
transport IDs and evidence keys are stripped before constructing model payloads.
No model is loaded here. Model-produced evidence must have matching provenance.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

FIELDS={'sample_id','question','qa_id','form','split','video','foa','binaural','mono','frame_indices','frame_times_s'}
PROFILES={'raw_av','av_mono','level1','level2'}

def digest(value):return hashlib.sha256(value).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def pcm(path):
    wave,rate=sf.read(path,dtype='float32',always_2d=True)
    if rate!=16000 or not np.isfinite(wave).all():raise ValueError('invalid complete 16k PCM')
    return wave.astype('<f4'),rate
def audio_identity(path):
    a,rate=pcm(path)
    return {'pcm_sha256':digest(a.tobytes()),'sample_rate_hz':rate,'sample_count':len(a),'channel_count':a.shape[1]}
def video_pixels(path):
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames','-show_entries','stream=width,height,avg_frame_rate,duration:frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(path)],text=True))
    stream=meta['streams'][0];width,height=stream['width'],stream['height']
    raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'])
    pixels=np.frombuffer(raw,np.uint8).reshape(-1,height,width,3)
    pts=[float(x['best_effort_timestamp_time']) for x in meta['frames']]
    if len(pts)!=len(pixels) or len(pts)<40:raise ValueError('video clock/frame mismatch')
    return pixels,pts,float(stream['duration'])
def video_identity(path):
    a,pts,duration=video_pixels(path)
    hashes=[digest(f.tobytes()) for f in a]
    return {'frame_sha256':hashes,'decoded_sha256':digest(canonical(hashes)),
       'frame_count':len(a),'resolution_hw':list(a.shape[1:3]),'pts_seconds':pts,'duration_seconds':duration}
def media_path(public,ref):
    p=Path(public).resolve()/str(ref)
    if not p.resolve().is_relative_to(Path(public).resolve()) or not p.is_file():raise ValueError('media root escape/missing media')
    return p
def check_request(row):
    if set(row)!=FIELDS or row['split'] not in {'pilot','test'} or row['qa_id']!='QA-27' or row['form']!='open':raise ValueError('request whitelist or frozen protocol mismatch')
    if len(row['frame_indices'])!=40 or len(row['frame_times_s'])!=40:raise ValueError('40-frame receipt required')
    if not isinstance(row['question'],str) or not row['question']:raise ValueError('missing public question')
def evidence_keys(public,row,producer,settings):
    check_request(row)
    audio={k:audio_identity(media_path(public,row[k]))['pcm_sha256'] for k in ('foa','binaural','mono')}
    video=video_identity(media_path(public,row['video']))['decoded_sha256']
    common={'producer':str(producer),'settings':settings}
    return {'M1':digest(canonical({**common,'decoded_audio_pcm':audio})),
            'M2':digest(canonical({**common,'video_pixels':video,'frame_indices':row['frame_indices']})),
            'cross_modal':digest(canonical({**common,'decoded_audio_pcm':audio,'video_pixels':video,'frame_indices':row['frame_indices']}))}
def build_model_payload(public,row,profile,*,evidence=None,producer=None,settings=None):
    if profile not in PROFILES:raise ValueError('unknown input profile')
    check_request(row)
    frames,pts,duration=video_pixels(media_path(public,row['video']))
    indices=row['frame_indices']
    if len(set(indices))!=40 or max(indices)>=len(frames) or min(indices)<0:raise ValueError('invalid frame selection')
    if not np.allclose([pts[i] for i in indices],row['frame_times_s'],atol=1e-6,rtol=0):raise ValueError('selected frame timestamps differ')
    mono,rate=pcm(media_path(public,row['mono']))
    binaural,_=pcm(media_path(public,row['binaural']))
    foa,_=pcm(media_path(public,row['foa']))
    if foa.shape!=(160000,4) or binaural.shape!=(160000,2) or mono.shape!=(160000,1):raise ValueError('full 10s layouts required')
    if not np.array_equal(mono[:,0],binaural.mean(axis=1)):raise ValueError('mono conversion mismatch')
    payload={'text':row['question'],'frames':frames[indices].copy(),'audio':mono[:,0].copy(),
       'frame_times_s':list(row['frame_times_s']),'fps':40/duration}
    if profile=='level2':payload['foa']=foa
    if profile=='level1':
        if evidence is None or producer is None or settings is None:raise ValueError('level1 requires media-derived model evidence; no oracle fallback')
        keys=evidence_keys(public,row,producer,settings)
        if evidence['keys']!=keys:raise ValueError('evidence provenance mismatch')
        if set(evidence)-{'keys','heard','seen','cross_modal'}:raise ValueError('unrecognized evidence fields')
        payload['evidence_text']={k:evidence[k] for k in ['heard','seen','cross_modal'] if k in evidence}
    # No filenames, group/member IDs, cache keys, truth fields or scoring metadata.
    return payload
