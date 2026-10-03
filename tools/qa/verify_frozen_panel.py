#!/usr/bin/env python3
"""Independent CPU verifier. Does not import the producer/certifier or open answers."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from shapely.geometry import shape, Point

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from avengine.qa.frozen_panel_inputs import build_model_payload, evidence_keys, FIELDS
from avengine.rooms.qa_evidence import build_pixel_appearance_review

def require(value,reason):
    if not value:raise ValueError(reason)
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def pcm_identity(path):
    a,rate=sf.read(path,dtype='float32',always_2d=True)
    require(rate==16000 and np.isfinite(a).all(),'PCM invalid')
    return {'pcm_sha256':sha(a.astype('<f4').tobytes()),'sample_rate_hz':rate,'sample_count':a.shape[0],'channel_count':a.shape[1]}
def video_identity(path):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames',
         '-show_entries','stream=width,height,duration:frame=best_effort_timestamp_time','-of','json',str(path)],text=True))
    s=info['streams'][0];shape=(s['height'],s['width'],3);size=int(np.prod(shape));hashes=[]
    p=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],stdout=subprocess.PIPE)
    while frame:=p.stdout.read(size):
        require(len(frame)==size,'partial decoded video frame');hashes.append(sha(frame))
    require(p.wait()==0,'video decode failed')
    pts=[float(f['best_effort_timestamp_time']) for f in info['frames']]
    require(len(hashes)==len(pts),'decoded timestamps mismatch')
    return {'frame_sha256':hashes,'decoded_sha256':sha(canonical(hashes)),'frame_count':len(hashes),
      'resolution_hw':list(shape[:2]),'pts_seconds':pts,'duration_seconds':float(s['duration'])}
def normalized(plan):
    p=copy.deepcopy(plan);p.pop('episode_id')
    for actor in p['visual_plan']['actors']:
        for k in ('asset_id','asset_revision','realized_attributes','display_label'):actor.pop(k,None)
        actor['emitter_binding'].pop('asset_id',None)
    return p
def media_path(public,reference):
    p=(public/reference).resolve();require(p.is_relative_to(public.resolve()) and p.is_file(),'public path escaped root');return p

def verify_audio_construction(cert):
    recipe=read(cert['provenance']['recipe']);g=Path(cert['provenance']['recipe']).parent
    ar=cert['construction']['audio_receipt'];dry=[]
    for row in ar['dry_audio']:
        require(filehash(row['path'])==row['sha256'],'dry speech changed')
        a,sr=sf.read(row['path'],dtype='float32');require(sr==16000 and a.ndim==1,'dry speech invalid');dry.append(a)
    foa=[np.load(g/'audio'/f'position_{i}.npy',allow_pickle=False) for i in range(2)]
    binaural=[np.load(g/'audio'/f'binaural_position_{i}.npy',allow_pickle=False) for i in range(2)]
    basis=read(Path(cert['provenance']['capture_roots'][0])/'neutral_readback.json')['camera'][0]['basis'];f=np.asarray(basis['forward'],float).copy();f[1]=0;f/=np.linalg.norm(f)
    up=np.array([0.,1.,0.]);right=np.cross(f,up)
    rotation=np.stack((f,-right,up));n=160000;gain=ar['gain_shared_all_columns']
    for ai in range(2):
        for kind,filters,channels in (('foa',foa,4),('binaural',binaural,2)):
            mix=np.zeros((channels,n),float)
            for di,wave in enumerate(dry):
                position=di if ai==0 else 1-di
                convolution=np.stack([fftconvolve(wave,ch) for ch in filters[position]])
                lo=round(recipe['start_seconds'][di]*16000);hi=min(n,lo+convolution.shape[1]);mix[:,lo:hi]+=convolution[:,:hi-lo]
            mix*=gain
            if kind=='foa':
                xyz=rotation@(mix[[3,1,2]]/np.sqrt(3));mix=np.stack((mix[0],xyz[1],xyz[2],xyz[0]))
            observed,_=sf.read(g/'audio'/f'a{ai}_{kind}.wav',dtype='float32',always_2d=True)
            require(np.array_equal(observed,mix.T.astype('<f4')),'full audio is not declared physical routing/convolution')
    require(filehash(ar['hrtf']['path'])==ar['hrtf']['sha256'],'binaural HRTF changed')

def verify_one(root,cert_path):
    started=time.monotonic();cpu=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    cert=read(cert_path);public=root/'public';members=cert['media']['members']
    require(len(members)==4,'quartet must have four inputs')
    require({(m['video_index'],m['audio_index']) for m in members}=={(0,0),(0,1),(1,0),(1,1)},'not a Cartesian quartet')
    require(len({m['sample_id'] for m in members})==4,'transport IDs collide')
    construction=cert['construction'];plans=[read(p) for p in construction['plan_paths']]
    verify_audio_construction(cert)
    config=read(root/'inputs/config.json') if (root/'inputs/config.json').exists() else read(root/'inputs/shard_config.json')
    house=next(h for h in config['houses'] if h['house']==cert['house'])
    require(cert['room'] in {r['room_label'] for r in house['rooms']},'not a checked room')
    source=read(house['acceptance_result'])['input'];room=next(r for r in source['acoustic_rooms'] if r['room_label']==cert['room'])
    floor=shape(room['floor_polygon']);geometry=construction['geometry']
    require(all(floor.covers(Point(q[0],q[2])) for q in [geometry['camera_m'],*geometry['feet_m']]),'pose leaves admitted room floor')
    recipe=read(cert['provenance']['recipe'])
    presets=[x['voice_preset'] for x in recipe['speech']]
    require((presets[0]==presets[1])==(cert['quota']['voice']=='same'),'voice quota disagrees with admitted clips')
    require(recipe['voice']==cert['quota']['voice'] and recipe['side']==cert['quota']['side'],'quota metadata differs')
    bearings=geometry['bearings_deg'];gap=abs(bearings[0]-bearings[1])
    if cert['quota']['side']=='same_side':require(bearings[0]*bearings[1]>0 and 12<=gap<=34,'same-side quota invalid')
    else:require(cert['quota']['side']=='opposite' and bearings[0]*bearings[1]<0 and gap>=18,'opposite-side quota invalid')
    fixed=[normalized(p) for p in plans]
    require(fixed[0]==fixed[1],'unallowed construction change')
    require(sha(canonical(fixed[0]))==construction['fixed_plan_sha256'],'fixed-field commitment differs')
    for vi in range(2):
        own=plans[vi]['visual_plan']['actors'];other=plans[1-vi]['visual_plan']['actors']
        require(own[0]['asset_id']==other[1]['asset_id'] and own[1]['asset_id']==other[0]['asset_id'],'not clothing exchange')
    captures=[Path(p) for p in cert['provenance']['capture_roots']]
    registry=read(config['registry']);registry=registry['assets'] if isinstance(registry,dict) else registry
    for vi in range(2):
        appearance=build_pixel_appearance_review(captures[vi],plans[vi],asset_registry={a['asset_id']:a for a in registry})
        expected={'source1':'blue' if vi==0 else 'green','source2':'green' if vi==0 else 'blue'}
        for slot,label in expected.items():
            actor=appearance['actors'][slot]
            require(actor['value'].casefold()==label and actor['status']=='reviewed' and
                    all(c['status']=='pass' for c in actor['checks']),'actual clothing pixels do not realize scoring labels')
    for key,record in construction['appearance_geometry_invariants'].items():
        arrays=[np.load(p/f'{key}.npy',mmap_mode='r',allow_pickle=False) for p in captures]
        require(np.array_equal(*arrays),f'native {key} changed')
        require([sha(x.tobytes()) for x in arrays]==record['sha256'],f'native {key} identity changed')
    rgb=[np.load(p/'rgb.npy',mmap_mode='r',allow_pickle=False)[...,:3] for p in captures]
    sem=np.load(captures[0]/'semantic.npy',mmap_mode='r',allow_pickle=False)
    ids=[v['semantic_id'] for v in read(captures[0]/'pixel_visibility_truth.json')['per_instance'].values()]
    change=np.any(rgb[0]!=rgb[1],axis=-1);outside=int(np.count_nonzero(change&~np.isin(sem,ids)))
    require(outside==0 and int(change.sum())==construction['rgb_changed_pixels'],'appearance pixels invalid')
    observed={};keys=[];episode_relations={};inconsistent=[]
    for member in members:
        row=member['request'];sid=member['sample_id']
        require(set(row)==FIELDS,'unexpected public fields')
        require(row==read(public/'groups'/sid/'request.json'),'certificate/request mismatch')
        require(row['question']==construction['question'],'question changes by member')
        for k in ('video','foa','binaural','mono'):
            parts=Path(row[k]).parts
            require(parts[0]=='groups' and parts[1]==sid and len(parts)==3,'episode key/path ambiguity')
        actual={'video':video_identity(media_path(public,row['video']))}
        for k in ('foa','binaural','mono'):actual[k]=pcm_identity(media_path(public,row[k]))
        require(actual==member['media'],'decoded media certificate mismatch')
        require(actual['video']['frame_count']==150 and actual['video']['duration_seconds']==10,'video is not full 10s')
        require(np.allclose(actual['video']['pts_seconds'],np.arange(150)/15,atol=1e-6,rtol=0),'video PTS clock invalid')
        for k,c in (('foa',4),('binaural',2),('mono',1)):
            require(actual[k]['channel_count']==c and actual[k]['sample_count']==160000,'audio not full 10s')
        vi,ai=member['video_index'],member['audio_index']
        require(actual['video']['frame_sha256']==[sha(f.tobytes()) for f in rgb[vi]],'encoded pixels differ from native video')
        for profile in ('raw_av','av_mono','level2'):
            payload=build_model_payload(public,row,profile)
            require(set(payload)<= {'text','frames','audio','foa','frame_times_s','fps'},'private/transport field leaked')
            require(payload['text']==row['question'],'model text changed')
            require([sha(f.tobytes()) for f in payload['frames']]==member['input_receipt']['frames']['rgb_sha256'],'model frames differ')
        key=evidence_keys(public,row,'uninitialized-model-producer',{'schema':'frozen_media_v1','frame_count':40})
        require(key==member['evidence_keys'],'evidence cache identity invalid');keys.append(key)
        pair=(actual['video']['decoded_sha256'],actual['foa']['pcm_sha256'])
        episode=Path(row['video']).parts[1]
        if episode in episode_relations and episode_relations[episode]!=pair:inconsistent.append(episode)
        episode_relations[episode]=pair;observed[(vi,ai)]=actual
    require(inconsistent==[],'members collapsed into inconsistent episode media')
    for vi in range(2):require(observed[(vi,0)]['video']==observed[(vi,1)]['video'],'same video differs by audio')
    for ai in range(2):
        for k in ('foa','binaural','mono'):require(observed[(0,ai)][k]==observed[(1,ai)][k],'same audio differs by video')
    require(observed[(0,0)]['foa']!=observed[(0,1)]['foa'],'target audio did not change')
    require(observed[(0,0)]['video']!=observed[(1,0)]['video'],'target video did not change')
    require(len({k['M1'] for k in keys})==2 and len({k['M2'] for k in keys})==2 and len({k['cross_modal'] for k in keys})==4,'parent-level cache reuse')
    require(cert['joint_solvability']['status']=='pass_automatic','joint QA not admitted')
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    parent={'house':cert['house'],'camera_m':geometry['camera_m'],'feet_m':sorted(geometry['feet_m']),'basis':geometry['basis']}
    return {'group':cert['group'],'status':'pass','certificate_sha256':filehash(cert_path),'three_layers_verified':True,
      'member_count':4,'inconsistent_episode_relations':inconsistent,'cache_counts':{'M1':2,'M2':2,'cross_modal':4},
      'physical_parent_sha256':sha(canonical(parent)),
      'video_pair_sha256':sha(canonical(sorted(observed[(vi,0)]['video']['decoded_sha256'] for vi in range(2)))),
      'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
      'child_cpu_seconds':after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime}

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--certificate',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();paths=[a.certificate] if a.certificate else sorted((root/'scoring_private/certificates').glob('*.json'))
    require(bool(paths),'no certificates');results=[]
    for path in paths:
        try:results.append(verify_one(root,path))
        except Exception as exc:results.append({'certificate':str(path),'status':'fail','reason':f'{type(exc).__name__}: {exc}'})
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    successful=[r for r in results if r['status']=='pass'];independent=True
    for key in ('physical_parent_sha256','video_pair_sha256'):
        if len({r[key] for r in successful})!=len(successful):independent=False
    config=read(root/'inputs/config.json') if (root/'inputs/config.json').exists() else read(root/'inputs/shard_config.json')
    if a.certificate is None:
        expected=len(config['houses'])*config['groups_per_house']
        if config['split']=='test' and (root/'run_state.json').exists():expected=len(read(root/'run_state.json')['indices'])
        require(len(paths)==expected,'full verification lacks requested parent certificates')
    require(all(r['physical_parent_sha256'] not in config.get('excluded_physical_parent_sha256',[]) and
                r['video_pair_sha256'] not in config.get('excluded_video_pair_sha256',[]) for r in successful),'formal panel reused a pilot parent')
    scoring_audit=None
    if (root/'public/COMMITMENTS.sha256').exists():
        # The scoring script alone opens answer bytes, including commitment checks.
        scoring_audit=json.loads(subprocess.check_output([sys.executable,str(REPO/'tools/qa/score_frozen_panel.py'),
          'audit','--root',str(root)],text=True))
        require(scoring_audit['status']=='pass','scoring-private seal audit failed')
    report={'status':'pass' if all(r['status']=='pass' for r in results) and independent else 'fail',
      'groups':results,'distinct_physical_parents':len({r['physical_parent_sha256'] for r in successful}),
      'distinct_video_pairs':len({r['video_pair_sha256'] for r in successful}),
      'answers_opened':False,'scoring_script_audit':scoring_audit,'models_run':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    if report['status']!='pass':sys.exit(1)
if __name__=='__main__':main()
