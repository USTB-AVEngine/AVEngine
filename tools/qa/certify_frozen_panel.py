#!/usr/bin/env python3
"""Construct media certificates and isolated model requests; never read answers."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import secrets
import subprocess
import sys
import time
import uuid

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from avengine.qa.frozen_panel_inputs import (audio_identity, canonical, digest,
    video_identity, video_pixels, build_model_payload, evidence_keys)
from avengine.rooms.qa_evidence import build_pixel_appearance_review
from avengine.acoustics.house_acceptance_metrics import foa_direction, rir_metrics

def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
def fixed_plan(plan):
    x=copy.deepcopy(plan);x.pop('episode_id')
    for a in x['visual_plan']['actors']:
        for k in ('asset_id','asset_revision','realized_attributes','display_label'):a.pop(k,None)
        a['emitter_binding'].pop('asset_id',None)
    return x
def encode(capture, path):
    a=np.load(capture/'rgb.npy',mmap_mode='r',allow_pickle=False)[...,:3]
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.shape[2]}x{a.shape[1]}',
         '-r','15','-i','pipe:0','-map_metadata','-1','-an','-c:v','libx264rgb','-crf','0',
         '-preset','veryfast','-threads','2','-movflags','+faststart',str(path)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as p:
        for frame in a:p.stdin.write(frame.tobytes())
        p.stdin.close()
        if p.wait():raise RuntimeError('lossless video encoding failed')
    decoded,_,_=video_pixels(path)
    if not np.array_equal(decoded,a):raise ValueError('decoded video changed native RGB pixels')

def certify(config,index):
    start=time.monotonic();cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    root=Path(config['output']);group=f'{config.get("panel_id","pilot")}_{index:03d}';g=root/'production'/group
    private=root/'scoring_private';private.mkdir(exist_ok=True);private.chmod(0o700)
    public=root/'public';public.mkdir(exist_ok=True);(root/'production').chmod(0o700)
    recipe=read(g/'recipe.json');plans=[read(g/'visual'/f'v{i}'/'plan/episode_plan.json') for i in range(2)]
    if fixed_plan(plans[0])!=fixed_plan(plans[1]):raise ValueError('visual swap changes fields outside whitelist')
    if read(g/'native_equivalence.json')['status']!='pass':raise ValueError('native geometry equivalence failed')
    captures=[g/'visual'/f'v{i}'/'capture' for i in range(2)]
    arrays={};invariants={}
    for name in ('depth','semantic','actor_root_readbacks','emitter_positions_m','actor_joint_readbacks_source1','actor_joint_readbacks_source2'):
        a=[np.load(c/f'{name}.npy',mmap_mode='r',allow_pickle=False) for c in captures]
        same=np.array_equal(*a);invariants[name]={'equal':bool(same),'sha256':[digest(x.tobytes()) for x in a]}
        if not same:raise ValueError(f'appearance changes native {name}')
        if name=='semantic':arrays['semantic']=a[0]
    rgb=[np.load(c/'rgb.npy',mmap_mode='r',allow_pickle=False)[...,:3] for c in captures]
    truth=[read(c/'pixel_visibility_truth.json') for c in captures]
    ids=[v['semantic_id'] for v in truth[0]['per_instance'].values()]
    mask=np.isin(arrays['semantic'],ids);changed=np.any(rgb[0]!=rgb[1],axis=-1)
    outside=int(np.count_nonzero(changed&~mask));changed_pixels=int(np.count_nonzero(changed))
    if outside or changed_pixels<512:raise ValueError('RGB changes outside people or clothing has no support')
    registry=read(config['registry']);registry=registry['assets'] if isinstance(registry,dict) else registry
    appearance=[]
    for vi in range(2):
        q=build_pixel_appearance_review(captures[vi],plans[vi],asset_registry={a['asset_id']:a for a in registry})
        if len(q['actors'])!=2 or any(a['status']!='reviewed' or any(x['status']!='pass' for x in a['checks']) for a in q['actors'].values()):
            raise ValueError(f'joint visual answerability rejected: {q["actors"]}')
        appearance.append(q)
    rb=[read(c/'neutral_readback.json') for c in captures];clock=rb[0]['clock']
    if rb[1]['clock']!=clock or clock['frame_count']!=150 or clock['sample_count']!=160000:raise ValueError('clock not full 10s')
    ar=read(g/'audio/receipt.json');directions=[];acoustics=[]
    for pi in range(2):
        f=np.load(g/'audio'/f'position_{pi}.npy',allow_pickle=False)
        direction=foa_direction(f,ar['listener_native_m'],ar['positions_native_m'][pi],recipe['geometry']['basis']['forward'])
        distance=float(np.linalg.norm(np.array(ar['listener_native_m'])-ar['positions_native_m'][pi]))
        metrics,_=rir_metrics(f,16000,distance,config['simulation']['max_ir_seconds'])
        if direction['check']['angle_deg'] is None or direction['check']['angle_deg']>10:raise ValueError('FOA target bearing invalid')
        if metrics['c50_db']<0:raise ValueError('speech clarity C50 < 0 dB')
        directions.append(direction);acoustics.append(metrics)
    target=[]
    for ai in range(2):
        s,_=sf.read(g/'audio'/f'a{ai}_speech0_stem.wav',dtype='float32',always_2d=True)
        other,_=sf.read(g/'audio'/f'a{ai}_speech1_stem.wav',dtype='float32',always_2d=True)
        lo=round(recipe['start_seconds'][0]*16000);hi=lo+round(recipe['speech'][0]['duration_s']*16000)
        e=float(np.square(s[lo:hi,0].astype(float)).sum());noise=float(np.square(other[lo:hi,0].astype(float)).sum())
        snr=10*np.log10(max(e,1e-30)/max(noise,1e-30))
        if e<=1e-12 or snr<12:raise ValueError('target sentence unclear/overlapping')
        target.append({'pcm_sha256':digest(s.astype('<f4').tobytes()),'target_to_competitor_db':float(snr),
            'full_dry_start_s':recipe['start_seconds'][0],'full_dry_end_s':hi/16000,'routing_position':ai})
    if target[0]['pcm_sha256']==target[1]['pcm_sha256']:raise ValueError('swapping target position leaves target stem unchanged')
    videos=[]
    for vi in range(2):
        path=g/'visual'/f'v{vi}'/'video.mp4'
        if not path.exists():encode(captures[vi],path)
        videos.append(video_identity(path))
    if videos[0]['decoded_sha256']==videos[1]['decoded_sha256']:raise ValueError('clothing swap does not change decoded video')
    audios=[]
    for ai in range(2):audios.append({k:audio_identity(g/'audio'/f'a{ai}_{k}.wav') for k in ('foa','binaural','mono')})
    if audios[0]['foa']['pcm_sha256']==audios[1]['foa']['pcm_sha256']:raise ValueError('audio routing swap has no observation change')
    indices=np.rint(np.linspace(0,149,40)).astype(int).tolist();members=[];answers=[]
    for vi in range(2):
        for ai in range(2):
            sid=uuid.uuid4().hex;folder=public/'groups'/sid;folder.mkdir(parents=True)
            os.link(g/'visual'/f'v{vi}'/'video.mp4',folder/'video.mp4')
            for key in ('foa','binaural','mono'):os.link(g/'audio'/f'a{ai}_{key}.wav',folder/f'{key}.wav')
            row={'sample_id':sid,'question':recipe['question_zh'],'qa_id':'QA-27','form':'open','split':config['split'],
                 'video':f'groups/{sid}/video.mp4','foa':f'groups/{sid}/foa.wav','binaural':f'groups/{sid}/binaural.wav','mono':f'groups/{sid}/mono.wav',
                 'frame_indices':indices,'frame_times_s':[videos[vi]['pts_seconds'][i] for i in indices]}
            write(folder/'request.json',row)
            framehash=[videos[vi]['frame_sha256'][i] for i in indices]
            receipt={'model_inference':'not_run','frames':{'count':40,'frame_indices':indices,'pts_seconds':row['frame_times_s'],'rgb_sha256':framehash},
                'audio':audios[ai],'text_fields':{k:row[k] for k in ('question','qa_id','form','split')},
                'payload_text_fields':['text'],'transport_fields_stripped':['sample_id','all media paths','cache keys','group/member/physical IDs'],
                'profiles':{'raw_av':'40 RGB frames + complete native binaural mean mono + public question','av_mono':'same content with no spatial FOA',
                    'level2':'same + complete camera ACN/SN3D FOA; no teacher/oracle state','level1':'deferred until model-produced heard/seen/cross evidence with verified hashes'},
                'derived_evidence':{'model_derived':'not_generated','oracle':'forbidden','M1':'actual decoded audio identity','M2':'actual decoded video identity + selected frames','cross_modal':'actual member audio/video combination'}}
            for profile in ('raw_av','av_mono','level2'):
                payload=build_model_payload(public,row,profile)
                if digest(payload['frames'].tobytes())!=digest(np.asarray([rgb[vi][i] for i in indices]).tobytes()):raise ValueError('request selected frames changed')
            keys=evidence_keys(public,row,'uninitialized-model-producer',{'schema':'frozen_media_v1','frame_count':40})
            members.append({'sample_id':sid,'video_index':vi,'audio_index':ai,'request':row,'media':{'video':videos[vi],**audios[ai]},'input_receipt':receipt,'evidence_keys':keys})
            answer='穿蓝色上衣的人' if vi==ai else '穿绿色上衣的人'
            answers.append({'sample_id':sid,'group':group,'video_index':vi,'audio_index':ai,'physical_person_id':f'{group}:source{ai+1}',
                'question':recipe['question_zh'],'correct_answer_zh':answer,'answer_label':'blue' if vi==ai else 'green'})
    cert={'schema':'frozen_binding_quartet_certificate_v1','pilot':config['split']=='pilot','exclude_final_panel':config['split']=='pilot',
      'group':group,'independent_parent':group,'house':recipe['house'],'room':recipe['room'],'quota':{'voice':recipe['voice'],'side':recipe['side']},
      'construction':{'allowed_visual_changes':['actors[*].asset_id','asset_revision','realized_attributes','display_label','emitter_binding.asset_id'],
       'allowed_audio_changes':['dialogue_index -> physical_emitter_slot; recompute convolution using that position RIR'],
       'fixed_plan_sha256':digest(canonical(fixed_plan(plans[0]))),'all_other_plan_fields_fixed':True,'metadata_only_changes':['episode_id'],
       'fixed_fields':['house/room mesh and acoustic materials','body geometry','root/joint trajectories','mouth positions','camera and light','clock and lengths','dry words/voices/start times/gain','question'],
       'question':recipe['question_zh'],'answer_relation':'label(video_index,audio_index)=label0 when XOR=0, label1 when XOR=1; private scoring script audits actual answers',
       'appearance_geometry_invariants':invariants,'rgb_changed_pixels':changed_pixels,'rgb_changed_outside_actor_masks':outside,
       'native_equivalence':read(g/'native_equivalence.json'),'geometry':recipe['geometry'],'audio_receipt':ar,
       'plan_paths':[str(g/'visual'/f'v{i}'/'plan/episode_plan.json') for i in range(2)]},
      'media':{'clock':clock,'members':members},
      'joint_solvability':{'status':'pass_automatic','whole_group_rejection':True,'selection_by_model':False,'appearance':appearance,
        'foa_direction':directions,'rir_clarity':acoustics,'target_stems':target,'admitted_dry_speech':ar['dry_audio'],
        'time_relation':'target dry sentence preserved at fixed start time; all selected frames show both stationary people; no lip or motion change with audio',
        'human_listening_review':'not_run; automatic clarity/admitted transcript proxies, no model inference'},
      'evaluation_inputs':{'status':'preprocessed_and_hash_checked; model_inference_not_run','members':members,
        'no_parent_scene_cache':True,'answer_reader':'tools/qa/score_frozen_panel.py'},
      'provenance':{'recipe':str(g/'recipe.json'),'capture_roots':[str(c) for c in captures],'code_git_base':config.get('code_git_base','c21c387')}}
    write(private/'certificates'/f'{group}.json',cert)
    write(private/'answers_parts'/f'{group}.json',{'nonce':secrets.token_hex(32),'pilot':cert['pilot'],'members':answers})
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    timing={'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
      'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime}
    write(g/'certify_timing.json',timing);print(json.dumps({'certified':group,'timing':timing}),flush=True)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('certification is CPU only')
    certify(read(a.config),a.index)
if __name__=='__main__':main()
