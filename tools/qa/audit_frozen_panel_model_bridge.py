#!/usr/bin/env python3
"""CPU audit of real AVModel-hub input helpers; no weights, inference or answers.

Imports only its numpy dataset/input layer. AST-extracted pure functions execute
unchanged source bodies without importing the training entrypoint or torch.
Exported ready specs bypass the legacy world-N3D adapter: panel FOA is already
camera-frame ACN/SN3D. Do not feed them through the legacy adapter again.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np
import soundfile as sf

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from avengine.qa.frozen_panel_inputs import build_model_payload, check_request, media_path, digest

def require(value,reason):
    if not value:raise ValueError(reason)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def pure_functions(path,names,extra=None):
    tree=ast.parse(Path(path).read_text());selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    require({n.name for n in selected}==set(names),'actual helper functions missing')
    namespace={'Mapping':Mapping,'Any':Any,'np':np,'Path':Path,'hashlib':hashlib,'json':json,'threading':threading}
    namespace.update(extra or {})
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace
def build_ready_spec(build_llm_input,public,row,presentation):
    paths={key:str(media_path(public,row[field])) for key,field in
      (('video','video'),('binaural_audio','binaural'),('foa_audio','foa'),('mono_audio','mono'))}
    view={'id':row['sample_id'],'question_id':row['sample_id'],'status':'available','presentation':presentation,
      'qa_id':row['qa_id'],'form':row['form'],'split':row['split'],'prompt':row['question'],'teacher_answer':None,
      'paths':paths,'public_sidecars':{'sampling':{'frame_indices':row['frame_indices'],'actual_times_s':row['frame_times_s']}}}
    spec=build_llm_input(view,facts_path=None,query_frames=False)
    if presentation in {'av','a_only'}:
        spec['model_input']['foa_adapter_status']='ready'
        spec['model_input']['foa_normalization']='sn3d'
    return spec
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--model-hub-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--export-dir',type=Path)
    a=p.parse_args();start=time.monotonic();cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    public=a.root.resolve()/'public';source=a.model_hub_source.resolve();sys.path.insert(0,str(source))
    from avledger.dataset import _episode_key
    from avledger.llm_inputs import build_llm_input,load_video_frames_at_indices,PRESENTATIONS,FORBIDDEN_MODEL_KEYS
    runtime=source/'scripts/avledger/run_llm_fast.py'
    funcs=pure_functions(runtime,{'_mono_av_waveform','_feature','_normalize_foa_for_encoder'})
    cache_source=source/'avledger/media_cache.py'
    cache=pure_functions(cache_source,{'_digest','file_digest','video_key','audio_key'},
      {'SCHEMA':'avledger_media_cache_v1','VIDEO_MAX_PIXELS':100352,'AUDIO_MAX_SAMPLES':320000,
       '_DIGESTS':{},'_DIGEST_LOCK':threading.Lock()})
    rows=[read(q) for q in sorted(public.glob('groups/*/request.json'))]
    require(bool(rows),'no public requests');episodes={};inconsistent=[];results=[];exports={k:[] for k in PRESENTATIONS}
    decoded={}
    for row in rows:
        check_request(row);sid=row['sample_id'];payload=build_model_payload(public,row,'level2')
        # The groups/ segment is authority; deliberately unusable fallback IDs.
        episode,kind=_episode_key({'binding_group_id':'transport_only'}, {}, str(public/'groups'/sid/'delivery'))
        pair=(digest(payload['frames'].tobytes()),digest(payload['foa'].tobytes()))
        if episode in episodes and episodes[episode]!=pair:inconsistent.append(episode)
        episodes[episode]=pair;require(kind=='member','actual episode builder selected wrong granularity')
        video=media_path(public,row['video']);identity=(video.stat().st_dev,video.stat().st_ino,tuple(row['frame_indices']))
        if identity not in decoded:
            # Actual helper's ffmpeg default thread count is constrained by CPU affinity (8 cores).
            affinity=os.sched_getaffinity(0)
            try:
                os.sched_setaffinity(0,set(sorted(affinity)[:8]))
                frames,duration=load_video_frames_at_indices(str(video),row['frame_indices'])
            finally:os.sched_setaffinity(0,affinity)
            decoded[identity]=(digest(frames.tobytes()),duration)
        frame_sha,duration=decoded[identity]
        require(frame_sha==digest(payload['frames'].tobytes()) and duration==10,'actual model frame helper differs')
        profile_receipts={}
        for presentation in PRESENTATIONS:
            spec=build_ready_spec(build_llm_input,public,row,presentation);mi=spec['model_input']
            require(spec['teacher_answer'] is None and not(set(mi)&FORBIDDEN_MODEL_KEYS),'private fields in model spec')
            require(mi['prompt']==row['question'] and sid not in mi['prompt'],'transport ID leaked into prompt')
            require(mi['frame_indices']==row['frame_indices'] and mi['frame_times_s']==row['frame_times_s'],'sampling receipt changed')
            if presentation=='av_mono':
                mono,_=funcs['_mono_av_waveform'](mi,presentation)
                require(np.array_equal(mono,payload['audio']) and len(mono)==160000,'actual mono reader changed or cropped audio')
            if presentation in {'av','a_only'}:
                feature=funcs['_feature'](spec,answer='')
                foa,sr=sf.read(feature['foa_path'],dtype='float32',always_2d=True)
                require(sr==16000 and np.array_equal(foa,payload['foa']),'FOA feature routing changed')
                require(feature['foa_normalization']=='sn3d','FOA normalization tag lost')
                normalized,normalization=funcs['_normalize_foa_for_encoder'](foa,source_normalization='sn3d',target_normalization='sn3d')
                require(np.array_equal(normalized,foa) and normalization['conversion']=='none','native SN3D should not be scaled twice')
                binaural,sr=sf.read(feature['content_audio_path'],dtype='float32',always_2d=True)
                require(sr==16000 and np.array_equal(binaural.mean(axis=1),payload['audio']),'content binaural conversion differs')
            # Paths and opaque IDs are transport; only decoded arrays and prompt go to towers/tokens.
            profile_receipts[presentation]={'selected_modalities':mi['selected_modalities'],'audio_layout':mi['audio_layout'],
              'camera_foa_normalization':'sn3d' if mi['foa_path'] else None,'frame_count':40 if mi['video_path'] else 0,
              'audio_samples':160000 if mi['audio_path'] else 0,'teacher_answer':None}
            exports[presentation].append(spec)
        results.append({'sample_id':sid,'episode_key':episode,'profiles':profile_receipts,'selected_frame_sha256':frame_sha,
          'actual_model_cache_keys':{'video':cache['video_key'](str(video),row['frame_indices'],'uninitialized-model'),
             'audio':cache['audio_key'](str(media_path(public,row['mono'])),'av_mono','uninitialized-model')}})
    require(inconsistent==[] and len(episodes)==len(rows),'actual episode builder collapsed members')
    if a.export_dir:
        a.export_dir.mkdir(parents=True,exist_ok=False)
        for presentation,specs in exports.items():
            with (a.export_dir/f'{presentation}.jsonl').open('x',encoding='utf8') as f:
                for spec in specs:f.write(json.dumps(spec,ensure_ascii=False)+'\n')
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    report={'status':'pass','members':len(rows),'unique_member_episodes':len(episodes),'inconsistent_episode_relations':inconsistent,
      'model_hub_commit':(source/'COMMIT').read_text().strip() if (source/'COMMIT').exists() else 'not_recorded',
      'source_files_sha256':{str(q.relative_to(source)):hashlib.sha256(q.read_bytes()).hexdigest() for q in
        (source/'avledger/dataset.py',source/'avledger/llm_inputs.py',runtime,cache_source)},
      'reader_execution':'real CPU input helpers; pure runtime functions extracted unchanged via AST; no weights imported',
      'legacy_cli_compatibility':'use exported ready specs and frozen_panel_inputs; legacy _prepare_runtime_foa_adapter expects world N3D/private facts and must not run on this panel',
      'processor_limits':{'VIDEO_MAX_PIXELS':100352,'AUDIO_MAX_SAMPLES':320000,'source_frames':40,'full_audio_samples':160000},
      'tokenizer_resize_encoder_execution':'not_run','answers_opened':False,'models_run':False,'records':results,
      'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
      'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime}
    a.output.parent.mkdir(parents=True,exist_ok=True);write(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
