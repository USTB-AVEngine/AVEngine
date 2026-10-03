#!/usr/bin/env python3
"""Adapter to the existing v6 plan / Habitat / RLR / QA functions.

Requests remain explicit: this infrastructure does not implement the other
agent's v7 30-second sampler, static-source IR optimization, or test protocol.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO=Path(os.environ.get('AVENGINE_GENERATOR_CODE_ROOT',str(Path(__file__).resolve().parents[2]))).resolve()
sys.path.insert(0,str(REPO/'src'))
from resources import write_json
from generate import sha

def load(path):return json.loads(Path(path).read_text())

def inventory(directory):
    return [{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(Path(directory).rglob('*')) if p.is_file()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['plan','render','audio','deliver'],required=True)
    p.add_argument('--request',required=True);p.add_argument('--attempt',required=True);p.add_argument('--upstream');p.add_argument('--world-root');p.add_argument('--gpu',default='0')
    a=p.parse_args();attempt=Path(a.attempt);request=load(a.request);started=time.monotonic()
    # The native simulation request, not only OMP, must be single-threaded.
    simulation_path=request.get('simulation_request')
    if not simulation_path:raise ValueError('v6 request must explicitly select its RLR simulation configuration')
    simulation=load(simulation_path)
    if isinstance(simulation.get('simulation'),dict):simulation['simulation']['thread_count']=1
    elif 'thread_count' in simulation:simulation['thread_count']=1
    else:raise ValueError('cannot locate RLR thread_count in simulation request')
    sim=attempt/'simulation_single_thread.json';write_json(sim,simulation);request['simulation_request']=str(sim)
    request.setdefault('runtime',{})['rlr_threads']=1
    request['runtime']['graphics_adapter']=int(a.gpu) if a.phase=='render' else request['runtime'].get('graphics_adapter',0)  # EGL uses the physical CUDA device number, not the masked ordinal.
    request.setdefault('production',{}).setdefault('stage_resources',{}).setdefault('audio',{})['rlr_threads']=1
    request_path=attempt/'effective_request.json';write_json(request_path,request)
    from avengine.dataset.binding_group_native import plan_visual_variant,capture_visual_plan,build_audio_assignment_plan,materialize_audio_variant,finalize_audio_assignment
    if a.phase=='plan':
        result=plan_visual_variant(request_path,attempt/'episode',label='cci_plan',log=attempt/'plan.log')
    elif a.phase=='render':
        src=Path(a.upstream)/'episode'
        shutil.copytree(src,attempt/'episode',symlinks=False)
        case=attempt/'episode/plan/habitat_execution/case_manifest.json'
        if case.is_file():
            value=load(case);value['m1_request_path']='m1_capture_request.json';write_json(case,value)
        write_json(attempt/'episode/request.json',request)
        result=capture_visual_plan(request,attempt/'episode',label='cci_render',log=attempt/'render.log')
    elif a.phase=='audio':
        src=Path(a.upstream)/'episode';plan=load(src/'plan/episode_plan.json')
        targets={str(e['event_id']):str(e['actor_id']) for e in plan.get('audio_events',[])}
        audio_plan,audio_request=build_audio_assignment_plan(plan,request,'a0',assignment_targets={'a0':targets},require_authoritative_endpoints=False)
        audio_request['simulation_request']=str(sim);audio_request.setdefault('runtime',{})['rlr_threads']=1
        materialized=materialize_audio_variant({'capture':str(src/'capture'),'visual_video':str(src/'capture/ue_visual_only.mp4') if (src/'capture/ue_visual_only.mp4').exists() else None},attempt/'episode',audio_plan,audio_request,member_id=request['episode_id'])
        result=finalize_audio_assignment(materialized,audio_request,verify_materialized=True)
        if result['delivered_audio_layouts'].get('status')!='pass':raise RuntimeError('audio readback did not pass')
        receipt=Path(materialized)/'delivery/audio/research_receipt.json'
        if not receipt.is_file():raise RuntimeError('native audio research receipt is missing')
        if receipt.is_file():
            value=load(receipt)
            def threads(v):
                if isinstance(v,dict):
                    for k,x in v.items():
                        if k=='thread_count':yield x
                        yield from threads(x)
                elif isinstance(v,list):
                    for x in v:yield from threads(x)
            found=list(threads(value))
            if not found or any(n!=1 for n in found):raise RuntimeError('native receipt did not confirm single-threaded RLR')
    else:
        if not a.world_root:raise ValueError('delivery requires a durable world root')
        src=Path(a.upstream)/'episode/delivery';dest=Path(a.world_root)
        dest.mkdir(parents=True,exist_ok=True)
        # A fresh delivery attempt is complete only after all media/certificates
        # are copied and read back. Existing attempts are never overwritten.
        final=dest/('delivery-'+attempt.name)
        shutil.copytree(src,final,symlinks=False,ignore=shutil.ignore_patterns('ir_cache','rir_cache','cache','*.npy','*.npz'))
        records=inventory(final)
        if not any(r['path'].endswith('.wav') for r in records) or not any(r['path'].endswith('.mp4') for r in records) or not any('questions' in r['path'] and r['path'].endswith('.json') for r in records):raise RuntimeError('delivery lacks audio, video or questions')
        # This is the durable media index. Original certificates retain the
        # original scratch paths as provenance; consumers use this index.
        index={'schema':'cci_delivery_index_v1','media':[r for r in records if Path(r['path']).suffix.lower() in ('.wav','.mp4')],'certificates':[r for r in records if Path(r['path']).suffix.lower() not in ('.wav','.mp4')],'source_request_sha256':sha(a.request)}
        write_json(final/'DELIVERY_INDEX.json',index)
        records=inventory(final)
        result={'delivery':str(final),'index':str(final/'DELIVERY_INDEX.json'),'files':records}
        write_json(dest/'ARTIFACTS.json',result)
    write_json(attempt/'stage.json',{'phase':a.phase,'result':result,'files':inventory(attempt),'wall_seconds':time.monotonic()-started,'rlr_threads':1})
if __name__=='__main__':main()
