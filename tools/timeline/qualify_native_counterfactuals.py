#!/usr/bin/env python3
"""Native 2x2 appearance/association canary; audit actual media and answers."""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
from copy import deepcopy
import numpy as np
import soundfile as sf
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.timeline.v7_candidate_generation import stage,audio_stage,member_scene,read_json
from avengine.timeline.v7_native_questions import make_questions
from avengine.timeline.v7_production_contract import audit_counterfactuals
from avengine.timeline.continuous_pilot import write_json
from avengine.spatial_audio.audio import write_float32_wav


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','native-root','windows-root','delivery-root','assets','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--visual-only',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only canary')
    inputs=read_json(a.inputs);house=inputs['houses'][0];root=a.output;root.mkdir(parents=True,exist_ok=True)
    assets=read_json(a.assets)['assets'];base_dir=a.native_root/house['house']
    base=read_json(base_dir/'scene_plan_input.json');observed=read_json(a.windows_root/(house['house']+'__observed_scene.json'))
    v0=read_json(base_dir/'native_visual_evidence.json')
    variant=deepcopy(assets);variant.reverse()
    for index,asset in enumerate(variant):asset['endpoint_id']='voice'+str(index+1)
    def render(attempt):
        renderer=NativeContinuousVideo(house,base,inputs,variant)
        try:result=renderer.encode(base,attempt/'scene_video.mp4')
        finally:renderer.close()
        return {'wall_seconds':result['wall_seconds'],'renderer':result['renderer']},['scene_video.mp4','native_visual_evidence.json']
    v1dir,_,_=stage(root,'visual_v1',{'scene':base,'assets':variant},render)
    v1=read_json(v1dir/'native_visual_evidence.json')
    for endpoint in ('voice1','voice2'):
        if v0['observed_mouth_trajectories'][endpoint]!=v1['observed_mouth_trajectories'][endpoint]:
            raise ValueError('appearance swap changed observed mouth trajectory; audio reuse invalid')
    if a.visual_only:print(str(v1dir));return
    delivery=a.delivery_root/house['house'];ar=read_json(delivery/'audio_completed.json')
    a0dir=delivery/ar['attempt']
    a1scene=member_scene(observed,v0,'v0_a1')
    a1dir,_,_=stage(root,'audio_a1',{'scene':a1scene,'inputs':inputs},
        lambda attempt:audio_stage(a1scene,house,inputs,delivery/'rir_cache',attempt))
    shared={'house':house['house'],'room':house['room']['room_label'],
            'materials':house['acoustic_package_manifest'],'camera':[base['listener_m'],base['camera_forward']],
            'background':str(base_dir),'trajectory':v0['observed_mouth_trajectories'],
            'event_times':[(e['start_tick'],e['end_tick']) for e in observed['events']], 'split':house['split']}
    members=[];golds=[];question_sets={};logical_keys={};media={};start=round(base['window_starts_s'][0]*5)/5
    for member in ('v0_a0','v0_a1','v1_a0','v1_a1'):
        visuals=v1 if member.startswith('v1') else v0
        scene=member_scene(observed,visuals,member);movie=(v1dir if member.startswith('v1') else base_dir)/'scene_video.mp4'
        audio_dir=a1dir if member.endswith('a1') else a0dir
        destination=root/member;destination.mkdir(exist_ok=True);video=destination/'video.mp4'
        if not video.exists():
            first=round(start*15)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-threads','1','-i',str(movie),
                '-vf',f'select=between(n\\,{first}\\,{first+149}),setpts=N/(15*TB)','-frames:v','150','-an',
                '-c:v','libx264','-threads','1','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(video)],check=True)
        audio,sr=sf.read(audio_dir/'scene_foa_full_tail.wav',dtype='float64',always_2d=True)
        samples=audio[round(start*sr):round(start*sr)+160000]
        if not (destination/'audio_foa.wav').exists():write_float32_wav(destination/'audio_foa.wav',samples,16000,channel_axis=1)
        if not (destination/'audio_mono.wav').exists():write_float32_wav(destination/'audio_mono.wav',samples[:,:1],16000,channel_axis=1)
        appearances={1:'黄色上衣的人',2:'蓝色上衣的人'} if member.startswith('v1') else {1:'蓝色上衣的人',2:'黄色上衣的人'}
        questions,answers,check=make_questions(scene,start,'native_cf__'+member,visuals,appearances,
                    {'video':'video.mp4','audio_foa':'audio_foa.wav'},quota_index=0)
        for name,value in [('questions.json',questions),('private_answers.json',answers),('evidence.json',check)]:
            if (destination/name).exists():
                if read_json(destination/name)!=value:raise ValueError('counterfactual rerun changed truth')
            else:write_json(destination/name,value)
        answer_by_id={g['question_id']:g for g in answers};question_sets[member]={}
        for q in questions:
            g=answer_by_id[q['question_id']];key=q['qa_id']+'/'+g['comparison_key']
            golds.append({'member':member,'comparison_key':key,'canonical_answer':g['canonical_answer']})
            question_sets[member][key]=q['modality'];logical_keys[(member,key)]=q['logical_question_id']
        members.append({**shared,'member':member,'visual_nuisance':int(member[1]),'emitter_association':int(member[-1]),'audio_nuisance':0})
        media[member]={'video':video.read_bytes(),'audio':samples}
    if media['v0_a0']['video']!=media['v0_a1']['video'] or media['v1_a0']['video']!=media['v1_a1']['video']:
        raise ValueError('irrelevant audio changed native encoded pixels')
    if not np.array_equal(media['v0_a0']['audio'],media['v1_a0']['audio']) or not np.array_equal(media['v0_a1']['audio'],media['v1_a1']['audio']):
        raise ValueError('irrelevant appearance changed FOA samples')
    if np.array_equal(media['v0_a0']['audio'],media['v0_a1']['audio']):raise ValueError('association swap did not change native FOA')
    comparisons=[]
    for left,right,kind,modality in [('v0_a0','v0_a1','vision','vision'),('v1_a0','v1_a1','vision','vision'),
                                   ('v0_a0','v1_a0','audio','audio'),('v0_a1','v1_a1','audio','audio'),
                                   ('v0_a0','v0_a1','binding','audiovisual'),('v1_a0','v1_a1','binding','audiovisual')]:
        common=set(question_sets[left])&set(question_sets[right])
        for key in sorted(common):
            if question_sets[left][key]==modality:
                comparisons.append({'left':left,'right':right,'kind':kind,'comparison_key':key})
    result=audit_counterfactuals(members,golds,comparisons)
    binding_ids=sorted({logical_keys[(c[side],c['comparison_key'])] for c in result['checks']
                        if c['kind']=='binding' and c['pass'] for side in ('left','right')})
    result.update(group_members=4,logical_questions=sum(len(x) for x in question_sets.values()),encoded_visual_invariance=True,
                  sample_exact_audio_invariance=True,association_audio_changed=True,
                  binding_logical_questions_with_changed_answers=len(binding_ids),
                  verified_binding_logical_question_ids=binding_ids,formal_dataset_count=0,
                  claim_boundary='native media and deterministic answer audit, not learned-model modality ablation')
    write_json(root/'counterfactual_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':main()
