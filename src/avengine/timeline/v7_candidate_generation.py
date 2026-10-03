"""Seed-driven native CPU candidates with durable stages, never formal release.

Each parent has one writer, assigned by the calling shard scheduler. Restarted
stages create new attempt directories and reuse exact-pose native RIR progress.
Completed stages are immutable. An AFS scheduler must establish single-writer
ownership; this module does not assume local SQLite locks protect remote jobs.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json, os, resource, subprocess, time, uuid, multiprocessing
import numpy as np
from avengine.assets.source_materials import MATERIAL_REVISION
from avengine.capture.continuous_native_visibility import VISIBILITY_REVISION
from avengine.timeline.continuous_pilot import (
    make_scene, write_json, PositionRIRCache, FormantSpeech, make_dry_buses,
)
from avengine.timeline.continuous_scene import (
    plan_rir_keyframes, partitioned_convolve, raw_world_to_camera_sn3d,
)
from avengine.timeline.v7_native_questions import make_questions
from avengine.spatial_audio.audio import write_float32_wav

FORMAL_BLOCKERS = (
    'moving_RIR_energy_admission_failed',
    'all_28_v6_question_families_and_native_evidence_admission_incomplete',
    'native_M2_assets_remain_research_and_full_motion_contact_admission_pending',
)


def read_json(path):
    return json.loads(Path(path).read_text())


def stage(root, name, recipe, run):
    """Publish one complete stage; interrupted attempt files cannot become truth."""
    recipe=json.loads(json.dumps(recipe,ensure_ascii=False,allow_nan=False))
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    pointer=root/(name+'_completed.json')
    if pointer.exists():
        completed=read_json(pointer)
        if completed['recipe']!=recipe:raise ValueError('completed stage recipe changed')
        directory=root/completed['attempt']
        if any(not (directory/p).is_file() for p in completed['required_files']):
            raise ValueError('completed stage lost a required artifact')
        return directory,completed['result'],True
    attempt=root/(name+'_attempt_'+uuid.uuid4().hex);attempt.mkdir()
    result,required=run(attempt)
    if not required or any(not (attempt/p).is_file() for p in required):
        raise ValueError('stage cannot publish without all required artifacts')
    completed={'recipe':recipe,'attempt':attempt.name,'required_files':required,'result':result}
    write_json(pointer,completed)
    return attempt,result,False


def seeded_scene(row, house):
    from avengine.timeline.v7_scene_quotas import QUOTA_INPUT_VERSION,validate_quota_row
    if row.get('schema')==QUOTA_INPUT_VERSION:
        validate_quota_row(row)
        raise RuntimeError('v4 scenario quotas require native layout, dry class and QA-family admission; v3 two-person fallback is forbidden')
    if row.get('schema') not in ('v7-production-inputs-20261003-v2','v7-production-inputs-20261003-v3'):
        raise ValueError('unsupported input seed schema')
    if row['duration_s']!=30 or row['questions_per_window'] not in (12,14):
        raise ValueError('unsupported duration/question schedule')
    if row['assignment_status']!='admitted' or row['house']!=house['house']:
        raise ValueError('seed row has no admitted physical house')
    if house['acoustic_status']!='pass' or row['room_id'] not in house['checked_room_ids']:
        raise ValueError('seed row is outside the measured house/room scope')
    hs={'development':'dev'}.get(house['split'],house['split'])
    if hs!=row['split']:raise ValueError('physical house crossed splits')
    if house.get('known_exposure') and row['split']!='train':
        raise ValueError('known exposed house cannot enter dev/test')
    selected=next(r for r in house['checked_rooms'] if r['room_label']==row['room_id'])
    working=deepcopy(house);working['room']=selected
    scene=make_scene(working,int(row['seeds']['trajectory']%3),row['duration_s'])
    scene.update(scene_id=row['scene_id'],split=row['split'],seed=row['scene_seed'],paired_length_comparison=False,
                 independent_parent_scene_id=row['scene_id'],visual_status='native_CPU_research_candidate',
                 window_starts_s=[x/10 for x in row['window_start_ticks'][:row['window_count']]])
    # The canary has legal idle/walk M2 figures; do not invent a clap gesture.
    scene['sources'].pop('clap1')
    scene['events']=[e for e in scene['events'] if e['endpoint_id']!='clap1']
    for endpoint,voice in (('voice1','en-us+m1'),('voice2','en-us+m3')):
        scene['sources'][endpoint].update(class_id=1,voice=voice)
    for e in scene['events']:e['class_id']=1
    if row['schema'].endswith('-v3'):
        trajectory_rng=np.random.default_rng(row['seeds']['trajectory'])
        a=np.asarray(selected['placement']['source_1_m']);b=np.asarray(selected['placement']['source_2_m'])
        far=b+trajectory_rng.uniform(.4,.6)*(a-b)
        for knot in scene['sources']['voice2']['trajectory']:
            if knot['time_s']%10 in (6,8):knot['position_m']=far.tolist()
        event_rng=np.random.default_rng(row['seeds']['event_schedule'])
        for e in scene['events']:
            block=(e['start_tick']//100)*100
            if e['start_tick']!=block:e['start_tick']+=int(event_rng.integers(-1,2))
            if e['end_tick']%100!=99:e['end_tick']+=int(event_rng.integers(-1,2))
            if e['end_tick']<=e['start_tick']:raise ValueError('jitter erased an activity segment')
        camera_rng=np.random.default_rng(row['seeds']['camera'])
        theta=np.radians(camera_rng.uniform(-4,4));c,s=np.cos(theta),np.sin(theta)
        scene['camera_forward']=(np.array([[c,0,s],[0,1,0],[-s,0,c]])@scene['camera_forward']).tolist()
        audio_rng=np.random.default_rng(row['seeds']['audio'])
        voices=['en-us+m1','en-us+m3']
        if audio_rng.integers(0,2):voices.reverse()
        for endpoint,voice in zip(('voice1','voice2'),voices,strict=True):
            scene['sources'][endpoint]['voice']=voice
            scene['sources'][endpoint]['dry_gain']=float(audio_rng.uniform(.8,1.2))
            scene['sources'][endpoint]['voice_identity_id']=voice
        scene['actor_pose_phase_ticks']={endpoint:int(row['seeds']['pose']+i*7919) for i,endpoint in enumerate(('voice1','voice2'))}
    scene['sampling_limitations']=['two male figures and speech only','three trajectory templates' if row['schema'].endswith('-v2') else 'single legal line segment with three start/stop cycles',
                                   'appearance/pose coverage is not a full-set implementation']
    return working,scene


def member_scene(base, visual, member):
    scene=deepcopy(base)
    for key,trajectory in visual['observed_mouth_trajectories'].items():
        scene['sources'][key]['trajectory']=trajectory
    if member.endswith('a1'):
        # Swap only dry content/activity association between the two physical
        # emitters; roots, bone poses, event times and background stay fixed.
        for e in scene['events']:
            e['endpoint_id']={'voice1':'voice2','voice2':'voice1'}[e['endpoint_id']]
            e['source_id']=scene['sources'][e['endpoint_id']]['actor_id']
        first,second=scene['sources']['voice1']['voice'],scene['sources']['voice2']['voice']
        scene['sources']['voice1']['voice']=second;scene['sources']['voice2']['voice']=first
        for field in ('dry_gain','voice_identity_id'):
            if field in scene['sources']['voice1']:
                x,y=scene['sources']['voice1'][field],scene['sources']['voice2'][field]
                scene['sources']['voice1'][field]=y;scene['sources']['voice2'][field]=x
    scene['member']=member
    return scene


def _dry_worker(scene, output):
    # A fresh CPU process fixes speech-engine initial state independently of
    # how many parents this shard worker already generated.
    output=Path(output);dry,receipt=make_dry_buses(scene,FormantSpeech())
    for endpoint,samples in dry.items():
        samples*=scene['sources'][endpoint].get('dry_gain',1.)
        with (output/(endpoint+'.npy')).open('xb') as f:
            np.save(f,samples,allow_pickle=False);f.flush();os.fsync(f.fileno())
    write_json(output/'dry_receipt.json',receipt)


def parent_dry(root, base):
    recipe={'seed':base['seed'],'duration_s':base['duration_s'],'events':base['events'],
            'voices':{k:{f:v[f] for f in ('voice','dry_gain','class_id') if f in v} for k,v in base['sources'].items()}}
    def build(attempt):
        child=multiprocessing.get_context('spawn').Process(target=_dry_worker,args=(base,str(attempt)))
        child.start();child.join()
        if child.exitcode!=0:raise RuntimeError('isolated CPU dry synthesis failed')
        receipt=read_json(attempt/'dry_receipt.json')
        return receipt,['dry_receipt.json',*[k+'.npy' for k in base['sources']]]
    directory,receipt,_=stage(root,'dry',recipe,build)
    dry={k:np.load(directory/(k+'.npy'),allow_pickle=False) for k in base['sources']}
    if any(v.shape!=(round(base['duration_s']*16000),) or not np.isfinite(v).all() for v in dry.values()):
        raise ValueError('cached dry activity is invalid')
    return dry,receipt


def audio_stage(scene, house, inputs, cache_root, attempt, renderer_factory=None, dry_override=None, dry_receipt=None):
    if renderer_factory is None:
        from tools.timeline.pilot_continuous_scenes import native_renderer
        renderer_factory=native_renderer
    start,cpu=time.perf_counter(),time.process_time()
    renderer,identity=renderer_factory(inputs,house)
    write_json(attempt/'native_RLR_setup.json',renderer.setup_report)
    cache=PositionRIRCache(cache_root,renderer,identity)
    if dry_override is None:
        dry,source_receipt=make_dry_buses(scene,FormantSpeech())
        for endpoint in dry:dry[endpoint]*=scene['sources'][endpoint].get('dry_gain',1.)
    else:
        dry,source_receipt=dry_override,dry_receipt
    stems={};plans={}
    for endpoint,spec in scene['sources'].items():
        # Keep every observed bone-anchor knot. This avoids silently decimating
        # animated mouths; interpolation itself still needs energy admission.
        keys=plan_rir_keyframes(spec['trajectory'],scene['duration_s'],np.asarray(scene['listener_m']),
                               max_seconds=.25,max_distance_m=.5)
        if all(np.array_equal(k['position_m'],keys[0]['position_m']) for k in keys):keys=keys[:1]
        filters=[cache.get(k['position_m']) for k in keys]
        stems[endpoint]=partitioned_convolve(dry[endpoint],filters,[k['sample'] for k in keys])
        plans[endpoint]={'keys':len(keys),'observed_native_mouth':True}
    length=max(x.shape[1] for x in stems.values());mix=np.zeros((4,length))
    for samples in stems.values():mix[:,:samples.shape[1]]+=samples
    camera=raw_world_to_camera_sn3d(mix,scene['camera_forward'])
    write_float32_wav(attempt/'scene_foa_full_tail.wav',camera,16000,channel_axis=0)
    result={'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,
            'RIR_cache':cache.stats.copy(),'RIR_plan':plans,'dry_assets_receipt':source_receipt,
            'formal_dataset_count':0,'blocking_reasons':list(FORMAL_BLOCKERS)}
    write_json(attempt/'audio_evidence.json',result)
    cache.db.close()
    return result,['scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json']


def audit_parent_group(row, base, members):
    from avengine.timeline.v7_production_contract import audit_counterfactuals
    import soundfile as sf
    if len(members)==1:return {'status':'not_grouped','verified_binding_logical_question_ids':[]}
    shared={'house':row['house'],'room':row['room_id'],'materials':'fixed_parent_recipe',
        'camera':[base['listener_m'],base['camera_forward']],'background':row['house'],
        'trajectory':{k:v['trajectory'] for k,v in base['sources'].items()},
        'event_times':[(e['start_tick'],e['end_tick']) for e in base['events']],'split':row['split']}
    declarations=[{**shared,'member':m['member'],'visual_nuisance':int(m['member'][1]),
                   'emitter_association':int(m['member'][-1]),'audio_nuisance':0} for m in members]
    answers=[];by_key={};logical={};media={}
    for m in members:
        name=m['member'];by_key[name]={}
        for i,w in enumerate(m['windows']):
            path=Path(m['windows_directory'])/w['window_id'];q=read_json(path/'questions.json')
            private={x['question_id']:x for x in read_json(path/'private_answers.json')}
            media[(name,i)]={'video':(path/'video.mp4').read_bytes(),'audio':sf.read(path/'audio_foa.wav',dtype='float32',always_2d=True)[0]}
            for item in q:
                gold=private[item['question_id']];key=f'{i}/{item["qa_id"]}/{gold["comparison_key"]}'
                by_key[name][key]=item['modality'];logical[(name,key)]=item['logical_question_id']
                answers.append({'member':name,'comparison_key':key,'canonical_answer':gold['canonical_answer']})
    checks=[]
    for left,right,kind,modality in [('v0_a0','v0_a1','vision','vision'),('v1_a0','v1_a1','vision','vision'),
            ('v0_a0','v1_a0','audio','audio'),('v0_a1','v1_a1','audio','audio'),
            ('v0_a0','v0_a1','binding','audiovisual'),('v1_a0','v1_a1','binding','audiovisual')]:
        for i in range(row['window_count']):
            if kind=='vision' and media[(left,i)]['video']!=media[(right,i)]['video']:raise ValueError('unrelated audio changed native video')
            if kind=='audio' and not np.array_equal(media[(left,i)]['audio'],media[(right,i)]['audio']):raise ValueError('unrelated picture changed FOA')
            if kind=='binding' and np.array_equal(media[(left,i)]['audio'],media[(right,i)]['audio']):raise ValueError('binding association did not change FOA')
        for key in sorted(set(by_key[left])&set(by_key[right])):
            if by_key[left][key]==modality:checks.append({'left':left,'right':right,'kind':kind,'comparison_key':key})
    result=audit_counterfactuals(declarations,answers,checks)
    result['verified_binding_logical_question_ids']=sorted({logical[(c[side],c['comparison_key'])] for c in result['checks']
             if c['kind']=='binding' and c['pass'] for side in ('left','right')})
    if result['status']!='pass':raise ValueError('observed counterfactual answers failed')
    return result


def generate_parent(row, house, inputs, assets, output, compact=False):
    from avengine.capture.continuous_native_video import NativeContinuousVideo
    from avengine.timeline.v7_production_contract import validate_question_counts
    start_cpu,start_wall=time.process_time(),time.perf_counter()
    working,base=seeded_scene(row,house)
    root=Path(output)/'parents'/row['scene_id'];root.mkdir(parents=True,exist_ok=True)
    write_json(root/f'worker_started_{os.getpid()}_{time.time_ns()}.json',
        {'pid':os.getpid(),'parent_pid':os.getppid(),'scene_id':row['scene_id'],'house':row['house'],
         'output_root':str(root.resolve()),'compute_device':'CPU'})
    recipe={'seed_row':row,'house':house,'runtime_inputs':inputs,'assets':assets,
            'visual_contract':{'source_material_revision':MATERIAL_REVISION,'pixel_visibility_revision':VISIBILITY_REVISION}}
    input_path=root/'input_recipe.json'
    if input_path.exists():
        if read_json(input_path)!=recipe:raise ValueError('parent input recipe changed')
    else:write_json(input_path,recipe)
    from avengine.timeline.v7_compaction import load_sealed,compact_parent
    sealed=load_sealed(root,recipe)
    if sealed is not None:return {**sealed['result'],'resumed_sealed_delivery':True}
    members=[];audio_by_factor={};reference_mouths=None
    dry_buses,dry_receipt=parent_dry(root,base) if row['schema'].endswith('-v3') else (None,None)
    for member in row['members']:
        variant=deepcopy(assets)
        appearance_flip=(row['schema'].endswith('-v3') and row['seeds']['appearance']%2==1)
        if member.startswith('v1') != appearance_flip:
            endpoints=[a['endpoint_id'] for a in variant]
            variant.reverse()
            for asset,endpoint in zip(variant,endpoints,strict=True):asset['endpoint_id']=endpoint
        appearances={base['sources'][a['endpoint_id']]['actor_id']:a.get('appearance_zh',{'blue':'蓝色上衣的人','yellow':'黄色上衣的人'}[a['appearance']['top_color']]) for a in variant}
        visual_name='visual_'+member[:2]
        def render(attempt):
            video=NativeContinuousVideo(working,base,inputs,variant,
                                        source_material_cache_directory=attempt/'source_material_cache')
            try:result=video.encode(base,attempt/'scene_video.mp4')
            finally:video.close()
            return {'renderer':result['renderer'],'wall_seconds':result['wall_seconds']},['scene_video.mp4','native_visual_evidence.json','source_materials.json','source_material_checks.json','native_visibility_evidence.json']
        visual_dir,vr,visual_resumed=stage(root,visual_name,{'seed_row':row,'assets':variant,'source_material_revision':MATERIAL_REVISION,'pixel_visibility_revision':VISIBILITY_REVISION},render)
        visual=read_json(visual_dir/'native_visual_evidence.json');scene=member_scene(base,visual,member)
        if reference_mouths is None:reference_mouths=visual['observed_mouth_trajectories']
        elif visual['observed_mouth_trajectories']!=reference_mouths:
            raise ValueError('counterfactual appearance changed a physical mouth trajectory')
        factor=member[-2:]
        def make_audio(attempt):
            if factor in audio_by_factor:
                original,previous=audio_by_factor[factor]
                for name in ('scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json'):
                    os.link(original/name,attempt/name)
                return {**previous,'reused_audio_variant':factor,'wall_seconds':0.,'process_cpu_seconds':0.,
                        'RIR_cache':{k:0 for k in previous['RIR_cache']}},['scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json']
            dry=dry_buses
            if dry is not None and factor=='a1':dry={'voice1':dry_buses['voice2'],'voice2':dry_buses['voice1']}
            return audio_stage(scene,working,inputs,root/'rir_cache',attempt,dry_override=dry,dry_receipt=dry_receipt),None
        def produce_audio(attempt):
            produced,files=make_audio(attempt)
            return produced if files is None else (produced,files)
        audio_dir,ar,audio_resumed=stage(root,'audio_'+member,{'scene':scene,'native_renderer_inputs':inputs},produce_audio)
        audio_by_factor[factor]=(audio_dir,ar)
        def windows(attempt):
            import soundfile as sf
            audio,sr=sf.read(audio_dir/'scene_foa_full_tail.wav',dtype='float64',always_2d=True)
            if sr!=16000 or audio.shape[1]!=4:raise ValueError('full-history FOA format changed')
            receipts=[];required=[]
            for i,start in enumerate(scene['window_starts_s']):
                wid=f'{row["scene_id"]}__{member}__w{i:02d}';dest=attempt/wid;dest.mkdir()
                first=round(start*15)
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-threads','1',
                    '-i',str(visual_dir/'scene_video.mp4'),'-vf',f'select=between(n\\,{first}\\,{first+149}),setpts=N/(15*TB)',
                    '-frames:v','150','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','18',
                    '-pix_fmt','yuv420p',str(dest/'video.mp4')],check=True)
                lo=round(start*16000);samples=audio[lo:lo+160000]
                if samples.shape!=(160000,4):raise ValueError('window lacks full audio history')
                write_float32_wav(dest/'audio_foa.wav',samples,16000,channel_axis=1)
                write_float32_wav(dest/'audio_mono.wav',samples[:,:1],16000,channel_axis=1)
                media={'video':'video.mp4','audio_foa':'audio_foa.wav','audio_mono':'audio_mono.wav','foa_format':'ACN_WYZX_SN3D_camera'}
                quota_index=row['question_quota_base']+row['members'].index(member)*row['window_count']+i
                questions,answers,check=make_questions(scene,start,wid,visual,appearances,media,
                    quota_index=quota_index,count=row['questions_per_window'])
                write_json(dest/'questions.json',questions);write_json(dest/'private_answers.json',answers)
                evidence={'seed_row':row,'member':member,'start_s':start,'scene':scene,'question_checks':check,
                          'native_visual_evidence':str(visual_dir/'native_visual_evidence.json'),
                          'native_audio_evidence':str(audio_dir/'audio_evidence.json'),
                          'formal_dataset_count':0,'blocking_reasons':list(FORMAL_BLOCKERS)}
                write_json(dest/'evidence.json',evidence)
                receipts.append({'window_id':wid,**validate_question_counts(questions,answers)})
                required += [f'{wid}/{name}' for name in ('video.mp4','audio_foa.wav','audio_mono.wav','questions.json','private_answers.json','evidence.json')]
            return {'windows':receipts,'formal_dataset_count':0},required
        wd,wr,wr_resumed=stage(root,'windows_'+member,{'scene':scene,'appearances':appearances,'quota':row['questions_per_window']},windows)
        members.append({'member':member,'windows_directory':str(wd),'windows':wr['windows'],
                        'resumed_stages':{'visual':visual_resumed,'audio':audio_resumed,'windows':wr_resumed},'audio':ar})
    group=audit_parent_group(row,base,members)
    group_path=root/'counterfactual_audit.json'
    if group_path.exists():
        if read_json(group_path)!=group:raise ValueError('completed counterfactual audit changed')
    else:write_json(group_path,group)
    receipt={'scene_id':row['scene_id'],'status':'candidate_only','formal_dataset_count':0,
            'members':members,'counterfactual_audit':group,'blocking_reasons':list(FORMAL_BLOCKERS),
            'wall_seconds':time.perf_counter()-start_wall,'process_cpu_seconds':time.process_time()-start_cpu,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    if compact:return compact_parent(root,receipt)['result']
    return receipt
