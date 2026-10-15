"""Recorded production sampling with actual registered standing identities.

Profile requests, realized native sources and delivery counts remain separate.
No unknown action, missing sound class or requested quota is silently admitted.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math
import numpy as np
from avengine.timeline.v7_audio_sources import PROGRAM_SCHEMA,filter_admission

REGISTERED_INPUT_SCHEMA='v7-recorded-registered-standing-v1'

def choose_assets(catalog,row):
    n=row['production_scenario']['speaker_count'];count=max(3,n+1)
    ids=row['production_scenario'].get('asset_ids')
    by_id={x['asset_id']:x for x in catalog}
    if ids: assets=[deepcopy(by_id[x]) for x in ids]
    else:
        gender='female' if row['production_scenario']['gender']=='F' else 'male'
        bodies=sorted({x['body_identity_id'] for x in catalog if x['dialogue_voice_binding_eligible'] and x['visual_gender']==gender and x['color']=='blue'})
        rng=np.random.default_rng(row['seeds']['appearance']);body=str(rng.choice(bodies))
        # Shirt colour must not encode speaking order or gender. The three
        # referable (hue-verifiable) colours are shuffled over the first three
        # actors; silent actors get a random gender; extra actors draw white or
        # black at random. Index-fixed colours made blue/yellow the speakers and
        # green/black female, white male (owner 2026-10-12).
        referable=[str(c) for c in rng.permutation(['blue','yellow','green'])]
        production_extra_colors=[str(c) for c in rng.permutation(['white','black'])] if row.get('production_sampling') else None
        assets=[];speech_genders=[]
        for i in range(count):
            if i<min(n,2):g=gender
            elif i<n:
                g=min(('female','male'),key=lambda k:(speech_genders.count(k),k))
            else:g=str(rng.choice(['female','male']))
            if i<n:speech_genders.append(g)
            available=sorted({x['body_identity_id'] for x in catalog if x['dialogue_voice_binding_eligible'] and x['visual_gender']==g and x['color']=='blue'})
            selected=body if i<2 else str(rng.choice(available))
            if production_extra_colors is not None and i>=5:
                pool=sorted((x for x in catalog if x['dialogue_voice_binding_eligible'] and x['visual_gender']==g
                             and x['color']=='original' and not x.get('controlled_shirt')),
                            key=lambda x:x['asset_id'])
                if not pool:raise ValueError('more than five people requires an admitted uncolored professional body')
                assets.append(deepcopy(pool[int(rng.integers(len(pool)))]))
                continue
            color=referable[i] if i<3 else (production_extra_colors[i-3] if production_extra_colors is not None else str(rng.choice(['white','black'])))
            assets.append(deepcopy(next(x for x in catalog if x['body_identity_id']==selected and x['color']==color)))
    additional=row['production_scenario'].get('additional_source_assets',[])
    for aid in additional:
        a=deepcopy(by_id[aid])
        if a.get('entity_class')!='articulated_animal' or a.get('species_id')!='dog' or not a.get('static_registered_reference_pass'):raise ValueError('additional source lacks explicit admitted dog intake')
        assets.append(a)
    for i,a in enumerate(assets):
        a['endpoint_id']='voice'+str(i+1)
        a['native_instance_id']=i+1
        if a['color'] in {'blue','yellow','green'}:
            a['appearance']={'top_color':a['color']};a['appearance_zh']={'blue':'蓝色上衣的人','yellow':'黄色上衣的人','green':'绿色上衣的人'}[a['color']]
    return assets

def _placements(layout,n):
    values=deepcopy(layout['valid'])
    if n>len(values):
        # New positions are candidate placements, not inherited admissions.
        # Every registered complete mesh is probed in its actual episode.
        from shapely.geometry import shape,Point
        polygon=shape(layout['room_floor_polygon'])
        floor=float(np.median([p['position_m'][1]-1.3 for p in values]))
        x0,z0,x1,z1=polygon.bounds
        candidates=[]
        for x in np.arange(x0+.3,x1,.65):
            for z in np.arange(z0+.3,z1,.65):
                if polygon.covers(Point(x,z)) and min(np.linalg.norm(np.array([x,z])-np.array([p['position_m'][0],p['position_m'][2]])) for p in values)>.7:
                    d=math.hypot(x-layout['listener_m'][0],z-layout['listener_m'][2]);candidates.append((abs(d-1.87),x,z))
        for _,x,z in sorted(candidates):
            if min(np.linalg.norm(np.array([x,z])-np.array([p['position_m'][0],p['position_m'][2]])) for p in values)<.65:continue
            yaw=math.degrees(math.atan2(layout['listener_m'][0]-x,-(layout['listener_m'][2]-z)))
            values.append({'position_m':[float(x),floor+1.3,float(z)],'body_yaw_deg':yaw,'status':'unqualified_candidate_requires_native_episode'})
            if len(values)>=n:break
    if len(values)<n:raise ValueError('room has too few separated candidate placements')
    return values[:n]

def semantic_record(sound,gender):
    from avengine.qa.v7_recorded_audio import natural_speech_blockers
    record=deepcopy(sound);record.update(source_type='author_tts',sound_class='speech_playback',gender=gender,original_id=sound['sound_asset_id'],normalized_path=sound['path'],duration_s=sound['synthesis_readback']['duration_s'],header={'sample_rate_hz':16000,'sample_count':sound['synthesis_readback']['sample_count'],'channels':1},native_placement_readback={'visual_gender':'female' if gender=='F' else 'male','scope':'source-to-registered-gender plan; episode attachment separately measured'})
    blockers=natural_speech_blockers(record)
    if blockers:raise ValueError('natural authored source blocked: '+str(blockers))
    ref=record['voice_reference'];record['speaker_id']=ref['speaker_id'];engine=record['synthesis_readback']['engine']
    record['screens']={'license':{'license':engine['model_license']+' + '+ref['license_spdx'],'original_id':record['original_id'],'source_uri':record['source_uri'],'retained_text':engine['model_license_path']}}
    return record

def build_registered_scene(row,house,records,assets,semantic_manifest=None,*,camera_evidence_policy=None):
    # The selected per-scene room must not alias the caller's frozen house.
    house=deepcopy(house)
    if row['schema']!=REGISTERED_INPUT_SCHEMA or house['house']!=row['house'] or house['split']=='test':raise ValueError('registered production row on train/dev house required')
    if house['acoustic_status']!='pass' or row['room_id'] not in house['checked_room_ids']:raise ValueError('unaccepted physical house/room')
    if {'development':'dev'}.get(house['split'],house['split'])!=row['split']:raise ValueError('house split crossed')
    if house.get('known_exposure') and row['split']!='train':raise ValueError('exposed heldout house')
    admitted,rejected=filter_admission(records)
    if rejected:raise ValueError('registered source pool contains blocked inputs: '+str(rejected[:3]))
    plan=row['production_scenario'];n=plan['speaker_count'];rng=np.random.default_rng(row['seeds']['audio'])
    layout=row['measured_static_layout'];starts=[t/10 for t in row['window_start_ticks'][:row['window_count']]]
    event_starts=[0.,10.,20.] if starts==[0.,5.,10.,15.,20.] else starts
    yaw=float(plan['camera_yaw_deg']);angle=math.radians(yaw)
    # Owner2026-10-15: all registered producers use one fixed camera.
    camera_policy='binding_static'
    if camera_policy not in {'uniform_static','binding_static','binding_sweep','review_sweep_360','pilot_binding_coverage'}:raise ValueError('unsupported camera policy')
    knots=[{'time_s':float(i),'yaw_deg':yaw+(36*i if camera_policy=='review_sweep_360' else 0)} for i in range(31)]
    scene={'scene_id':row['scene_id'],'independent_parent_scene_id':row['scene_id'],'seed':row['scene_seed'],'split':row['split'],'duration_s':30.,'video_fps':15,'sample_rate_hz':16000,'listener_m':layout['listener_m'],'camera_forward':[math.sin(angle),0.,-math.cos(angle)],'camera_yaw_knots':knots,'window_starts_s':starts,'sources':{},'events':[],'production_scenario':plan,'source_motion_policy':'registered standing only; independent objects require separate motion admission','sampling_limitations':['walk/sit/seat pending complete episode admission','unadmitted old sound classes are not generated','moving object episode intake pending','profile requests are not measured realized marginals']}
    if row.get('scene_demand'):scene['scene_demand']=deepcopy(row['scene_demand'])
    placements=_placements(layout,len(assets))
    if camera_policy in {'binding_static','binding_sweep','pilot_binding_coverage'}:
        vectors=[np.asarray(p['position_m'])-np.asarray(layout['listener_m']) for p in placements[:2]]
        forward=sum(v/np.linalg.norm(v) for v in vectors);forward[1]=0.;forward/=np.linalg.norm(forward)
        yaw=float(layout.get('fixed_camera_yaw_deg',math.degrees(math.atan2(forward[0],-forward[2]))+float(plan.get('binding_camera_yaw_offset_deg',0.))))
        a=math.radians(yaw);scene['camera_forward']=[math.sin(a),0.,-math.cos(a)]
        scene['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}]
        scene['camera_sampling']='one fixed camera for the complete30s; owner20261015'
    else:scene['camera_sampling']=camera_policy
    from shapely.geometry import shape,Point
    floor_poly=shape(layout['room_floor_polygon']).buffer(-.3)
    if layout.get('room_furniture_polygon'):floor_poly=floor_poly.difference(shape(layout['room_furniture_polygon']).buffer(.5))
    x0,z0,x1,z1=floor_poly.bounds
    height=float(np.median([p['position_m'][1] for p in layout['valid'][:3]]))
    candidates=[[float(x),height,float(z)] for x in np.arange(x0+.1,x1,.5) for z in np.arange(z0+.1,z1,.5) if floor_poly.covers(Point(x,z)) and math.hypot(x-layout['listener_m'][0],z-layout['listener_m'][2])>.5]
    candidates.sort(key=lambda p:abs(math.hypot(p[0]-layout['listener_m'][0],p[2]-layout['listener_m'][2])-1.87))
    scene['native_placement_candidates']=deepcopy(layout.get('native_candidates')or candidates[:192])
    from avengine.timeline.v7_camera_evidence import configure_camera_evidence
    configure_camera_evidence(scene,row,camera_evidence_policy)
    if plan.get('motion_requests'):scene['production_scenario']['motion_floor_polygon']=layout['room_floor_polygon']
    for i,(a,p) in enumerate(zip(assets,placements,strict=True)):
        ep=a['endpoint_id'];g='F' if a['visual_gender']=='female' else 'M'
        scene['sources'][ep]={'actor_id':i+1,'class_id':0 if g=='F' else 1,'content_kind':'speech' if i<n else 'silent','body_yaw_deg':p['body_yaw_deg'],'mechanism':'human_live_recorded','visual_gender':a['visual_gender'],'dry_gain':1.,'trajectory':[{'time_s':0.,'position_m':p['position_m']},{'time_s':30.,'position_m':p['position_m']}]}
    for a in assets:
        if a.get('entity_class')=='articulated_animal':
            from avengine.timeline.v7_environmental_sources import STRICT_CLASS13
            scene['sources'][a['endpoint_id']].update(class_id=STRICT_CLASS13.get(a['sound_class']),content_kind='recorded_animal',mechanism='animal_live_recorded',entity_class='articulated_animal',species_id=a['species_id'])
    program={'schema':PROGRAM_SCHEMA,'records':[],'events':[]};by_id={};used_voices=set();busy=[]
    def add(ep,when,record,whole=True):
        lo=round(max(0.,when)*10);duration=record['header']['sample_count']/16000 if whole else min(2.,record['duration_s'])
        ticks=math.ceil(duration*10);hi=lo+ticks
        if hi>300 or any(e==ep and lo<b and hi>a for e,a,b in busy):raise ValueError('complete source schedule conflicts')
        sid=f'event_{len(scene["events"]):04d}';spec=scene['sources'][ep]
        scene['events'].append({'event_id':sid,'endpoint_id':ep,'source_id':spec['actor_id'],'class_id':spec['class_id'],'start_tick':lo,'end_tick':hi,'mechanism':spec['mechanism']})
        program['events'].append({'event_id':sid,'sound_asset_id':record['sound_asset_id'],'crop_start_sample':0,**({'preserve_full_recorded_clip':True} if whole and record.get('source_type')!='author_tts' else {})})
        by_id[record['sound_asset_id']]=record;busy.append((ep,lo,hi))
    if plan.get('semantic_scenario_id'):
        if not semantic_manifest:raise ValueError('authored scene missing actual batch manifest')
        sounds=[s for s in semantic_manifest['sounds'] if s['scenario_id']==plan['semantic_scenario_id']]
        subset=plan.get('semantic_sound_subset_ids')
        if subset is not None:
            if len(subset)<2 or len(set(subset))!=len(subset) or not set(subset)<=set(s['sound_asset_id'] for s in sounds):raise ValueError('authored subset requires distinct admitted roles of the same scenario')
            sounds=[s for s in sounds if s['sound_asset_id'] in subset]
        if len(sounds)<2 or len(sounds)>n:raise ValueError('authored scene role deficit')
        order=plan.get('semantic_sound_ids')
        if order:
            if set(order)!=set(s['sound_asset_id'] for s in sounds):raise ValueError('authored sound selection does not match the actual scenario')
            sounds=[next(s for s in sounds if s['sound_asset_id']==sid) for sid in order]
        for i,s in enumerate(sounds):
            ep=assets[i]['endpoint_id'];g='F' if assets[i]['visual_gender']=='female' else 'M';r=semantic_record(s,g)
            if s['voice_reference']['speaker_id'] in used_voices:raise ValueError('authored roles share voice')
            used_voices.add(s['voice_reference']['speaker_id'])
            offsets=plan.get('semantic_role_start_offsets_s',[2.+j*.9 for j in range(len(sounds))])
            if len(offsets)!=len(sounds) or any(not math.isfinite(float(t)) or not 0.<=float(t)<=7. for t in offsets):raise ValueError('invalid complete authored role schedule')
            for start in event_starts:add(ep,start+float(offsets[i]),r)
    else:
        pool_split='train' if row['split']=='train' else 'eval'
        for i,a in enumerate(assets[:n]):
            g='F' if a['visual_gender']=='female' else 'M';pool=[r for r in admitted if r.get('dataset')=='VCTK-Corpus 0.92' and r.get('legacy_library_split')==pool_split and r.get('gender')==g and .8<=r['duration_s']<=2.5]
            voices=sorted({r['speaker_id'] for r in pool}-used_voices)
            if not voices:raise ValueError('distinct complete admitted voice deficit')
            speaker=str(rng.choice(voices));used_voices.add(speaker);clips=[r for r in pool if r['speaker_id']==speaker]
            ep=a['endpoint_id']
            for start in event_starts:
                times=[start-.8,start+2.,start+5.,start+7.5] if i==0 else [start+.5+(i-1)*.4,start+5.+(i-1)*.4]
                for when in times:
                    candidates=[r for r in clips if not any(e==ep and round(max(0.,when)*10)<b and round(max(0.,when)*10)+math.ceil(r['duration_s']*10)>a for e,a,b in busy)]
                    if not candidates:raise ValueError('speaker lacks a complete utterance fitting this activity slot')
                    add(ep,when,candidates[int(rng.integers(len(candidates)))])
    for animal in [a for a in assets if a.get('entity_class')=='articulated_animal']:
        from avengine.timeline.v7_environmental_sources import recording_header
        pool=[r for r in admitted if r.get('sound_class')==animal['sound_class']and r.get('duration_s',r.get('readback',{}).get('duration_s',0))>=2.]
        if not pool:raise ValueError('admitted recorded dog source deficit')
        r=recording_header(pool[int(rng.integers(len(pool)))])
        for start in event_starts:
            for offset in [.2,2.6,5.2,7.7]:add(animal['endpoint_id'],start+offset,r,whole=False)
    # Laughter is a recorded non-speech mouth emission; it is not a speaker.
    # No clapping/footstep/instrument capability is invented for standing people.
    from avengine.timeline.v7_environmental_sources import attach_environmental_program
    program['records']=list(by_id.values())
    attach_environmental_program(scene,program,plan,admitted,rng,starts)
    by_id.update({r['sound_asset_id']:r for r in program['records']})
    ns=plan.get('non_speech_class','laughter')
    if ns is not None:scene['admitted_sound_class_domain']=sorted({r['sound_class']for r in admitted if r.get('dataset')!='VCTK-Corpus 0.92'})
    if ns is None:
        if n==0 and not plan.get('environment_sound_classes'):raise ValueError('zero-speaking window still requires actual non-speech sound')
        program['records']=list(by_id.values());working=deepcopy(house);working['room']=next(r for r in house['checked_rooms'] if r['room_label']==row['room_id']);working['room']['placement']=deepcopy(working['room']['placement']);working['room']['placement']['camera_m']=scene['listener_m']
        return working,scene,program
    if ns!='laughter':raise ValueError('this standing profile admits only mouth-attached recorded laughter; other mechanisms need native object intake')
    pool=[r for r in admitted if r.get('sound_class')==ns and r.get('duration_s',r.get('readback',{}).get('duration_s',0))>=2.]
    if not pool:raise ValueError('admitted non-speech mouth-source deficit')
    ep=assets[2 if n<=2 else n]['endpoint_id'];scene['sources'][ep].update(class_id=4,content_kind='laughter',mechanism='human_live_recorded_laughter')
    r=deepcopy(pool[int(rng.integers(len(pool)))]);r['duration_s']=r.get('duration_s',r.get('readback',{}).get('duration_s'));r['header']={'sample_rate_hz':16000,'sample_count':r.get('readback',{}).get('sample_count',round(r['duration_s']*16000)),'channels':1}
    for start in starts:
        for offset in ([.2,2.6,5.2,7.7] if n==0 else [.2]):add(ep,start+offset,r,whole=False)
    program['records']=list(by_id.values());working=deepcopy(house);working['room']=next(r for r in house['checked_rooms'] if r['room_label']==row['room_id']);working['room']['placement']=deepcopy(working['room']['placement']);working['room']['placement']['camera_m']=scene['listener_m']
    return working,scene,program
