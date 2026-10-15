"""Phase-two visible source bindings, including two registered foot children."""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
from avengine.timeline.v7_visible_sources import build_visible_scene,SCHEMA,point,production_audio_record_views
PHASE2_SCHEMA='v7-visible-registered-pilot-phase2-v1'


def _validate_authored_motion_support(scene, program, requests):
    """Fail before native work when the requested TTS pose cannot be certified.

    The existing dialogue intake verifies only standing speaking humans. Its
    late registry check rejects a seated or walking TTS mesh; recorded speech and
    other moving scene participants do not use that standing-only intake.
    """
    records = {r['sound_asset_id']: r for r in program['records']}
    bindings = {b['event_id']: b for b in program['events']}
    authored = {e['endpoint_id'] for e in scene['events']
                if records[bindings[e['event_id']]['sound_asset_id']].get('source_type') == 'author_tts'}
    unsupported = [(ep, request['action']) for ep, request in requests.items()
                   if ep in authored and request['action'] in {'sit', 'walk'}
                   and not (request['action']=='walk' and request.get('allow_registered_walk_tts'))]
    scene['pending_registered_walk_tts']=[ep for ep,request in requests.items()
        if ep in authored and request.get('allow_registered_walk_tts')]
    if unsupported:
        poses = ', '.join(f'{ep} ({action})' for ep, action in unsupported)
        raise ValueError(
            'authored TTS motion lacks qualified pose lineage: ' + poses +
            '; dialogue intake currently requires standing speaking humans; '
            'use complete recorded VCTK speech for seated speakers')


def _validate_recorded_speech_voice(record, original):
    if (record.get('source_type')=='author_tts' or not record.get('transcript')
            or record.get('speaker_id')!=original.get('speaker_id')
            or record.get('gender')!=original.get('gender') or record['sound_class']!='speech_playback'):
        raise ValueError('recorded sequence must preserve the same admitted whole speaking voice')

def schedule_authored_speech(scene,program,assets,schedules):
    byid={a['asset_id']:a for a in assets};bindings={b['event_id']:b for b in program['events']}
    records={r['sound_asset_id']:r for r in program['records']}
    for aid,starts in schedules.items():
        a=byid[aid];ep=a['endpoint_id'];events=[e for e in scene['events']if e['endpoint_id']==ep]
        if a['kind']!='human' or not events:raise ValueError('authored schedule needs a real speaking human')
        first=events[0];binding=bindings[first['event_id']];record=records[binding['sound_asset_id']]
        if record.get('source_type')!='author_tts' or binding.get('crop_start_sample',0)!=0:
            raise ValueError('authored schedule cannot relabel or crop recorded speech')
        duration=math.ceil(record['header']['sample_count']/1600)
        ticks=[round(float(start)*10)for start in starts]
        if not ticks or any(not math.isfinite(float(t))or float(t)<0 or abs(float(t)*10-round(float(t)*10))>1e-7 for t in starts)or ticks!=sorted(set(ticks))or any(t+duration>300 for t in ticks):
            raise ValueError('each whole authored utterance must fit its real scene schedule')
        doomed={e['event_id']for e in events};scene['events']=[e for e in scene['events']if e['event_id']not in doomed]
        program['events']=[b for b in program['events']if b['event_id']not in doomed]
        for i,tick in enumerate(ticks):
            eid='authored_speech_'+ep+'_'+str(i)
            scene['events'].append({**deepcopy(first),'event_id':eid,'start_tick':tick,'end_tick':tick+duration})
            program['events'].append({**deepcopy(binding),'event_id':eid})

def footstep_walkers_for_row(row):
    motion=row.get('motion_requests',{})
    walkers=[]
    for aid,classes in row.get('source_classes',{}).items():
        if 'footsteps' not in classes:continue
        if classes!=['footsteps']:raise ValueError('one sound source per human: footsteps cannot be mixed with speech or another sound')
        if motion.get(aid,{}).get('action')!='walk':raise ValueError('footsteps require actual requested walking')
        walkers.append(aid)
    if not set(row.get('phase2_include_footsteps_asset_ids',[]))<=set(walkers):
        raise ValueError('one sound source per human: the footstep alias cannot add feet to a speaking or silent walker')
    return walkers

def build_phase2_scene(row,house,records,catalog,semantic_manifest=None):
    if row.get('schema')!=PHASE2_SCHEMA:raise ValueError('explicit phase-two schema required')
    from avengine.timeline.v7_production_inputs import validate_sampling_row
    production_intake=validate_sampling_row(row,catalog)
    motion=row.get('motion_requests',{})
    stub=deepcopy(row);stub['schema']=SCHEMA;stub['phase2_moving_asset_ids']=list(motion);stub.pop('motion_requests',None)
    # The public production interface names this canonical field explicitly.
    # Accept the earlier private pilot alias only by translating it before
    # production compilation, so recorded speech ordering is physically real.
    if 'semantic_offsets_s' in row:
        if 'semantic_start_offsets_s' in row and row['semantic_start_offsets_s']!=row['semantic_offsets_s']:
            raise ValueError('conflicting authored speech timing fields')
        stub['semantic_start_offsets_s']=deepcopy(row['semantic_offsets_s'])
    silent=set(row.get('silent_asset_ids',[]))
    stub['asset_ids']=[aid for aid in row['asset_ids'] if aid not in silent]
    walkers=footstep_walkers_for_row(row)
    footstep_walkers=walkers
    if any(motion.get(aid,{}).get('action')!='walk'for aid in footstep_walkers):raise ValueError('footsteps require actual requested walking')
    if row.get('semantic_scenario_id'):
        stub['asset_ids']=[aid for aid in stub['asset_ids'] if aid not in walkers]
    else:
        for aid in walkers:stub.setdefault('source_classes',{})[aid]=['speech_playback']
    working,scene,program,assets=build_visible_scene(stub,house,records,catalog,semantic_manifest)
    by_id={a['asset_id']:a for a in assets};all_catalog={a['asset_id']:a for a in catalog}
    for aid in row['asset_ids']:
        if aid not in silent and not (row.get('semantic_scenario_id') and aid in walkers):continue
        a=deepcopy(all_catalog[aid]);i=len(assets)+1;ep='voice'+str(i)
        a.update(endpoint_id=ep,native_instance_id=i,silent_decor=aid in silent)
        assets.append(a);by_id[aid]=a
        scene['sources'][ep]={'actor_id':i,'asset_id':aid,'entity_class':a['entity_class'],'species_id':a['species_id'],
            'class_id':None,'content_kind':'silent','mechanism':'visible_silent_bystander','silent_decor':aid in silent,
            'native_visual_kind':'registered_drawable','dry_gain':0.,
            'trajectory':[{'time_s':0.,'position_m':row['listener_m']},{'time_s':30.,'position_m':row['listener_m']}]}
    requests={}
    scene['planned_footsteps']=[]
    from avengine.timeline.v7_audio_sources import filter_admission
    admitted,rejected=filter_admission(production_audio_record_views(records))
    if rejected:raise ValueError('new sound input has rejected records')
    from avengine.timeline.v7_environmental_sources import recording_header,STRICT_CLASS13
    foot_pool=[r for r in admitted if r['sound_class']=='footsteps']
    rng=np.random.default_rng(row['scene_seed']+1147)
    for aid,request in motion.items():
        if aid not in by_id:raise ValueError('motion request lacks a registered scene body')
        a=by_id[aid];ep=a['endpoint_id'];request=deepcopy(request)
        if request['action']=='walk' and not a.get('walk_allowed'):raise ValueError('walking identity lacks roster authority: '+aid)
        if request['action']=='sit' and (a['kind']!='human' or not a.get('sit_allowed')):raise ValueError('sitting identity lacks roster authority')
        requests[ep]=request
        if request.get('animal_motion_admission'):
            scene['sources'][ep]['motion_metadata']=deepcopy(request['animal_motion_admission'])
            scene['sources'][ep]['motion_marker']=request['animal_motion_admission']['motion_marker']
        if aid not in footstep_walkers:continue
        if a['kind']!='human':raise ValueError('footsteps are reserved for walking registered humans')
        if not foot_pool:raise ValueError('admitted footsteps input missing')
        record=recording_header(foot_pool[int(rng.integers(len(foot_pool)))])
        doomed={e['event_id'] for e in scene['events'] if e['endpoint_id']==ep}if aid in walkers else set()
        scene['events']=[e for e in scene['events'] if e['event_id'] not in doomed]
        program['events']=[e for e in program['events'] if e['event_id'] not in doomed]
        a['allowed_classes']=sorted(set(a['allowed_classes'])|{'footsteps'})
        base=scene['sources'][ep]
        if aid in walkers:base.update(content_kind='silent',class_id=None,sound_class=None,mechanism='walking_human_mouth_silent')
        for side in ('L','R'):
            alias=ep+'_foot_'+side
            scene['sources'][alias]={**deepcopy(base),'owner_endpoint_id':ep,'emitter_part':'foot_'+side,
                'content_kind':'footsteps','sound_class':'footsteps','class_id':STRICT_CLASS13.get('footsteps'),
                'mechanism':'walking_human_registered_foot','silent_decor':False,'dry_gain':.6}
        scene['planned_footsteps'].append({'endpoint_id':ep,'asset_id':aid,'sound_asset_id':record['sound_asset_id'],'requested_events':max(6,int(request.get('footstep_events',6)))})
        program['records'].append(record)
    # Whole admitted utterances/effects may repeat on the constructor's clock.
    # This changes physical scheduling only; all samples and native truth remain.
    for aid,starts in row.get('phase2_source_event_schedule_s',{}).items():
        ep=by_id[aid]['endpoint_id'];old=[e for e in scene['events']if e['endpoint_id']==ep]
        if not old:raise ValueError('scheduled physical source has no admitted event')
        bindings={e['event_id']:e for e in program['events']};prototype=deepcopy(old[0]);binding=deepcopy(bindings[prototype['event_id']]);duration=prototype['end_tick']-prototype['start_tick'];ids={e['event_id']for e in old}
        scene['events']=[e for e in scene['events']if e['event_id']not in ids];program['events']=[e for e in program['events']if e['event_id']not in ids]
        for i,start in enumerate(starts):
            lo=round(float(start)*10)
            if abs(lo/10-float(start))>1e-8 or lo<0 or lo+duration>300:raise ValueError('complete source event schedule exceeds original tick clock')
            eid='scheduled_'+ep+'_'+str(i);scene['events'].append({**prototype,'event_id':eid,'start_tick':lo,'end_tick':lo+duration});program['events'].append({**binding,'event_id':eid})
    _validate_authored_motion_support(scene, program, requests)
    # Optional physical planning constraints; frozen QA sector predicates are
    # untouched and independently recompute the actual accepted positions.
    targets=row.get('semantic_azimuth_targets_deg',{})
    if not set(targets)<=set(row['asset_ids']):raise ValueError('azimuth target has no registered body')
    if any(not math.isfinite(float(v)) for v in targets.values()):raise ValueError('nonfinite physical target')
    scene['phase2_semantic_azimuth_targets_deg']=deepcopy(targets)
    scene['phase2_semantic_camera_sector_targets_deg']=deepcopy(row.get('semantic_camera_sector_targets_deg',{}))
    distance_targets=row.get('native_attachment_distance_targets_m',{})
    if not set(distance_targets)<=set(row['asset_ids']):
        raise ValueError('distance proposal target has no registered body')
    if any(not math.isfinite(float(v)) or float(v)<=0 for v in distance_targets.values()):
        raise ValueError('distance proposal must be finite and positive')
    if distance_targets:scene['native_attachment_distance_targets_m']=deepcopy(distance_targets)
    if row.get('phase2_effect_start_offsets_s'):
        starts=row['phase2_effect_start_offsets_s']
        if len(starts)!=6:raise ValueError('six effect starts required')
        authored_eps={a['endpoint_id'] for a in assets if a['kind']=='human'} if row.get('semantic_scenario_id') else set()
        for ep in scene['sources']:
            if ep in authored_eps:continue
            evs=[e for e in scene['events'] if e['endpoint_id']==ep]
            if not evs:continue
            if len(evs)!=6:raise ValueError('effect schedule differs from actual admitted events')
            for index,e in enumerate(evs):
                duration=e['end_tick']-e['start_tick'];lo=round(float(starts[index])*10)
                if lo<0 or lo+duration>300:raise ValueError('complete event does not fit')
                e.update(start_tick=lo,end_tick=lo+duration)
    # Explicit recorded-only speech timing preserves each whole admitted VCTK
    # utterance. The existing authored standing-only QA adapter remains intact.
    for aid,starts in row.get('phase2_recorded_speech_schedule_s',{}).items():
        a=by_id[aid];ep=a['endpoint_id']
        evs=[e for e in scene['events']if e['endpoint_id']==ep]
        if a['kind']!='human' or not evs or any(e['sound_class']!='speech_playback'for e in evs):
            raise ValueError('recorded schedule needs an actual registered human speech source')
        bindings={b['event_id']:b for b in program['events']};records_by_id={r['sound_asset_id']:r for r in program['records']}
        first=evs[0];binding=bindings[first['event_id']]
        if records_by_id[binding['sound_asset_id']].get('source_type')=='author_tts' or binding.get('crop_start_sample',0)!=0 or not binding.get('preserve_full_recorded_clip'):
            raise ValueError('recorded schedule cannot relabel TTS or a cropped utterance')
        doomed={e['event_id']for e in evs};scene['events']=[e for e in scene['events']if e['event_id']not in doomed]
        program['events']=[b for b in program['events']if b['event_id']not in doomed]
        sequence=row.get('phase2_recorded_sound_ids',{}).get(aid,[binding['sound_asset_id']]*len(starts))
        if len(sequence)!=len(starts):raise ValueError('whole recorded sequence length differs from its schedule')
        admitted_by_id={r['sound_asset_id']:r for r in admitted}
        original=records_by_id[binding['sound_asset_id']]
        for i,(start,sid) in enumerate(zip(starts,sequence,strict=True)):
            record=records_by_id.get(sid)
            if record is None:
                record=recording_header(admitted_by_id[sid]);records_by_id[sid]=record;program['records'].append(record)
            _validate_recorded_speech_voice(record,original)
            duration=math.ceil(record['header']['sample_count']/1600)
            lo=round(float(start)*10)
            if lo<0 or lo+duration>300:raise ValueError('whole recorded speech cannot fit')
            eid='recorded_speech_'+ep+'_'+str(i)
            scene['events'].append({**deepcopy(first),'event_id':eid,'start_tick':lo,'end_tick':lo+duration})
            program['events'].append({**deepcopy(binding),'event_id':eid,'sound_asset_id':sid})
    schedule_authored_speech(scene,program,assets,row.get('phase2_authored_speech_schedule_s',{}))
    if row.get('phase2_single_animal_utterance_crop'):
        bindings={b['event_id']:b for b in program['events']}
        records_by_id={r['sound_asset_id']:r for r in program['records']}
        for e in scene['events']:
            if e['sound_class'] not in {'dog_bark','cat_meow'}:continue
            b=bindings[e['event_id']];record=records_by_id[b['sound_asset_id']]
            intervals=record['screens']['automatic_activity']['activity_intervals_samples']
            # One full measured bark/meow with >=25ms source context, rounded
            # upward to the existing 100ms event clock. No waveform alteration.
            lo,hi=intervals[0];crop=max(0,int(lo)-400)
            ticks=math.ceil((int(hi)+400-crop)/1600)
            if 1<=ticks<e['end_tick']-e['start_tick'] and crop+ticks*1600<=record['header']['sample_count']:
                b['crop_start_sample']=crop
                e['end_tick']=e['start_tick']+ticks
                e['measured_single_utterance_crop']={'original_activity_interval_samples':[lo,hi],'original_crop_start_sample':crop,'complete_measured_activity_preserved':True,'waveform_modified':False}
    for aid,limit in row.get('phase2_animal_event_limit',{}).items():
        a=by_id[aid]
        if a['kind']!='animal' or int(limit)<1:raise ValueError('event limit requires a real calling animal')
        evs=[e for e in scene['events']if e['endpoint_id']==a['endpoint_id']]
        if int(limit)>len(evs):raise ValueError('animal event limit exceeds existing actual events')
        removed={e['event_id']for e in evs[int(limit):]}
        scene['events']=[e for e in scene['events']if e['event_id']not in removed]
        program['events']=[e for e in program['events']if e['event_id']not in removed]
    turn_hold=float(row.get('phase2_camera_turn_hold_s',0.))
    if not math.isfinite(turn_hold) or not 0.<=turn_hold<=.6:
        raise ValueError('physical camera turn hold must be finite and in 0 to 0.6 seconds')
    if turn_hold:scene['phase2_camera_turn_hold_s']=turn_hold
    if row.get('native_occlusion_relations'):scene['native_occlusion_relations']=deepcopy(row['native_occlusion_relations'])
    if row.get('category_focus_class'):scene['production_scenario']['category_focus_class']=row['category_focus_class']
    if production_intake:scene['production_sampling_intake']=production_intake
    if row.get('scene_demand'):scene['scene_demand']=deepcopy(row['scene_demand'])
    scene['phase2_profile']=True
    scene['production_scenario'].update(motion_requests=requests,motion_floor_polygon=working['room']['floor_polygon'])
    scene['admitted_sound_class_domain']=sorted({r['sound_class'] for r in admitted})
    scene['source_motion_policy']='registered admitted pose/foot/head nodes, actual native per-frame readbacks'
    scene['sampling_limitations']=['private pilot; no training-table export']
    used={b['sound_asset_id'] for b in program['events']}|{p['sound_asset_id'] for p in scene['planned_footsteps']}
    program['records']=[r for r in program['records'] if r['sound_asset_id'] in used]
    validate_phase2_bindings(scene,assets,allow_pending_feet=True)
    return working,scene,program,assets

def source_owners(scene,assets):
    by_ep={a['endpoint_id']:a for a in assets};owners={}
    for ep,spec in scene['sources'].items():
        base=spec.get('owner_endpoint_id',ep);a=by_ep.get(base)
        if a is None or spec.get('asset_id')!=a['asset_id'] or spec.get('actor_id')!=a['native_instance_id']:
            raise ValueError('source has no matching registered native entity')
        if spec.get('native_visual_kind')!='registered_drawable':raise ValueError('invisible source forbidden')
        if ep!=base and (spec.get('emitter_part') not in {'foot_L','foot_R'} or a['kind']!='human'):
            raise ValueError('unregistered secondary emitter')
        owners[ep]=a
    if not set(by_ep)<=set(scene['sources']):raise ValueError('registered native entity source metadata missing')
    return owners

def validate_phase2_bindings(scene,assets,allow_pending_feet=False):
    if not scene.get('phase2_profile'):raise ValueError('phase-two source authority missing')
    owners=source_owners(scene,assets)
    if len({a['native_instance_id'] for a in assets})!=len(assets):raise ValueError('duplicate physical body identity')
    footsteps=0
    for e in scene['events']:
        ep=e['endpoint_id'];a=owners.get(ep)
        if a is None or a.get('silent_decor') or e['sound_class'] not in a['allowed_classes']:
            raise ValueError('event class has no permitted physical registered source')
        if e['source_id']!=a['native_instance_id']:raise ValueError('event/body identity mismatch')
        if e['sound_class']=='footsteps':
            spec=scene['sources'][ep];req=scene['production_scenario']['motion_requests'].get(spec.get('owner_endpoint_id'))
            if not req or req['action']!='walk' or spec.get('emitter_part') not in {'foot_L','foot_R'}:
                raise ValueError('footsteps must come from registered walking human feet')
            footsteps+=1
    if scene.get('planned_footsteps') and not footsteps and not allow_pending_feet:
        raise ValueError('planned footsteps were not resolved onto actual foot contact intervals')
    return {'status':'pass','events':len(scene['events']),'registered_entities':len(assets),'invisible_sources':0,'footsteps_events':footsteps}

def resolve_footstep_events(scene,program,visual):
    """Use observed stance intervals, never sound during a planted pause.

    The original admitted effect is cropped once per actual stance interval.
    Source samples, crop offsets and activity are retained in the dry receipt.
    """
    records={r['sound_asset_id']:r for r in program['records']};proof=[]
    for planned in scene.get('planned_footsteps',[]):
        ep=planned['endpoint_id'];record=records[planned['sound_asset_id']]
        frames=visual['native_motion_admissions'][ep]['native_30Hz_readbacks']
        source_activity=record['screens']['automatic_activity']['activity_intervals_samples']
        crop=int(source_activity[0][0])
        if crop+1600>record['header']['sample_count']:raise ValueError('footstep crop lacks one measured tick')
        chosen=[]
        for tick in range(1,298):
            lo=tick*3;hi=(tick+1)*3
            if hi>=len(frames):break
            for side in ('L','R'):
                alias=ep+'_foot_'+side
                samples=[next(v for v in f['attached_emitter_readbacks'] if v['endpoint_id']==alias) for f in frames[lo:hi+1]]
                if all(f['moving'] for f in frames[lo:hi+1]) and all(v['in_contact'] and abs(v['floor_delta_m'])<.008 for v in samples):
                    if chosen and tick-chosen[-1][0]<5:continue
                    chosen.append((tick,side,samples))
                    break
        if not chosen:raise ValueError('walking motion has no actual admitted stance interval for footsteps')
        selection=chosen
        planned['event_unit']='footstep_bout'
        planned['maximum_adjacent_onset_gap_s']=.6
        planned['all_actual_step_contacts_retained']=True
        for tick,side,samples in selection:
            eid='footstep_event_'+str(len(scene['events'])).zfill(4);alias=ep+'_foot_'+side;spec=scene['sources'][alias]
            scene['events'].append({'event_id':eid,'endpoint_id':alias,'source_id':spec['actor_id'],'class_id':spec['class_id'],'sound_class':'footsteps','start_tick':int(tick),'end_tick':int(tick+1),'mechanism':spec['mechanism']})
            program['events'].append({'event_id':eid,'sound_asset_id':record['sound_asset_id'],'crop_start_sample':crop})
            proof.append({'event_id':eid,'native_body_endpoint_id':ep,'native_foot':side,'start_tick':int(tick),'crop_start_sample':crop,'actual_contact_frame_indices':list(range(tick*3,(tick+1)*3+1)),'contact_worst_error_m':max(abs(v['floor_delta_m']) for v in samples),'emitter_source':'actual original-skinned sole vertices and native attached foot child'})
    return {'status':'pass','events':proof,'thresholds_changed':False,'recording_cropped_not_repeated_or_time_stretched':True,
            'annotation_event_unit':'footstep_bout','maximum_adjacent_onset_gap_s':.6,
            'raw_step_count':len(proof),'bout_annotation_requires_actual_native_wet_support':True}

def audit_phase2_bindings(scene,assets,visual,allow_pending_feet=False):
    validate_phase2_bindings(scene,assets,allow_pending_feet=allow_pending_feet);owners=source_owners(scene,assets);base={a['endpoint_id']:a for a in assets}
    errors=[];worst=0.;count=0;visible={a['asset_id']:0 for a in assets};active={e['endpoint_id'] for e in scene['events']}
    for frame in visual['frames']:
        actual={r['endpoint_id']:r for r in frame['actors']}
        if set(actual)!=set(base):errors.append('registered native entity disappeared')
        for ep,a in base.items():
            r=actual.get(ep)
            if not r:continue
            if r['asset_id']!=a['asset_id'] or r['actor_id']!=a['native_instance_id']:errors.append('native registered identity changed')
            if r.get('visible_pixels',0)>=100:visible[a['asset_id']]+=1
        for ep in active:
            a=owners[ep];spec=scene['sources'][ep];owner=spec.get('owner_endpoint_id',ep);r=actual.get(owner)
            if r is None:errors.append('invisible event source');continue
            if ep==owner:
                local=r.get('emitter_local_m_frame',a['emitter_local_m']);observed=r['emitter_position_m']
            else:
                nodes={v['endpoint_id']:v for v in r.get('attached_emitter_readbacks',[])}
                if ep not in nodes:errors.append('foot child missing');continue
                child=nodes[ep];local=child['local_m'];observed=child['position_m']
                if child['native_object_id']!=r['native_object_id']:errors.append('foot child belongs to another body')
            expected=(np.asarray(r['body_to_world_m'])@np.r_[local,1.])[:3]
            err=float(np.linalg.norm(np.asarray(observed)-expected));worst=max(worst,err);count+=1
            if err>.0001:errors.append('native emitter readback exceeds unchanged 0.1mm')
    from avengine.capture.v7_offscreen_layout import coverage_errors
    errors.extend(coverage_errors(scene,assets,visual,visible))
    return {'status':'fail' if errors else 'pass','invisible_sources':sum('invisible' in e for e in errors),'emitter_readbacks':count,'emitter_worst_error_m':worst,'asset_visible_frame_counts':visible,'errors':sorted(set(errors))}


def qualify_registered_walk_tts(scene,visual):
    """Only the actual admitted walking body and attached mouth permit TTS."""
    reports=[]
    for ep in scene.get('pending_registered_walk_tts',[]):
        record=visual.get('native_motion_admissions',{}).get(ep,{})
        readbacks=record.get('native_30Hz_readbacks',[])
        if (record.get('status')!='pass' or len(readbacks)!=901
                or record.get('final_complete_scene_recheck',{}).get('status')!='pass'
                or not record.get('lineage',{}).get('master_sha256')
                or any(not r.get('native_emitter_present') or r.get('asset_id')!=scene['sources'][ep]['asset_id'] for r in readbacks)):
            raise ValueError('walking TTS needs complete actual native body, original skin and attached mouth admission')
        reports.append({'endpoint_id':ep,'status':'pass','native_root_and_mouth_readbacks':len(readbacks),
                        'motion_master_sha256':record['lineage']['master_sha256'],'footsteps_emitted':False})
    return {'status':'pass','walking_tts_sources':reports,'source_waveforms_or_QA_predicates_changed':False}
