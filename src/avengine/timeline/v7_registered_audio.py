"""Recorded alpha buses, static native RIRs and time-varying camera FOA.

No moving human or RIR interpolation is certified by this static profile.
Protocol coordinates are an explicit rigid transformation of measured native
mouths into a fixed reference camera, using observed 100 ms native camera
poses. House/RIR physics and the authority's predicates remain unchanged.
"""
from __future__ import annotations
from copy import deepcopy
import math,time
from pathlib import Path
import numpy as np
import soundfile as sf
from avengine.timeline.continuous_pilot import PositionRIRCache,write_json
from avengine.timeline.continuous_scene import partitioned_convolve,angular_distance
from avengine.capture.v7_alpha_native import rotate_world_foa
from avengine.spatial_audio.audio import write_float32_wav


def phase2_ray_multiplier(scene,dry):
    active_moving=[spec for ep,spec in scene['sources'].items() if np.any(dry[ep]) and any(
        not np.array_equal(k['position_m'],spec['trajectory'][0]['position_m']) for k in spec['trajectory'])]
    if not scene.get('phase2_profile') or not active_moving:return 1
    # Preserve the qualified human budget; fresh contexts prevent native
    # history dependence without changing any energy or direction criterion.
    return 9

def render_alpha_audio(scene,house,inputs,cache_directory,attempt,dry,receipt,camera_frames):
    from tools.timeline.pilot_continuous_scenes import native_renderer
    from avengine.timeline.v7_audio_sources import validate_dry_receipt
    from avengine.acoustics.house_acceptance_metrics import foa_direction
    validate_dry_receipt(receipt);start=time.perf_counter();cpu=time.process_time()
    effective_inputs=inputs;sampling_profile=None
    multiplier=phase2_ray_multiplier(scene,dry)
    if multiplier>1:
        # Same backend/materials and absolute gates, with more native rays.
        # Baseline same-position repeats exceed the 1dB octave criterion;
        # step refinement alone cannot remove that sampling variance.
        effective_inputs=deepcopy(inputs)
        original=deepcopy(inputs['simulation'])
        for key in ('direct_ray_count','indirect_ray_count','source_ray_count'):
            effective_inputs['simulation'][key]=multiplier*int(original[key])
        cache_directory=Path(cache_directory).with_name(Path(cache_directory).name+f'_phase2_native_rays_x{multiplier}')
        sampling_profile={'revision':f'same-native-engine-rays-x{multiplier}-all-reference-gates-v2',
            'baseline_simulation':original,'effective_simulation':effective_inputs['simulation'],
            'ray_counts_multiplier':multiplier,'thresholds_changed':False,'noise_budget_refitted':False}
    renderer,identity=native_renderer(effective_inputs,house)
    if sampling_profile is not None:
        from avengine.timeline.v7_phase2_fresh_rir import FreshNativeContextRenderer,REVISION as CONTEXT_REVISION
        renderer=FreshNativeContextRenderer(effective_inputs,house,identity,renderer.setup_report,attempt/'fresh_native_rirs')
        identity={**identity,'phase2_context_policy':CONTEXT_REVISION}
        cache_directory=Path(str(cache_directory)+'_fresh_context_v1')
        sampling_profile.update(context_policy=CONTEXT_REVISION,primary_and_reference_processes_independent=True)
    simulation=identity['simulation']
    if simulation['thread_count']!=1 or simulation['diffraction'] is not True or simulation['max_diffraction_order']!=10:
        raise ValueError('alpha acoustic recipe must keep the calibrated native one-thread diffraction setup')
    write_json(attempt/'native_RLR_setup.json',renderer.setup_report)
    cache=PositionRIRCache(cache_directory,renderer,identity);stems={};directions=[];plans={};wet_support={}
    from avengine.timeline.v7_wet_support import event_wet_support,WET_SUPPORT_REVISION
    for ep,spec in scene['sources'].items():
        if not np.any(dry[ep]):
            plans[ep]={'RIR_count':0,'reason':'visible silent bystander'};continue
        trajectory=spec['trajectory'];point=np.asarray(trajectory[0]['position_m'])
        if any(not np.array_equal(k['position_m'],point) for k in trajectory):
            if scene.get("phase2_profile"):
                from avengine.timeline.v7_phase2_motion_audio import moving_stem_phase2 as moving_stem
            else:
                from avengine.timeline.v7_registered_motion_audio import moving_stem
            stems[ep],motion,observed=moving_stem(spec,scene,effective_inputs,cache,dry[ep],attempt,ep,
                event_bindings=[b for b in receipt['events'] if b['endpoint_id']==ep])
            plans[ep]={'RIR_count':motion['RIR_count'],'motion':'admitted native 30Hz source; midpoint-qualified emission-aligned crossfade','moving_energy':motion}
            directions.extend(observed)
            wet_support.update(motion.get('event_wet_support',{}))
            continue
        rir=cache.get(point);distance=float(np.linalg.norm(point-np.asarray(scene['listener_m'])))
        direction=foa_direction(rir,scene["listener_m"],point,scene["camera_forward"],sample_rate=16000)
        target=(point-np.asarray(scene['listener_m']))/distance
        observed=direction['check']['foa_world_direction']
        if observed is None:raise ValueError('native FOA direct-direction readback missing')
        error=float(direction['check']['angle_deg'])
        directions.append({'endpoint_id':ep,'native_mouth_m':point.tolist(),'distance_m':distance,
            'foa_readback':direction,'expected_world_direction':target.tolist(),'angular_error_deg':error,'passed_5deg':error<=5.})
        if error>5.:
            write_json(attempt/'direction_rejection.json',{'status':'blocked','directions':directions,'unchanged_gate_deg':5.})
            cache.db.close()
            raise ValueError('alpha native FOA direction exceeds frozen 5deg criterion')
        stems[ep]=partitioned_convolve(dry[ep],[rir],[0]);plans[ep]={'RIR_count':1,'motion':'static native MJaw'}
        wet_support.update(event_wet_support(dry[ep],[b for b in receipt['events'] if b['endpoint_id']==ep],[rir]))
    if not stems:raise ValueError('alpha scene needs actual recorded sound')
    length=max(x.shape[1] for x in stems.values());world=np.zeros((4,length))
    for x in stems.values():world[:,:x.shape[1]]+=x
    camera=rotate_world_foa(world,camera_frames)
    before=world[[3,1,2]]/math.sqrt(3);after=camera[[3,1,2]]
    energy_error=float(np.max(np.abs(np.sum(before*before,axis=0)-np.sum(after*after,axis=0))))
    if energy_error>1e-10:raise ValueError('camera FOA rotation failed vector-energy conservation')
    write_float32_wav(attempt/'scene_foa_full_tail.wav',camera,16000,channel_axis=0)
    result={'status':'pass','static_only':not any('moving_energy'in p for p in plans.values()),'moving_energy_gate':'pass' if any('moving_energy'in p for p in plans.values()) else 'not_applicable_no_moving_sources',
        'directions':directions,'camera_rotation_energy_absolute_error':energy_error,
        'RIR_plan':plans,'RIR_cache':cache.stats.copy(),'dry_assets_receipt':receipt,
        'event_wet_support':wet_support,'wet_support_revision':WET_SUPPORT_REVISION,
        'wet_support_complete':set(wet_support)=={e['event_id'] for e in scene['events']},
        'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,
        'format':'ACN_WYZX_SN3D_camera','full_emission_history_and_tail':True}
    if sampling_profile is not None:
        result['phase2_native_sampling_profile']=sampling_profile
        result['fresh_native_process_evidence']=renderer.process_evidence
    write_json(attempt/'audio_evidence.json',result);cache.db.close()
    return result,['scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json']


def complete_native_occluder_targets(visual):
    """Retain fully measured target streams; unknown peers stay unavailable."""
    rows=visual.get('occluder_evidence',{}).get('frame_records',[])
    count=visual.get('frame_count',len(visual.get('frames',[])))
    groups={}
    for row in rows:
        actor=str(row.get('target_instance_id',row.get('actor_id','')))
        groups.setdefault(actor,[]).append(row)
    complete={actor for actor,track in groups.items() if actor and len(track)==count
        and {r.get('frame_index') for r in track}==set(range(count))
        and all(r.get('unknown_occluder_pixels')==0 for r in track)}
    return [deepcopy(r) for r in rows if str(r.get('target_instance_id',r.get('actor_id',''))) in complete],sorted(complete)


def reviewed_occluder_display_label(appearance,identity):
    if appearance.get('label_en'):return appearance['label_en']
    value=str(appearance.get('value','')).strip()
    kind=identity.get('entity_class')
    if kind=='articulated_human':
        from avengine.qa.unified_catalog import _appearance_phrases
        return _appearance_phrases({'value':value})[0]
    noun=identity.get('species_id') if kind=='articulated_animal' else kind
    noun=str(noun or 'registered object').replace('_',' ')
    return 'the '+value.replace('_',' ')+' '+noun


def source_qa_evidence(scene,program,dry_receipt,visual,foa_path,audio_evidence,*,question_episode_id,window_start_s=None):
    by_id={r['sound_asset_id']:r for r in program['records']};bindings={b['event_id']:b for b in dry_receipt['events']}
    events={};protocol_events=[]
    # The dry source's original activity clock is shifted only by actual
    # placement; padding never invents speech or new utterances.
    for event in scene['events']:
        binding=bindings[event['event_id']];record=by_id[binding['sound_asset_id']];offset=binding['output_start_sample']
        intervals=record.get('screens',{}).get('automatic_activity',{}).get('activity_intervals_samples') or record.get('source_normalization',{}).get('measured_activity',{}).get('source_activity_intervals_samples')
        if intervals is None:raise ValueError('source lacks measured activity')
        actual=[]
        for lo,hi in intervals:
            crop=binding.get('crop_start_sample',0)
            a=max(offset+lo-crop,binding['output_start_sample']);b=min(offset+hi-crop,binding['output_end_sample_exclusive'])
            if b>a:actual.append([int(a),int(b)])
        events[event['event_id']]={'sound_asset_id':record['sound_asset_id'],'transcript':record.get('transcript',''),
            'event_unit':'utterance' if record.get('transcript') else 'recorded_event','event_segmentation':record.get('segmentation',{'method':'complete admitted utterance','original_id':record.get('original_id')}),
            'source_qc':{'status':'pass','verdict':'pass','full_content_preserved':binding['semantic_content_complete'],
                'source':record['path'],'license':record['screens']['license']['license'],'original_id':record.get('original_id',record['screens']['license']['original_id'])},
            'activity_status':'observed','activity_intervals_samples_global':actual,
            'emission_mechanism':event.get('mechanism','human_live_recorded'), 'sound_class':record.get('sound_class'), 'native_endpoint_id':event['endpoint_id']}
        wet=audio_evidence.get('event_wet_support',{}).get(event['event_id']) if audio_evidence.get('wet_support_complete') is True else None
        if wet:
            events[event['event_id']]['wet_intervals_global_s']=deepcopy(wet['wet_intervals_global_s'])
            events[event['event_id']]['wet_support_readback']=deepcopy(wet)
        motion_meta=scene['sources'][event['endpoint_id']].get('motion_metadata')
        if motion_meta:events[event['event_id']]['motion_metadata']=deepcopy(motion_meta)
        if window_start_s is not None:
            lo=round(window_start_s*16000);hi=lo+160000
            whole=binding['output_start_sample']>=lo and binding['output_end_sample_exclusive']<=hi
            events[event['event_id']]['source_qc']['whole_utterance_in_window']=whole
            if not whole:
                events[event['event_id']].pop('transcript',None)
        ticks=set()
        for lo,hi in actual:ticks.update(range(lo//1600,math.ceil(hi/1600)))
        if ticks:
            ordered=sorted(ticks);start=previous=ordered[0]
            for tick in ordered[1:]+[ordered[-1]+2]:
                if tick!=previous+1:
                    protocol_events.append({'endpoint_id':event['endpoint_id'],'source_id':event['source_id'],
                        'class_id':event['class_id'],'start_tick':start,'end_tick':previous+1,'mechanism':event.get('mechanism','human_live_recorded')})
                    start=tick
                previous=tick
    protocol=deepcopy(scene);protocol['events']=[e for e in protocol_events if isinstance(e['class_id'],int) and 0<=e['class_id']<13]
    protocol['outside_class13_events']=[e for e in protocol_events if e not in protocol['events']]
    protocol['class13_scope']='only declared taxonomy targets; full audio program and class admission remain retained'
    # All source positions are static. Native camera poses at 100 ms are
    # measured by the renderer, then used in this exact change of coordinates.
    reference=math.degrees(math.atan2(scene['camera_forward'][0],-scene['camera_forward'][2]))
    cameras=visual['protocol_camera_readbacks']
    listener=np.asarray(scene['listener_m'])
    for ep,spec in protocol['sources'].items():
        from avengine.timeline.v7_registered_motion_audio import position_at
        track=[]
        for row in cameras:
            point=position_at(scene['sources'][ep]['trajectory'],row['time_s']);delta=point-listener
            angle=math.radians(row['yaw_deg']-reference);c,s=math.cos(angle),math.sin(angle)
            rotation=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
            track.append({'time_s':row['time_s'],'position_m':(listener+rotation@delta).tolist()})
        spec['trajectory']=track
    protocol['coordinate_system']='measured_native_camera_track_transformed_into_fixed_reference_basis'
    protocol['resolution_wh']=list(reversed(visual['renderer']['resolution_hw']))
    protocol['hfov_deg']=visual['renderer']['hfov_deg']
    proof={'events':events,'actor_profiles':{str(a['actor_id']):{'entity_class':scene['sources'].get(a['endpoint_id'],a).get('entity_class','articulated_human'),'species_id':scene['sources'].get(a['endpoint_id'],a).get('species_id','human'),'emission_mechanism':scene['sources'].get(a['endpoint_id'],{}).get('mechanism','silent_decor'),'native_object_id':a['native_object_id']} for a in visual['frames'][0]['actors']},
        'camera_frame_readbacks':visual['camera_frame_readbacks'],'protocol_scene':protocol,
        'question_episode_id':question_episode_id,'question_seed':str(scene['seed']),
        'foa_readback':{'path':str(foa_path),'format':'ACN_WYZX_SN3D_camera',
            'spatial_review':{'status':'pass','source':'self-contained native RIR direct-direction and camera rotation receipt',
                'directions':audio_evidence['directions'],'camera_frames':len(visual['camera_frame_readbacks'])}}}
    for actor in visual['frames'][0]['actors']:
        if actor.get('entity_class') in {'static_animal','static_object'}:
            spec=scene['sources'].get(actor['endpoint_id'],{'mechanism':'silent_decor'})
            proof['actor_profiles'][str(actor['actor_id'])]={'entity_class':actor['entity_class'],'species_id':actor.get('species_id'),
                'emission_mechanism':spec['mechanism'],'native_object_id':actor['native_object_id'],'visual_binding':'actual registered native UE mesh'}
    for ep,spec in scene['sources'].items():
        if spec.get('native_visual_kind')=='audio_only':
            proof['actor_profiles'][str(spec['actor_id'])]={'entity_class':'environmental_audio_point','species_id':None,'emission_mechanism':spec['mechanism'],'visual_binding':'none'}
    proof['window_start_s']=window_start_s
    occluder_frames=visual.get('occluder_evidence',{}).get('frame_records',[])
    proof['occluder_interface_status']={'status':'measured_partial' if any(r['unknown_occluder_pixels'] for r in occluder_frames) else 'measured_complete' if occluder_frames else 'not_run','frames':len(occluder_frames),'frames_with_unknown_pixels':sum(r['unknown_occluder_pixels']>0 for r in occluder_frames),'unknown_pixels':sum(r['unknown_occluder_pixels'] for r in occluder_frames),'retained_source':'native_alpha_visual_evidence.json#occluder_evidence','strict_QA10_policy':'incomplete measured IDs remain private; only complete unambiguous stream supplied'}
    complete_frames,complete_targets=complete_native_occluder_targets(visual)
    proof['occluder_interface_status']['complete_target_ids']=complete_targets
    if complete_frames:
        registry={}
        for actor in visual['frames'][0]['actors']:
            registry[str(actor['actor_id'])]={'native_instance_id':actor['actor_id'],
                'native_object_id':actor['native_object_id'],'semantic_id':actor['semantic_id'],
                'asset_id':actor['asset_id'],'entity_class':scene['sources'].get(actor['endpoint_id'],actor).get('entity_class','articulated_human'),'species_id':scene['sources'].get(actor['endpoint_id'],actor).get('species_id','human'),
                'status':'measured_native_registered_standing',
                'identity_source':'actual_native_object_and_semantic_readback'}
        reviewed=visual.get('native_appearance_review',{}).get('actors',{})
        for aid,record in registry.items():
            if aid in reviewed and reviewed[aid].get('status') == 'reviewed':
                appearance = reviewed[aid]
                record.update(label_zh=appearance['label_zh'],
                              display_label=reviewed_occluder_display_label(appearance,record),
                              appearance_value=appearance.get('value'))
        proof['occluder_registry']=registry
        proof['occluder_evidence']=deepcopy(visual['occluder_evidence'])
        proof['occluder_evidence']['frame_records']=complete_frames
        proof['occluder_evidence']['complete_target_ids']=complete_targets
        proof['occluder_evidence']['incomplete_targets_supplied_as_unknown']=True
    return proof
