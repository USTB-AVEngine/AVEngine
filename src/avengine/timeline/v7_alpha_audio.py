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


def render_alpha_audio(scene,house,inputs,cache_directory,attempt,dry,receipt,camera_frames):
    from tools.timeline.pilot_continuous_scenes import native_renderer
    from avengine.timeline.v7_audio_sources import validate_dry_receipt
    from avengine.acoustics.house_acceptance_metrics import foa_direction
    validate_dry_receipt(receipt);start=time.perf_counter();cpu=time.process_time()
    renderer,identity=native_renderer(inputs,house)
    simulation=identity['simulation']
    if simulation['thread_count']!=1 or simulation['diffraction'] is not True or simulation['max_diffraction_order']!=10:
        raise ValueError('alpha acoustic recipe must keep the calibrated native one-thread diffraction setup')
    write_json(attempt/'native_RLR_setup.json',renderer.setup_report)
    cache=PositionRIRCache(cache_directory,renderer,identity);stems={};directions=[];plans={}
    for ep,spec in scene['sources'].items():
        if not np.any(dry[ep]):
            plans[ep]={'RIR_count':0,'reason':'visible silent bystander'};continue
        trajectory=spec['trajectory'];point=np.asarray(trajectory[0]['position_m'])
        if any(not np.array_equal(k['position_m'],point) for k in trajectory):raise ValueError('unadmitted human movement')
        rir=cache.get(point);distance=float(np.linalg.norm(point-np.asarray(scene['listener_m'])))
        direction=foa_direction(rir,scene["listener_m"],point,scene["camera_forward"],sample_rate=16000)
        target=(point-np.asarray(scene['listener_m']))/distance
        observed=direction['check']['foa_world_direction']
        if observed is None:raise ValueError('native FOA direct-direction readback missing')
        error=float(direction['check']['angle_deg'])
        directions.append({'endpoint_id':ep,'native_mouth_m':point.tolist(),'distance_m':distance,
            'foa_readback':direction,'expected_world_direction':target.tolist(),'angular_error_deg':error,'passed_5deg':error<=5.})
        if error>5.:raise ValueError('alpha native FOA direction exceeds frozen 5deg criterion')
        stems[ep]=partitioned_convolve(dry[ep],[rir],[0]);plans[ep]={'RIR_count':1,'motion':'static native MJaw'}
    if not stems:raise ValueError('alpha scene needs actual recorded sound')
    length=max(x.shape[1] for x in stems.values());world=np.zeros((4,length))
    for x in stems.values():world[:,:x.shape[1]]+=x
    camera=rotate_world_foa(world,camera_frames)
    before=world[[3,1,2]]/math.sqrt(3);after=camera[[3,1,2]]
    energy_error=float(np.max(np.abs(np.sum(before*before,axis=0)-np.sum(after*after,axis=0))))
    if energy_error>1e-10:raise ValueError('camera FOA rotation failed vector-energy conservation')
    write_float32_wav(attempt/'scene_foa_full_tail.wav',camera,16000,channel_axis=0)
    result={'status':'pass','static_only':True,'moving_energy_gate':'not_applicable_no_moving_sources',
        'directions':directions,'camera_rotation_energy_absolute_error':energy_error,
        'RIR_plan':plans,'RIR_cache':cache.stats.copy(),'dry_assets_receipt':receipt,
        'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,
        'format':'ACN_WYZX_SN3D_camera','full_emission_history_and_tail':True}
    write_json(attempt/'audio_evidence.json',result);cache.db.close()
    return result,['scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json']


def source_qa_evidence(scene,program,dry_receipt,visual,foa_path,audio_evidence,*,question_episode_id,window_start_s=None):
    by_id={r['sound_asset_id']:r for r in program['records']};bindings={b['event_id']:b for b in dry_receipt['events']}
    events={};protocol_events=[]
    # The dry source's original activity clock is shifted only by actual
    # placement; padding never invents speech or new utterances.
    for event in scene['events']:
        binding=bindings[event['event_id']];record=by_id[binding['sound_asset_id']];offset=binding['output_start_sample']
        intervals=record['screens']['automatic_activity']['activity_intervals_samples'];actual=[]
        for lo,hi in intervals:
            a=max(offset+lo,binding['output_start_sample']);b=min(offset+hi,binding['output_end_sample_exclusive'])
            if b>a:actual.append([int(a),int(b)])
        events[event['event_id']]={'sound_asset_id':record['sound_asset_id'],'transcript':record['transcript'],
            'event_unit':'utterance','event_segmentation':{'method':'complete original recorded VCTK utterance','original_id':record['original_id']},
            'source_qc':{'status':'pass','verdict':'pass','full_content_preserved':binding['semantic_content_complete'],
                'source':record['path'],'license':'CC-BY-4.0','original_id':record['original_id']},
            'activity_status':'observed','activity_intervals_samples_global':actual,
            'emission_mechanism':'human_live_recorded'}
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
                        'class_id':event['class_id'],'start_tick':start,'end_tick':previous+1,'mechanism':'human_live_recorded'})
                    start=tick
                previous=tick
    protocol=deepcopy(scene);protocol['events']=protocol_events
    # All source positions are static. Native camera poses at 100 ms are
    # measured by the renderer, then used in this exact change of coordinates.
    reference=math.degrees(math.atan2(scene['camera_forward'][0],-scene['camera_forward'][2]))
    cameras=visual['protocol_camera_readbacks']
    listener=np.asarray(scene['listener_m'])
    for ep,spec in protocol['sources'].items():
        point=np.asarray(scene['sources'][ep]['trajectory'][0]['position_m']);delta=point-listener;track=[]
        for row in cameras:
            angle=math.radians(row['yaw_deg']-reference);c,s=math.cos(angle),math.sin(angle)
            rotation=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
            track.append({'time_s':row['time_s'],'position_m':(listener+rotation@delta).tolist()})
        spec['trajectory']=track
    protocol['coordinate_system']='measured_native_camera_track_transformed_into_fixed_reference_basis'
    protocol['resolution_wh']=list(reversed(visual['renderer']['resolution_hw']))
    protocol['hfov_deg']=visual['renderer']['hfov_deg']
    proof={'events':events,'actor_profiles':{str(a['actor_id']):{'entity_class':'articulated_human','species_id':'human','emission_mechanism':'human_live_recorded'} for a in visual['frames'][0]['actors']},
        'camera_frame_readbacks':visual['camera_frame_readbacks'],'protocol_scene':protocol,
        'question_episode_id':question_episode_id,'question_seed':str(scene['seed']),
        'foa_readback':{'path':str(foa_path),'format':'ACN_WYZX_SN3D_camera',
            'spatial_review':{'status':'pass','source':'self-contained native RIR direct-direction and camera rotation receipt',
                'directions':audio_evidence['directions'],'camera_frames':len(visual['camera_frame_readbacks'])}}}
    return proof
