"""Executable native animation-first/standing-slide request pipeline."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
from avengine.assets.animal_motion.policy import select_motion,MotionUnavailable,MotionRejected,attach_to_scene
from avengine.assets.animal_motion.native import run_habitat
from avengine.assets.animal_motion.checks import frame_checks


def completed_checks(native,audio,mode):
    checks=frame_checks(native['frames'],mode=mode,native=True)
    failures=list(checks['failures'])
    if not all(f.get('complete_silhouette_in_frame',False) for f in native['frames']):failures.append('full_native_silhouette_not_verified')
    if native.get('direction_negative_controls',{}).get('status')!='pass':failures.append('direction_negative_controls_not_pass')
    if audio.get('status')!='pass':failures.append('moving_RIR_not_pass:'+str(audio.get('error',audio.get('status'))))
    # A matching endpoint name does not prove an acoustic trajectory match.
    if audio.get('motion_receipt'):
        reference=json.loads(Path(audio['motion_receipt']).read_text())
        keys=('time_s','mouth_world_m')
        if [{k:f[k] for k in keys} for f in reference['frames']] != [{k:f[k] for k in keys} for f in native['frames']]:failures.append('RIR_native_mouth_trajectory_differs')
        if reference['listener_m']!=native['listener_m']:failures.append('RIR_listener_differs')
    else:failures.append('RIR_native_trajectory_lineage_missing')
    return {'status':'fail' if failures else 'pass','failures':failures,'geometry':checks}


def validate_reused_route(native,route,*,speed,seconds):
    from avengine.assets.roster_motion.contact import yaw_matrix
    frames=native['frames'];times=np.asarray([f['time_s'] for f in frames],float)
    if len(times)<2 or abs(times[0])>1e-10 or abs(times[-1]-seconds)>1e-10:
        raise ValueError('reused_native_duration_differs')
    nominal=np.outer(times,[0.,0.,-speed])
    local=np.asarray([f.get('world_root_m',nominal[i]) for i,f in enumerate(frames)],float)
    if not np.isfinite(local).all() or np.max(np.abs(local[:,[0,2]]-nominal[:,[0,2]]))>1e-5:
        raise ValueError('reused_native_route_or_speed_differs')
    expected=np.asarray(route['root_m'],float)+local@yaw_matrix(route['yaw_rad']).T
    observed=np.asarray([f['root_world_m'] for f in frames],float)
    if not np.isfinite(observed).all() or np.max(np.linalg.norm(observed-expected,axis=1))>1e-5:
        raise ValueError('reused_native_route_or_speed_differs')


def execute_request(record,house,inputs,route,output,*,speed=.05,seconds=4.8,ray_multiplier=9,reuse=None,audio_cores=None,validation_policy="strict"):
    """Compile and qualify one requested animal in its actual single-source room.

    The explicit per-room native request is required for both modes. No static
    registry flag or certificate from another room can admit the translation.
    ``reuse`` may map mode to this task's native/audio receipts; their complete
    trajectory identity is checked before accepting them.
    """
    if validation_policy != "strict":
        from avengine.assets.animal_motion.motion_only_service import execute_motion_only_request
        return execute_motion_only_request(record,house,inputs,route,output,speed=speed,
            seconds=seconds,reuse=reuse,policy=validation_policy)
    root=Path(output);root.mkdir(parents=True,exist_ok=False);reuse=reuse or {};attempt_files={}
    def build(mode,actual_speed):
        if mode=='walk_animation' and not record.get('has_walking'):
            raise MotionUnavailable('registered_target_has_no_skeleton_or_Walking')
        directory=root/mode
        if mode in reuse:
            npath=Path(reuse[mode]['native']);native=json.loads(npath.read_text())
            if native['asset_id']!=record['asset_id'] or native['house']!=house['house'] or native['motion_mode']!=mode:
                raise ValueError('reused_native_asset_house_or_mode_differs')
            validate_reused_route(native,route,speed=actual_speed,seconds=seconds)
            if mode=='walk_animation':
                files=record['files'];key=next(k for k in ('motion_master_glb','motion_runtime_glb','walking_candidate_master') if k in files)
                if Path(native['source_master']).resolve()!=Path(files[key]['path']).resolve():
                    raise ValueError('reused_native_motion_source_differs')
                if native.get('source_provenance',{}).get('explicit_motion_profile')!=record.get('motion_profile'):
                    raise ValueError('reused_native_target_motion_profile_differs')
        else:
            npath=directory/'native_motion.json';native=run_habitat(record,house,inputs,route,directory,speed=actual_speed,seconds=seconds,mode=mode)
        if native.get('checks',{}).get('status')!='pass':
            if native.get('status')=='failed' and native.get('failure_kind')!='motion_rejected':raise RuntimeError('native_execution_error:'+str(native.get('error')))
            raise MotionRejected('native_motion_failed:'+str(native.get('error') or ','.join(native.get('checks',{}).get('failures',[])[:12])))
        if mode in reuse and reuse[mode].get('audio'):
            apath=Path(reuse[mode]['audio']);audio=json.loads(apath.read_text())
        else:
            directory.mkdir(exist_ok=True);hp=directory/'house_input.json';ip=directory/'runtime_input.json';hp.write_text(json.dumps(house,indent=2));ip.write_text(json.dumps(inputs,indent=2));apath=directory/'audio/audio.json'
            if audio_cores:
                base=[sys.executable,'-m','tools.assets.prepare_animal_audio_endpoints','--motion',str(npath),'--house',str(hp),'--runtime',str(ip),'--cores',str(audio_cores)]
                endpoints=directory/'audio_endpoints'
                result=subprocess.run([*base,'--output',str(endpoints),'--ray-multiplier',str(ray_multiplier)])
                if result.returncode:raise RuntimeError('native_audio_endpoints_failed:'+str(result.returncode))
                result=subprocess.run([sys.executable,'-m','tools.assets.qualify_animal_audio_parallel','--motion',str(npath),'--house',str(hp),'--runtime',str(ip),'--cores',str(audio_cores),'--previous',str(endpoints),'--output',str(apath.parent)])
            else:
                result=subprocess.run([sys.executable,'-m','tools.assets.validate_animal_motion_audio','--motion',str(npath),'--house',str(hp),'--runtime',str(ip),'--output',str(apath.parent),'--ray-multiplier',str(ray_multiplier),'--fresh-process','--frequency-coherence'])
            if not apath.exists():raise RuntimeError('audio_execution_failed_without_receipt:'+str(result.returncode))
            audio=json.loads(apath.read_text())
        attempt_files[mode]={'native':str(npath),'audio':str(apath)}
        return {'native':native,'audio':audio,'receipt_paths':attempt_files[mode]}
    choice=select_motion(asset_id=record['asset_id'],requested_speed_m_s=speed,
        attempt_animation=lambda s:build('walk_animation',s),attempt_slide=lambda s:build('slide_no_animation',s),
        qualify=lambda p,m:completed_checks(p['native'],p['audio'],m))
    receipt={'asset_id':record['asset_id'],'species':record['species'],'house':house['house'],**choice.metadata(),'receipts':attempt_files,'status':'pass' if choice.motion_mode!='cannot_move' else 'blocked'}
    if choice.payload:
        native=choice.payload['native'];scene={'sources':{'animal':{'asset_id':record['asset_id'],'trajectory':[{'time_s':f['time_s'],'position_m':f['mouth_world_m']} for f in native['frames']]}},'listener_m':native['listener_m'],'duration_s':native['frames'][-1]['time_s']}
        scene['camera_forward']=native['camera_forward']
        scene['sources']['animal']['animal_motion_audio']=choice.payload['receipt_paths']
        scene['sources']['animal']['native_frame_source']=choice.payload['receipt_paths']['native']
        attach_to_scene(scene,'animal',choice);(root/'motion_scene.json').write_text(json.dumps(scene,indent=2))
    (root/'admission.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return receipt
