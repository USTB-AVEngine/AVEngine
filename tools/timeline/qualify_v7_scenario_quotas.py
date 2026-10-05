#!/usr/bin/env python3
"""CPU native static-crowd diagnostics for actual v0.2 speaker/spatial questions.

This is a qualification panel, not a 12--15-item production window. Research
M2/formant admission and moving RIR qualification remain explicit blockers.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from avengine.timeline.continuous_pilot import write_json,FormantSpeech,make_dry_buses,PositionRIRCache
from avengine.timeline.continuous_scene import generate_v02_questions,partitioned_convolve,raw_world_to_camera_sn3d
from avengine.timeline.v7_scene_quotas import add_quotas,scene_from_native_layout,frustum_geometry
from avengine.spatial_audio.audio import write_float32_wav
from avengine.acoustics.runtime import load_habitat_runtime
from tools.timeline.pilot_continuous_scenes import native_renderer

def read(path):return json.loads(Path(path).read_text())

def find_layout(house,inputs,renderer):
    hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
    pf=hs.PathFinder()
    if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('missing readonly navmesh')
    place=house['room']['placement'];a=np.asarray(place['source_1_m']);b=np.asarray(place['source_2_m']);listener=np.asarray(place['camera_m'])
    forward=(a+b)/2-listener;forward[1]=0;forward/=np.linalg.norm(forward)
    # Initialize the raw-scan intersection context before querying LOS.
    renderer.render([a.tolist()])
    # Use the measured floor of the retained placement, not its nominal
    # 1.3 m source offset (which differs from the native M2 mouth height).
    base_floor=np.asarray(pf.snap_point(a-[0.,1.3,0.]),float)[1]
    if not np.isfinite(base_floor):raise ValueError('retained placement has no native floor')
    points=[]
    for x in np.arange(min(a[0],b[0],listener[0])-.6,max(a[0],b[0],listener[0])+.61,.65):
        for z in np.arange(min(a[2],b[2],listener[2])-.6,max(a[2],b[2],listener[2])+.61,.65):
            nominal=np.array([x,base_floor,z]);foot=np.asarray(pf.snap_point(nominal),float)
            if not np.isfinite(foot).all() or np.linalg.norm(foot-nominal)>.1:continue
            point=foot+[0.,1.3,0.];delta=listener-point;distance=np.linalg.norm(delta)
            if distance<.8 or distance>7:continue
            hit=renderer.context.trace_ray_first_hit(point.tolist(),(delta/distance).tolist(),.001,max(.002,distance-.001))
            points.append({'position_m':point.tolist(),'native_LOS_measured':True,'blocked':bool(hit.hit),
                'navmesh_error_m':float(np.linalg.norm(foot-nominal)),
                'in_frustum':frustum_geometry(point,listener,forward)})
    clear=[p for p in points if not p['blocked']]
    clear.sort(key=lambda p:(not p['in_frustum'],np.linalg.norm(np.array(p['position_m'])-listener),p['position_m']))
    selected=[]
    for p in clear:
        if all(np.linalg.norm(np.array(p['position_m'])-np.array(old['position_m']))>=.64 for old in selected):selected.append(p)
    if len(selected)<9:raise ValueError(f'native layout has {len(selected)} distinct clear slots; need 8 speakers plus bystander')
    return {'status':'native_measured','listener_m':listener.tolist(),'camera_forward':forward.tolist(),
        'speaker_anchors_m':[p['position_m'] for p in selected[1:9]],'bystander_anchor_m':selected[0]['position_m'],
        'candidate_slots':points,'read_only_navmesh':house['input']['navmesh_source'],
        'room_scope':house['room']['room_label'],'scope_status':'new native placement diagnostic; not additional acoustic house admission'}

def work(task):
    ip,ap,out,house_id,layout_only,camera_yaw_deg=task;inputs=read(ip);assets=read(ap)['assets'];house=next(h for h in inputs['houses'] if h['house']==house_id)
    root=Path(out)/house_id;root.mkdir(parents=True,exist_ok=True)
    write_json(root/f'worker_{os.getpid()}.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'compute_device':'CPU','command':'qualify_v7_scenario_quotas.py'})
    renderer,identity=native_renderer(inputs,house);write_json(root/'native_RLR_setup.json',renderer.setup_report)
    layout=find_layout(house,inputs,renderer)
    if camera_yaw_deg:
        theta=np.radians(camera_yaw_deg);c,s=np.cos(theta),np.sin(theta)
        layout['camera_forward']=(np.array([[c,0,s],[0,1,0],[-s,0,c]])@layout['camera_forward']).tolist()
        for point in layout['candidate_slots']:
            point['in_frustum']=frustum_geometry(point['position_m'],layout['listener_m'],layout['camera_forward'])
    layout['diagnostic_camera_yaw_deg']=camera_yaw_deg
    write_json(root/'native_layout.json',layout)
    if layout_only:return {'house':house_id,'native_layout':layout,'formal_dataset_count':0}
    cache=PositionRIRCache(root/'rir_cache',renderer,identity);tts=FormantSpeech();rows=[]
    from avengine.capture.continuous_native_video import NativeContinuousVideo
    for n in range(9):
        dest=root/f'speakers_{n}';dest.mkdir()
        clock,cpu=time.perf_counter(),time.process_time()
        original={'schema':'v7-production-inputs-20261003-v3','scene_index':n,'scene_id':f'{house_id}__speakers_{n}',
            'house':house_id,'split':house['split'],'scene_seed':2026100400+n,'seeds':{'camera':2026100400+n},
            'duration_s':30,'window_count':1,'window_start_ticks':[40]}
        row=add_quotas([original])[0]
        specs=[{'actor_id':i,'class_id':0 if i==8 else 1,'voice':(['en-us+m1','en-us+m2','en-us+m3','en-us+m4','en-us+m5','en-us+m6','en-us+m7','en-us+f3'][i-1]),
            'mechanism':'human_live_formant','voice_identity_perceptually_verified':False} for i in range(1,n+1)]
        scene=scene_from_native_layout(row,house,layout,specs)
        scene['sources']['bystander']={'actor_id':n+1,'class_id':1,'mechanism':'silent_bystander','trajectory':[
            {'time_s':0.,'position_m':layout['bystander_anchor_m']},{'time_s':30.,'position_m':layout['bystander_anchor_m']}]}
        declarations=[]
        for index,endpoint in enumerate(scene['sources']):
            asset=deepcopy(assets[index%len(assets)]);asset['endpoint_id']=endpoint;declarations.append(asset)
        # All roots/joints are static. The ten native seconds are the published
        # crop; no imagined moving or lip-synchronization evidence is asserted.
        visual_scene=deepcopy(scene);visual_scene['duration_s']=10.
        import inspect
        kwargs={'source_material_cache_directory':root/'source_material_cache'} if 'source_material_cache_directory' in inspect.signature(NativeContinuousVideo).parameters else {}
        video=NativeContinuousVideo(house,visual_scene,inputs,declarations,**kwargs)
        try:visual=video.encode(visual_scene,dest/'video.mp4',review_frames=[0,75,149])
        finally:video.close()
        for endpoint,track in visual['observed_mouth_trajectories'].items():
            p=track[0]['position_m']
            if any(np.linalg.norm(np.array(k['position_m'])-p)>1e-8 for k in track):raise ValueError('static diagnostic moved a native mouth')
            scene['sources'][endpoint]['trajectory']=[{'time_s':0.,'position_m':p},{'time_s':30.,'position_m':p}]
        dry,receipt=make_dry_buses(scene,tts);mix=np.zeros((4,480000+64000-1));directions=[]
        for endpoint,samples in dry.items():
            if not np.any(samples):continue
            ir=cache.get(scene['sources'][endpoint]['trajectory'][0]['position_m'])
            from avengine.acoustics.house_acceptance_metrics import foa_direction
            direction=foa_direction(ir,np.array(scene['listener_m']),np.array(scene['sources'][endpoint]['trajectory'][0]['position_m']),np.array(scene['camera_forward']))
            directions.append({'endpoint_id':endpoint,**direction})
            if direction['check']['angle_deg']>5:raise ValueError('native static FOA direction disagrees with camera geometry')
            signal=partitioned_convolve(samples,[ir],[0]);mix[:,:signal.shape[1]]+=signal
        camera=raw_world_to_camera_sn3d(mix,scene['camera_forward']);crop=camera[:,64000:224000]
        write_float32_wav(dest/'audio_foa.wav',crop,16000,channel_axis=0)
        write_float32_wav(dest/'audio_mono.wav',crop[:1],16000,channel_axis=0)
        media={'video':'video.mp4','audio_foa':'audio_foa.wav','audio_mono':'audio_mono.wav','foa_format':'ACN_WYZX_SN3D_camera'}
        public,private,audit=generate_v02_questions(scene,4.,scene['scene_id'],media)
        public=[q for q in public if q['form']=='open'];ids={q['question_id'] for q in public};private=[q for q in private if q['question_id'] in ids]
        for q in public:q['logical_question_id']=q['question_id'].removesuffix('__open')
        observed=next(a['canonical_answer'] for a in private if '__STARSS-SPEAKERS__' in a['question_id'])
        if int(observed)!=n:raise ValueError('observed stable actor count differs from requested population')
        write_json(dest/'questions.json',public);write_json(dest/'private_answers.json',private)
        write_json(dest/'evidence.json',{'scene':scene,'native_visual':str(dest/'native_visual_evidence.json'),
            'dry_receipt':receipt,'RIR_cache':cache.stats.copy(),'protocol_audit':audit,'native_static_FOA_directions':directions,
            'camera_yaw_diagnostic_deg':camera_yaw_deg,'FOA_and_all_derived_reads_camera_forward':scene['camera_forward'],
            'static_native_frames_align_to_crop':True,'formal_dataset_count':0,
            'material_pipeline':visual['renderer'].get('source_material_rendering','legacy round3 Flat diagnostic; official material integration pending'),
            'limitations':['eight formant configurations; perceptual voice identity and gender not admitted',
                'static diagnostic uses the retained camera, not the requested full360 camera quota',
                'research M2 packages; no motion/contact/rights admission','diagnostic panel lacks 12-15 question production quota',
                'not a generated v7-S parent; not a full 13/34 class/media quota acceptance']})
        rows.append({'requested_speakers':n,'actual_speaker_answer':int(observed),'logical_questions':len(public),
            'questions':public,'answers':private,'wall_seconds':time.perf_counter()-clock,'CPU_seconds':time.process_time()-cpu,
            'output':str(dest),'formal_dataset_count':0})
        print('QUOTA_NATIVE_DONE',house_id,n,len(public),flush=True)
    result={'house':house_id,'rows':rows,'RIR_cache':cache.stats.copy(),'formal_dataset_count':0}
    write_json(root/'summary.json',result);cache.db.close();return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--layout-only',action='store_true');p.add_argument('--camera-yaw-deg',type=float,default=0.);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    inputs=read(a.inputs);a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(work,[(str(a.inputs),str(a.assets),str(a.output),h['house'],a.layout_only,a.camera_yaw_deg) for h in inputs['houses']]))
    write_json(a.output/'summary.json',{'houses':rows,'formal_dataset_count':0})
if __name__=='__main__':main()
