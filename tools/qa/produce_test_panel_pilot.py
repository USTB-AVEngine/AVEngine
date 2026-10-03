#!/usr/bin/env python3
"""Frozen binding pilot adapter: checked HM3D poses, existing Habitat and RLR.

All external paths are supplied in --config. Every parent is independent;
only a group's two modalities are reused across its crossed members.
No model, ASR, TTS model or model predictions are loaded.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import threading
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import fftconvolve
import soundfile as sf
from shapely.geometry import shape, Point

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))

def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def elapsed(start,cpu): return {'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu}
def runtime(config):
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    return prepare_installed_habitat_runtime(**config['runtime'],allow_mp3d_environment=False)
def renderer(config,result,listener,*,binaural=False,orientation=None):
    from avengine.acoustics.runtime import load_compiled_acoustic_scene,RLRSimulationConfig
    from avengine.acoustics.rir_cache import _NativeRIRBatchRenderer
    listener=np.asarray(listener,float).tolist()
    scene=load_compiled_acoustic_scene(result['acoustic_package_manifest'],allow_nonpassing_research_qa=True)
    sim=dict(config['simulation']);sim['thread_count']=8
    # Only keys accepted by the existing simulation contract are retained.
    if 'threads' in sim:sim['threads']=8
    sim=RLRSimulationConfig.from_mapping(sim)
    return _NativeRIRBatchRenderer(scene,sim,batch_size=1,initial_positions_m=[listener],listener_position_m=listener,
        listener_orientation_wxyz=orientation or [1.,0.,0.,0.],layout_type='binaural' if binaural else 'ambisonics',channel_count=2 if binaural else 4,hrtf_file_path=config['hrtf'] if binaural else '',
        source_radius_m=0.,listener_radius_m=0.,**{k:v for k,v in config['runtime'].items() if k!='mp3d_root'})

def prepare(config,index):
    from avengine.assets.mp3d_region_actor_tracks import materialize_common_plan_habitat
    from avengine.dataset.binding_group_native import select_controlled_swap_assets
    from avengine.timeline.continuous_scene import camera_basis,published_bearing
    root=Path(config['output']);group=f'{config.get("panel_id", "pilot")}_{index:03d}'
    out=root/'production'/group;out.mkdir(parents=True,exist_ok=False)
    per=int(config['groups_per_house']);hi=index//per;j=index%per;house=config['houses'][hi]
    # 3/3/2/2 in the first house, 2/2/3/3 in the second: five per cell overall.
    cell=(j+2*hi)%4;voice='same' if cell<2 else 'different';side='opposite' if cell%2==0 else 'same_side'
    h=read(house['acceptance_result']);h['acoustic_package_manifest']=house['acoustic_package_manifest'];rooms=house['rooms']
    room=rooms[(j+1)%len(rooms)]
    rt=runtime(config);pf=rt.habitat_sim.PathFinder()
    if not pf.load_nav_mesh(h['input']['navmesh_source']):raise ValueError('navmesh unavailable')
    p=room['placement'];listener=np.array(p['camera_m'],float);listener[1]-=1.5
    source=np.array([p['source_1_m'],p['source_2_m']],float);source[:,1]-=1.2
    rng=np.random.default_rng(config['seed']+index)
    admitted=next(r for r in h['input']['acoustic_rooms'] if r['room_label']==room['room_label'])
    polygon=shape(admitted['floor_polygon'])
    # Pool contains only measured semantic floor points with native navmesh support.
    pool=[];bounds=polygon.bounds
    for xx in np.arange(math.floor(bounds[0]*4)/4+.125,bounds[2],.25):
        for zz in np.arange(math.floor(bounds[1]*4)/4+.125,bounds[3],.25):
            if not polygon.covers(Point(xx,zz)):continue
            query=np.array([xx,listener[1],zz]);q=np.asarray(pf.snap_point(query),float)
            if not np.isfinite(q).all() or np.linalg.norm(q-query)>.10 or not polygon.covers(Point(q[0],q[2])):continue
            clearance=pf.distance_to_closest_obstacle(q,.5)
            if clearance>=.15:pool.append(q)
    pool=np.asarray(pool);cam_pool=[q for q in pool if pf.distance_to_closest_obstacle(q,.5)>=.25]
    if len(pool)<3 or not cam_pool:raise ValueError('quota geometry unavailable: insufficient checked floor support')
    native=renderer(config,h,listener+[0,1.55,0]);native.render([(source[0]+[0,1.64,0]).tolist()])
    failures={};geometry=None
    for attempt in range(1600):
        l=listener.copy();s=source.copy()
        if attempt or j or config['split']=='test':
            l[[0,2]]+=rng.uniform(-.32,.32,2);s[:,[0,2]]+=rng.uniform(-.32,.32,(2,2))
        if side=='same_side' or attempt>=120:
            # No interpolation through furniture. Every candidate is on this checked room's floor.
            l=np.asarray(cam_pool[int(rng.integers(len(cam_pool)))]).copy()
            choices=rng.choice(len(pool),size=2,replace=False);s=pool[choices].copy()
        checks=[];points=[l,*s]
        for q in points:
            snap=np.asarray(pf.snap_point(q),float);error=float(np.linalg.norm(snap-q))
            checks.append({'query_m':q.tolist(),'snap_m':snap.tolist(),'error_m':error})
        if any(not np.isfinite(x['error_m']) or x['error_m']>.10 or not polygon.covers(Point(x['snap_m'][0],x['snap_m'][2])) for x in checks):
            failures['navigation']=failures.get('navigation',0)+1;continue
        l=np.asarray(checks[0]['snap_m']);s=np.array([x['snap_m'] for x in checks[1:]])
        c=l+[0,1.55,0];mouth=s+[0,1.641311,0]
        if np.linalg.norm(s[0]-s[1])<.8 or any(not 1.4<=np.linalg.norm(q-c)<=4.2 for q in mouth):continue
        v=mouth-c;v[:,1]=0;u=v/np.linalg.norm(v,axis=1)[:,None]
        f=(u[0]+u[1]);f/=np.linalg.norm(f)
        gap=math.degrees(math.acos(np.clip(u[0]@u[1],-1,1)))
        if side=='same_side':
            if not 12<=gap<=34:continue
            yaw=math.radians(gap/2+5)
            f=Rotation.from_rotvec([0,yaw,0]).apply(f)
        if side=='opposite' and gap<18:continue
        f,r,up=camera_basis(f)
        angles=[published_bearing(x,c,f) for x in mouth]
        if max(abs(x) for x in angles)>49:continue
        if side=='same_side' and angles[0]*angles[1]<=0:continue
        if side=='opposite' and angles[0]*angles[1]>=0:continue
        raychecks=[]
        for foot in s:
            for height in [1.10,1.641311]:
                end=foot+[0,height,0];d=end-c;dist=np.linalg.norm(d)
                hit=native.context.trace_ray_any_hit(c.tolist(),(d/dist).tolist(),.02,float(dist-.08)).hit
                raychecks.append({'endpoint_m':end.tolist(),'blocked':bool(hit)})
        if any(x['blocked'] for x in raychecks):
            failures['occlusion']=failures.get('occlusion',0)+1;continue
        parent={'house':house['house'],'camera_m':c.tolist(),'feet_m':sorted(s.tolist()),
                'basis':{'forward':f.tolist(),'right':r.tolist(),'up':up.tolist()}}
        parent_sha=hashlib.sha256(json.dumps(parent,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        if parent_sha in config.get('excluded_physical_parent_sha256',[]):
            failures['excluded_pilot_parent']=failures.get('excluded_pilot_parent',0)+1;continue
        geometry={'camera_m':c.tolist(),'feet_m':s.tolist(),'mouths_planned_m':mouth.tolist(),
          'basis':{'forward':f.tolist(),'right':r.tolist(),'up':up.tolist()},'bearings_deg':angles,
          'navigation':checks,'clear_body_rays':raychecks,'attempt':attempt,'failed_candidates':failures,
          'checked_room_floor_polygon':admitted['floor_polygon'],'checked_floor_pool_count':len(pool),'floor_offsets_m':[x[1] for x in s],'floor_measured_first':True,
          'raw_hm3d_to_common':'[x,y,z] -> [x,z,-y]', 'navmesh_common_up_axis':'+Y'};break
    del native
    if geometry is None:raise ValueError(f'quota geometry unavailable {room["room_label"]}: {failures}')
    speech=read(config['speech']);scenarios=speech['dialogues']['scenarios']
    scenario=scenarios[j%len(scenarios)]
    candidates=[s for s in speech['sounds'] if s['scenario_id']==scenario['id'] and s.get('voice_gender')=='male']
    choice=None
    for a in candidates:
        for b in candidates:
            if a['utterance_index']==b['utterance_index'] or (a['voice_preset']==b['voice_preset'])!=(voice=='same'):continue
            if a['duration_s']+b['duration_s']<=8.:choice=[a,b];break
        if choice:break
    if choice is None:raise ValueError('admitted full speech clips do not fit voice quota')
    times=[.4,.4+choice[0]['duration_s']+.6]
    write(out/'recipe.json',{'pilot':config['split']=='pilot','exclude_final_panel':config['split']=='pilot','group':group,'independent_parent':group,
      'house':house['house'],'room':room['room_label'],'voice':voice,'side':side,'geometry':geometry,
      'speech':choice,'start_seconds':times,'scenario':scenario,'target_utterance':0,'question_zh':
      scenario['reverse_question_zh'].format(answer=choice[0]['semantic_answer']['label_zh']),
      'producer_sha256':sha(__file__),'code_adapter':'checked_pose_to_existing_common_plan_habitat','no_model_selection':True})
    template=read(config['template_plan']);old_package=template['resources']['room_package']
    house_out=root/'inputs'/house['house'];house_out.mkdir(parents=True,exist_ok=True)
    dc=house_out/'pilot.scene_dataset_config.json';rm=house_out/'room_manifest.json'
    if not dc.exists():
        dataset=read(config['template_dataset']);dataset['stages']['paths']['.glb']=[str(Path(h['input']['scene_directory']+'/'+h['input']['scan_id']+'.glb').resolve())];dataset['stages']['default_attributes'].update(nav_asset=h['input']['navmesh_source'],semantic_asset=h['input']['semantic_source'],semantic_descriptor_filename=h['input']['annotation_source']);dataset['stages']['configs']=[{'original_file':dataset['stages']['paths']['.glb'][0],'template_handle':dataset['stages']['paths']['.glb'][0],'attributes':copy.deepcopy(dataset['stages']['default_attributes'])}];write(dc,dataset)
        manifest=read(house['room_manifest']);manifest['scene']['dataset_config_path']=str(dc)
        for asset in manifest['assets']:
            if asset['role']=='scene_dataset_config':asset['path']=str(dc)
        write(rm,manifest)
    package=copy.deepcopy(old_package)
    package.update(room_id=house['house'],acoustic_package=h['acoustic_package_manifest'],subrooms=[{'subroom_id':room['room_label']}])
    package['visual_scene']={'scene_glb':read(rm)['scene']['scene_id'],'dataset_config':str(dc),'navmesh':h['input']['navmesh_source']}
    package['planning_inputs']={'room_manifest':str(rm),**package['visual_scene']}
    package['walkable_space']={'kind':'habitat_navmesh','path':h['input']['navmesh_source']}
    package['floor_reference']={'status':'measured','floor_height_m':geometry['feet_m'][0][1]}
    package['static_geometry']={'representation':'real_surface_mesh','source_manifest':h['acoustic_package_manifest'],'source_to_common_frame':'x_y_z_to_x_z_-y_v1'}
    for k in list(package):
        if k.endswith('_template'):del package[k]
    assets=[a['asset_id'] for a in template['visual_plan']['actors']]
    select_controlled_swap_assets(read(config['registry']),entity_class='articulated_human',prefer=assets)
    for vi in range(2):
        vroot=out/'visual'/f'v{vi}';vroot.mkdir(parents=True)
        cam={**copy.deepcopy(template['visual_plan']['camera']),'position_m':geometry['camera_m'],'basis':geometry['basis'],
             'resolution_hw':[360,640],'horizontal_fov_deg':110.,'candidate_id':group,'motion':'static'}
        actors=copy.deepcopy(template['visual_plan']['actors'])
        annotation_ids=[int(line.split(',')[0]) for line in Path(h['input']['annotation_source']).read_text().splitlines() if line.split(',')[0].strip().isdigit()]
        semantic_base=max(annotation_ids,default=0)+100
        for i,a in enumerate(actors):a['semantic_id']=semantic_base+i
        if vi:
            for i,a in enumerate(actors):
                other=template['visual_plan']['actors'][1-i]
                for key in ['asset_id','asset_revision','realized_attributes','display_label']:a[key]=copy.deepcopy(other[key])
                a['emitter_binding']['asset_id']=a['asset_id']
        frames=[]
        for fi in range(template['clock']['frame_count']):
            states=copy.deepcopy(template['visual_plan']['frames'][0]['actor_states'])
            for i,state in enumerate(states):
                foot=np.array(geometry['feet_m'][i]);c=np.array(geometry['camera_m']);to=c-foot;to[1]=0
                yaw=math.atan2(to[0],to[2]);rot=Rotation.from_rotvec([0,yaw,0]);mat=np.eye(4);mat[:3,:3]=rot.as_matrix();mat[:3,3]=foot
                state['root_transform']={'matrix_row_major':mat.reshape(-1).tolist(),'translation_m':foot.tolist(),'rotation_xyzw':rot.as_quat().tolist(),'scale':[1.,1.,1.]}
                state.update(frame_index=fi,action_time_ticks=0,action_phase=0.,moving=False)
                state['planned_emitter_m']=(foot+[0,1.641311,0]).tolist();state['emitter_transform']={'position_m':state['planned_emitter_m']}
            frames.append({'frame_index':fi,'pts_ticks':fi*template['clock']['ticks_per_frame'],'actor_states':states,'camera_state':cam})
        plan={'kind':'avengine_question_driven_episode','plan_coordinates':'renderer_neutral','coordinate_frame':template['coordinate_frame'],
          'episode_id':f'{group}_v{vi}','seed':config['seed']+index,'status':'research_candidate','clock':template['clock'],
          'scene':{'scene_id':house['house'],'room_id':house['house']},'visual_plan':{'backend_role':'production_visual','camera':cam,'actors':actors,'frames':frames,'render':template['visual_plan']['render']},
          'resources':{'room_package':package,'renderer':'habitat','backend':'habitat'}}
        request={'allow_research_candidate_assets':True,'source_registry':config['registry'],'runtime':config['runtime'],
                 'episode_id':plan['episode_id'],'profile':{'anchor_visibility':'in_fov','competitor_visibility':'in_fov'}}
        write(vroot/'request.json',request);write(vroot/'plan/episode_plan.json',plan);write(vroot/'plan/room_package.json',package);write(vroot/'plan/habitat_room_manifest.json',read(rm))
        materialize_common_plan_habitat(plan=plan,room_manifest=rm,runtime_registry=config['registry'],output=vroot/'plan/habitat_execution',allow_research_candidate=True)
    print(json.dumps({'prepared':group,'quota':[voice,side],'house':house['house'],'room':room['room_label']}),flush=True)

def _gpu_samples(gpu, stop, rows):
    owner=os.getpid()
    while not stop.is_set():
        try:
            xml=subprocess.check_output(['nvidia-smi','-i',str(gpu),'-q','-x'],text=True)
            for item in ET.fromstring(xml).findall('.//process_info'):
                pid=int(item.findtext('pid')); q=pid;owned=False
                for _ in range(20):
                    if q==owner:owned=True;break
                    if q<=1:break
                    stat=Path(f'/proc/{q}/stat').read_text().rsplit(')',1)[1].split();q=int(stat[1])
                if owned:
                    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
                    if 'capture_mp3d_multi_actor.py' not in cmd:continue
                    used=item.findtext('used_memory','0 MiB').split()[0]
                    if used.isdigit():rows.append({'pid':pid,'memory_mib':int(used),'command':cmd,'monotonic':time.monotonic()})
        except (OSError,ValueError,subprocess.CalledProcessError):pass
        stop.wait(.5)

def capture(config,index,gpu):
    from avengine.dataset.binding_group_native import capture_visual_plan,compare_native_visuals
    out=Path(config['output'])/'production'/f'{config.get("panel_id", "pilot")}_{index:03d}';records=[]
    for vi in range(2):
        v=out/'visual'/f'v{vi}'
        if (v/'capture/neutral_readback.json').exists():continue
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<config['min_free_vram_mb']:raise RuntimeError(f'GPU {gpu} lacks {config["min_free_vram_mb"]} MiB free: {free}')
        req=read(v/'request.json');req['runtime']['graphics_adapter']=gpu
        start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
        samples=[];stop=threading.Event();monitor=threading.Thread(target=_gpu_samples,args=(gpu,stop,samples),daemon=True);monitor.start()
        try:record=capture_visual_plan(req,v,label=f'v{vi}',log=out/f'v{vi}.capture.log')
        finally:stop.set();monitor.join(timeout=5)
        peak=max((s['memory_mib'] for s in samples),default=None)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(v/'timing.json',{'wall_seconds':time.monotonic()-start,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'gpu':gpu,'gpu_peak_process_mib':peak,'gpu_memory_samples':samples,'command':record['process']})
        records.append(record)
    paths=[{'neutral_readback':str(out/'visual'/f'v{i}'/'capture/neutral_readback.json')} for i in range(2)]
    if not (out/'native_equivalence.json').exists():write(out/'native_equivalence.json',compare_native_visuals(*paths))
    print(json.dumps({'captured':out.name}),flush=True)

def audio(config,index):
    from avengine.timeline.continuous_scene import raw_world_to_camera_sn3d
    out=Path(config['output'])/'production'/f'{config.get("panel_id", "pilot")}_{index:03d}';recipe=read(out/'recipe.json')
    rb=read(out/'visual/v0/capture/neutral_readback.json');c=rb['camera'][0]['position_m'];positions=[]
    for slot in ['source1','source2']:
        rows=rb['entities'][slot];points=np.array([r['emitter'] for r in rows]);
        if not np.allclose(points,points[0],rtol=0,atol=1e-7):raise ValueError('pilot static emitter moved')
        positions.append(points[0].tolist())
    house=config['houses'][index//int(config['groups_per_house'])];h=read(house['acceptance_result']);h['acoustic_package_manifest']=house['acoustic_package_manifest']
    start=time.monotonic();cpu=time.process_time();r=renderer(config,h,c)
    filters=[];native=[]
    for point in positions:
        batch=r.render([point]);filters.append(batch.samples[0]);native.append({'wall_seconds':batch.wall_seconds,'cpu_seconds':batch.process_cpu_seconds})
    setup=r.setup_report;del r;ad=out/'audio';ad.mkdir(exist_ok=False)
    for i,f in enumerate(filters):np.save(ad/f'position_{i}.npy',f,allow_pickle=False)
    dry=[]
    for sound in recipe['speech']:
        a,sr=sf.read(sound['path'],dtype='float32')
        if sr!=16000 or a.ndim!=1:raise ValueError('admitted dry speech rate/layout changed')
        dry.append(a)
    basis=rb['camera'][0]['basis'];matrix=np.column_stack([basis['right'],basis['up'],-np.array(basis['forward'])])
    xyzw=Rotation.from_matrix(matrix).as_quat().tolist()
    br=renderer(config,h,c,binaural=True,orientation=[xyzw[3],*xyzw[:3]])
    bin_filters=[]
    for i,point in enumerate(positions):
        batch=br.render([point]);bin_filters.append(batch.samples[0]);native.append({'layout':'binaural','wall_seconds':batch.wall_seconds,'cpu_seconds':batch.process_cpu_seconds})
    del br
    for i,f in enumerate(bin_filters):np.save(ad/f'binaural_position_{i}.npy',f,allow_pickle=False)
    n=rb['clock']['sample_count'];peak=0.;tracks=[];stems=[];bin_tracks=[]
    for ai in range(2):
        mix=np.zeros((4,n),float);parts=[]
        for di,a in enumerate(dry):
            position=di if ai==0 else 1-di
            f=filters[position];rendered=np.stack([fftconvolve(a,ch) for ch in f])
            stem=np.zeros_like(mix);lo=round(recipe['start_seconds'][di]*16000);hi=min(n,lo+rendered.shape[1]);stem[:,lo:hi]=rendered[:,:hi-lo]
            parts.append(stem);mix+=stem
        bmix=np.zeros((2,n),float)
        for di,a in enumerate(dry):
            position=di if ai==0 else 1-di
            rendered=np.stack([fftconvolve(a,ch) for ch in bin_filters[position]])
            lo=round(recipe['start_seconds'][di]*16000);hi=min(n,lo+rendered.shape[1]);bmix[:,lo:hi]+=rendered[:,:hi-lo]
        bin_tracks.append(bmix)
        tracks.append(mix);stems.append(parts);peak=max(peak,float(np.abs(mix).max()),float(np.abs(bmix).max()))
    gain=min(1.,.75/peak)
    for ai,mix in enumerate(tracks):
        sn=raw_world_to_camera_sn3d(mix*gain,rb['camera'][0]['basis']['forward']).T.astype('<f4')
        sf.write(ad/f'a{ai}_foa.wav',sn,16000,subtype='FLOAT')
        # Current av/av_mono content input is native binaural_downmix.
        binaural=(bin_tracks[ai]*gain).T.astype('<f4')
        sf.write(ad/f'a{ai}_binaural.wav',binaural,16000,subtype='FLOAT')
        sf.write(ad/f'a{ai}_mono.wav',binaural.mean(axis=1),16000,subtype='FLOAT')
        for di,stem in enumerate(stems[ai]):sf.write(ad/f'a{ai}_speech{di}_stem.wav',raw_world_to_camera_sn3d(stem*gain,rb['camera'][0]['basis']['forward']).T.astype('<f4'),16000,subtype='FLOAT')
    write(ad/'receipt.json',{'timing':elapsed(start,cpu),'native':native,'setup':setup,'positions_native_m':positions,'listener_native_m':c,
       'sample_count':n,'sample_rate':16000,'gain_shared_all_columns':gain,'dry_audio':[{'path':s['path'],'sha256':sha(s['path']),'transcript':s['transcript'],'voice':s['voice_preset'],'admission':s.get('transcript_status')} for s in recipe['speech']],
       'layout':'ACN SN3D camera: W,Y(left),Z(up),X(forward)','conversion':'existing raw_world_to_camera_sn3d; raw native N3D -> SN3D; directional axes projected on camera basis',
       'hrtf':{'path':config['hrtf'],'sha256':sha(config['hrtf'])},'mono_conversion':'arithmetic mean of native binaural L/R; FOA never averaged','dynamic_body_acoustic_proxy':'none; both visual assets use one fixed room mesh/material and point mouth emitters',
       'full_convolution_retained_until_10_second_clip_end':True,'model_inference':'not_run'})
    print(json.dumps({'audio':out.name,'timing':elapsed(start,cpu)}),flush=True)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','capture','audio']);p.add_argument('--config',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--gpu',type=int)
    a=p.parse_args();config=read(a.config)
    if not 0<=a.index<len(config['houses'])*int(config['groups_per_house']):raise ValueError('index outside configured independent parents')
    if a.stage=='capture':
        if a.gpu is None:raise ValueError('capture requires explicit --gpu')
        capture(config,a.index,a.gpu)
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU stage requires CUDA_VISIBLE_DEVICES empty')
        start=time.monotonic();cpu=time.process_time()
        (prepare if a.stage=='prepare' else audio)(config,a.index)
        out=Path(config['output'])/'production'/f'{config.get("panel_id", "pilot")}_{a.index:03d}'
        write(out/f'{a.stage}_timing.json',elapsed(start,cpu))
if __name__=='__main__':main()
