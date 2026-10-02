#!/usr/bin/env python3
"""Read-only Habitat overheads for curation entries, isolated per house."""
import argparse,json,math,os,sys,subprocess,time,csv
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_r8_overhead_rgb import prepare_installed_habitat_runtime, look_at

def worker(job,output):
    rt=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg',mp3d_root='/data/datasets/habitat_data',allow_mp3d_environment=False)
    hs=rt.habitat_sim;cfg=hs.SimulatorConfiguration();cfg.scene_id=job['glb'];cfg.load_semantic_mesh=False
    spec=hs.CameraSensorSpec();spec.uuid='rgb';spec.sensor_type=hs.SensorType.COLOR;spec.sensor_subtype=hs.SensorSubType.ORTHOGRAPHIC
    spec.resolution=[1024,1024];spec.position=[0.,0.,0.];spec.ortho_scale=1.;spec.near=.1;spec.far=100.
    ac=hs.agent.AgentConfiguration();ac.sensor_specifications=[spec]
    sim=hs.Simulator(hs.Configuration(cfg,[ac]));agent=sim.get_agent(0)
    sensor=sim._sensors['rgb']._sensor_object
    print('sensor methods', [n for n in dir(sensor) if 'projection' in n or 'ortho' in n],flush=True)
    print('initial projection',np.array(sensor.render_camera.projection_matrix),flush=True)
    entries={}
    for room in job['rooms']:
        key=job['house']+'/'+room['label']; dest=output/job['house'];dest.mkdir(parents=True,exist_ok=True)
        lo,hi=np.array(room['bbox_xz_m'],float);cx,cz=(lo+hi)/2;floor=float(room['floor_y_m'])
        span=max(12.,math.ceil((max(hi-lo)+1.)*2)/2)
        # Match the verified R8 reference: world -X right, world +Z up.
        state=agent.get_state();state.position=np.array([cx,floor+30.,cz],dtype=np.float32)
        state.rotation=look_at([0.,-1.,0.],rt.quaternion.quaternion);state.sensor_states={}
        agent.set_state(state,True)
        images=[]
        for suffix,cut in [('overview',1.8),('lower',.8)]:
            spec.ortho_scale=1.;spec.near=30.-cut;spec.far=31.
            # Habitat orthographic scale is the reciprocal of horizontal span.
            spec.ortho_scale=1./span
            sensor.set_projection_params(spec)
            matrix=np.array(sensor.render_camera.projection_matrix)
            actual_span=2./abs(matrix[0,0])
            if abs(actual_span-span)>.001:raise RuntimeError(f'ortho span mismatch {actual_span} != {span}')
            rgb=np.asarray(sim.get_sensor_observations()['rgb'])[...,:3]
            rel=f'{key}_{suffix}.png';file=output/rel;tmp=file.with_suffix('.tmp.png');Image.fromarray(rgb).save(tmp);tmp.replace(file)
            black=float(np.all(rgb<8,axis=2).mean())
            images.append({'path':rel,'title':f'完整范围 · 地面上方 {cut:.1f} 米剖切','cut_height_above_floor_m':cut,'span_m':float(span),'black_fraction':black,'projection':matrix.tolist()})
        entries[key]={'house':job['house'],'room_label':room['label'],'source_glb':job['glb'],'rooms_source':job['rooms_source'],'bbox_xz_m':room['bbox_xz_m'],'floor_y_m':floor,'size_px':[1024,1024],'orientation':'image_right=-X,image_up=+Z','view':np.array(sensor.render_camera.camera_matrix).tolist(),'images':images}
        (dest/(room['label']+'.json')).write_text(json.dumps(entries[key],ensure_ascii=False,indent=2))
        print('wrote',key,flush=True)
    sim.close()

def batch(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    inv=json.loads(args.inventory.read_text());jobs=[]
    for h in inv['houses']:
        house=h['house'];_,split,index,scene=house.split('_',3)
        base=Path('/data/avengine_external/studio/tasks')/h['task_id']/'output/render'
        src=next(base.glob('**/'+house+'/rooms.json'))
        rs={f"R{r['region_id']}":r for r in json.loads(src.read_text())['rooms']}
        rooms=[dict(rs[r['label']],label=r['label']) for r in h['rooms']]
        job={'house':house,'glb':f'/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/{split}/{index}-{scene}/{scene}.glb','rooms_source':str(src),'rooms':rooms}
        jobs.append(job)
    if args.limit:jobs=jobs[:args.limit]
    for i,job in enumerate(jobs,1):
        if all((output/job['house']/(r['label']+'.json')).is_file() for r in job['rooms']):continue
        jf=output/(job['house']+'.job.json');jf.write_text(json.dumps(job))
        print('house',i,len(jobs),job['house'],flush=True)
        with (output/(job['house']+'.log')).open('w') as log:
            try:r=subprocess.run([sys.executable,__file__,'--output',str(output),'--job',str(jf)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
            except subprocess.TimeoutExpired:print('TIMEOUT',job['house'],flush=True);continue
        if r.returncode:print('FAILED',job['house'],r.returncode,flush=True)
    entries={};missing=[]
    for job in jobs:
        for r in job['rooms']:
            key=job['house']+'/'+r['label'];f=output/(key+'.json')
            if f.is_file():entries[key]=json.loads(f.read_text())
            else:missing.append(key)
    (output/'manifest.json').write_text(json.dumps({'entries':entries,'missing':missing},ensure_ascii=False,indent=2))
    with (output/'files.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['house','room','file','source_glb','cut_height_m','span_m','black_fraction'])
        for entry in entries.values():
            for im in entry['images']:w.writerow([entry['house'],entry['room_label'],str(output/im['path']),entry['source_glb'],im['cut_height_above_floor_m'],im['span_m'],im['black_fraction']])
    print('DONE',len(entries),'missing',missing,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inventory',type=Path);p.add_argument('--job',type=Path);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.job:worker(json.loads(args.job.read_text()),args.output)
    else:batch(args)
