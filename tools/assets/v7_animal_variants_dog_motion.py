"""Rerun every original dog contact frame and mouth node on owned coat variants."""
import argparse,copy,hashlib,io,json,math,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from v7_human_roster_materials import read_glb,write_glb,viewbytes
from v7_human_roster_pose import accessor,pose,rigid_glb
from v7_pet_birds_hy3d_habitat import make_object,yaw_matrix
from v7_human_roster_habitat import camera
from avengine.assets.source_materials import configure_source_pbr,measure_source_appearance,source_palette
from v7_animal_variants_habitat import floor_at

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def foot_masks(d,b):
    names=[d['nodes'][j].get('name','').replace('_',' ') for j in d['skins'][0]['joints']];spec={'FL':[' L Hand',' L Finger0'],'FR':[' R Hand',' R Finger0'],'HL':[' L Foot',' L Toe0'],'HR':[' R Foot',' R Toe0']};out=[]
    for n in d['nodes']:
        if 'mesh' not in n:continue
        for p in d['meshes'][n['mesh']]['primitives']:
            js=accessor(d,b,p['attributes']['JOINTS_0']).astype(int);ws=accessor(d,b,p['attributes']['WEIGHTS_0']);m={f:(np.isin(js,[i for i,x in enumerate(names) if any(x.endswith(t) for t in tails)])*ws).sum(1)>.15 for f,tails in spec.items()};m['HEAD']=(np.isin(js,[i for i,x in enumerate(names) if x.endswith(' Head') or 'Mouth' in x])*ws).sum(1)>.15;out.append(m)
    return out

def frame_geometry(d,b,action,t,masks,planned,fi):
    parts,ev=pose(d,b,action,t,ground=False,canonical_flip=False,emitter_bone='Xtra Mouth');root=np.array(planned['path_points_m'][fi]);yaw=planned['yaw_rad'][fi];rot=yaw_matrix(yaw);feet={}
    for side in ['FL','FR','HL','HR']:
        points=np.concatenate([p['positions'][m[side]] for p,m in zip(parts,masks) if m[side].any()]);minimum=points[:,1].min();bottom=points[points[:,1]<minimum+.003];local=np.array([np.median(bottom[:,0]),minimum,np.median(bottom[:,2])]);world=root+rot@local;old=planned['frames'][fi]['feet'][side];feet[side]={'semantic_skin_vertices':len(points),'local_support_m':local.tolist(),'world_support_m':world.tolist(),'in_contact':old['in_contact'],'source_planner_support_world_m':old.get('support_world_m',old['ankle_world_m'])}
    head=np.concatenate([p['positions'][m['HEAD']] for p,m in zip(parts,masks) if m['HEAD'].any()]);mouth=np.array(ev['mouth_m']);lo=head.min(0)-.08;hi=head.max(0)+.08;mouth_in_head=bool(np.all(mouth>=lo)&np.all(mouth<=hi));return parts,ev,root,yaw,feet,mouth_in_head

def externalize(d,b,folder):
    out=copy.deepcopy(d);folder.mkdir(parents=True,exist_ok=True)
    for i,im in enumerate(out.get('images',[])):
        if 'bufferView' not in im:continue
        target=folder/f'image_{i}.png';Image.open(io.BytesIO(viewbytes(d,b,im['bufferView']))).save(target);im.clear();im['uri']='../textures/'+target.name
    return out

def compact(path):
    d,b=read_glb(path);ids=set()
    for mesh in d['meshes']:
        for p in mesh['primitives']:
            ids.update(p['attributes'].values())
            if 'indices' in p:ids.add(p['indices'])
    ids=sorted(ids);mapping={old:i for i,old in enumerate(ids)};accessors=[copy.deepcopy(d['accessors'][i]) for i in ids];views=sorted({a['bufferView'] for a in accessors});viewmap={old:i for i,old in enumerate(views)};newviews=[];payload=bytearray()
    for vi in views:
        payload.extend(bytes((-len(payload))%4));v=copy.deepcopy(d['bufferViews'][vi]);raw=viewbytes(d,b,vi);v.update(buffer=0,byteOffset=len(payload),byteLength=len(raw));newviews.append(v);payload.extend(raw)
    for a in accessors:a['bufferView']=viewmap[a['bufferView']]
    for mesh in d['meshes']:
        for p in mesh['primitives']:
            p['attributes']={k:mapping[v] for k,v in p['attributes'].items()}
            if 'indices' in p:p['indices']=mapping[p['indices']]
    d['accessors']=accessors;d['bufferViews']=newviews;d['buffers']=[{'byteLength':len(payload)}];write_glb(path,d,bytes(payload))

def cpu_check(T,reference,meta,house,family):
    d,b=read_glb(meta['motion_runtime_glb']);planned=json.loads((reference/f"round2/assets/dogs/{meta['body_id']}/{house}_{family}.json").read_text());action=next(x for x in d['animations'] if x['name']==family+'_Contact_'+house);masks=foot_masks(d,b);frames=[]
    for fi in range(planned['frame_count']):
        parts,ev,root,yaw,feet,mhead=frame_geometry(d,b,action,fi/30.,masks,planned,fi);frames.append({'frame':fi,'time_s':fi/30.,'feet':feet,'mouth_local_m':ev['mouth_m'],'mouth_nearest_surface_m':ev['mouth_nearest_surface_m'],'mouth_in_semantic_head':mhead,'body_floor_y_m':ev['floor_y_m']})
    rec={'asset_id':meta['asset_id'],'house':house,'family':family,'frame_count':len(frames),'source_clip':action['name'],'source_planner':str(reference/f"round2/assets/dogs/{meta['body_id']}/{house}_{family}.json"),'new_motion_glb':meta['motion_runtime_glb'],'new_motion_sha256':hashlib.sha256(Path(meta['motion_runtime_glb']).read_bytes()).hexdigest(),'frames':frames,'maximum_mouth_nearest_surface_m':max(x['mouth_nearest_surface_m'] for x in frames),'mouth_head_fail_frames':sum(not x['mouth_in_semantic_head'] for x in frames),'status':'pass_cpu' if all(len(x['feet'])==4 and x['mouth_in_semantic_head'] and x['mouth_nearest_surface_m']<.08 for x in frames) else 'fail_cpu','native_floor_status':'pending'};dump(T/'evidence/dog_motion_cpu'/house/(meta['asset_id']+'_'+family+'.json'),rec);return rec

def native_check(T,reference,meta,house,family,view):
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
    from avengine.rooms.contracts import load_and_validate_inputs
    d,b=read_glb(meta['motion_runtime_glb']);planned=json.loads((reference/f"round2/assets/dogs/{meta['body_id']}/{house}_{family}.json").read_text());action=next(x for x in d['animations'] if x['name']==family+'_Contact_'+house);masks=foot_masks(d,b);motion_sha=hashlib.sha256(Path(meta['motion_runtime_glb']).read_bytes()).hexdigest();case_path=T/'evidence/dog_motion_native'/house/(meta['asset_id']+'_'+family+'.json');dest=case_path.with_suffix('')/motion_sha[:16];folder=dest/'frames';folder.mkdir(parents=True,exist_ok=True);ed=externalize(d,b,dest/'textures');r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=view['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(view['room_manifest']),Path(view['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,dest,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
    for agent in cfg.agents:
        for spec in agent.sensor_specifications:spec.resolution=[640,640]
    record={'asset_id':meta['asset_id'],'house':house,'family':family,'source_clip':action['name'],'scene':str(scene),'new_motion_glb':meta['motion_runtime_glb'],'new_motion_sha256':motion_sha,'frames':[],'screenshots':[]};selected={0,planned['frame_count']//3,2*planned['frame_count']//3,planned['frame_count']-1};palette=source_palette(d,b,Path(meta['motion_runtime_glb']))
    with r.habitat_sim.Simulator(cfg) as sim:
        shader=configure_source_pbr(cfg,r.habitat_sim,metadata_mediator=sim.metadata_mediator)
        for fi in range(planned['frame_count']):
            parts,ev,root,yaw,feet,mhead=frame_geometry(d,b,action,fi/30.,masks,planned,fi);fpath=folder/f'{fi:04d}.glb';rigid_glb(ed,b,parts,fpath);compact(fpath);rigid_sha=hashlib.sha256(fpath.read_bytes()).hexdigest();obj=make_object(sim,r,fpath,root,yaw,7400211);obj.override_collision_group(r.habitat_sim.physics.CollisionGroups.Dynamic);child=obj.root_scene_node.create_child();child.translation=r.magnum.Vector3(ev['mouth_m']);expected=root+yaw_matrix(yaw)@np.array(ev['mouth_m']);native=np.array(child.absolute_translation);eerror=float(np.linalg.norm(native-expected))
            for foot in feet.values():
                p=np.array(foot['world_support_m']);floor=floor_at(sim,r,p,ignore=[obj.object_id]);foot['native_floor_y_m']=floor;foot['native_support_delta_m']=None if floor is None else float(p[1]-floor)
            obj.translation=r.magnum.Vector3(root+[0,.04,0]);collision=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root);frame={'frame':fi,'time_s':fi/30.,'feet':feet,'native_emitter_world_m':native.tolist(),'emitter_readback_error_m':eerror,'mouth_in_semantic_head':mhead,'mouth_nearest_surface_m':ev['mouth_nearest_surface_m'],'raised_body_collision_4cm':collision,'rigid_pose_sha256':rigid_sha,'rigid_pose_path':str(fpath),'rigid_pose_retained':fi in selected};record['frames'].append(frame)
            if fi in selected:
                target=root+[0,meta['dimensions_m'][1]*.55,0];chosen_camera=None
                for angle in [0,-.4,.4,-.8,.8,1.2,-1.2]:
                    if chosen_camera:break
                    for mult in [.7,1.,1.3,.5]:
                        if chosen_camera:break
                        for elevation in [.8,1.05,1.3]:
                            distance=max(meta['dimensions_m'])*1.8*mult;eye=root+yaw_matrix(yaw)@np.array([math.sin(angle)*distance,meta['dimensions_m'][1]*elevation,-math.cos(angle)*distance]);sim.initialize_agent(0,camera(r,eye,target));obs=sim.get_sensor_observations();mask=np.array(obs[uuids['semantic']])==7400211;yy,xx=np.where(mask)
                            if len(xx)>1000 and min(xx.min(),yy.min(),639-xx.max(),639-yy.max())>3:chosen_camera=True;break
                rgb=np.array(obs[uuids['rgb']])[...,:3];ip=dest/f'{fi:04d}.jpg';Image.fromarray(rgb).save(ip,quality=86);appearance=measure_source_appearance(rgb,mask,palette=palette,max_highlight_ratio=.02,strict=False);record['screenshots'].append({'frame':fi,'image':str(ip),'visible_pixels':int(mask.sum()),'production_appearance':appearance,'status':'pass' if chosen_camera and mask.sum()>1000 and appearance['highlight_ratio']<=.02 and appearance['color_observable'] else 'fail'})
            sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
            if fi not in selected:fpath.unlink()
    contacts=[abs(f['native_support_delta_m']) for row in record['frames'] for f in row['feet'].values() if f['in_contact'] and f['native_support_delta_m'] is not None];missing=sum(f['native_floor_y_m'] is None for row in record['frames'] for f in row['feet'].values());minimum=min((f['native_support_delta_m'] for row in record['frames'] for f in row['feet'].values() if f['native_support_delta_m'] is not None),default=-1);passed=bool(contacts) and max(contacts)<=.012 and minimum>=-.012 and missing==0 and all(x['emitter_readback_error_m']<1e-4 and x['mouth_in_semantic_head'] and x['mouth_nearest_surface_m']<.08 for x in record['frames']) and all(x['status']=='pass' for x in record['screenshots']);record.update(status='pass' if passed else 'fail',frame_count=len(record['frames']),maximum_native_contact_delta_m=max(contacts) if contacts else None,minimum_native_foot_delta_m=minimum,missing_stage_contacts=missing,maximum_emitter_readback_error_m=max(x['emitter_readback_error_m'] for x in record['frames']),raised_collision_frames=sum(x['raised_body_collision_4cm'] for x in record['frames']),collision_scope='native 4cm lifted proxy recorded; four-feet/mouth gate does not treat a separate path clearance issue as a change to source rig',shader=shader,audio='not_run');dump(dest/'receipt.json',record);dump(case_path,record);return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--native',action='store_true');p.add_argument('--only');a=p.parse_args();T=a.task_root;plan=json.loads((T/'plan.json').read_text());views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());results=[]
    for item in plan['animals']:
        if item['mode']!='rocketbox_coat_variant' or (a.only and a.only!=item['asset_id']):continue
        path=T/'assets'/item['asset_id']/'asset.json'
        if not path.exists():continue
        meta=json.loads(path.read_text())
        for house in ['mp3d','hm3d']:
            for family in ['Idle','Walking']:
                try:r=native_check(T,a.reference_root,meta,house,family,next(x for x in views if x['house']==house)) if a.native else cpu_check(T,a.reference_root,meta,house,family);results.append({k:v for k,v in r.items() if k not in ['frames']})
                except BaseException as e:results.append({'asset_id':item['asset_id'],'house':house,'family':family,'status':'fail','error':repr(e),'traceback':traceback.format_exc()})
                dump(T/'evidence'/('dog_motion_native_results.json' if a.native else 'dog_motion_cpu_results.json'),results);print('DOG_REPLAY',item['asset_id'],house,family,results[-1]['status'],flush=True)
    expected=4 if a.only else 16
    return 0 if len(results)==expected and all(x['status'].startswith('pass') for x in results) else 1
if __name__=='__main__':raise SystemExit(main())
