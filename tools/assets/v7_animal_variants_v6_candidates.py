"""Read-only v6 source reuse in independent candidates; actual full-key-frame native feet and head QA."""
import argparse,copy,hashlib,itertools,json,math,sys,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from PIL import Image
from v7_human_roster_pose import read_glb,pose,animation_times,globals_at,accessor,rigid_glb
from v7_human_roster_materials import geometry_animation_digest
from v7_animal_variants_normalize import png_compat
from v7_animal_variants_dog_motion import externalize,compact
from v7_animal_variants_habitat import floor_at
from v7_pet_birds_hy3d_habitat import make_object
from v7_human_roster_habitat import camera
from avengine.assets.source_materials import prepare_source_glb,configure_source_pbr,source_palette,measure_source_appearance

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def descendants(d,index):
    out={index}
    for child in d['nodes'][index].get('children',[]):out|=descendants(d,child)
    return out

def semantic_masks(d,b,manifest):
    anchors={x['anchor_id']:x['joint_id'] for x in manifest['anchors']};name_to_id={n.get('name'):i for i,n in enumerate(d['nodes'])};parts=[]
    for node in d['nodes']:
        if 'mesh' not in node:continue
        for p in d['meshes'][node['mesh']]['primitives']:
            if 'skin' not in node:raise RuntimeError('non-skinned primitive cannot inherit semantic paw masks')
            joints=d['skins'][node['skin']]['joints'];js=accessor(d,b,p['attributes']['JOINTS_0']).astype(int);ws=accessor(d,b,p['attributes']['WEIGHTS_0']);m={}
            for key in ['head','paw_front_left','paw_front_right','paw_hind_left','paw_hind_right']:
                ids=descendants(d,name_to_id[anchors[key]]);skin_ids=[i for i,node_id in enumerate(joints) if node_id in ids];m[key]=(np.isin(js,skin_ids)*ws).sum(1)>.15
            parts.append(m)
    return parts,name_to_id[anchors['head']]

def measure_frame(d,b,action,t,masks,head_node,emitter_head_local,rotation,translation):
    parts,ev=pose(d,b,action,float(t),ground=False,canonical_flip=False);g=globals_at(d,b,action,float(t));mouth=g[head_node][:3,:3]@np.array(emitter_head_local)+g[head_node][:3,3];mouth=rotation@mouth+translation
    for p in parts:
        p['positions']=p['positions']@rotation.T+translation
        if p['normals'] is not None:p['normals']=p['normals']@rotation.T
    feet={}
    for key in ['paw_front_left','paw_front_right','paw_hind_left','paw_hind_right']:
        sets=[p['positions'][m[key]] for p,m in zip(parts,masks) if m[key].any()]
        if not sets:raise RuntimeError('semantic paw region missing:'+key)
        v=np.concatenate(sets);bottom=v[v[:,1]<=v[:,1].min()+.003];point=np.median(bottom,axis=0);point[1]=v[:,1].min();feet[key]={'support_local_m':point.tolist(),'semantic_skin_vertices':len(v)}
    head=np.concatenate([p['positions'][m['head']] for p,m in zip(parts,masks) if m['head'].any()]);allv=np.concatenate([p['positions'] for p in parts]);surface=float(np.linalg.norm(head-mouth,axis=1).min());inside=bool(np.all(mouth>=head.min(0)-.03)&np.all(mouth<=head.max(0)+.03));return parts,feet,mouth,{'mouth_to_head_surface_m':surface,'mouth_in_semantic_head':inside,'minimum_body_y_m':float(allv[:,1].min()),'dimensions_m':np.ptp(allv,axis=0).tolist()}

def build(T):
    A=json.loads((T/'evidence/A_provenance.json').read_text());sources=json.loads((T/'evidence/v6_sources.json').read_text());records=[]
    for row in A['animals']:
        if row['classification']!='全部宽松许可':continue
        aid=row['asset_id'];D=T/'C_candidates'/aid;D.mkdir(parents=True,exist_ok=True)
        try:
            s=next(x for x in sources if x['asset_id']==aid);original=Path(s['source_artifacts'][0]['path']);assert sha(original)==s['source_artifacts'][0]['sha256'];owned=D/'source.glb';owned.write_bytes(original.read_bytes());d0,b0=read_glb(owned);compat=D/'source_png.glb';png_compat(owned,compat);runtime,mat=prepare_source_glb(compat,asset_id=aid,entity_class='articulated_animal',output_directory=D/'materials');motion=D/'motion_runtime.glb';motion.write_bytes(Path(runtime).read_bytes());d,b=read_glb(motion);assert geometry_animation_digest(d,b)==geometry_animation_digest(d0,b0);manifest=json.loads(Path(s['package_manifest']).read_text());masks,head_node=semantic_masks(d,b,manifest);idle=next(x for x in d['animations'] if x['name']=='Idle');parts,_=pose(d,b,idle,0,ground=False,canonical_flip=False);head=np.concatenate([p['positions'][m['head']] for p,m in zip(parts,masks) if m['head'].any()]);allv=np.concatenate([p['positions'] for p in parts]);head_direction=head.mean(0)-allv.mean(0);head_direction[1]=0;head_direction/=max(np.linalg.norm(head_direction),1e-9);projection=head@head_direction;mouth=head[projection>=np.quantile(projection,.985)].mean(0);g=globals_at(d,b,idle,0);emitter_local=(np.linalg.inv(g[head_node])@np.r_[mouth,1])[:3];angle=math.pi-math.atan2(head_direction[0],head_direction[2]);c,sn=math.cos(angle),math.sin(angle);rot=np.array([[c,0,sn],[0,1,0],[-sn,0,c]]);translation=np.array([0,-allv[:,1].min(),0]);meta={'asset_id':aid,'name':row['name'],'source_glb':str(original),'source_sha256':sha(original),'source_package_manifest':s['package_manifest'],'source_package_sha256':sha(s['package_manifest']),'motion_runtime_glb':str(motion),'motion_runtime_sha256':sha(motion),'geometry_skin_all_animations_unchanged':True,'production_material_receipt':mat,'source_licensing':row['licenses'],'head_node':head_node,'emitter_head_local_m':emitter_local.tolist(),'rotation':rot.tolist(),'translation_m':translation.tolist(),'scale':'unchanged source metres; C does not rewrite original body scale','downstream_ai_training_admission':'pending_native_checks','CPU_actions':[]}
            contact=json.loads((Path(meta['source_package_manifest']).parent/'contacts/contact_phases.json').read_text());dump(D/'source_contact_phases.json',contact)
            for family in ['Idle','Walking']:
                action=next(x for x in d['animations'] if x['name']==family);times=animation_times(d,b,action);frames=[]
                for fi,t in enumerate(times):
                    ps,feet,emitter,ev=measure_frame(d,b,action,float(t),masks,head_node,emitter_local,rot,translation);frames.append({'frame':fi,'time_s':float(t),'feet':feet,'mouth_local_m':emitter.tolist(),**ev})
                    if family=='Idle' and fi==0:rigid_glb(d,b,ps,D/'standing.glb');meta['dimensions_m']=ev['dimensions_m']
                meta['CPU_actions'].append({'family':family,'frame_count':len(frames),'frames':frames,'mouth_status':'pass' if all(x['mouth_in_semantic_head'] and x['mouth_to_head_surface_m']<.08 for x in frames) else 'fail'})
            dump(D/'candidate.json',meta);records.append({'asset_id':aid,'status':'built_pending_native','frames':sum(x['frame_count'] for x in meta['CPU_actions'])});print('C_BUILD',aid,records[-1],flush=True)
        except BaseException as ex:records.append({'asset_id':aid,'status':'build_fail','error':repr(ex),'traceback':traceback.format_exc()});print('C_BUILD_FAIL',aid,repr(ex),flush=True)
        dump(T/'reports/C_build_results.json',records)
    return records

def native(T):
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
    from avengine.rooms.contracts import load_and_validate_inputs
    plan=json.loads((T/'reports/C_build_results.json').read_text());views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());placements=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/v7_human_roster_20261004/round3/evidence/chair_native_placements.json').read_text());records=[]
    for view in views:
        house=view['house'];r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=view['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(view['room_manifest']),Path(view['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,T/'evidence/C_native'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
        for agent in cfg.agents:
            for spec in agent.sensor_specifications:spec.resolution=[640,640]
        with r.habitat_sim.Simulator(cfg) as sim:
            shader=configure_source_pbr(cfg,r.habitat_sim,metadata_mediator=sim.metadata_mediator)
            for entry in plan:
                if entry['status']!='built_pending_native':continue
                aid=entry['asset_id'];D=T/'C_candidates'/aid;meta=json.loads((D/'candidate.json').read_text());d,b=read_glb(meta['motion_runtime_glb']);manifest=json.loads(Path(meta['source_package_manifest']).read_text());masks,head_node=semantic_masks(d,b,manifest);rotation=np.array(meta['rotation']);translation=np.array(meta['translation_m']);root=None;floor_probes=[]
                for cand in [p for p in placements if p['house']==house and p['status']=='pass_candidate'][:8]:
                    rp=np.array(cand['root_m']);fy=floor_at(sim,r,rp)
                    if fy is None:continue
                    rp[1]=fy;footpoints=meta['CPU_actions'][0]['frames'][0]['feet'];deltas=[]
                    for f in footpoints.values():
                        p=rp+np.array(f['support_local_m']);floor=floor_at(sim,r,p);deltas.append(None if floor is None else float(p[1]-floor))
                    floor_probes.append({'root_m':rp.tolist(),'deltas_m':deltas})
                    if all(v is not None for v in deltas):root=rp;break
                for family in ['Idle','Walking']:
                    receipt=T/'evidence/C_native'/house/(aid+'_'+family+'.json');rec={'asset_id':aid,'house':house,'family':family,'motion_runtime_sha256':meta['motion_runtime_sha256'],'status':'fail','frames':[],'screenshots':[],'floor_probes':floor_probes,'shader':shader,'source_geometry_skin_all_clips_unchanged':True,'audio':'not_run'}
                    try:
                        if root is None:raise RuntimeError('four stage floor rays unavailable')
                        action=next(x for x in d['animations'] if x['name']==family);times=animation_times(d,b,action);dest=D/'native'/house/family;folder=dest/'frames';folder.mkdir(parents=True,exist_ok=True);ed=externalize(d,b,dest/'textures');selected={0,len(times)//3,2*len(times)//3,len(times)-1};palette=source_palette(d,b,Path(meta['motion_runtime_glb']));source_contact=json.loads((D/'source_contact_phases.json').read_text());source_action=next(x for x in source_contact['actions'] if x.get('semantic_action_id',x.get('action_id'))==('idle' if family=='Idle' else 'walk'))
                        for fi,t in enumerate(times):
                            parts,feet,emitter,ev=measure_frame(d,b,action,float(t),masks,head_node,meta['emitter_head_local_m'],rotation,translation);fpath=folder/f'{fi:04d}.glb';rigid_glb(ed,b,parts,fpath);compact(fpath);obj=make_object(sim,r,fpath,root,0,7700401);child=obj.root_scene_node.create_child();child.translation=r.magnum.Vector3(emitter);emit_error=float(np.linalg.norm(np.array(child.absolute_translation)-(root+emitter)));source_frame=min(source_action['frames'],key=lambda row:abs(row['source_time_seconds']-float(t)));contact_flags={x['contact_id']:x['in_contact'] for x in source_frame['contacts']}
                            for key,f in feet.items():
                                point=root+np.array(f['support_local_m']);fy=floor_at(sim,r,point,ignore=[obj.object_id]);f.update(native_floor_y_m=fy,native_delta_m=None if fy is None else float(point[1]-fy),in_contact=True if family=='Idle' else bool(contact_flags[key]))
                            rec['frames'].append({'frame':fi,'time_s':float(t),'feet':feet,'emitter_readback_error_m':emit_error,'rigid_pose_sha256':sha(fpath),'rigid_pose_retained':fi in selected,**ev})
                            if fi in selected:
                                target=root+np.array([0,meta['dimensions_m'][1]*.6,0]);chosen=False
                                for angle in [0,-.4,.4,-.8,.8,1.2,-1.2,math.pi]:
                                    if chosen:break
                                    for dist in [max(meta['dimensions_m'])*x for x in [1.5,2.,1.,2.5]]:
                                        if chosen:break
                                        for elevation in [.8,1.1,1.4]:
                                            eye=root+np.array([math.sin(angle)*dist,meta['dimensions_m'][1]*elevation,-math.cos(angle)*dist]);sim.initialize_agent(0,camera(r,eye,target));obs=sim.get_sensor_observations();mask=np.array(obs[uuids['semantic']])==7700401;yy,xx=np.where(mask)
                                            if len(xx)>1000 and min(xx.min(),yy.min(),639-xx.max(),639-yy.max())>3:chosen=True;break
                                rgb=np.array(obs[uuids['rgb']])[...,:3];ip=dest/f'{fi:04d}.jpg';Image.fromarray(rgb).save(ip,quality=88);appearance=measure_source_appearance(rgb,mask,palette=palette,max_highlight_ratio=.02,strict=False);rec['screenshots'].append({'frame':fi,'image':str(ip),'image_sha256':sha(ip),'complete_visible':chosen,'visible_pixels':int(mask.sum()),'production_appearance':appearance,'status':'pass' if chosen and appearance['highlight_ratio']<=.02 and appearance['color_observable'] else 'fail'})
                            sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
                            if fi not in selected:fpath.unlink()
                        deltas=[f['native_delta_m'] for x in rec['frames'] for f in x['feet'].values() if f['native_delta_m'] is not None];contacts=[abs(f['native_delta_m']) for x in rec['frames'] for f in x['feet'].values() if f['in_contact'] and f['native_delta_m'] is not None];missing=sum(f['native_delta_m'] is None for x in rec['frames'] for f in x['feet'].values());passed=bool(contacts) and max(contacts)<=.012 and min(deltas)>=-.012 and missing==0 and all(x['mouth_in_semantic_head'] and x['mouth_to_head_surface_m']<.08 and x['emitter_readback_error_m']<1e-4 for x in rec['frames']) and all(x['status']=='pass' for x in rec['screenshots']);rec.update(status='pass' if passed else 'fail',frame_count=len(rec['frames']),maximum_contact_delta_m=max(contacts),minimum_foot_delta_m=min(deltas),missing_floor_contacts=missing,maximum_mouth_to_head_surface_m=max(x['mouth_to_head_surface_m'] for x in rec['frames']),contact_policy='Idle all four supports; Walking nearest authored original package contact phase, retained with source warnings; no per-frame floor correction or joint editing')
                    except BaseException as ex:
                        rec.update(error=repr(ex),traceback=traceback.format_exc())
                        for obj in list(sim.get_rigid_object_manager().get_objects_by_handle_substring('').values()):sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
                    dump(receipt,rec);records.append(rec);dump(T/'reports/C_native_results.json',records);print('C_NATIVE',aid,house,family,rec['status'],rec.get('maximum_contact_delta_m'),rec.get('error'),flush=True)
    summary=[]
    for entry in plan:
        aid=entry['asset_id'];cases=[x for x in records if x['asset_id']==aid];pass_all=len(cases)==4 and all(x['status']=='pass' for x in cases);row={'asset_id':aid,'status':'admitted' if pass_all else 'rejected','downstream_ai_training_admission':'granted' if pass_all else 'not_granted','source_checks':'A all output-compatible licenses','native_cases':[{'path':str(T/'evidence/C_native'/x['house']/(aid+'_'+x['family']+'.json')),'sha256':sha(T/'evidence/C_native'/x['house']/(aid+'_'+x['family']+'.json')),'status':x['status'],'maximum_contact_delta_m':x.get('maximum_contact_delta_m'),'minimum_foot_delta_m':x.get('minimum_foot_delta_m'),'error':x.get('error')} for x in cases],'source_original_unchanged':True};summary.append(row)
    roster={'schema':'avengine_v7_v6_clean_animal_candidates_20261012_v1','artifact_root':str(T),'candidates':summary,'animals':[x for x in summary if x['status']=='admitted'],'existing_packages_modified':False,'production_sampler_integration':False,'audio_bound':False};dump(T/'registrations/v7_asset_roster_v6_clean_candidates_20261012.json',roster);dump(T/'reports/C_status.json',{'status':'completed','clean_v6_candidates':len(plan),'admitted':sum(x['status']=='admitted' for x in summary),'rejected':sum(x['status']=='rejected' for x in summary),'native_cases':len(records),'native_frames':sum(x.get('frame_count',0) for x in records),'candidate_roster':str(T/'registrations/v7_asset_roster_v6_clean_candidates_20261012.json')});return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--native',action='store_true');a=p.parse_args();native(a.task_root) if a.native else build(a.task_root)
if __name__=='__main__':main()
