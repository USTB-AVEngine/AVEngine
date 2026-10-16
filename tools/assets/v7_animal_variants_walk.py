"""Bounded one-hour v6 walking experiment: whole-frame vertical translation only."""
import argparse,copy,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from PIL import Image
from v7_animal_variants_v6_candidates import semantic_masks,sha,dump,measure_frame as source_measure
from v7_human_roster_pose import read_glb,pose,animation_times,globals_at,rigid_glb
from v7_animal_variants_dog_motion import externalize,compact
from v7_animal_variants_habitat import floor_at
from v7_pet_birds_hy3d_habitat import make_object
from v7_human_roster_habitat import camera
from v7_animal_variants_standing import tilt_matrix
from avengine.assets.source_materials import configure_source_pbr,source_palette,measure_source_appearance
CURRENT_META=None

def whole_frame_y_correction(deltas,contact_keys):
 """Minimize contact error with ONE body translation, respecting 12 mm penetration."""
 values=list(deltas.values());contacts=[deltas[k] for k in contact_keys]
 dy=-(min(contacts)+max(contacts))/2 if contacts else -min(values)
 return float(max(dy,-.012-min(values)))

def measure_frame(d,b,action,t,masks,head_node,emitter_local,rotation,translation):
 parts,feet,mouth,ev=source_measure(d,b,action,t,masks,head_node,emitter_local,rotation,translation);scale=np.array(CURRENT_META['scale_xyz'])
 for p in parts:
  p['positions']*=scale
  if p['normals'] is not None:p['normals']/=scale;p['normals']/=np.maximum(np.linalg.norm(p['normals'],axis=1,keepdims=True),1e-12)
 for f in feet.values():f['support_local_m']=(np.array(f['support_local_m'])*scale).tolist()
 mouth*=scale;phase=CURRENT_META['contact_action'];sf=min(phase['frames'],key=lambda x:abs(x['source_time_seconds']-t));flags={x['contact_id']:x['in_contact'] for x in sf['contacts']};dy=whole_frame_y_correction({k:f['support_local_m'][1] for k,f in feet.items()},[k for k,on in flags.items() if on])
 for p in parts:p['positions'][:,1]+=dy
 for f in feet.values():f['support_local_m'][1]+=dy
 mouth[1]+=dy;v=np.concatenate([p['positions'] for p in parts]);head=np.concatenate([p['positions'][m['head']] for p,m in zip(parts,masks) if m['head'].any()]);ev.update(mouth_to_head_surface_m=float(np.linalg.norm(head-mouth,axis=1).min()),mouth_in_semantic_head=bool(np.all(mouth>=head.min(0)-.03)&np.all(mouth<=head.max(0)+.03)),minimum_body_y_m=float(v[:,1].min()),dimensions_m=np.ptp(v,axis=0).tolist(),whole_frame_vertical_translation_m=dy)
 return parts,feet,mouth,ev

def build(T):
 global CURRENT_META
 P=T/'append_20261012';start=time.time();rows=[]
 for mp in json.loads((P/'reports/all_final_metadata.json').read_text()):
  sm=json.loads(Path(mp).read_text())
  if sm['mode']!='v6_idle_static_derivative':continue
  aid=sm['asset_id'];D=P/'walking_frames'/aid;D.mkdir(parents=True,exist_ok=True);orig=T/'C_candidates'/aid
  if not (orig/'candidate.json').exists():orig=P/'C_first_four'/aid
  cm=json.loads((orig/'candidate.json').read_text());fix=sm['idle_derivative'];cm=copy.deepcopy(cm);cm['rotation']=(tilt_matrix(fix['pitch_deg'],fix['roll_deg'])@np.array(cm['rotation'])).tolist();cm['translation_m']=fix['translation_m'];cm['scale_xyz']=sm['size_consistency']['scale_xyz'];cm['dimensions_m']=sm['dimensions_m'];cm['source_static_metadata']=mp;contact=json.loads((orig/'source_contact_phases.json').read_text());cm['contact_action']=next(x for x in contact['actions'] if x.get('semantic_action_id',x.get('action_id'))=='walk');dump(D/'source_contact_phases.json',contact);CURRENT_META=cm;d,b=read_glb(cm['motion_runtime_glb']);manifest=json.loads(Path(cm['source_package_manifest']).read_text());masks,hn=semantic_masks(d,b,manifest);action=next(x for x in d['animations'] if x['name']=='Walking');frames=[]
  for fi,t in enumerate(animation_times(d,b,action)):
   parts,feet,mouth,ev=measure_frame(d,b,action,float(t),masks,hn,cm['emitter_head_local_m'],np.array(cm['rotation']),np.array(cm['translation_m']));sf=min(cm['contact_action']['frames'],key=lambda x:abs(x['source_time_seconds']-float(t)));flags={x['contact_id']:x['in_contact'] for x in sf['contacts']};contacts=[abs(f['support_local_m'][1]) for k,f in feet.items() if flags[k]];frames.append({'frame':fi,'time_s':float(t),'feet':feet,'contact_flags':flags,'maximum_contact_delta_m':max(contacts) if contacts else 0.,'minimum_foot_delta_m':min(f['support_local_m'][1] for f in feet.values()),'mouth_local_m':mouth.tolist(),**ev})
  cm['CPU_actions']=[{'family':'Walking','frame_count':len(frames),'frames':frames}];dump(D/'candidate.json',cm);triangles=157840 if aid.endswith('shiba_inu_red_medium_standard_adult_research_v2') else sm['triangles'];row={'asset_id':aid,'status':'built_pending_native' if triangles<=100000 else 'not_run_animation_lod_required','source_animation_triangles':triangles,'CPU_corrected_maximum_contact_delta_m':max(f['maximum_contact_delta_m'] for f in frames),'vertical_only_feasible_on_flat_floor':all(f['maximum_contact_delta_m']<=.012+1e-9 and f['minimum_foot_delta_m']>=-.012-1e-9 for f in frames),'frame_count':len(frames),'source_static_metadata':mp,'method':'constant admitted size and rigid heading, then ONLY whole-frame Y translation, package contact phases unchanged'};rows.append(row);dump(P/'reports/walking_build_results.json',rows);print('WALK_VERTICAL_CPU',row,flush=True)
 dump(P/'evidence/walking_budget_start.json',{'started_unix':start,'started_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'maximum_seconds':3600,'build_seconds':time.time()-start});return rows

def native(T):
    global CURRENT_META
    budget=json.loads((T/'append_20261012/evidence/walking_budget_start.json').read_text())
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
    from avengine.rooms.contracts import load_and_validate_inputs
    plan=json.loads((T/'append_20261012/reports/walking_build_results.json').read_text());views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());placements=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/v7_human_roster_20261004/round3/evidence/chair_native_placements.json').read_text());records=[]
    for view in views:
        house=view['house'];r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=view['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(view['room_manifest']),Path(view['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,T/'append_20261012/evidence/walking'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
        for agent in cfg.agents:
            for spec in agent.sensor_specifications:spec.resolution=[640,640]
        with r.habitat_sim.Simulator(cfg) as sim:
            shader=configure_source_pbr(cfg,r.habitat_sim,metadata_mediator=sim.metadata_mediator)
            for entry in plan:
                if entry['status']!='built_pending_native':continue
                aid=entry['asset_id'];D=T/'append_20261012/walking_frames'/aid;meta=json.loads((D/'candidate.json').read_text());CURRENT_META=meta;d,b=read_glb(meta['motion_runtime_glb']);manifest=json.loads(Path(meta['source_package_manifest']).read_text());masks,head_node=semantic_masks(d,b,manifest);rotation=np.array(meta['rotation']);translation=np.array(meta['translation_m']);root=None;floor_probes=[]
                for cand in [p for p in placements if p['house']==house and p['status']=='pass_candidate'][:8]:
                    rp=np.array(cand['root_m']);fy=floor_at(sim,r,rp)
                    if fy is None:continue
                    rp[1]=fy;footpoints=meta['CPU_actions'][0]['frames'][0]['feet'];deltas=[]
                    for f in footpoints.values():
                        p=rp+np.array(f['support_local_m']);floor=floor_at(sim,r,p);deltas.append(None if floor is None else float(p[1]-floor))
                    floor_probes.append({'root_m':rp.tolist(),'deltas_m':deltas})
                    if all(v is not None for v in deltas):root=rp;break
                for family in ['Walking']:
                    if time.time()-budget['started_unix']>3600:raise RuntimeError('one-hour walking experiment budget reached')
                    receipt=T/'append_20261012/evidence/walking'/house/(aid+'_'+family+'.json');rec={'asset_id':aid,'house':house,'family':family,'motion_runtime_sha256':meta['motion_runtime_sha256'],'status':'fail','frames':[],'screenshots':[],'floor_probes':floor_probes,'shader':shader,'source_geometry_skin_all_clips_unchanged':True,'audio':'not_run'}
                    try:
                        if root is None:raise RuntimeError('four stage floor rays unavailable')
                        action=next(x for x in d['animations'] if x['name']==family);times=animation_times(d,b,action);dest=D/'native'/house/family;folder=dest/'frames';folder.mkdir(parents=True,exist_ok=True);ed=externalize(d,b,dest/'textures');selected={0,len(times)//3,2*len(times)//3,len(times)-1};palette=source_palette(d,b,Path(meta['motion_runtime_glb']));source_contact=json.loads((D/'source_contact_phases.json').read_text());source_action=next(x for x in source_contact['actions'] if x.get('semantic_action_id',x.get('action_id'))==('idle' if family=='Idle' else 'walk'))
                        for fi,t in enumerate(times):
                            parts,feet,emitter,ev=measure_frame(d,b,action,float(t),masks,head_node,meta['emitter_head_local_m'],rotation,translation)
                            source_frame=min(source_action['frames'],key=lambda row:abs(row['source_time_seconds']-float(t)));contact_flags={x['contact_id']:x['in_contact'] for x in source_frame['contacts']};native_before={}
                            for key,f in feet.items():
                                point=root+np.array(f['support_local_m']);fy=floor_at(sim,r,point)
                                if fy is None:raise RuntimeError('native per-frame floor ray unavailable')
                                native_before[key]=float(point[1]-fy)
                            native_dy=whole_frame_y_correction(native_before,[key for key,on in contact_flags.items() if on])
                            for part in parts:part['positions'][:,1]+=native_dy
                            for foot in feet.values():foot['support_local_m'][1]+=native_dy
                            emitter[1]+=native_dy;ev['minimum_body_y_m']+=native_dy;ev['native_additional_whole_frame_vertical_translation_m']=native_dy
                            fpath=folder/f'{fi:04d}.glb';rigid_glb(ed,b,parts,fpath);compact(fpath);obj=make_object(sim,r,fpath,root,0,7700401);child=obj.root_scene_node.create_child();child.translation=r.magnum.Vector3(emitter);emit_error=float(np.linalg.norm(np.array(child.absolute_translation)-(root+emitter)));source_frame=min(source_action['frames'],key=lambda row:abs(row['source_time_seconds']-float(t)));contact_flags={x['contact_id']:x['in_contact'] for x in source_frame['contacts']}
                            for key,f in feet.items():
                                point=root+np.array(f['support_local_m']);fy=floor_at(sim,r,point,ignore=[obj.object_id]);f.update(native_floor_y_m=fy,native_delta_m=None if fy is None else float(point[1]-fy),in_contact=True if family=='Idle' else bool(contact_flags[key]))
                            rec['frames'].append({'frame':fi,'time_s':float(t),'feet':feet,'emitter_readback_error_m':emit_error,'rigid_pose_sha256':sha(fpath),'rigid_pose_retained':True,**ev})
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
                            # Retain every owned native rigid frame for an actually qualified future walking cache.
                        deltas=[f['native_delta_m'] for x in rec['frames'] for f in x['feet'].values() if f['native_delta_m'] is not None];contacts=[abs(f['native_delta_m']) for x in rec['frames'] for f in x['feet'].values() if f['in_contact'] and f['native_delta_m'] is not None];missing=sum(f['native_delta_m'] is None for x in rec['frames'] for f in x['feet'].values());passed=bool(contacts) and max(contacts)<=.012 and min(deltas)>=-.012 and missing==0 and all(x['mouth_in_semantic_head'] and x['mouth_to_head_surface_m']<.08 and x['emitter_readback_error_m']<1e-4 for x in rec['frames']) and all(x['status']=='pass' for x in rec['screenshots']);rec.update(status='pass' if passed else 'fail',frame_count=len(rec['frames']),maximum_contact_delta_m=max(contacts),minimum_foot_delta_m=min(deltas),missing_floor_contacts=missing,maximum_mouth_to_head_surface_m=max(x['mouth_to_head_surface_m'] for x in rec['frames']),contact_policy='Idle all four supports; Walking nearest authored original package contact phase, retained with source warnings; per-frame whole-body vertical translation, no joint editing; original 12 mm C gates unchanged')
                    except BaseException as ex:
                        rec.update(error=repr(ex),traceback=traceback.format_exc())
                        for obj in list(sim.get_rigid_object_manager().get_objects_by_handle_substring('').values()):sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
                    dump(receipt,rec);records.append(rec);dump(T/'append_20261012/reports/walking_native_results.json',records);print('C_NATIVE',aid,house,family,rec['status'],rec.get('maximum_contact_delta_m'),rec.get('error'),flush=True)
    summary=[]
    for entry in plan:
        aid=entry['asset_id'];cases=[x for x in records if x['asset_id']==aid];pass_all=len(cases)==2 and all(x['status']=='pass' for x in cases);row={'asset_id':aid,'status':'admitted' if pass_all else ('not_run' if entry['status']!='built_pending_native' else 'rejected'),'build_status':entry['status'],'CPU_corrected_maximum_contact_delta_m':entry['CPU_corrected_maximum_contact_delta_m'],'downstream_ai_training_admission':'granted' if pass_all else 'not_granted','source_checks':'owner-authorized all v6 provenance; standing admission already independently qualified','native_cases':[{'path':str(T/'append_20261012/evidence/walking'/x['house']/(aid+'_'+x['family']+'.json')),'sha256':sha(T/'append_20261012/evidence/walking'/x['house']/(aid+'_'+x['family']+'.json')),'status':x['status'],'maximum_contact_delta_m':x.get('maximum_contact_delta_m'),'minimum_foot_delta_m':x.get('minimum_foot_delta_m'),'error':x.get('error')} for x in cases],'source_original_unchanged':True};summary.append(row)
    roster={'schema':'avengine_v7_v6_whole_vertical_walking_candidates_20261012_v1','artifact_root':str(T),'candidates':summary,'animals':[x for x in summary if x['status']=='admitted'],'existing_packages_modified':False,'production_sampler_integration':False,'audio_bound':False};dump(T/'append_20261012/reports/v6_walking_candidates_20261012.json',roster);dump(T/'append_20261012/reports/walking_status.json',{'status':'completed','owner_authorized_v6_candidates':len(plan),'admitted':sum(x['status']=='admitted' for x in summary),'rejected':sum(x['status']=='rejected' for x in summary),'not_run':sum(x['status']=='not_run' for x in summary),'native_cases':len(records),'native_frames':sum(x.get('frame_count',0) for x in records),'candidate_roster':str(T/'append_20261012/reports/v6_walking_candidates_20261012.json')});return records

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--native',action='store_true');a=p.parse_args();native(a.task_root) if a.native else build(a.task_root)
if __name__=='__main__':main()
