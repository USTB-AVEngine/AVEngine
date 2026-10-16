"""Sample official gesture/turn adapters in both native Habitat scenes, with seated chairs."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import pose,rigid_glb
from v7_human_roster_habitat import make_object,camera,project
from v7_roster_habitat_motion import externalize
from v7_roster_contact_ik import yaw_matrix
from v7_roster_native_seating import StageFloor
from v7_roster_disk import check
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
def run(T):
 R=T/'round3';rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());places=json.loads((R/'evidence/chair_native_placements.json').read_text());chairs={c['asset_id']:c for c in json.loads((R/'registrations/chair_roster.json').read_text())['chairs']};out=[]
 for v in views:
  house=v['house'];check(T,1024*1024**2);r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,R/'images/habitat_extra'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
  for ag in cfg.agents:
   for s in ag.sensor_specifications:s.resolution=[640,640]
  med=r.habitat_sim.metadata.MetadataMediator(cfg.sim_cfg);m=med.pbr_shader_template_manager;pb=m.create_template('v7_extra_direct',register_template=False);pb.enable_ibl=False;pb.enable_direct_lights=True;pb.map_mat_txtr_to_linear=True;pb.map_output_to_srgb=True;m.register_template(pb,'v7_extra_direct');med.set_curr_default_pbr_attributes_handle('v7_extra_direct');cfg.metadata_mediator=med
  with r.habitat_sim.Simulator(cfg)as sim:
   for row in rows:
    aid=row['asset_id'];d,b=read_glb(T/'round2/assets/humans'/aid/'final_ue_root.glb');dest=R/'assets/native_extra'/house/aid;ed=externalize(d,b,dest/'textures')
    for family in ['Standing_Talk','Turning_Left90','Sitting_Talk']:
     plan=json.loads((R/'evidence/extra_actions'/aid/(family+'.json')).read_text());clip=next(a for a in d['animations']if a['name']==plan['clip']);rec={'asset_id':aid,'house':house,'family':family,'source_geometric_status':plan['geometric_status'],'frames':[],'status':'fail','native_scope':'four sampled keys; full 30Hz geometry and UE bone evaluation have separate receipts'};chobj=None;view=json.loads((T/'evidence'/('habitat_view_'+house+'.json')).read_text());root=np.asarray(view['object_position_m']);yaw=view['yaw_rad'];rot=yaw_matrix(yaw)
     if family=='Sitting_Talk':
      f=plan['chair_fit'];pl=next(x for x in places if x['house']==house and x['asset_id']==f['chair_id']and x['status']=='pass_candidate');cr=np.asarray(pl['root_m']);yaw=0;rot=yaw_matrix(yaw);root=cr+np.asarray(f['human_local_offset_m']);cd,cb=read_glb(Path(chairs[f['chair_id']]['habitat_glb']));cd=copy.deepcopy(cd)
      for n in cd['nodes']:
       if 'mesh'in n:n['scale']=[1,f['instance_scale_y'],1]
      cp=dest/'chair.glb';write_glb(cp,cd,cb);chobj=make_object(sim,r,cp,cr,yaw,semantic=6200002)
     folder=dest/family/'frames';folder.mkdir(parents=True,exist_ok=True)
     # rigid_glb texture URIs are relative to frames; place each family texture link inside the task.
     tex=folder.parent/'textures'
     if not tex.exists():tex.symlink_to(dest/'textures',target_is_directory=True)
     for i in sorted(set([0,plan['frame_count']//3,2*plan['frame_count']//3,plan['frame_count']-1])):
      parts,ev=pose(d,b,clip,i/30,ground=False);gp=folder/(str(i)+'.glb');rigid_glb(ed,b,parts,gp);obj=make_object(sim,r,gp,root,yaw,semantic=6200001);em=obj.root_scene_node.create_child();em.translation=r.magnum.Vector3(ev['mouth_m']);mouth=np.asarray(em.absolute_translation);me=float(np.linalg.norm(mouth-(root+rot@np.asarray(ev['mouth_m']))));obj.translation=r.magnum.Vector3(root+[0,.04,0]);coll=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root);gaps=[]
      if family!='Sitting_Talk':
       for side,ft in plan['frames'][i]['feet'].items():
        if ft['in_contact']:
         p=root+rot@np.asarray(ft.get('support_world_m',ft['ankle_world_m']));gap=float(p[1]-StageFloor(sim,r,[obj.object_id])(p));gaps.append(gap)
      else:gaps=[float(parts[j]['positions'][:,1].min()+root[1]-StageFloor(sim,r,[obj.object_id,chobj.object_id])(root))for j in range(len(parts))];gaps=[min(gaps)]
      eye=root+rot@np.array([0,1.12,-2.75]);sim.initialize_agent(0,camera(r,eye,root+[0,.85,0]));obs=sim.get_sensor_observations();rgb=np.asarray(obs[uuids['rgb']])[...,:3];mask=np.asarray(obs[uuids['semantic']])==6200001;x,y=project(r,sim,uuids['rgb'],mouth);im=Image.fromarray(rgb);dr=ImageDraw.Draw(im);dr.ellipse((x-3,y-3,x+3,y+3),outline='lime',width=2);ip=R/'images/habitat_extra'/house/aid/family/(str(i)+'.jpg');ip.parent.mkdir(parents=True,exist_ok=True);im.save(ip,quality=85);rec['frames'].append({'frame':i,'time_s':i/30,'mouth_world_m':mouth.tolist(),'maximum_mouth_readback_error_m':me,'native_body_collision_4cm':coll,'support_gaps_m':gaps,'visible_pixels':int(mask.sum()),'highlight_ratio_rgb_ge_245':float(np.mean(np.all(rgb[mask]>=245,axis=1)))if mask.any()else None,'image':str(ip)});sim.get_rigid_object_manager().remove_object_by_id(obj.object_id);gp.unlink()
     if chobj:sim.get_rigid_object_manager().remove_object_by_id(chobj.object_id)
     rec['status']='pass'if plan['geometric_status']=='pass'and all(not f['native_body_collision_4cm']and f['maximum_mouth_readback_error_m']<.0001 and f['visible_pixels']>0 and all(abs(q)<.012 for q in f['support_gaps_m'])for f in rec['frames'])else'fail';json_write(R/'evidence/habitat_extra'/house/aid/(family+'.json'),rec);out.append({k:v for k,v in rec.items()if k!='frames'});print('HAB_EXTRA',house,aid,family,rec['status'],flush=True)
 json_write(R/'evidence/habitat_extra_action_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
