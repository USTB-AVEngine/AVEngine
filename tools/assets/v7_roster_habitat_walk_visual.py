"""Reselect native Habitat cameras that show sampled walk mouths and both feet."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,rigid_glb
from v7_human_roster_habitat import make_object,camera,project
from v7_roster_habitat_motion import externalize
from v7_roster_contact_ik import yaw_matrix
from v7_roster_disk import check
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
def near(mask,p,radius=7):
 x,y=map(lambda a:int(round(a)),p)
 return 0<=x<mask.shape[1]and 0<=y<mask.shape[0]and bool(mask[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1].any())
def run(T):
 R=T/'round3';rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());out=[]
 for v in views:
  house=v['house'];check(T,256*1024**2);r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,uuids,_,_= _make_configuration(inputs,None,R/'images/habitat_walk_review'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.gpu_device_id=0;cfg.sim_cfg.enable_physics=True
  for ag in cfg.agents:
   for s in ag.sensor_specifications:s.resolution=[640,640]
  med=r.habitat_sim.metadata.MetadataMediator(cfg.sim_cfg);m=med.pbr_shader_template_manager;pb=m.create_template('v7_walk_visible',register_template=False);pb.enable_ibl=False;pb.enable_direct_lights=True;pb.map_mat_txtr_to_linear=True;pb.map_output_to_srgb=True;m.register_template(pb,'v7_walk_visible');med.set_curr_default_pbr_attributes_handle('v7_walk_visible');cfg.metadata_mediator=med
  with r.habitat_sim.Simulator(cfg)as sim:
   for row in rows:
    aid=row['asset_id'];d,b=read_glb(T/'round2/assets/humans'/aid/'final_ue_root.glb');plan=json.loads((T/'round2/assets/humans'/aid/('walking_'+house+'.json')).read_text());pa=json.loads(Path(plan.get('path_manifest')or T/'round2/registrations'/('path_'+house+'.json')).read_text());dest=R/'assets/walk_visible'/house/aid;ed=externalize(d,b,dest/'textures');clip=next(a for a in d['animations']if a['name']==plan['clip']);rec={'asset_id':aid,'house':house,'family':'Walking','frames':[],'status':'fail','scope':'four sampled cameras; original full native geometric replay remains separate'}
    for i in sorted(set([0,len(plan['frames'])//3,2*len(plan['frames'])//3,len(plan['frames'])-1])):
     root=np.asarray(pa['points_m'][i]);yaw=pa['yaw_rad'][i];parts,ev=pose(d,b,clip,i/30,ground=False);gp=dest/'frames'/(str(i)+'.glb');rigid_glb(ed,b,parts,gp);obj=make_object(sim,r,gp,root,yaw,semantic=6500001);mouth=np.asarray(plan['frames'][i]['mouth_world_m']);feet=[np.asarray(f.get('support_world_m',f['ankle_world_m']))for f in plan['frames'][i]['feet'].values()];best=None;attempt=0
     for angle in [0,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3,math.pi/2,-math.pi/2,math.pi,2*math.pi/3,-2*math.pi/3]:
      if best:break
      for dist in [2.5,2.,1.7,3.]:
       attempt+=1;eye=root+yaw_matrix(yaw+angle)@np.array([0,1.12,-dist]);sim.initialize_agent(0,camera(r,eye,root+[0,.85,0]));obs=sim.get_sensor_observations();rgb=np.asarray(obs[uuids['rgb']])[...,:3];mask=np.asarray(obs[uuids['semantic']])==6500001;mp=project(r,sim,uuids['rgb'],mouth);fp=[project(r,sim,uuids['rgb'],f)for f in feet]
       if int(mask.sum())>=1000 and near(mask,mp)and all(near(mask,p)for p in fp):
        best=(rgb,mask,mp,fp,eye);break
     frame={'frame':i,'camera_attempts':attempt,'status':'fail'}
     if best:
      rgb,mask,mp,fp,eye=best;im=Image.fromarray(rgb);dr=ImageDraw.Draw(im)
      for p,col in [(mp,'lime')]+[(p,'cyan')for p in fp]:x,y=p;dr.ellipse((x-3,y-3,x+3,y+3),outline=col,width=2)
      ip=R/'images/habitat_walk_review'/house/aid/(str(i)+'.jpg');ip.parent.mkdir(parents=True,exist_ok=True);im.save(ip,quality=85);frame.update(status='pass',image=str(ip),visible_pixels=int(mask.sum()),mouth_pixel=list(mp),foot_pixels=[list(p)for p in fp],camera_eye_m=eye.tolist(),highlight_ratio_rgb_ge_245=float(np.mean(np.all(rgb[mask]>=245,axis=1))),method='native semantic visibility plus projected mouth and both foot support/ankle 15px neighbourhoods')
     rec['frames'].append(frame);sim.get_rigid_object_manager().remove_object_by_id(obj.object_id);gp.unlink()
    rec['status']='pass'if all(f['status']=='pass'for f in rec['frames'])else'fail';json_write(R/'evidence/habitat_walk_visual'/house/(aid+'.json'),rec);out.append({k:v for k,v in rec.items()if k!='frames'});print('WALK_VISIBLE',house,aid,rec['status'],[f['camera_attempts']for f in rec['frames']],flush=True)
 json_write(R/'evidence/habitat_walk_visual_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
