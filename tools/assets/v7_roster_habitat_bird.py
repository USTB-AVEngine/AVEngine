"""Native Habitat qualification of the MIT library sparrow on its silent perch."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import json_write
from v7_human_roster_habitat import make_object,camera,project
from v7_roster_contact_ik import yaw_matrix
from v7_roster_disk import check
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
def run(T):
 R=T/'round3';meta=json.loads((R/'assets/bird_sparrow_fallback/source.json').read_text());placements=json.loads((R/'evidence/chair_native_placements.json').read_text());views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());out=[]
 for v in views:
  house=v['house'];check(T,32*1024**2);pl=next(r for r in placements if r['house']==house and r['status']=='pass_candidate');root=np.asarray(pl['root_m']);yaw=0;rot=yaw_matrix(yaw);r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,R/'images/bird'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
  for ag in cfg.agents:
   for s in ag.sensor_specifications:s.resolution=[640,640]
  med=r.habitat_sim.metadata.MetadataMediator(cfg.sim_cfg);m=med.pbr_shader_template_manager;pb=m.create_template('v7_bird_direct',register_template=False);pb.enable_ibl=False;pb.enable_direct_lights=True;pb.map_mat_txtr_to_linear=True;pb.map_output_to_srgb=True;m.register_template(pb,'v7_bird_direct');med.set_curr_default_pbr_attributes_handle('v7_bird_direct');cfg.metadata_mediator=med
  with r.habitat_sim.Simulator(cfg)as sim:
   obj=make_object(sim,r,R/'assets/bird_sparrow_fallback/runtime.glb',root,yaw,semantic=6400001);child=obj.root_scene_node.create_child();child.translation=r.magnum.Vector3(meta['beak_local_glb_m']);native=np.asarray(child.absolute_translation);expected=root+rot@np.asarray(meta['beak_local_glb_m']);ee=float(np.linalg.norm(native-expected));obj.translation=r.magnum.Vector3(root+[0,.04,0]);collision=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root);supports=[]
   for x,z in [(0,0),(-.07,0),(.07,0),(0,-.07),(0,.07)]:
    p=root+[x,0,z];hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(p+[0,.15,0]),r.magnum.Vector3(0,-1,0)),.5).hits;hits=[h for h in hits if h.object_id!=obj.object_id];supports.append(None if not hits else float(p[1]-hits[0].point[1]))
   eye=root+[0,.45,-.90];sim.initialize_agent(0,camera(r,eye,root+[0,.28,0]));obs=sim.get_sensor_observations();rgb=np.asarray(obs[uuids['rgb']])[...,:3];mask=np.asarray(obs[uuids['semantic']])==6400001;ratio=float(np.mean(np.all(rgb[mask]>=245,axis=1)))if mask.any()else None;x,y=project(r,sim,uuids['rgb'],native);p=R/'images/bird'/house/'sparrow_fallback.jpg';p.parent.mkdir(parents=True,exist_ok=True);im=Image.fromarray(rgb);dr=ImageDraw.Draw(im);dr.ellipse((x-4,y-4,x+4,y+4),outline='lime',width=2);im.save(p,quality=85)
   passed=not collision and all(q is not None and abs(q)<.012 for q in supports)and ee<.0001 and mask.any()and ratio is not None and ratio<.02
   rec={'asset_id':meta['asset_id'],'house':house,'status':'pass'if passed else'fail','image':str(p),'bird_height_m':meta['height_bird_m'],'total_height_m':meta['height_bird_m']+meta['perch_top_m'],'root_m':root.tolist(),'yaw_rad':yaw,'beak_local_glb_m':meta['beak_local_glb_m'],'beak_world_m':native.tolist(),'beak_pixel':[x,y],'maximum_beak_readback_error_m':ee,'support_deltas_m':supports,'raised_body_collision_4cm':collision,'visible_pixels':int(mask.sum()),'highlight_ratio_rgb_ge_245':ratio,'movement':'static','species':'sparrow_library_fallback_not_pet_parrot','perch_produces_sound':False,'audio_admission_owner':'v7-families','beak_visual_status':'native projection candidate requires visual assessment'}
   out.append(rec);print('BIRD_NATIVE',house,rec['status'],supports,ratio,flush=True)
   sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
   for aid in ['v7_lx_hand_drum_player','v7_lx_pianist']:
    lv=json.loads((T/'evidence'/('habitat_view_'+house+'.json')).read_text());lr=np.asarray(lv['object_position_m']);lo=make_object(sim,r,T/'assets/runtime'/aid/'original/habitat.glb',lr,0,semantic=6400003)
    for name,offset in [('rear',[0,1.05,1.6]),('side',[1.6,1.05,0]),('front',[0,1.05,-1.6])]:
     sim.initialize_agent(0,camera(r,lr+offset,lr+[0,.60,0]));oo=sim.get_sensor_observations();ip=R/'images/lx_seat_review'/house/aid/(name+'.jpg');ip.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.asarray(oo[uuids['rgb']])[...,:3]).save(ip,quality=85)
    sim.get_rigid_object_manager().remove_object_by_id(lo.object_id)
 json_write(R/'evidence/bird_native_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
