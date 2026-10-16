"""Qualify the licensed static pet parrot in native Habitat reference houses."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import json_write
from v7_human_roster_habitat import make_object,camera,project
from v7_roster_disk import check
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
def run(T):
 R=T/'round4';meta=json.loads((R/'assets/parrot/source.json').read_text());placements=json.loads((T/'round3/evidence/chair_native_placements.json').read_text());views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());out=[]
 for v in views:
  house=v['house'];check(T,32*1024**2);pl=next(r for r in placements if r['house']==house and r['status']=='pass_candidate');root=np.asarray(pl['root_m']);runtime=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,R/'images/habitat_parrot'/house,mp3d_root=runtime.mp3d_root,include_audio_sensor=False,physics_config_path=runtime.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
  for ag in cfg.agents:
   for s in ag.sensor_specifications:s.resolution=[640,640]
  med=runtime.habitat_sim.metadata.MetadataMediator(cfg.sim_cfg);m=med.pbr_shader_template_manager;pb=m.create_template('v7_parrot_direct',register_template=False);pb.enable_ibl=False;pb.enable_direct_lights=True;pb.map_mat_txtr_to_linear=True;pb.map_output_to_srgb=True;m.register_template(pb,'v7_parrot_direct');med.set_curr_default_pbr_attributes_handle('v7_parrot_direct');cfg.metadata_mediator=med
  with runtime.habitat_sim.Simulator(cfg)as sim:
   obj=make_object(sim,runtime,R/'assets/parrot/runtime.glb',root,0,semantic=6400011);child=obj.root_scene_node.create_child();child.translation=runtime.magnum.Vector3(meta['beak_local_glb_m']);native=np.asarray(child.absolute_translation);expected=root+np.asarray(meta['beak_local_glb_m']);ee=float(np.linalg.norm(native-expected));obj.translation=runtime.magnum.Vector3(root+[0,.04,0]);collision=bool(obj.contact_test());obj.translation=runtime.magnum.Vector3(root)
   patch=meta['perch_patch']['center_m'];supports=[]
   for x,z in [(0,0),(-.07,0),(.07,0),(0,-.07),(0,.07)]:
    p=root+[patch[0]+x,0,patch[2]+z];hits=sim.cast_ray(runtime.habitat_sim.geo.Ray(runtime.magnum.Vector3(p+[0,.15,0]),runtime.magnum.Vector3(0,-1,0)),.5).hits;hits=[h for h in hits if h.object_id!=obj.object_id];supports.append(None if not hits else float(p[1]-hits[0].point[1]))
   screenshots=[]
   for name,offset in [('front',[0,.72,-1.35]),('three_quarter',[.75,.72,-1.20])]:
    accepted=None
    angles=[0,.12,-.12] if name=='front' else [math.pi/6,-math.pi/6,math.pi/4,-math.pi/4,math.pi/12,-math.pi/12]
    for angle in angles:
     if accepted is not None:break
     for dist in [1.35,1.10,.95,.80]:
      eye=root+np.array([math.sin(angle)*dist,.72,-math.cos(angle)*dist]);sim.initialize_agent(0,camera(runtime,eye,root+[0,.43,0]));candidate=sim.get_sensor_observations();cmask=np.asarray(candidate[uuids['semantic']])==6400011;xx,yy=project(runtime,sim,uuids['rgb'],native);px,py=int(round(xx)),int(round(yy))
      if cmask.sum()>1000 and 0<=px<640 and 0<=py<640 and cmask[max(0,py-5):py+6,max(0,px-5):px+6].any():accepted=(eye,angle,dist);break
    if accepted is None:raise RuntimeError('No native source-visible pet bird camera for '+house+'/'+name)
    eye,angle,dist=accepted;sim.initialize_agent(0,camera(runtime,eye,root+[0,.43,0]));obs=sim.get_sensor_observations();rgb=np.asarray(obs[uuids['rgb']])[...,:3];mask=np.asarray(obs[uuids['semantic']])==6400011;ratio=float(np.mean(np.all(rgb[mask]>=245,axis=1)))if mask.any()else None;x,y=project(runtime,sim,uuids['rgb'],native);px,py=int(round(x)),int(round(y));beak_visible=bool(0<=px<640 and 0<=py<640 and mask[max(0,py-5):py+6,max(0,px-5):px+6].any())
    p=R/'images/habitat_parrot'/house/(name+'.jpg');p.parent.mkdir(parents=True,exist_ok=True);im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.ellipse((x-4,y-4,x+4,y+4),outline='lime',width=2);im.save(p,quality=85)
    screenshots.append({'view':name,'eye_glb_m':eye.tolist(),'azimuth_rad':angle,'distance_m':dist,'image':str(p),'visible_pixels':int(mask.sum()),'highlight_ratio_rgb_ge_245':ratio,'beak_pixel':[x,y],'beak_neighborhood_visible':beak_visible,'status':'pass'if mask.sum()>1000 and ratio is not None and ratio<.02 and beak_visible else'fail'})
   passed=not collision and all(q is not None and abs(q)<.012 for q in supports)and ee<.0001 and all(x['status']=='pass'for x in screenshots)and all(abs(f['minimum_gap_to_perch_m'])<.001 and f['contact_points_inside_perch']for f in meta['foot_perch_qa'])
   rec={'asset_id':meta['asset_id'],'house':house,'status':'pass'if passed else'fail','source':str(R/'assets/parrot/source.json'),'root_m':root.tolist(),'beak_local_glb_m':meta['beak_local_glb_m'],'beak_world_m':native.tolist(),'maximum_beak_readback_error_m':ee,'support_deltas_m':supports,'raised_body_collision_4cm':collision,'foot_perch_qa':meta['foot_perch_qa'],'screenshots':screenshots,'movement':'static','bird_height_m':meta['height_bird_m'],'total_height_m':meta['total_height_m'],'perch_produces_sound':False,'highlight_method':'native RGB saturated pixel proxy, not specular AOV','scope':'two native static reference placements, source-weighted foot and beak points, raised-body scene contact proxy'}
   out.append(rec);json_write(R/'evidence/habitat_pet_parrot_results.json',out);print('PARROT_NATIVE',house,rec['status'],supports,flush=True)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
