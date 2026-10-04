"""Native Habitat review in real MP3D/HM3D rooms; explicit RGB and garment mask."""
import argparse,copy,io,json,math,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import read_glb,write_glb,append_image,material_body_image,json_write
from v7_human_roster_pose import accessor,materialize
from v7_human_roster_roi import roi
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs

def camera(r,eye,target):
 forward=np.array(target)-np.array(eye);forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,forward);state=r.habitat_sim.AgentState();state.position=np.array(eye);state.rotation=r.quaternion.from_rotation_matrix(np.column_stack([right,up,-forward]));return state

def shirt_glb(T,base,row):
 path=T/'assets/masks'/row['asset_id']/'shirt.glb'
 if path.exists():return path
 d,b=read_glb(base);ii=material_body_image(d,row['prefix']);m=np.asarray(Image.open(T/'masks'/row['asset_id']/'shirt_main_color.png').convert('L'));buf=io.BytesIO();Image.fromarray(np.repeat(m[...,None],3,axis=2)).save(buf,format='PNG');b=append_image(d,b,ii,buf.getvalue())
 for mat in d['materials']:
  p=mat.setdefault('pbrMetallicRoughness',{});p['metallicFactor']=0;p['roughnessFactor']=1;body=mat.get('name','').split('.')[0]==row['prefix']+'_body'
  p['baseColorFactor']=[1,1,1,1] if body else [0,0,0,1]
  if not body:p.pop('baseColorTexture',None)
  mat.pop('normalTexture',None);mat.pop('emissiveTexture',None);mat['emissiveFactor']=[0,0,0];mat.pop('extensions',None)
 write_glb(path,d,b);return path

def make_object(sim,r,path,pos,yaw,shader='pbr',semantic=211):
 manager=sim.get_object_template_manager();handle=str(path)+'.'+shader;t=manager.create_template(handle,register_template=False);t.render_asset_handle=str(path);t.collision_asset_handle=str(path);t.compute_COM_from_shape=False;t.shader_type=shader;t.semantic_id=semantic;t.scale=r.magnum.Vector3(1,1,1);manager.register_template(t,handle);obj=sim.get_rigid_object_manager().add_object_by_template_handle(handle);assert obj is not None
 obj.motion_type=r.habitat_sim.physics.MotionType.KINEMATIC;obj.translation=r.magnum.Vector3(pos);obj.rotation=r.magnum.Quaternion.rotation(r.magnum.Rad(yaw),r.magnum.Vector3.y_axis());obj.semantic_id=semantic
 def semantic_node(node):
  node.semantic_id=semantic
  for child in node.children:semantic_node(child)
 obj.root_scene_node.semantic_id=semantic
 return obj

def project(r,sim,uuid,point):
 sensor=sim._sensors[uuid]._sensor_object;node=sensor.node;world=np.r_[point,1];view=np.linalg.inv(np.asarray(node.absolute_transformation()));p=np.asarray(sensor.render_camera.projection_matrix)@view@world;p=p[:3]/p[3];h,w=sensor.specification().resolution
 return [float((p[0]+1)*w/2),float((1-p[1])*h/2)]

def run(a):
 T=a.task_root;selection=json.loads((T/'registrations/selection.json').read_text());selected={r['asset_id']:r for r in selection['avatars']}
 metas=json.loads(a.case_manifest.read_text()) if a.case_manifest else [json.loads(p.read_text())for p in sorted((T/'assets/runtime').glob('*/*/poses.json'))]
 if a.only:metas=[r for r in metas if a.only in r['asset_id']]
 metas=metas[a.offset:]
 if a.limit:metas=metas[:a.limit]
 views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());results=[]
 for v in views:
  if a.house and v['house']!=a.house:continue
  r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg')
  inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']))
  config,uuids,_,scene=_make_configuration(inputs,None,T/'renders'/a.render_subdir/v['house'],mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);config.sim_cfg.enable_physics=True;config.sim_cfg.gpu_device_id=0
  for ag in config.agents:
   for sensor in ag.sensor_specifications:sensor.resolution=[640,640]
  med=r.habitat_sim.metadata.MetadataMediator(config.sim_cfg);m=med.pbr_shader_template_manager;pbr=m.create_template('v7_human_direct',register_template=False);pbr.enable_ibl=False;pbr.enable_direct_lights=True;pbr.map_mat_txtr_to_linear=True;pbr.map_output_to_srgb=True;m.register_template(pbr,'v7_human_direct');med.set_curr_default_pbr_attributes_handle('v7_human_direct');config.metadata_mediator=med
  with r.habitat_sim.Simulator(config)as sim:
   pos=np.array(v['object_position'],float);snap=np.asarray(sim.pathfinder.snap_point(pos));assert np.isfinite(snap).all()
   pos[1]=snap[1];floorhits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(pos+[0,.25,0]),r.magnum.Vector3(0,-1,0)),.7).hits
   if not floorhits:raise RuntimeError('No physical stage floor under selected point')
   pos[1]=float(floorhits[0].point[1]);direction=np.array(v['camera_position'])-pos;direction[1]=0;direction/=np.linalg.norm(direction);eye=pos+direction*3.0+[0,1.12,0];target=pos+[0,.98,0];yaw=math.atan2(-direction[0],-direction[2]);sim.initialize_agent(0,camera(r,eye,target))
   view={'house':v['house'],'scene':str(scene),'object_position_m':pos.tolist(),'camera_position_m':eye.tolist(),'yaw_rad':yaw,'navmesh_snap_m':snap.tolist(),'resolution':[640,640],'shader':'PBR direct lights, IBL disabled; texture and output sRGB conversions enabled'}
   cam=sim._sensors[uuids['rgb']]._sensor_object
   view['rgb_view_matrix']=np.linalg.inv(np.asarray(cam.node.absolute_transformation())).tolist();view['rgb_projection_matrix']=np.asarray(cam.render_camera.projection_matrix).tolist()
   json_write(T/'evidence'/('habitat_view_'+v['house']+'.json'),view)
   for meta in metas:
    aid=meta['asset_id'];color=meta['color'];base=T/meta['habitat_glb'];dest=T/'renders'/a.render_subdir/v['house']/aid/color;dest.mkdir(parents=True,exist_ok=True);proof=dest/'native.json'
    if proof.exists() and not a.force:results.append(json.loads(proof.read_text()));continue
    obj=make_object(sim,r,base,pos,yaw);obs=sim.get_sensor_observations();rgb=np.asarray(obs[uuids['rgb']])[...,:3].copy();mask=np.asarray(obs[uuids['semantic']])==211;objid=obj.object_id
    mouth=np.array(meta['poses']['stand']['mouth_m']) if meta['poses']['stand']['mouth_m']is not None else None
    c=math.cos(yaw);s=math.sin(yaw);R=np.array([[c,0,s],[0,1,0],[-s,0,c]]);worldmouth=pos+R@mouth if mouth is not None else None
    pixel=project(r,sim,uuids['rgb'],worldmouth) if worldmouth is not None else None
    Image.fromarray(rgb).save(dest/'rgb.png');Image.fromarray(mask.astype('uint8')*255).save(dest/'mask.png')
    marked=Image.fromarray(rgb);dr=ImageDraw.Draw(marked)
    if pixel is not None:
     x,y=pixel;dr.ellipse((x-4,y-4,x+4,y+4),outline=(0,255,0),width=2)
    marked.save(dest/'mouth.png');visible=int(mask.sum());yy,xx=np.where(mask)
    if visible:Image.fromarray(rgb[max(0,yy.min()-15):min(640,yy.max()+16),max(0,xx.min()-15):min(640,xx.max()+16)]).save(dest/'crop.jpg',quality=90)
    bbox=obj.aabb;miny=float(bbox.min[1])+pos[1];maxy=float(bbox.max[1])+pos[1]
    # Native floor ray through the actor's support point; ignore its own collider.
    hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(pos+[0,.15,0]),r.magnum.Vector3(0,-1,0)),.5).hits;stagehits=[h for h in hits if h.object_id!=objid];floor_y=float(stagehits[0].point[1]) if stagehits else None
    rec={**view,'asset_id':aid,'color':color,'visible_pixels':visible,'native_height_m':maxy-miny,'native_floor_delta_m':miny-pos[1],'floor_raycast_y_m':floor_y,'floor_raycast_status':'pass' if floor_y is not None and abs(floor_y-pos[1])<.025 else 'qualified','mouth_world_m':None if worldmouth is None else worldmouth.tolist(),'mouth_pixel':pixel,'mouth_status':'native_projection_pending_review','highlight_ratio_rgb_ge_245':float(np.mean(np.all(rgb[mask]>=245,axis=1))) if visible else None,'glb':str(base.relative_to(T)),'motion_family':meta.get('review_family','stand'),'motion_frame':meta.get('review_frame'),'status':'pass' if visible>1500 and abs(miny-pos[1])<.015 else 'fail','rgb':str((dest/'rgb.png').relative_to(T))}
    sim.get_rigid_object_manager().remove_object_by_id(objid)
    if aid in selected and selected[aid]['controlled_color'] and not meta.get('review_family'):
     cache=T/'evidence/rendered_shirt_roi'/v['house']/(aid+'.png');cache.parent.mkdir(parents=True,exist_ok=True)
     if cache.exists():shirt=np.asarray(Image.open(cache))>0
     else:
      shirt=roi(base,T/'masks'/aid/'shirt_main_color.png',selected[aid]['prefix'],view,mask);Image.fromarray(shirt.astype('uint8')*255).save(cache)
     shirt &=mask;Image.fromarray(shirt.astype('uint8')*255).save(dest/'shirt_mask.png');rec['shirt_pixels']=int(shirt.sum());rec['shirt_median_srgb']=np.median(rgb[shirt],axis=0).tolist() if shirt.any() else None;rec['shirt_roi_method']='perspective-correct UV mask with whole-human z-buffer and native semantic visibility, eroded one pixel'
    json_write(proof,rec);results.append(rec);print('NATIVE',v['house'],aid,color,rec['status'],visible,flush=True)
 json_write(T/'evidence'/a.result_name,results)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--house');p.add_argument('--case-manifest',type=Path);p.add_argument('--render-subdir',default='habitat');p.add_argument('--only');p.add_argument('--limit',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--force',action='store_true');p.add_argument('--result-name',default='habitat_native_results.json');run(p.parse_args())
