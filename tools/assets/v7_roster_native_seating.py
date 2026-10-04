"""Native seat fitting with floor support, furniture collision and retained navigation."""
import argparse,copy,json,math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import pose,rigid_glb,accessor
from v7_human_roster_habitat import make_object,camera
from v7_roster_habitat_motion import externalize
from v7_roster_contact_ik import FootPlant,yaw_matrix
from v7_roster_collision import intersections,clipped_surface_minimum
from v7_roster_disk import check
from v7_roster_floor_grid import require_physical_ground
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
class StageFloor:
 def __init__(self,sim,r,ignore=(),ground=None):self.sim,self.r,self.ignore,self.ground=sim,r,set(ignore),ground
 def __call__(self,p):
  hits=self.sim.cast_ray(self.r.habitat_sim.geo.Ray(self.r.magnum.Vector3(np.asarray(p)+[0,.20,0]),self.r.magnum.Vector3(0,-1,0)),.85).hits;hits=[h for h in hits if h.object_id not in self.ignore]
  if not hits:raise RuntimeError('Missing actual floor')
  y=float(hits[0].point[1])
  if self.ground is not None:require_physical_ground(y,self.ground)
  return y
 def many(self,p):return np.array([self(x)for x in p])
def triangles(d,b,parts,offset):
 out=[]
 for p in parts:
  prim=d['meshes'][d['nodes'][p['node']]['mesh']]['primitives'][p['primitive']];ix=accessor(d,b,prim['indices']).astype(int).reshape(-1,3);out.append((p['positions']+offset)[ix])
 return np.concatenate(out)
def fit(d,b,root,yaw,sim,r,ground):
 ac=next(a for a in d['animations']if a['name']=='Sitting_Contact_hm3d');solver=FootPlant(d,b,ac,[0,1/30],[root.tolist()]*2,[yaw]*2,0,0,1/30,StageFloor(sim,r,ground=ground))
 for a in solver.active.values():a[:]=True
 g,p,e=solver.frame(0);return p,e,solver.max_reach
def run(T):
 R=T/'round3';rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];chairs={c['asset_id']:c for c in json.loads((R/'registrations/chair_roster.json').read_text())['chairs']};compat={r['asset_id']:r for r in json.loads((R/'evidence/chair_body_compatibility.json').read_text())};survey={r['house']:r for r in json.loads((R/'evidence/existing_seats_native.json').read_text())};placements=json.loads((R/'evidence/chair_native_placements.json').read_text());views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());out=[]
 for v in views:
  house=v['house'];check(T,512*1024**2);r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,R/'images/seated'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
  for ag in cfg.agents:
   for s in ag.sensor_specifications:s.resolution=[640,640]
  med=r.habitat_sim.metadata.MetadataMediator(cfg.sim_cfg);m=med.pbr_shader_template_manager;pb=m.create_template('v7_sit_direct',register_template=False);pb.enable_ibl=False;pb.enable_direct_lights=True;pb.map_mat_txtr_to_linear=True;pb.map_output_to_srgb=True;m.register_template(pb,'v7_sit_direct');med.set_curr_default_pbr_attributes_handle('v7_sit_direct');cfg.metadata_mediator=med
  with r.habitat_sim.Simulator(cfg)as sim:
   path=json.loads((T/'round2/registrations'/('path_'+house+'.json')).read_text());route=np.asarray(path['points_m']);tree=cKDTree(route[:,[0,2]])
   for row in rows:
    aid=row['asset_id'];check(T,32*1024**2);dest=R/'assets/seated_native'/house/aid;folder=dest/'frames';folder.mkdir(parents=True,exist_ok=True);d,b=read_glb(T/'round2/assets/humans'/aid/'final_ue_root.glb');ed=externalize(d,b,dest/'textures');sc=json.loads((T/'round2/assets/humans'/aid/'seated_contact/hm3d/seated.json').read_text());cx,_,cz=sc['chair_boxes'][0]['center_m'];fits={f['chair_id']:f for f in compat[aid]['styles']if f['status']=='pass'};candidates=[]
    for seat in survey[house]['seats']:
     if seat['status']=='surface_candidate':
      p=seat['seat_surface_center_m'];candidates.append({'kind':'existing','seat':seat,'root':[p[0],p[1]-sc['seat_height_m'],p[2]],'offset':[-cx,0,-cz]})
    for pl in placements:
     if pl['house']==house and pl['status']=='pass_candidate'and pl['asset_id']in fits:
      candidates.append({'kind':'fallback','seat':chairs[pl['asset_id']],'root':pl['root_m'],'offset':fits[pl['asset_id']]['human_local_offset_m']})
    accepted=None;failures=[]
    for cand in candidates:
     cr=np.asarray(cand['root']);offset=np.asarray(cand['offset']);targets=survey[house]['facing_targets'];target=min(targets,key=lambda t:np.linalg.norm(np.asarray(t['center_m'])[[0,2]]-cr[[0,2]]))if targets else None;baseyaw=math.atan2(-(target['center_m'][0]-cr[0]),-(target['center_m'][2]-cr[2]))if target else 0
     for yaw in [baseyaw,baseyaw+math.pi/2,baseyaw-math.pi/2,baseyaw+math.pi]:
      sim.pathfinder.load_nav_mesh(path['navmesh']);rot=yaw_matrix(yaw);hr=cr+rot@offset;body=None;chair=None
      try:
       np0=np.asarray(sim.pathfinder.snap_point(cr));ground=StageFloor(sim,r)(np0);parts,ev,reach=fit(d,b,hr,yaw,sim,r,ground);tri=triangles(d,b,parts,offset)
       if reach>.001 or max(abs(f['floor_delta_m'])for f in ev['feet'].values())>.008:raise RuntimeError('Foot contact or reach failed')
       if cand['kind']=='fallback':
        ch=cand['seat'];point=clipped_surface_minimum(tri,[0,0],[ch['seat_width_m'],ch['seat_depth_m']],.25)
        if point is None:raise RuntimeError('No seat surface')
        h=float(point[1]);scale=h/ch['seat_height_m'];hits=0
        for bx in ch['solid_boxes']:
         c=np.array(bx['center_m']);sz=np.array(bx['size_m']);c[1]*=scale;sz[1]*=scale;hits+=len(intersections(tri,c,sz,bx.get('yaw_rad',0),.004))
        if hits:raise RuntimeError('Human/chair triangle penetration')
        cd,cb=read_glb(Path(ch['habitat_glb']));cd=copy.deepcopy(cd)
        for n in cd['nodes']:
         if 'mesh'in n:n['scale']=[1,scale,1]
        write_glb(dest/'chair.glb',cd,cb);chair=make_object(sim,r,dest/'chair.glb',cr,yaw,semantic=6100002);chair.translation=r.magnum.Vector3(cr+[0,.05,0]);bad=chair.contact_test();sim.get_rigid_object_manager().remove_object_by_id(chair.object_id);chair=None
        if bad:raise RuntimeError('Chair/scene collision')
       humanfile=folder/'human.glb';rigid_glb(ed,b,parts,humanfile);body=make_object(sim,r,humanfile,hr,yaw,semantic=6100001);body.translation=r.magnum.Vector3(hr+[0,.04,0])
       if body.contact_test():raise RuntimeError('Body/scene collision')
       body.translation=r.magnum.Vector3(hr);worldparts=np.concatenate([p['positions']for p in parts])@rot.T+hr;clear=float(tree.query(worldparts[:,[0,2]])[0].min())
       if clear<.60:raise RuntimeError('Protected walking corridor occupied')
       if cand['kind']=='fallback':
        chair=make_object(sim,r,dest/'chair.glb',cr,yaw,semantic=6100002);leg=[]
        for p in ch['support_points_m']:
         wp=cr+rot@np.asarray(p);leg.append(float(wp[1]-StageFloor(sim,r,[body.object_id,chair.object_id])(wp)))
        if any(abs(x)>.012 for x in leg):raise RuntimeError('Chair leg not grounded')
        seatgap=0.
       else:
        point=clipped_surface_minimum(tri,[0,0],[.40,.28],.25)
        if point is None:raise RuntimeError('Seat contact point absent')
        wp=cr+rot@point;seatgap=float(wp[1]-StageFloor(sim,r,[body.object_id])(wp));h=cand['seat']['seat_height_above_floor_m'];scale=None;leg=[]
        if abs(seatgap)>.012:raise RuntimeError('Existing seat gap')
       # Rebuild the private in-memory mesh with added objects and retain the reference route.
       body.motion_type=r.habitat_sim.physics.MotionType.STATIC
       if chair:chair.motion_type=r.habitat_sim.physics.MotionType.STATIC
       ns=r.habitat_sim.NavMeshSettings();ns.set_defaults();ns.agent_height=path['agent_height_m'];ns.agent_radius=path['agent_radius_m'];ns.agent_max_climb=.12;ns.agent_max_slope=10;ns.include_static_objects=True
       if not sim.recompute_navmesh(sim.pathfinder,ns):raise RuntimeError('Seat NavMesh rebuild failed')
       snaps=np.asarray([sim.pathfinder.snap_point(p)for p in route]);ne=float(np.linalg.norm((snaps-route)[:,[0,2]],axis=1).max())
       if not np.isfinite(ne)or ne>.002:raise RuntimeError('Seat blocks reference navigation path')
       emitter=body.root_scene_node.create_child();emitter.translation=r.magnum.Vector3(ev['mouth_m']);mouth=np.asarray(emitter.absolute_translation);me=float(np.linalg.norm(mouth-(hr+rot@np.asarray(ev['mouth_m']))))
       eye=cr+rot@np.array([0,1.05,-2.6]);sim.initialize_agent(0,camera(r,eye,cr+[0,.75,0]));obs=sim.get_sensor_observations();rgb=np.asarray(obs[uuids['rgb']])[...,:3];mask=np.asarray(obs[uuids['semantic']])==6100001;ip=R/'images/seated'/house/(aid+'.jpg');ip.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(ip,quality=85)
       if not mask.any()or me>.0001:raise RuntimeError('Native visibility or FOA failed')
       accepted={'asset_id':aid,'house':house,'status':'pass','seat_source':cand['kind'],'seat_type':cand['seat'].get('style')or cand['seat'].get('category'),'seat_id':cand['seat'].get('asset_id')or cand['seat'].get('semantic_id'),'human_glb':str(humanfile),'chair_glb':str(dest/'chair.glb')if chair else None,'human_root_m':hr.tolist(),'seat_root_m':cr.tolist(),'yaw_rad':yaw,'seat_height_m':h,'chair_scale_y':scale,'seat_contact_gap_m':seatgap,'physical_floor_reference_y_m':ground,'foot_floor_semantics':'all support rays must match nearby navigation-supported physical ground within 15mm, furniture surfaces rejected','maximum_foot_contact_gap_m':max(abs(f['floor_delta_m'])for f in ev['feet'].values()),'chair_support_deltas_m':leg,'maximum_ik_reach_error_m':reach,'body_scene_collision':False,'minimum_body_reference_route_distance_m':clear,'protected_route_navmesh_horizontal_error_m':ne,'mouth_m':ev['mouth_m'],'mouth_world_m':mouth.tolist(),'mouth_readback_error_m':me,'image':str(ip),'visible_pixels':int(mask.sum()),'highlight_ratio_rgb_ge_245':float(np.mean(np.all(rgb[mask]>=245,axis=1))),'facing_target':target,'chair_produces_sound':False,'motion':'static seated contact pose','contact_qualification':'lowest actual seated surface; anatomical buttock region check remains separate','failed_candidate_count':len(failures)}
      except Exception as exc:failures.append({'seat_source':cand['kind'],'reason':str(exc),'yaw_rad':yaw});accepted=None
      finally:
       for obj in [chair,body]:
        if obj is not None:sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
       sim.pathfinder.load_nav_mesh(path['navmesh'])
      if accepted:break
     if accepted:break
    rec=accepted or {'asset_id':aid,'house':house,'status':'fail','failed_candidates':failures};json_write(R/'evidence/seated_native'/house/(aid+'.json'),rec);out.append(rec);print('NATIVE_SEATED',house,aid,rec['status'],rec.get('seat_source'),len(failures),flush=True)
 json_write(R/'evidence/seated_native_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
