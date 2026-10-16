"""Recast planning on actual Kujiale USD triangles; UE remains RGB authority."""
import argparse,json,math
from pathlib import Path
import numpy as np
from v7_human_roster_materials import write_glb,json_write
from v7_roster_native_paths import sample_curve
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
def main(T):
 R=T/'round2';p=R/'assets/kujiale_navigation_geometry_v2.glb'
 if not p.exists():
  tri=np.load(T/'evidence/ue_source_triangles.npz')['triangles'];verts=tri[:,:,[1,2,0]].copy();verts[:,:,2]*=-1;verts/=100;verts=verts.reshape(-1,3).astype(np.float32);ix=np.arange(len(verts),dtype=np.uint32)
  pos=verts.tobytes();ib=ix.tobytes();d={'asset':{'version':'2.0','generator':'task-owned Kujiale USD navigation proxy, no production pixels'},'buffers':[{'byteLength':len(pos)+len(ib)}],'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':len(pos),'target':34962},{'buffer':0,'byteOffset':len(pos),'byteLength':len(ib),'target':34963}],'accessors':[{'bufferView':0,'componentType':5126,'type':'VEC3','count':len(verts),'min':verts.min(0).tolist(),'max':verts.max(0).tolist()},{'bufferView':1,'componentType':5125,'type':'SCALAR','count':len(ix)}],'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1,'mode':4}]}],'nodes':[{'mesh':0}],'scenes':[{'nodes':[0]}],'scene':0};write_glb(p,d,pos+ib)
 v=json.loads((R/'registrations/path_mp3d.json').read_text())['view_input']
 r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg')
 inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,_,_,_=_make_configuration(inputs,None,R/'navmesh/kujiale',mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path)
 stage=p.with_name('kujiale_navigation_v3.stage_config.json');json_write(stage,{'render_asset':str(p),'collision_asset':str(p),'up':[0,1,0],'front':[0,0,-1],'scale':[1,1,1]});cfg.sim_cfg.scene_id=str(stage);cfg.sim_cfg.scene_dataset_config_file='default';cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.create_renderer=False;cfg.sim_cfg.gpu_device_id=0
 with r.habitat_sim.Simulator(cfg)as sim:
  ns=r.habitat_sim.NavMeshSettings();ns.set_defaults();ns.agent_height=2.05;ns.agent_radius=.65;ns.agent_max_climb=.12;ns.agent_max_slope=10
  assert sim.recompute_navmesh(sim.pathfinder,ns);sim.pathfinder.seed(20261004);result=None
  for i in range(4000):
   a=sim.pathfinder.get_random_navigable_point();b=sim.pathfinder.get_random_navigable_point()
   if not 2.7<=np.linalg.norm((np.asarray(b)-np.asarray(a))[[0,2]])<=5 or abs(a[1]-b[1])>.08:continue
   result=sample_curve(sim,r,a,b)
   if result and abs(float(np.median(np.asarray(result['points_m'])[:,1])))<.05:break
   result=None
  if result is None:raise RuntimeError('No cleared curved Kujiale path')
  nav=R/'navmesh/kujiale_v3.navmesh';nav.parent.mkdir(parents=True,exist_ok=True);sim.pathfinder.save_nav_mesh(str(nav))
  rec={**result,'house':'kujiale','scene':str(stage),'source_usd':'/data/avengine_external/ue-assets/kujiale_0020_full_home_current_d702c47_20260821T0130Z/derived/kujiale_0020_full_home_preview_surface.usda','source_triangles':1891369,'source_to_navigation_m':'UEcm(x,y,z) -> (y,z,-x)/100; reflection restores source USD winding','navigation_to_ue_cm':'100*(-z,x,y)','agent_height_m':2.05,'agent_radius_m':.65,'navmesh':str(nav),'candidate_attempts':i+1,'native_navigation_backend':'Recast on task-owned exact USD visible geometry','visual_backend':'UE/SPEAR native Kujiale; Habitat RGB not used'}
  rec['stage_up']=[0,1,0];rec['stage_front']=[0,0,-1];rec['coordinate_revision']='explicit_stage_orientation_v3';json_write(R/'registrations/path_kujiale_v3.json',rec);print('KUJIALE_PATH_DONE',rec['duration_s'])
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--task-root',type=Path,required=True);main(a.parse_args().task_root)
