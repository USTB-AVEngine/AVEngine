"""Sample real native stage floors near each verified navigation path."""
import argparse,json
from pathlib import Path
import numpy as np
from v7_human_roster_materials import json_write
def open_scene(T,path):
 from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
 from avengine.rooms.contracts import load_and_validate_inputs
 R=T/'round2';v=path.get('view_input')or json.loads((R/'registrations/path_mp3d.json').read_text())['view_input']
 r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg')
 inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,_,_,_=_make_configuration(inputs,None,R/'navmesh'/path['house'],mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.create_renderer=False;cfg.sim_cfg.gpu_device_id=0
 if path['house']=='kujiale':cfg.sim_cfg.scene_id=path['scene'];cfg.sim_cfg.scene_dataset_config_file='default'
 return r,r.habitat_sim.Simulator(cfg)
def main(T):
 R=T/'round2';out=[]
 for house in ['mp3d','hm3d','kujiale']:
  name='path_kujiale_v3.json'if house=='kujiale'else'path_'+house+'.json';path=json.loads((R/'registrations'/name).read_text());points=np.array(path['points_m']);xs=np.arange(points[:,0].min()-1.2,points[:,0].max()+1.201,.02);zs=np.arange(points[:,2].min()-1.2,points[:,2].max()+1.201,.02);ys=np.full((len(xs),len(zs)),np.nan,np.float32);top=float(points[:,1].max())+.25;bottom=float(points[:,1].min())-.35
  r,sim=open_scene(T,path)
  with sim:
   for i,x in enumerate(xs):
    for j,z in enumerate(zs):
     hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(float(x),top,float(z)),r.magnum.Vector3(0,-1,0)),top-bottom).hits
     if hits:ys[i,j]=float(hits[0].point[1])
  dest=R/'evidence'/('floor_grid_'+house+('_v3'if house=='kujiale'else'_wide')+'.npz');np.savez_compressed(dest,x=xs,z=zs,y=ys);rec={'house':house,'grid':str(dest),'resolution_m':.02,'sample_count':int(ys.size),'valid_count':int(np.isfinite(ys).sum()),'native_physics_ray':True,'scene':path['scene'],'source_path_manifest':str(R/'registrations'/name),'ray_start_y_m':top,'ray_end_y_m':bottom};out.append(rec);print('NATIVE_FLOOR_GRID',house,rec['sample_count'],flush=True)
 json_write(R/'evidence/native_floor_grids.json',out)
class FloorGrid:
 def __init__(self,p):
  with np.load(p)as z:self.x=z['x'];self.z=z['z'];self.y=z['y']
 def many(self,p):
  p=np.asarray(p)
  if np.any((p[:,0]<self.x[0])|(p[:,0]>self.x[-1])|(p[:,2]<self.z[0])|(p[:,2]>self.z[-1])):raise RuntimeError("Point outside verified native floor grid")
  ix=np.clip(np.searchsorted(self.x,p[:,0])-1,0,len(self.x)-2);iz=np.clip(np.searchsorted(self.z,p[:,2])-1,0,len(self.z)-2);a=(p[:,0]-self.x[ix])/(self.x[ix+1]-self.x[ix]);b=(p[:,2]-self.z[iz])/(self.z[iz+1]-self.z[iz]);result=self.y[ix,iz]*(1-a)*(1-b)+self.y[ix+1,iz]*a*(1-b)+self.y[ix,iz+1]*(1-a)*b+self.y[ix+1,iz+1]*a*b
  if not np.isfinite(result).all():raise RuntimeError('Missing floor samples under the shoe')
  return result
 def __call__(self,p):
  return float(self.many(np.asarray(p).reshape(1,3))[0])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);main(p.parse_args().task_root)



def require_physical_ground(measured_y, reference_y, tolerance_m=.015):
 """Reject furniture-top support even when an unlabelled ray hit is horizontal."""
 if not np.isfinite(measured_y) or not np.isfinite(reference_y):
  raise ValueError("Physical floor reference is missing or non-finite")
 if abs(float(measured_y)-float(reference_y))>tolerance_m:
  raise ValueError("Foot ray hit furniture instead of navigation-supported physical ground")
 return float(measured_y)
