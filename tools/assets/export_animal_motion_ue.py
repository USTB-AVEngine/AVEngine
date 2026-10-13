"""Prepare metric skeletal clips and validate a Kujiale route before import."""
import argparse,json,time
from pathlib import Path
import numpy as np
from avengine.assets.animal_motion.profiles import Rig
from avengine.assets.animal_motion.walk import ConstrainedWalk
from avengine.assets.animal_motion.export import export_walk
from avengine.assets.animal_motion.native import StageClearance
from avengine.assets.animal_motion.stage import OriginalVerticalStageRay
from avengine.assets.animal_motion.ue_floor import NativeFloorReference
from avengine.assets.roster_motion.episode import triangle_mesh

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--route',type=Path,required=True);p.add_argument('--speed',type=float,default=.08);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.process_time()
 record=next(x for x in json.loads(a.inventory.read_text())['assets'] if x['asset_id']==a.asset);h=json.loads(a.house.read_text());decl=h['input']['ue_config']['native_floor_surface'];native=NativeFloorReference(decl);mesh=native.mesh
 p0=np.asarray(json.loads(a.route.read_text())['root_m'],float);rays=OriginalVerticalStageRay(Path(decl['native_editor_audit']),mesh=mesh);clearance=StageClearance(mesh,p0[1])
 class Floor:
  def __call__(self,p):return float(rays.many([np.asarray(p)+p0])[0]-p0[1])
 rig=Rig.load(record);walk=ConstrainedWalk(rig,speed=a.speed,floor=Floor());times=np.arange(145)/30
 result=export_walk(rig,walk,times,a.output/'motion.glb');minimum=1.;minskin=1.
 for i,t in enumerate(times):
  parts,row=walk.frame(float(t));v,f=triangle_mesh(rig.document,rig.binary,parts);world=v+np.asarray(row['world_root_m'])+p0;minimum=min(minimum,clearance.check(world,f));near=world[world[:,1]<p0[1]+.08];minskin=min(minskin,float(np.min(near[:,1]-rays.many(near))))
  if minskin<-.00101:raise ValueError('Kujiale_source_geometry_penetrates_floor')
 result.update(asset_id=a.asset,status='cpu_route_pass_pending_native_UE',root_m=p0.tolist(),yaw_rad=0.,house=h['house'],house_source=str(a.house),clip_seconds=4.8,frame_count=145,minimum_nonfloor_clearance_m=minimum,minimum_full_skin_floor_m=minskin,cpu_s=time.process_time()-start,source_rig=str(rig.path),motion_glb=str(a.output/'motion.glb'))
 (a.output/'motion.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ('frames','bone_names','basis_baked_in_mesh_and_inverse_binds')},flush=True)
if __name__=='__main__':main()
