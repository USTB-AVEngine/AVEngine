"""Apply the owner amendment to this task's derived metadata; retain removed evidence."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from v7_object_prepare import dump
from measure_static_upright_correction import load_glb
EXCLUDED={'bathtub','floor_drain','sink_with_tap','toilet','fireplace'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);a=p.parse_args();T=a.task_root;q=T/'prepared_existing.json';rows=json.loads(q.read_text());active=[]
 for m in rows:
  if m['object_type'] in EXCLUDED:continue
  if m['object_type']=='air_conditioner':m['silent_decor']=True;m['emitter']=None
  if m['object_type']=='doorbell_chime_unit':
   vertices,_=load_glb(Path(m['derived_path']));bbox=np.asarray(m['geometry']['bbox_m']);target=bbox[0]+np.array([.92,.5,.5])*(bbox[1]-bbox[0]);point=vertices[np.argmin(np.linalg.norm(vertices-target,axis=1))];m['silent_decor']=False;m['emitter']={'local_m':point.tolist(),'nearest_vertex_distance_m':0.,'physical_rationale':'机身中心高度与左右中心，靠前的最近网格表面','active':True}
  dump(T/'assets'/m['asset_id']/'asset.json',m);active.append(m)
 assert len(active)==30;dump(q,active);print('OWNER_SCOPE',len(active),'active;',sum(m['silent_decor'] for m in active),'silent; removed evidence retained')
if __name__=='__main__':main()
