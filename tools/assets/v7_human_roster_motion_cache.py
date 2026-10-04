"""Repair source-motion caches using every animation channel's key times."""
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import animation_times,pose
def one(args):
 T,aid,color=args;T=Path(T);base=T/'assets/runtime'/aid/color/'runtime.glb';d,b=read_glb(base);action=next(x for x in d['animations']if x['name']=='Walking');sampled=animation_times(d,b,action);cache=T/'assets/motion'/aid;manifest=json.loads((cache/'motion.json').read_text());before=manifest['walk']['frame_count'];data={};records=[]
 for fi,t in enumerate(sampled):
  ps,ev=pose(d,b,action,float(t));records.append(ev)
  for pi,p in enumerate(ps):
   data[f'p_{fi}_{pi}']=p['positions'].astype('float32')
   if p['normals']is not None:data[f'n_{fi}_{pi}']=p['normals'].astype('float32')
 np.savez_compressed(cache/'walk.npz',**data);manifest['walk']={'source_action':'Walking','frame_count':len(records),'times_s':sampled.tolist(),'frames':records,'cache':str((cache/'walk.npz').relative_to(T)),'sampling':'union of all source animation channel key times'};json_write(cache/'motion.json',manifest)
 return {'asset_id':aid,'before_frames':before,'after_frames':len(records),'duration_s':float(sampled[-1]-sampled[0]),'all_floor_y_m':max(abs(r['floor_y_m'])for r in records)}
def run(T,workers):
 rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];args=[(str(T),r['asset_id'],'red'if r['controlled_color']else'original')for r in rows];records=[]
 with ProcessPoolExecutor(max_workers=workers)as pool:
  for rec in pool.map(one,args):records.append(rec);print('MOTION',json.dumps(rec),flush=True)
 json_write(T/'evidence/motion_cache_sampling_repair.json',records)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args();run(a.task_root,min(a.workers,3))
