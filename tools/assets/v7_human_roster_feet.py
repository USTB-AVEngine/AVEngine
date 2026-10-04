"""Check semantic human shoe vertices, not just whole-asset minimum height."""
import argparse,json
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import accessor,pose,animation_times
def run(T):
 rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];results=[]
 for row in rows:
  aid=row['asset_id'];c='red'if row['controlled_color']else'original';base=T/'assets/runtime'/aid/c/'runtime.glb';d,b=read_glb(base);masks=[]
  for node in d['nodes']:
   if 'mesh'not in node:continue
   for p in d['meshes'][node['mesh']]['primitives']:
    feet={}
    if 'skin'in node:
     skin=d['skins'][node['skin']];names=[d['nodes'][j].get('name','').lower().replace(' ','_')for j in skin['joints']];js=accessor(d,b,p['attributes']['JOINTS_0']).astype(int);ws=accessor(d,b,p['attributes']['WEIGHTS_0'])
     for side in ['l','r']:
      ids=[i for i,n in enumerate(names)if '_'+side+'_'in n and('foot'in n or'toe'in n)];feet[side]=(np.isin(js,ids)*ws).sum(axis=1)>.15
    masks.append(feet)
  action=next(x for x in d['animations']if x['name']=='Standing_Idle');times=animation_times(d,b,action);posed,_=pose(d,b,action,float(times[0]));cache=T/'assets/motion'/aid;manifest=json.loads((cache/'motion.json').read_text());frames=[]
  def floor(parts):
   minima={}
   for side in ['l','r']:
    ys=[p[:,1][m[side]]for p,m in zip(parts,masks)if side in m and m[side].any()]
    minima[side]=float(np.concatenate(ys).min())if ys else None
   return minima
  standing=floor([p['positions']for p in posed])
  with np.load(cache/'walk.npz')as data:
   for fi in range(manifest['walk']['frame_count']):frames.append(floor([data[f'p_{fi}_{pi}']for pi in range(len(masks))]))
  passed=all(v is not None and abs(v)<.015 for v in standing.values())and all(all(v is not None and v>=-.001 for v in f.values())and min(f.values())<.015 for f in frames)
  rec={'asset_id':aid,'standing_shoe_min_y_m':standing,'walking_frames':frames,'status':'pass'if passed else'fail','threshold_stand_each_shoe_m':.015,'threshold_walk_at_least_one_shoe_m':.015,'method':'native source skin weights identify Foot/Toe vertices; posed rigid geometry and every walking cache frame'};results.append(rec);print('FEET',aid,rec['status'],standing,flush=True)
 json_write(T/'evidence/rocketbox_semantic_foot_qa.json',results)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
