"""Normalize extended human rigs and emit complete source-motion pose metadata."""
from pathlib import Path
import argparse,sys,json,numpy as np
from concurrent.futures import ProcessPoolExecutor
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import pose,animation_times,accessor
from v7_human_roster_ue_root import promote
def one(arg):
 T,aid=arg;T=Path(T);R=T/'round2';D=R/'assets/humans'/aid
 sys.path.insert(0,'/data/jzy/code/SPEAR-lead-b-m6-atomic-audit/tools')
 from normalize_rocketbox_glb_for_ue import normalize_in_place_grounded_metric_glb_bytes
 d,b=read_glb(D/'source_units.glb')
 r=next(i for i,n in enumerate(d['nodes'])if n.get('name')=='HumanRosterRoot');root=d['nodes'][r];arm=root['children'][0];n=d['nodes'][arm]
 n['translation']=[x+y for x,y in zip(n.get('translation',[0,0,0]),root.get('translation',[0,0,0]))];root['children']=[];d['scenes'][d.get('scene',0)]['nodes']=[arm]
 write_glb(D/'pre_metric.glb',d,b);raw,normal=normalize_in_place_grounded_metric_glb_bytes((D/'pre_metric.glb').read_bytes())
 (D/'runtime.glb').write_bytes(raw);json_write(D/'normalization.json',normal)
 d,b=read_glb(D/'runtime.glb');nd,nb,info=promote(d,b);write_glb(D/'ue_root.glb',nd,nb)
 results={}
 for ac in d['animations']:
  ts=animation_times(d,b,ac);samples=sorted(set([float(ts[0]),float(ts[len(ts)//2]),float(ts[-1])]))
  frames=[]
  for t in samples:
   _,ev=pose(d,b,ac,t);frames.append(ev)
  results[ac['name']]={'duration_s':float(ts[-1]-ts[0]),'first_time_s':float(ts[0]),'frame_count':len(ts),'samples':frames}
 json_write(D/'actions.json',{'asset_id':aid,'actions':results,'ue_root_promotion':info})
 return {'asset_id':aid,'status':'pass','actions':list(results),'frame_counts':{k:v['frame_count']for k,v in results.items()}}
def run(T):
 rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];args=[(str(T),r['asset_id'])for r in rows if(T/'round2/assets/humans'/r['asset_id']/'bake.json').exists()]
 with ProcessPoolExecutor(max_workers=3)as pool:out=list(pool.map(one,args))
 json_write(T/'round2/evidence/extended_normalization.json',out);print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
