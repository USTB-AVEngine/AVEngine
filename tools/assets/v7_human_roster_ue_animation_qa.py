"""Compare UE-native animation anchors to the source metric glTF rig."""
import argparse,json
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import globals_at
def normal(s):return s.lower().replace(' ','').replace('_','').replace('-','')
def run(T,readback_name="ue_animation_readback.json",result_name="ue_animation_source_equivalence.json"):
 result=[]
 for row in json.loads((T/'evidence'/readback_name).read_text()):
  aid=row['asset_id'];d,b=read_glb(T/'assets/metric'/aid/'runtime.glb');actions={a['name']:a for a in d['animations']};mouth=next(i for i,n in enumerate(d['nodes'])if n.get('name','').endswith('MUpperLip'));feet={side:next(i for i,n in enumerate(d['nodes'])if normal(n.get('name','')).endswith(side.lower()+'foot'))for side in ['L','R']}
  for action in row['actions']:
   name=action['asset'].split('/')[-1].split('.')[0];a=actions[name]
   for frame in action['frames']:
    g=globals_at(d,b,a,frame['time_s']);native={'mouth':frame['mouth_cm'],**frame['ankles_cm']};errs={}
    for key,ni in {'mouth':mouth,**feet}.items():
     v=g[ni][:3,3]*100;expected=v[[0,2,1]];errs[key]=float(np.linalg.norm(np.array(native[key])-expected))
    result.append({'asset_id':aid,'action':name,'time_s':frame['time_s'],'anchor_error_cm':errs,'status':'pass'if max(errs.values())<.3 else'fail'})
 json_write(T/'evidence'/result_name,{'samples':result,'sample_count':len(result),'maximum_anchor_error_cm':max(max(r['anchor_error_cm'].values())for r in result),'status':'pass'if all(r['status']=='pass'for r in result)else'fail','threshold_cm':.3,'comparison':'native AnimPose world/component-space mouth and ankle positions versus metric source glTF channel evaluation; no scene-colour comparison'})
 print(len(result),max(max(r['anchor_error_cm'].values())for r in result),{r['status']for r in result})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--readback-name',default='ue_animation_readback.json');p.add_argument('--result-name',default='ue_animation_source_equivalence.json');a=p.parse_args();run(a.task_root,a.readback_name,a.result_name)
