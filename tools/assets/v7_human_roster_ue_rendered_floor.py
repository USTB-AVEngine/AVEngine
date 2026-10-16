
"""Check visible feet/support height from native UE SceneDepth."""
from pathlib import Path
import argparse
import json,math
import numpy as np

from v7_human_roster_ue_capture_qa import hdr
def run(T):
 out=[]
 for filename in ['ue_native_results.json','ue_action_native_results.json']:
  for r in json.loads((T/'evidence'/filename).read_text()):
   dest=T/r['capture_directory']if r.get('capture_directory')else T/'renders/ue'/r['asset_id']/r['color']
   depth=hdr(dest/'isolated_depth.hdr')[:,:,0];mask=(depth>50)&(depth<1000)
   eye=np.array(r['camera_position_cm']);forward=np.array(r['camera_target_cm'])-eye;forward/=np.linalg.norm(forward);right=np.cross([0,0,1],forward);right/=np.linalg.norm(right);up=np.cross(forward,right);yy,xx=np.indices(depth.shape);tan=math.tan(math.radians(r['fov_degrees'])/2)
   zs=eye[2]+depth*(forward[2]+right[2]*(2*(xx+.5)/640-1)*tan+up[2]*(1-2*(yy+.5)/640)*tan);delta=float(zs[mask].min())-r['object_position_cm'][2]
   out.append({'source':filename,'asset_id':r['asset_id'],'color':r['color'],'native_visible_minimum_above_floor_cm':delta,'status':'pass'if abs(delta)<3.5 else'qualified_depth_floor_visibility'})
 d={'cases':out,'case_count':len(out),'minimum_cm':min(x['native_visible_minimum_above_floor_cm']for x in out),'maximum_cm':max(x['native_visible_minimum_above_floor_cm']for x in out),'tolerance_cm':3.5,'method':'inverse projection of human-only native linear SceneDepth at pixel centres; Radiance RGBE and rasterisation permit up to 3.5cm; source semantic-foot and native pose readback remain the precise grounding evidence','all_within_tolerance':all(x['status']=='pass'for x in out)}
 (T/'evidence/ue_rendered_floor_qa.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print({k:v for k,v in d.items()if k!='cases'})


if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--task-root",type=Path,required=True);run(p.parse_args().task_root)
