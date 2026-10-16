"""Audit native JPEG visibility/highlight proxies without UE texture-colour matching."""
import argparse,json
from pathlib import Path
import numpy as np
from v7_roster_contact_ik import yaw_matrix
from PIL import Image
from v7_human_roster_materials import json_write
def run(T):
 rows=json.loads((T/'round3/evidence/ue_native_capture_results.json').read_text());out=[]
 for row in rows:
  rec=dict(row);p=Path(row['isolated']);im=np.asarray(Image.open(p).convert('RGB'));mask=im.max(2)>12;count=int(mask.sum());ratio=float(np.mean(np.all(im[mask]>=245,axis=1)))if count else None
  if row['kind']=='human'and row['family']!='Walking':
   family='Sitting_Talk'if row['family']=='Sitting'else row['family'];plan=json.loads((T/'round3/evidence/extra_actions'/row['asset_id']/(family+'.json')).read_text());local=np.asarray(plan['frames'][int(round(row['time_s']*30))]['mouth_m']);point=np.asarray(row['root_m'])+yaw_matrix(row['yaw_rad'])@local;expected=np.array([-100*point[2],100*point[0],100*point[1]]);rec['mouth_expected_world_cm']=expected.tolist();rec['mouth_error_cm']=float(np.linalg.norm(np.asarray(row['mouth_world_cm'])-expected))
  rec.update(isolated_visible_proxy_pixels=count,highlight_ratio_rgb_ge_245=ratio,proxy_method='native show-only JPEG silhouette estimated by RGB max>12; saturation >=245 proxy, not a specular AOV; no UE texture colour difference test',qa_status='pass'if row['status']=='pass_capture'and rec.get('mouth_error_cm',0)<.3 and count>=128 and ratio is not None and ratio<.02 and row.get('material_slot_compatibility',True)else'fail')
  out.append(rec)
 json_write(T/'round3/evidence/ue_native_capture_qa.json',out);print('UE_CAPTURE_QA',len(out),sum(r['qa_status']=='pass'for r in out),max([r['highlight_ratio_rgb_ge_245']for r in out if r['highlight_ratio_rgb_ge_245']is not None]or[0]))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
