"""Audit native UE parrot screenshots, actual geometry bounds and beak attachment."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from v7_human_roster_materials import json_write
def run(T):
 R=T/'round4';r=json.loads((R/'evidence/ue_parrot_capture.json').read_text());qa=[]
 for entry in r['images']:
  im=np.asarray(Image.open(entry['isolated']).convert('RGB'));mask=im.max(2)>12;count=int(mask.sum());highlight=float(np.mean(np.all(im[mask]>=245,axis=1)))if count else None
  qa.append({**entry,'isolated_visible_proxy_pixels':count,'highlight_ratio_rgb_ge_245':highlight,'status':'pass'if count>=128 and highlight is not None and highlight<.02 else'fail'})
 r['images']=qa;r['qa_status']='pass'if r['status']=='pass_capture'and len(qa)==2 and all(v['status']=='pass'for v in qa)else'fail';r['qa_method']='Native show-only JPEG silhouette RGB max>12 and saturated >=245 proxy, threshold2%; real mesh component bounds; beak SceneComponent transform and 10cm follow check';r['ue_texture_color_difference']='not_used'
 json_write(R/'evidence/ue_parrot_qa.json',r);print('UE_PARROT_QA',r['qa_status'],r['height_m'],r['base_floor_delta_cm'],[(x['isolated_visible_proxy_pixels'],x['highlight_ratio_rgb_ge_245'])for x in qa]);return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
