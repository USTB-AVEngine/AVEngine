"""Fit silent chair styles to exact seated surfaces, then check triangle penetration."""
import argparse,json
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,accessor
from v7_roster_collision import intersections,clipped_surface_minimum
def run(T):
 chairs=json.loads((T/'round3/registrations/chair_roster.json').read_text())['chairs'];out=[]
 for row in json.loads((T/'registrations/selection.json').read_text())['avatars']:
  aid=row['asset_id'];D=T/'round2/assets/humans'/aid;d,b=read_glb(D/'final_ue_root.glb');ac=next(a for a in d['animations']if a['name']=='Sitting_Contact_hm3d');p,e=pose(d,b,ac,0,ground=False);sc=json.loads((D/'seated_contact/hm3d/seated.json').read_text());cx,_,cz=sc['chair_boxes'][0]['center_m'];tri=[]
  for part in p:
   primitive=d['meshes'][d['nodes'][part['node']]['mesh']]['primitives'][part['primitive']];idx=accessor(d,b,primitive['indices']).astype(int).reshape(-1,3);tri.append((part['positions']-[cx,0,cz])[idx])
  original=np.concatenate(tri);rec={'asset_id':aid,'source_seat_height_m':sc['seat_height_m'],'supported_styles':[],'styles':[]}
  for ch in chairs:
   best=None
   for dz in [0,.025,-.025,.05,-.05,.075,-.075,.10]:
    triangles=original-[0,0,dz];point=clipped_surface_minimum(triangles,[0,0],[ch['seat_width_m'],ch['seat_depth_m']],.25)
    if point is None or not .29<point[1]<.56:continue
    height=float(point[1]);scale=height/ch['seat_height_m']
    if not .70<scale<1.3:continue
    hits={}
    for bx in ch['solid_boxes']:
     center=np.array(bx['center_m']);size=np.array(bx['size_m']);center[1]*=scale;size[1]*=scale;n=len(intersections(triangles,center,size,bx.get('yaw_rad',0),.004))
     if n:hits[bx['name']]=n
    candidate={'chair_id':ch['asset_id'],'chair_style':ch['style'],'seat_height_m':height,'instance_scale_y':scale,'human_local_offset_m':[-cx,0,-cz-dz],'contact_surface_point_local_m':point.tolist(),'triangle_solid_box_intersections':hits,'contact_gap_m':0.,'status':'pass'if not hits else'fail','qualification':'exact clipped-triangle contact height, conservative inner boxes with 4mm surface tolerance; scene placement and animated gestures require separate receipts'}
    if best is None or sum(hits.values())<sum(best['triangle_solid_box_intersections'].values()):best=candidate
    if not hits:break
   if best is None:best={'chair_id':ch['asset_id'],'chair_style':ch['style'],'status':'fail','reason':'no ergonomic contact height in allowed range'}
   rec['styles'].append(best)
   if best['status']=='pass':rec['supported_styles'].append(ch['style'])
  out.append(rec)
 json_write(T/'round3/evidence/chair_body_compatibility.json',out);print('COMPATIBILITY',len(out),sum(len(r['supported_styles'])for r in out),sum(bool(r['supported_styles'])for r in out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
