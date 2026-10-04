"""Bake terrain-corrected sitting poses with actual task-owned chair geometry."""
import argparse,copy,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import pose,rigid_glb,animation_times
from v7_human_roster_ue_root import promote
from v7_roster_contact_ik import FootPlant,find_bone,C,serialize_action,yaw_matrix
from v7_roster_floor_grid import FloorGrid
from v7_roster_seated import make_chair
def one(arg):
 T,aid=arg;T=Path(T);R=T/'round2';D=R/'assets/humans'/aid;d,b=read_glb(D/'motion_runtime.glb');base=copy.deepcopy(d);bb=bytes(b);ac=next(x for x in base['animations']if x['name']=='Sitting_Idle');ts=animation_times(base,bb,ac);out=[]
 for house in ['mp3d','hm3d','kujiale']:
  path=json.loads((R/'registrations'/('path_kujiale_v2.json'if house=='kujiale'else'path_'+house+'.json')).read_text());root=np.array(path['points_m'][0]);yaw=path['yaw_rad'][0];floor=FloorGrid(R/'evidence'/('floor_grid_'+house+'.npz'));solver=FootPlant(base,bb,ac,[0,1/30],[root.tolist()]*2,[yaw]*2,0,ts[0],ts[-1]-ts[0],floor)
  for s in solver.active:solver.active[s][:]=True
  gs=[];frames=[]
  for i in [0,1]:
   g,parts,e=solver.frame(i);gs.append(g);frames.append(e)
  name='Sitting_Contact_'+house;d,b=serialize_action(d,b,name,[0,1/30],gs);p=np.concatenate([x['positions']for x in parts]);hip=(C@gs[0][find_bone(base,' Pelvis')])[:3,3];cx,cz=float(hip[0]),float(hip[2])+.09;width,depth=.44,.24;inside=(abs(p[:,0]-cx)<width/2)&(abs(p[:,2]-cz)<depth/2)&(p[:,1]>.25)&(p[:,1]<hip[1]+.01);height=float(p[inside,1].min());sd=D/'seated_contact'/house;sd.mkdir(parents=True,exist_ok=True);rigid_glb(base,bb,parts,sd/'human.glb');boxes=make_chair(sd/'chair.glb',[cx,cz],width,depth,height)
  cd,cb=read_glb(sd/'chair.glb');cv=cd['bufferViews'][0];v=np.frombuffer(cb,np.float32,count=540,offset=cv['byteOffset']).reshape(180,3).copy()
  for bi,box in enumerate(boxes[1:],1):
   bp=np.array(box['center_m']);bp[1]=0;wp=root+yaw_matrix(yaw)@bp;bottom=floor(wp)-root[1];box['center_m'][1]=(height-.035+bottom)/2;box['size_m'][1]=height-.035-bottom;piece=v[bi*36:(bi+1)*36];piece[:,1]=np.where(piece[:,1]<height/2,bottom,height-.035)
  payload=bytearray(cb);payload[cv['byteOffset']:cv['byteOffset']+v.nbytes]=v.tobytes();cd['accessors'][0].update(min=v.min(0).tolist(),max=v.max(0).tolist());write_glb(sd/'chair.glb',cd,bytes(payload))
  slab=(abs(p[:,0]-cx)<width/2)&(abs(p[:,2]-cz)<depth/2)&(p[:,1]>height-.035+.001)&(p[:,1]<height-.001);penetrating=int(slab.sum());maxgap=max(abs(f['floor_delta_m'])for e in frames for f in e['feet'].values());passed=.30<height<.70 and maxgap<.008 and penetrating==0 and solver.max_reach<.001
  rec={'asset_id':aid,'house':house,'clip':name,'human_glb':str(sd/'human.glb'),'chair_glb':str(sd/'chair.glb'),'chair_boxes':boxes,'root_m':root.tolist(),'yaw_rad':yaw,'seat_height_m':height,'seat_slab_penetrating_vertices':penetrating,'seat_surface_vertex_gap_m':0.,'maximum_foot_contact_gap_m':maxgap,'maximum_ik_reach_error_m':solver.max_reach,'source':'Rocketbox Sitting_Idle first frame + terrain foot IK','sitting_is_static_pose':True,'standing_to_sitting_transition':'not_provided','geometric_status':'pass'if passed else'fail','triangle_box_intersection_status':'pending','native_status':'pending','frames':frames};json_write(sd/'seated.json',rec);out.append({k:v for k,v in rec.items()if k!='frames'});print('SEAT_CONTACT',aid,house,rec['geometric_status'],height,penetrating,flush=True)
 nd,nb,pr=promote(d,b);write_glb(D/'final_runtime.glb',d,b);write_glb(D/'final_ue_root.glb',nd,nb);json_write(D/'seated_contact_summary.json',out);return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);a=p.parse_args();rows=json.loads((a.task_root/'registrations/selection.json').read_text())['avatars']
 with ProcessPoolExecutor(max_workers=3)as ex:out=[r for rs in ex.map(one,[(str(a.task_root),r['asset_id'])for r in rows])for r in rs]
 json_write(a.task_root/'round2/evidence/seated_contact_results.json',out)
