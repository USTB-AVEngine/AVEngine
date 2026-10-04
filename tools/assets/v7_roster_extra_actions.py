"""Build floor-fixed speech gestures and turn actions from official Rocketbox clips."""
import argparse,copy,json,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import pose,globals_at,animation_times,accessor
from v7_roster_contact_ik import FootPlant,find_bone,serialize_action
from v7_roster_collision import intersections,clipped_surface_minimum
from v7_roster_disk import check
class Flat:
 def __call__(self,p):return 0.
 def many(self,p):return np.zeros(len(p))
def triangles(d,b,parts,offset):
 return np.concatenate([(p['positions']+offset)[accessor(d,b,d['meshes'][d['nodes'][p['node']]['mesh']]['primitives'][p['primitive']]['indices']).astype(int).reshape(-1,3)]for p in parts])
def one(args):
 T,aid=args;T=Path(T);D=T/'round2/assets/humans'/aid;d,b=read_glb(D/'final_ue_root.glb');base=copy.deepcopy(d);bb=bytes(b);fits=next(r for r in json.loads((T/'round3/evidence/chair_body_compatibility.json').read_text())if r['asset_id']==aid);chairs={c['asset_id']:c for c in json.loads((T/'round3/registrations/chair_roster.json').read_text())['chairs']};f=next((r for r in fits['styles']if r['status']=='pass'and r['chair_id']=='v7_chair_stool'),next(r for r in fits['styles']if r['status']=='pass'));ch=chairs[f['chair_id']];offset=np.asarray(f['human_local_offset_m']);out=[]
 for family in ['Standing_Talk','Turning_Left90','Sitting_Talk']:
  frames=[];gs=[];seats=[]
  if family=='Sitting_Talk':
   ac=next(a for a in base['animations']if a['name']=='Sitting_Gesture');sit=next(a for a in base['animations']if a['name']=='Sitting_Contact_kujiale');bg=globals_at(base,bb,sit,0);parents={}
   for i,n in enumerate(base['nodes']):
    for c in n.get('children',[]):parents[c]=i
   chosen=set()
   for suffix in [' L Clavicle',' R Clavicle',' Neck']:
    try:chosen.add(find_bone(base,suffix))
    except StopIteration:pass
   while True:
    old=len(chosen);chosen.update(c for k in list(chosen)for c in base['nodes'][k].get('children',[]))
    if len(chosen)==old:break
   duration=min(4.,float(animation_times(base,bb,ac)[-1]));times=np.arange(int(duration*30)+1)/30
   for i,t in enumerate(times):
    source=globals_at(base,bb,ac,float(t));g={}
    def build(k):
     if k in g:return g[k]
     par=parents.get(k);origin=source if k in chosen else bg;local=origin[k]if par is None else np.linalg.inv(origin[par])@origin[k];g[k]=local if par is None else build(par)@local;return g[k]
    for k in bg:build(k)
    ps,ev=pose(base,bb,globals_override=g,ground=False);tri=triangles(base,bb,ps,offset);point=clipped_surface_minimum(tri,[0,0],[ch['seat_width_m'],ch['seat_depth_m']],.25);h=float(point[1])if point is not None else float('nan');scale=f['instance_scale_y'];hits={}
    for bx in ch['solid_boxes']:
     c=np.asarray(bx['center_m']).copy();sz=np.asarray(bx['size_m']).copy();c[1]*=scale;sz[1]*=scale;n=len(intersections(tri,c,sz,bx.get('yaw_rad',0),.004))
     if n:hits[bx['name']]=hits.get(bx['name'],0)+n
    ev.update(time_s=float(t),seat_contact_gap_m=h-f['seat_height_m'],chair_intersections=hits,chair_id=f['chair_id']);frames.append(ev);gs.append(g)
   contact=max(abs(r['seat_contact_gap_m'])for r in frames);floor=min(r['bounds_min_m'][1]for r in frames);bad=sum(bool(r['chair_intersections'])for r in frames);passed=contact<.008 and floor>-.008 and bad==0;metrics={'maximum_seat_gap_m':contact,'minimum_mesh_floor_y_m':floor,'chair_collision_frames':bad,'chair_fit':f,'fixed_lower_body':True,'source_transplant':'official Sitting_Gesture arm chains and Neck descendants, full base lower body and spine retained'}
  else:
   ac=next(a for a in base['animations']if a['name']==family);ts=animation_times(base,bb,ac);duration=min(6. if family=='Standing_Talk'else 3.,float(ts[-1]-ts[0]));times=np.arange(int(duration*30)+1)/30
   solver=FootPlant(base,bb,ac,times,np.zeros((len(times),3)),np.zeros(len(times)),1,float(ts[0]),float(ts[-1]-ts[0])+.00001,Flat())
   for side in solver.active:
    if family=='Standing_Talk':solver.active[side][:]=True
    else:
     heights=np.array([r[2][side]for r in solver.source]);solver.active[side]=heights<max(.015,float(np.quantile(heights,.4)))
   for i,t in enumerate(times):g,ps,ev=solver.frame(i);gs.append(g);frames.append(ev)
   active=[f for r in frames for f in r['feet'].values()if f['in_contact']];contact=max(abs(r['floor_delta_m'])for r in active)if active else 100;low=min(f['floor_delta_m']for r in frames for f in r['feet'].values());lock=max(r['support_lock_error_m']for r in active)if active else 100
   passed=contact<.008 and low>-.008 and lock<.003 and solver.max_reach<.001;metrics={'maximum_contact_floor_delta_m':contact,'minimum_foot_floor_delta_m':low,'maximum_support_lock_error_m':lock,'maximum_ik_reach_error_m':solver.max_reach}
  name=family+'_Contact';d['animations']=[a for a in d['animations']if a['name']!=name];d,b=serialize_action(d,b,name,times,gs);rec={'asset_id':aid,'family':family,'clip':name,'frame_count':len(frames),'duration_s':float(times[-1]),'geometric_status':'pass'if passed else'fail','native_status':'pending','frames':frames,**metrics};json_write(T/'round3/evidence/extra_actions'/aid/(family+'.json'),rec);out.append({k:v for k,v in rec.items()if k!='frames'});print('EXTRA_ACTION',aid,family,rec['geometric_status'],metrics,flush=True)
 check(T,64*1024**2);temp=D/'extra_actions_pending.glb';write_glb(temp,d,b);temp.replace(D/'final_ue_root.glb');return out
def run(T):
 rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];check(T,1024*1024**2)
 with ProcessPoolExecutor(max_workers=3)as pool:out=[r for rs in pool.map(one,[(str(T),r['asset_id'])for r in rows])for r in rs]
 json_write(T/'round3/evidence/extra_action_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
