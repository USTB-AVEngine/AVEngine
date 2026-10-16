"""Audit dog four-paw contact using their native meshes and donor bone angles."""
import argparse,copy,json,math
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import pose,rigid_glb,accessor,animation_times,globals_at
from v7_roster_contact_ik import FootPlant,find_bone,serialize_action
from v7_roster_floor_grid import FloorGrid
def one(T,breed):
 D=T/'round2/assets/dogs'/breed;d,b=read_glb(D/'runtime.glb');base=copy.deepcopy(d);bb=bytes(b);chains={}
 for s,p in [('FL',[' L UpperArm',' L Forearm',' L Hand',' L Finger0']),('FR',[' R UpperArm',' R Forearm',' R Hand',' R Finger0']),('HL',[' L Thigh',' L Calf',' L Foot',' L Toe0']),('HR',[' R Thigh',' R Calf',' R Foot',' R Toe0'])]:chains[s]=[find_bone(d,x)for x in p]
 masks=[]
 for node in d['nodes']:
  if 'mesh'not in node:continue
  for p in d['meshes'][node['mesh']]['primitives']:
   js=accessor(d,b,p['attributes']['JOINTS_0']).astype(int);w=accessor(d,b,p['attributes']['WEIGHTS_0']);joints=d['skins'][node['skin']]['joints'];masks.append({s:(np.isin(js,[joints.index(k)for k in chain[2:]])*w).sum(1)>.15 for s,chain in chains.items()})
 from v7_roster_disk import check
 check(T,512*1024**2)
 records=[];poses=D/'poses';poses.mkdir(exist_ok=True)
 for house in ['hm3d','mp3d','kujiale']:
  pathfile=T/'round3/registrations/dog_paths'/breed/(house+'.json');pathfile=pathfile if pathfile.exists()else T/'round2/registrations'/('path_kujiale_v3.json'if house=='kujiale'else'path_'+house+'.json');path=json.loads(pathfile.read_text());floor=FloorGrid(T/'round2/evidence'/('floor_grid_'+house+('_v3'if house=='kujiale'else'_wide')+'.npz'))
  for family in ['Idle','Walking']:
   ac=next(a for a in base['animations']if a['name']==family);ts=animation_times(base,bb,ac)
   if family=='Idle':times=np.arange(61)/30;positions=[path['points_m'][0]]*len(times);yaws=[path['yaw_rad'][0]]*len(times);rate=1.
   else:
    target=.3 if breed=='beagle'else .8;n=int(math.ceil(path['length_m']/target*30));times=np.arange(n+1)/30;old=np.asarray(path['time_s']);phase=np.linspace(old[0],old[-1],n+1);rawp=np.asarray(path['points_m']);positions=np.column_stack([np.interp(phase,old,rawp[:,j])for j in range(3)]).tolist();yaws=np.interp(phase,old,np.unwrap(path['yaw_rad'])).tolist()
    velocity=[]
    for chain in chains.values():
     vs=np.asarray([globals_at(base,bb,ac,float(t))[chain[2]][2,3]for t in ts]);vv=np.gradient(vs,ts);velocity.extend(vv[vv>.08])
    nominal=float(np.median(velocity));rate=target/nominal

   solver=FootPlant(base,bb,ac,times,positions,yaws,rate,ts[0],ts[-1]-ts[0],floor,basis=np.eye(4),chains=chains,masks=masks,canonical_flip=False,emitter_bone='Xtra Mouth',backward_threshold=.08)
   if family=='Idle':
    for side in chains:solver.active[side][:]=True
   else:
    for side in chains:
     # Keep authored swing/stance timing but include raised donor paws as contact candidates.
     heights=np.array([s[2][side]for s in solver.source]);
     if breed!='beagle':solver.active[side]|=heights<np.quantile(heights,.3)
   gs=[];frames=[]
   for fi,t in enumerate(times):
    g,p,ev=solver.frame(fi);gs.append(g);frames.append(ev)
    if fi in [0,len(times)//2,len(times)-1]:rigid_glb(base,bb,p,poses/(house+'_'+family+'_'+str(fi)+'.glb'))
   name=family+'_Contact_'+house;d,b=serialize_action(d,b,name,times,gs);active=[ft for e in frames for ft in e['feet'].values()if ft['in_contact']];hi=max(abs(f['floor_delta_m'])for f in active);lo=min(f['floor_delta_m']for e in frames for f in e['feet'].values());lock=max(f['support_lock_error_m']for f in active)
   passed=hi<.008 and lo>-.008 and lock<.003 and solver.max_reach<.001
   rec={'asset_id':breed,'house':house,'family':family,'clip':name,'frame_count':len(times),'duration_s':float(times[-1]),'maximum_contact_floor_delta_m':hi,'minimum_foot_floor_delta_m':lo,'maximum_support_lock_error_m':lock,'maximum_reach_error_m':solver.max_reach,'geometric_status':'pass'if passed else'fail','native_status':'pending','frames':frames,'path_points_m':positions,'yaw_rad':list(yaws),'source_play_rate':float(rate),'root_speed_m_s':0. if family=='Idle'else target}
   json_write(D/(house+'_'+family+'.json'),rec);records.append({k:v for k,v in rec.items()if k!='frames'});print('DOG_CONTACT',breed,house,family,rec['geometric_status'],round(hi,4),round(solver.max_reach,4),flush=True)

 from scipy.spatial.transform import Rotation
 for node in d['nodes']:
  if 'matrix'not in node:continue
  m=np.asarray(node.pop('matrix')).reshape(4,4).T;scale=np.linalg.norm(m[:3,:3],axis=0);r=m[:3,:3]/scale
  if np.linalg.det(r)<0:scale[0]*=-1;r[:,0]*=-1
  node.update(translation=m[:3,3].tolist(),rotation=Rotation.from_matrix(r).as_quat().tolist(),scale=scale.tolist())
 from v7_human_roster_ue_root import promote
 d,b,_=promote(d,b);write_glb(D/'motion_runtime.glb',d,b);json_write(D/'motion_summary.json',records);return records
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--only');a=p.parse_args();out=[]
 for breed in ([a.only]if a.only else ['beagle','german_shepherd']):out.extend(one(a.task_root,breed))
 json_write(a.task_root/'round2/evidence/dog_motion_results.json',out)
