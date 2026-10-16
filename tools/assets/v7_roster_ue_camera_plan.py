"""Choose UE review cameras with clear rays through the actual source-room mesh."""
import argparse,json,math
from pathlib import Path
import numpy as np
from v7_roster_contact_ik import yaw_matrix
from v7_human_roster_materials import json_write
def visible(o,p,tri,lo,hi):
 delta=p-o;length=np.linalg.norm(delta);v=delta/length;keep=np.all(hi>=np.minimum(o,p)-.002,1)&np.all(lo<=np.maximum(o,p)+.002,1);t=tri[keep]
 if not len(t):return True
 e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0];h=np.cross(np.broadcast_to(v,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h);good=abs(det)>1e-9;inv=np.where(good,1/np.where(good,det,1),0);s=o-t[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);w=q@v*inv;depth=np.einsum('ij,ij->i',e2,q)*inv
 return not np.any(good&(u>=0)&(w>=0)&(u+w<=1)&(depth>.0001)&(depth<length-.03))
def run(T):
 raw=np.load(T/'evidence/ue_source_triangles.npz')['triangles'].astype(float)/100;tri=np.stack([raw[:,:,1],raw[:,:,2],-raw[:,:,0]],axis=2);lo=tri.min(1);hi=tri.max(1);cp=json.loads((T/'evidence/ue_source_floor_plan.json').read_text())['selected'];cm=cp['location_cm'];cr=np.array([cm[1],cm[2],-cm[0]])/100;jobs=[]
 for r in json.loads((T/'registrations/selection.json').read_text())['avatars']:
  aid=r['asset_id'];p=json.loads((T/'round2/assets/humans'/aid/'walking_kujiale.json').read_text());path=json.loads(Path(p.get('path_manifest')or T/'round2/registrations/path_kujiale_v3.json').read_text())
  for i in sorted(set([0,len(p['frames'])//3,2*len(p['frames'])//3,len(p['frames'])-1])):jobs.append((aid,'Walking',i,np.asarray(path['points_m'][i]),path['yaw_rad'][i],'human'))
  f=json.loads((T/'round3/evidence/extra_actions'/aid/'Sitting_Talk.json').read_text())['chair_fit'];hr=cr+np.asarray(f['human_local_offset_m'])
  for family in ['Sitting','Sitting_Talk','Standing_Talk','Turning_Left90']:jobs.append((aid,family,0,hr if family.startswith('Sitting')else cr,0,'human'))
 for breed in ['beagle','german_shepherd']:
  for family in ['Idle','Walking']:
   p=json.loads((T/'round2/assets/dogs'/breed/('kujiale_'+family+'.json')).read_text())
   for i in sorted(set([0,len(p['frames'])//2,len(p['frames'])-1])):jobs.append((breed,family,i,np.asarray(p['path_points_m'][i]),p['yaw_rad'][i],'dog'))
 for p in(T/'round3/assets/chairs').glob('*.glb'):jobs.append((p.stem,'Static',0,cr,0,'chair'))
 jobs.append(('v7_rocketbox_bird_sparrow_fallback_01','Static',0,cr,0,'bird'));out={}
 for aid,family,i,root,yaw,kind in jobs:
  target=root+[0,.42 if kind=='dog'else .28 if kind=='bird'else .52 if kind=='chair'else .90,0];accepted=None
  for delta in [0,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3,math.pi/2,-math.pi/2,math.pi]:
   if accepted:break
   for dist,fov in ([(.85,45)]if kind=='bird'else [(2.8,50),(2.4,50),(2.,60),(1.7,65)]):
    height=.45 if kind=='bird'else .85 if kind=='dog'else 1.22;eye=root+yaw_matrix(yaw+delta)@np.array([0,height,-dist]);points=[target,target+[0,.25,0],target-[0,.25,0],target+[.18,0,0],target-[.18,0,0],root+[0,.15,0],root+[0,1.65 if kind=='human'else .70 if kind=='chair'else .35,0]]
    if all(visible(eye,p,tri,lo,hi)for p in points):accepted={'eye_glb_m':eye.tolist(),'target_glb_m':target.tolist(),'fov_degrees':fov,'source_ray_count':len(points),'source_clear_rays':True};break
  out[aid+'/'+family+'/'+str(i)]=accepted or {'status':'no_clear_camera','source_clear_rays':False};print('UE_CAMERA',aid,family,i,bool(accepted),flush=True)
 json_write(T/'round3/registrations/ue_camera_plan.json',{'source':str(T/'evidence/ue_source_triangles.npz'),'method':'five exact triangle rays to the actor center band; native images and source geometry bounds agree in a separate receipt','cameras':out})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
