"""Create seat geometry fitted to source sitting pose and a seated upper-body talk clip."""
import argparse,copy,json,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import accessor,globals_at,pose,rigid_glb,animation_times
from v7_roster_contact_ik import find_bone,subtree,serialize_action,C
def make_chair(path,center,width,depth,height):
 cx,cz=center;th=.035;boxes=[([cx,height-th/2,cz],[width,th,depth])]
 for sx in [-1,1]:
  for sz in [-1,1]:boxes.append(([cx+sx*(width/2-.035),height/2-th/2,cz+sz*(depth/2-.035)],[.045,height-th,.045]))
 verts=[];normals=[];faces=[([0,1,2,0,2,3],[0,0,-1]),([5,4,7,5,7,6],[0,0,1]),([4,0,3,4,3,7],[-1,0,0]),([1,5,6,1,6,2],[1,0,0]),([3,2,6,3,6,7],[0,1,0]),([4,5,1,4,1,0],[0,-1,0])]
 for center,size in boxes:
  x,y,z=np.array(size)/2;v=np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])+center
  for ids,n in faces:verts.extend(v[ids]);normals.extend([n]*6)
 verts=np.asarray(verts,np.float32);normals=np.asarray(normals,np.float32);b=verts.tobytes()+normals.tobytes();d={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0,'name':'V7SeatChair'}],'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1},'material':0}]}],'materials':[{'name':'V7SeatWood','pbrMetallicRoughness':{'baseColorFactor':[.24,.105,.032,1],'metallicFactor':0,'roughnessFactor':.85},'doubleSided':True}],'buffers':[{'byteLength':len(b)}],'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':verts.nbytes},{'buffer':0,'byteOffset':verts.nbytes,'byteLength':normals.nbytes}],'accessors':[{'bufferView':0,'componentType':5126,'type':'VEC3','count':len(verts),'min':verts.min(0).tolist(),'max':verts.max(0).tolist()},{'bufferView':1,'componentType':5126,'type':'VEC3','count':len(normals)}]};write_glb(path,d,b)
 return [{'center_m':list(np.asarray(c,float)),'size_m':list(np.asarray(s,float))}for c,s in boxes]
def one(arg):
 T,aid=arg;T=Path(T);D=T/'round2/assets/humans'/aid;d,b=read_glb(D/'runtime.glb');sit=next(a for a in d['animations']if a['name']=='Sitting_Idle');talk=next(a for a in d['animations']if a['name']=='Standing_Talk');t0=float(animation_times(d,b,sit)[0]);base=globals_at(d,b,sit,t0);parts,ev=pose(d,b,sit,t0);correction=ev['ground_correction_m']
 for m in base.values():m[1,3]+=correction
 pelvis=find_bone(d,' Pelvis');hip=(C@base[pelvis])[:3,3];p=np.concatenate([x['positions']for x in parts]);cx,cz=float(hip[0]),float(hip[2])+.09;width=.44;depth=.24
 inside=(abs(p[:,0]-cx)<width/2)&(abs(p[:,2]-cz)<depth/2)&(p[:,1]>hip[1]-.25)&(p[:,1]<hip[1]+.01);patch=p[inside]
 if not len(patch):raise ValueError('No buttock/thigh surface near pelvis')
 height=float(patch[:,1].min());center=[cx,cz];upper=subtree(d,find_bone(d,' Spine1'));parents={j:i for i,n in enumerate(d['nodes'])for j in n.get('children',[])};roots=d['scenes'][d.get('scene',0)]['nodes']
 for root in roots:parents.pop(root,None)
 times=np.arange(121)/30;gs=[];frames=[];maxpenetration=0;contacts=[];seatdir=D/'seated';seatdir.mkdir(exist_ok=True)
 for i,t in enumerate(times):
  tg=globals_at(d,b,talk,8+float(t));g=copy.deepcopy(base)
  for j in upper:
   par=parents.get(j);local=np.linalg.inv(tg[par])@tg[j]if par is not None else tg[j];slocal=np.linalg.inv(base[par])@base[j]if par is not None else base[j];local[:3,3]=slocal[:3,3];g[j]=g[par]@local if par is not None else local
  out,e=pose(d,b,None,float(t),globals_override=g,ground=False);v=np.concatenate([x['positions']for x in out]);on=(abs(v[:,0]-cx)<width/2)&(abs(v[:,2]-cz)<depth/2)&(v[:,1]>height-.035)&(v[:,1]<height+.2);ys=v[on,1];gap=float(ys.min()-height)if len(ys)else float('inf');penetration=max(0,-gap);contacts.append(gap);maxpenetration=max(maxpenetration,penetration);e.update(time_s=float(t),seat_contact_gap_m=gap);frames.append(e);gs.append(g)
  if i in [0,40,80,120]:rigid_glb(d,b,out,seatdir/(str(i).zfill(3)+'.glb'))
 nd,nb=serialize_action(d,b,'Sitting_Chair',[0,1/30],[base,base]);nd,nb=serialize_action(nd,nb,'Sitting_Talk_Chair',times,gs);write_glb(D/'seated_runtime.glb',nd,nb);boxes=make_chair(seatdir/'chair.glb',center,width,depth,height)
 pp,_=pose(d,b,None,0,globals_override=base,ground=False);rigid_glb(d,b,pp,seatdir/'sitting.glb')
 rec={'asset_id':aid,'seat_height_m':height,'seat_width_m':width,'seat_depth_m':depth,'seat_center_xz_m':center,'chair':str(seatdir/'chair.glb'),'chair_boxes':boxes,'sitting_pose_source':'Rocketbox Sitting_Idle first frame','seated_talk_source':'Rocketbox Sitting_Idle lower body plus Standing_Talk upper spine descendants, source seconds 8 through 12','seated_talk_is_layered':True,'audio_driven_visemes':False,'duration_s':4,'frame_count':121,'seat_contact_search_note':'actual seat slab upper surface; vertices below bottom face are not penetration','maximum_seat_penetration_m':maxpenetration,'maximum_seat_contact_gap_m':max(contacts),'minimum_seat_contact_gap_m':min(contacts),'foot_floor_gap_m':float(p[:,1].min()),'geometric_contact_status':'pass'if .30<height<.70 and maxpenetration<.008 and max(abs(x)for x in contacts)<.012 else'fail','native_seat_contact_status':'pending','frames':frames}
 json_write(D/'seated.json',rec);print('SEATED',aid,rec['geometric_contact_status'],round(height,4),round(maxpenetration,4),round(max(contacts),4),flush=True);return {k:v for k,v in rec.items()if k!='frames'}
def main(a):
 T=a.task_root;rows=json.loads((T/'registrations/selection.json').read_text())['avatars']
 with ProcessPoolExecutor(max_workers=3)as ex:out=list(ex.map(one,[(str(T),r['asset_id'])for r in rows]))
 json_write(T/'round2/evidence/seated_geometric_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);main(p.parse_args())
