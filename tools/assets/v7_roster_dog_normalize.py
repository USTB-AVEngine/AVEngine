"""Normalize native dog rigs, preserving their own skin and complete donor motion."""
import argparse,copy,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import accessor,globals_at,animation_times,pose
from v7_roster_contact_ik import rotation_between
def one(T,aid,source):
 R=T/'round2';D=R/'assets/dogs'/aid;D.mkdir(parents=True,exist_ok=True);d,b=read_glb(source);skin=d['skins'][0];joints=skin['joints'];parent={c:i for i,n in enumerate(d['nodes'])for c in n.get('children',[])}
 ni=next(i for i,n in enumerate(d['nodes'])if n.get('skin')==0);meshworld=globals_at(d,b)[ni];ibm=accessor(d,b,skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1);bind=[np.linalg.inv(m)for m in ibm]
 yaw=np.array([[0,0,1],[0,1,0],[-1,0,0]])
 idle=next(a for a in d['animations']if a['name']=='Idle');its=animation_times(d,b,idle);parts,_=pose(d,b,idle,float(its[0]),canonical_flip=False,emitter_bone='Xtra Mouth')
 groups={}
 specs={'FL':[' L Hand',' L Finger0'],'FR':[' R Hand',' R Finger0'],'HL':[' L Foot',' L Toe0'],'HR':[' R Foot',' R Toe0']}
 names=[d['nodes'][j].get('name','').replace('_',' ')for j in joints];footmask=[]
 for node in d['nodes']:
  if 'mesh'not in node:continue
  for primitive in d['meshes'][node['mesh']]['primitives']:
   js=accessor(d,b,primitive['attributes']['JOINTS_0']).astype(int);ws=accessor(d,b,primitive['attributes']['WEIGHTS_0'])
   footmask.append({f:(np.isin(js,[i for i,n in enumerate(names)if any(n.endswith(t)for t in tails)])*ws).sum(1)>.15 for f,tails in specs.items()})
 supports=[]
 for f in specs:
  p=np.concatenate([part['positions'][m[f]]for part,m in zip(parts,footmask)if m[f].any()]);lo=p[:,1].min();q=p[p[:,1]<lo+.004].mean(0);supports.append(q@yaw.T)
 supports=np.asarray(supports);centre=supports.mean(0);_,_,vh=np.linalg.svd(supports-centre);normal=vh[-1]
 if normal[1]<0:normal=-normal
 level=rotation_between(normal,[0,1,0]);mat=level@yaw;tilt=float(np.arccos(np.clip(normal[1],-1,1))*180/math.pi);levelled=supports@level.T;residual=float(np.ptp(levelled[:,1]));shift=np.array([0,-levelled[:,1].min(),0])
 Q=np.eye(4);Q[:3,:3]=mat;Q[:3,3]=shift
 def normal_global(g):
  out=g.copy();sc=np.linalg.norm(out[:3,:3],axis=0);out[:3,:3]/=sc;return Q@out
 gbind=[normal_global(g)for g in bind];nd={k:copy.deepcopy(d[k])for k in ['asset','images','textures','samplers','materials','meshes','bufferViews','accessors','extensionsUsed','extensionsRequired']if k in d};nd.update(buffers=[{'byteLength':0}],nodes=[],skins=[],scenes=[{'nodes':[0,len(joints)+1]}],scene=0,animations=[]);payload=bytearray(b)
 def add(arr,atype):
  x=np.asarray(arr,np.float32);payload.extend(bytes((-len(payload))%4));vi=len(nd['bufferViews']);nd['bufferViews'].append({'buffer':0,'byteOffset':len(payload),'byteLength':x.nbytes});payload.extend(x.tobytes());ai=len(nd['accessors']);a={'bufferView':vi,'componentType':5126,'count':len(x),'type':atype}
  if atype=='VEC3':a.update(min=x.min(0).tolist(),max=x.max(0).tolist())
  nd['accessors'].append(a);return ai
 nd['nodes'].append({'name':'V7MetricRig','children':[]});mapping={old:i+1 for i,old in enumerate(joints)}
 for i,j in enumerate(joints):
  p=parent.get(j)
  while p is not None and p not in mapping:p=parent.get(p)
  local=np.linalg.inv(gbind[joints.index(p)])@gbind[i]if p is not None else gbind[i]
  nd['nodes'].append({'name':d['nodes'][j].get('name','joint_'+str(i)),'matrix':local.T.reshape(16).tolist(),'children':[]})
  nd['nodes'][mapping[p]if p is not None else 0]['children'].append(mapping[j])
 nd['nodes'].append({'name':'native_dog_mesh','mesh':d['nodes'][ni]['mesh'],'skin':0})
 newibm=np.stack([np.linalg.inv(g).T.reshape(16)for g in gbind]);nd['skins']=[{'joints':list(range(1,len(joints)+1)),'inverseBindMatrices':add(newibm,'MAT4'),'skeleton':next(i for i in nd['nodes'][0]['children'])}]
 for primitive in nd['meshes'][d['nodes'][ni]['mesh']]['primitives']:
  p=accessor(d,b,primitive['attributes']['POSITION']);world=p@mat.T+shift;primitive['attributes']['POSITION']=add(world,'VEC3')
  if 'NORMAL'in primitive['attributes']:
   n=accessor(d,b,primitive['attributes']['NORMAL'])@mat.T;n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);primitive['attributes']['NORMAL']=add(n,'VEC3')
 for ac in d['animations']:
  ts=animation_times(d,b,ac);ti=add(ts[:,None],'SCALAR');nd['accessors'][ti].update(min=[float(ts[0])],max=[float(ts[-1])]);nac={'name':ac['name'],'samplers':[],'channels':[]};frames=[]
  for t in ts:
   old=globals_at(d,b,ac,float(t));frames.append([normal_global(old[j])for j in joints])
  for i,j in enumerate(joints):
   p=parent.get(j)
   while p is not None and p not in mapping:p=parent.get(p)
   locals=np.stack([np.linalg.inv(g[joints.index(p)])@g[i]if p is not None else g[i]for g in frames]);quat=Rotation.from_matrix(locals[:,:3,:3]).as_quat()
   for k in range(1,len(quat)):
    if quat[k]@quat[k-1]<0:quat[k]*=-1
   for path,x,typ in [('translation',locals[:,:3,3],'VEC3'),('rotation',quat,'VEC4')]:
    oi=add(x,typ);si=len(nac['samplers']);nac['samplers'].append({'input':ti,'output':oi,'interpolation':'LINEAR'});nac['channels'].append({'sampler':si,'target':{'node':mapping[j],'path':path}})
  nd['animations'].append(nac)
 nd['buffers'][0]['byteLength']=len(payload);write_glb(D/'runtime.glb',nd,bytes(payload))
 equivalence=[];footqa=[]
 for ac in d['animations']:
  nac=next(x for x in nd['animations']if x['name']==ac['name']);ts=animation_times(d,b,ac)
  for t in ts:
   old,_=pose(d,b,ac,float(t),canonical_flip=False,emitter_bone='Xtra Mouth');new,ev=pose(nd,bytes(payload),nac,float(t),canonical_flip=False,ground=False,emitter_bone='Xtra Mouth')
   error=max(float(np.linalg.norm((op['positions']@mat.T+shift)-npart['positions'],axis=1).max())for op,npart in zip(old,new));# old pose was auto-grounded; undo that diagnostic grounding for equivalence.
   rawold,oe=pose(d,b,ac,float(t),canonical_flip=False,ground=False,emitter_bone='Xtra Mouth');error=max(float(np.linalg.norm((op['positions']@mat.T+shift)-npart['positions'],axis=1).max())for op,npart in zip(rawold,new));equivalence.append(error)
   ft={f:float(np.concatenate([p['positions'][m[f],1]for p,m in zip(new,footmask)if m[f].any()]).min())for f in specs};footqa.append({'action':ac['name'],'time_s':float(t),'foot_min_y_m':ft,'mouth_m':ev['mouth_m'],'whole_floor_y_m':ev['floor_y_m']})
 json_write(D/'normalization.json',{'asset_id':aid,'source':str(source),'up_axis':'+Y','front_axis':'-Z','units':'metres','native_weights_preserved':True,'native_mesh_topology_preserved':True,'support_plane_tilt_deg':tilt,'four_paw_nonplanarity_range_m':residual,'four_paw_source_supports_after_heading_m':supports.tolist(),'source_pose':'Idle first frame; audited library rig, no TokenRig required','maximum_complete_cycle_surface_equivalence_error_m':max(equivalence),'support_plane_status':'pass'if tilt<10 and residual<.03 else'fail','actions':['Idle','Walking']})
 json_write(D/'foot_qa.json',footqa);print('DOG_NORMALIZED',aid,tilt,residual,max(equivalence),flush=True)
def main(T):
 one(T,'beagle',T/'round2/assets/dogs/beagle/source_units.glb')
 one(T,'german_shepherd',T/'round2/assets/dogs/german_shepherd/source_units.glb')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);main(p.parse_args().task_root)
