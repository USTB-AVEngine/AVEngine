"""Bounded target-native envelope rigging trial; original geometry stays intact."""
import argparse,copy,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from avengine.assets.roster_motion.io import read_glb,write_glb
from avengine.assets.roster_motion.pose import pose,rigid_glb,accessor
from avengine.assets.animal_motion.profiles import Rig,PAWS,ANCHORS
from avengine.assets.animal_motion.slide import registration_record

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--donor',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 inv=json.loads(a.inventory.read_text());record=copy.deepcopy(next(x for x in inv['assets'] if x['asset_id']==a.asset));donor=Rig.load(next(x for x in inv['assets'] if x['asset_id']==a.donor));row=registration_record(record);source=Path(record['files'].get('runtime_glb',record['files'].get('habitat_glb'))['path'])
 d,b=read_glb(source)
 if d.get('skins'):raise ValueError('prototype is restricted to genuinely unskinned targets')
 parts=pose(d,b,None,ground=False,canonical_flip=False)[0];original=np.concatenate([p['positions'] for p in parts]);rigid_glb(d,b,parts,a.output/'rest.glb');d,b=read_glb(a.output/'rest.glb');binary=bytearray(b)
 def add(x,kind,component=5126):
  dtype={5126:np.float32,5123:np.uint16}[component];arr=np.asarray(x,dtype);binary.extend(bytes((-len(binary))%4));view=len(d['bufferViews']);d['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':arr.nbytes});binary.extend(arr.tobytes());idx=len(d['accessors']);ac={'bufferView':view,'componentType':component,'count':len(arr),'type':kind}
  if kind=='SCALAR':ac.update(min=[float(arr.min())],max=[float(arr.max())])
  d['accessors'].append(ac);return idx
 landmarks=row['landmarks_local_m'];feet=np.asarray([landmarks[f'foot_{i}'] for i in range(4)],float);order=np.argsort(feet[:,2]);order=np.r_[order[:2][np.argsort(feet[order[:2],0])],order[2:][np.argsort(feet[order[2:],0])]];feet=feet[order];height=float(np.ptp(original[:,1]));head=np.asarray(row['head_center_local_m']);mouth=np.asarray(row['emit_point_local_m']);rear=original[original[:,2]>=np.quantile(original[:,2],.99)];tail=np.mean(rear,axis=0)
 names=['Body','Head','Tail'];positions=[np.zeros(3),head,tail];parents=[None,0,0];chains={}
 for side,foot in zip(PAWS,feet):
  hip=np.array([foot[0]*.85,height*.58,foot[2]*.9]);ankle=foot+[0,min(.05,height*.10),0];knee=(hip+ankle)/2;knee[2]+=(.03 if side.startswith('F') else -.03);toe=foot+[0,.004,0];ids=[]
  for name,pos in zip(('Hip','Knee','Ankle','Toe'),(hip,knee,ankle,toe)):
   idx=len(names);names.append(name+'_'+side);positions.append(pos);parents.append(ids[-1] if ids else 0);ids.append(idx)
  chains[side]=ids
 positions=np.asarray(positions);nodes=[]
 for i,name in enumerate(names):
  parent=parents[i];node={'name':name,'translation':(positions[i]-positions[parent] if parent is not None else positions[i]).tolist()}
  nodes.append(node)
 for i,parent in enumerate(parents):
  if parent is not None:nodes[parent].setdefault('children',[]).append(i)
 mesh_nodes=[];weight_summary=[]
 for old in d['nodes']:
  if 'mesh' not in old:continue
  mi=old['mesh'];idx=len(nodes);nodes.append({'name':'OwnTargetMesh_'+str(mi),'mesh':mi,'skin':0});nodes[0].setdefault('children',[]).append(idx);mesh_nodes.append(idx)
  for prim in d['meshes'][mi]['primitives']:
   v=accessor(d,bytes(binary),prim['attributes']['POSITION']).astype(float);weights=np.zeros((len(v),len(names)));weights[:,0]=1
   separation=np.linalg.norm(feet[:,None,[0,2]]-feet[None,:,[0,2]],axis=-1);np.fill_diagonal(separation,np.inf);radius=min(.07,float(separation.min())*.4)
   dist=np.linalg.norm(v[:,None,[0,2]]-feet[None,:,[0,2]],axis=-1);nearest=dist.argmin(1)
   for pi,side in enumerate(PAWS):
    hip,knee,ankle,toe=chains[side];ids=np.flatnonzero(nearest==pi);u=np.clip((v[ids,1]-feet[pi,1])/(positions[hip,1]-feet[pi,1]),0,1)
    strength=np.clip((1-u)*2.0,0,1)*np.exp(-2*(dist[ids,pi]/radius)**2)
    sole=(v[ids,1]<=feet[pi,1]+.015)&(dist[ids,pi]<radius);strength[sole]=1.
    low=u<.25;high=u>.7;middle=~(low|high);weights[ids,0]=1-strength
    weights[ids[low],toe]=strength[low]*.7;weights[ids[low],ankle]=strength[low]*.3
    weights[ids[middle],knee]=strength[middle]
    weights[ids[high],hip]=strength[high]
   hm=(v[:,2]<head[2]+height*.08)&(v[:,1]>height*.35);weights[hm,:]=0;weights[hm,1]=1
   selected=np.argsort(weights,axis=1)[:,-4:];values=np.take_along_axis(weights,selected,axis=1);values/=values.sum(1,keepdims=True)
   prim['attributes']['JOINTS_0']=add(selected,'VEC4',5123);prim['attributes']['WEIGHTS_0']=add(values,'VEC4')
   weight_summary.append({'vertices':len(v),'rigid_body_vertices':int((weights[:,0]>.99).sum()),'head_vertices':int(hm.sum()),'leg_vertices':int((weights[:,3:].sum(1)>.5).sum())})
 d['nodes']=nodes;d['scenes']=[{'nodes':[0]}];d['scene']=0;ibm=np.tile(np.eye(4),(len(names),1,1));ibm[:,:3,3]=-positions;d['skins']=[{'name':'OwnTargetRig','joints':list(range(len(names))),'inverseBindMatrices':add(ibm.transpose(0,2,1).reshape(-1,16),'MAT4'),'skeleton':0}]
 times=np.linspace(0,4/3,41);ti=add(times[:,None],'SCALAR');actions=[]
 idle_g=donor.globals(0,'Idle')
 for action_name in ('Idle','Walking'):
  ac={'name':action_name,'channels':[],'samplers':[]}
  def channel(node,path,values,kind):
   oi=add(values,kind);idx=len(ac['samplers']);ac['samplers'].append({'input':ti,'output':oi,'interpolation':'LINEAR'});ac['channels'].append({'sampler':idx,'target':{'node':node,'path':path}})
  bob=np.zeros((len(times),3))
  if action_name=='Walking':bob[:,1]=.002*np.sin(2*np.pi*times/times[-1])
  channel(0,'translation',bob,'VEC3')
  for side,c in chains.items():
   angles=[]
   for t in times:
    if action_name=='Idle':angles.append(0.);continue
    m=donor.globals(t)[donor.chains[side][0]][:3,:3];z=idle_g[donor.chains[side][0]][:3,:3];m=m/np.linalg.norm(m,axis=0);z=z/np.linalg.norm(z,axis=0)
    angles.append(float(np.clip(Rotation.from_matrix(m@z.T).as_rotvec()[0]*.6,-.25,.25)))
   quats=Rotation.from_rotvec(np.c_[angles,np.zeros(len(times)),np.zeros(len(times))]).as_quat();channel(c[0],'rotation',quats,'VEC4')
  actions.append(ac)
 d['animations']=actions;d['buffers'][0]['byteLength']=len(binary);path=a.output/'motion.glb';write_glb(path,d,bytes(binary));rest=pose(d,bytes(binary),actions[0],0,ground=False,canonical_flip=False)[0];error=float(np.max(np.abs(original-np.concatenate([p['positions'] for p in rest]))))
 if error>1e-6:raise ValueError('target rest geometry changed')
 manifest={'schema':'owned_target_rig_candidate_not_registration','anchors':[{'anchor_id':'body','joint_id':'Body'},{'anchor_id':'head','joint_id':'Head'},*[{'anchor_id':name,'joint_id':'Toe_'+side} for side,name in zip(PAWS,ANCHORS)]]};mp=a.output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
 candidate={'rotation':np.eye(3).tolist(),'translation_m':[0,0,0],'scale_xyz':[1,1,1],'emitter_head_local_m':(mouth-head).tolist(),'source_package_manifest':str(mp)};cp=a.output/'candidate.json';cp.write_text(json.dumps(candidate,indent=2));record['has_skeleton']=True;record['has_walking']=True;record['walking_candidate']=str(cp);record['files']['motion_runtime_glb']={'path':str(path),'skins':1,'animations':['Idle','Walking']};(a.output/'inventory.json').write_text(json.dumps({'assets':[record]},indent=2))
 receipt={'asset_id':a.asset,'status':'rigged_candidate_not_animation_admitted','source_standing_glb':str(source),'motion_donor':str(donor.path),'donor_license':'existing Rocketbox MIT rig with Quaternius CC0 Walking; no donor geometry copied','method':'own-target registered paw/head landmarks, deterministic four-limb envelope skinning, retargeted donor hip flexion; downstream constrained gait requires full native admission','joints':len(names),'rest_vertex_max_error_m':error,'triangle_topology_preserved':True,'weights':weight_summary,'inventory':str(a.output/'inventory.json')};(a.output/'rigging.json').write_text(json.dumps(receipt,indent=2));print(receipt,flush=True)
if __name__=='__main__':main()
