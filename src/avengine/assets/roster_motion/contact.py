"""World-space foot planting and two-bone IK for source-authored motion."""
import copy,math
import numpy as np
from scipy.spatial.transform import Rotation
from avengine.assets.roster_motion.pose import globals_at,pose,accessor
C=np.diag([-1.,1.,-1.,1.])
def yaw_matrix(a):
 c,s=math.cos(a),math.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rotation_between(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);a/=np.linalg.norm(a);b/=np.linalg.norm(b);v=np.cross(a,b);dot=np.clip(a@b,-1,1)
 if dot>.999999:return np.eye(3)
 if dot<-.999999:
  axis=np.cross(a,[1,0,0]if abs(a[0])<.8 else[0,1,0]);axis/=np.linalg.norm(axis);return Rotation.from_rotvec(axis*math.pi).as_matrix()
 skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
 return np.eye(3)+skew+skew@skew/(1+dot)
def find_bone(d,suffix):
 return next(i for i,n in enumerate(d['nodes'])if n.get('name','').replace('_',' ').endswith(suffix))
def subtree(d,i):
 out=[i]
 for j in d['nodes'][i].get('children',[]):out.extend(subtree(d,j))
 return out
def rotate_subtree(d,g,i,rot):
 p=g[i][:3,3].copy();m=np.eye(4);m[:3,:3]=rot;m[:3,3]=p-rot@p
 for j in subtree(d,i):g[j]=m@g[j]
def two_bone(d,g,hip,knee,ankle,target,foot_rotation,toe=None,toe_relative=None):
 h=g[hip][:3,3].copy();k=g[knee][:3,3].copy();f=g[ankle][:3,3].copy();l1=np.linalg.norm(k-h);l2=np.linalg.norm(f-k)
 desired=np.asarray(target);delta=desired-h;distance=np.linalg.norm(delta);reach_error=max(0,distance-l1-l2)
 distance=np.clip(distance,abs(l1-l2)+1e-5,l1+l2-1e-5);axis=delta/np.linalg.norm(delta);along=(l1*l1-l2*l2+distance*distance)/(2*distance)
 bend=(k-h)-axis*((k-h)@axis)
 if np.linalg.norm(bend)<1e-6:bend=np.cross(axis,[1,0,0])
 bend/=np.linalg.norm(bend);newk=h+axis*along+bend*math.sqrt(max(0,l1*l1-along*along))
 rotate_subtree(d,g,hip,rotation_between(k-h,newk-h))
 k=g[knee][:3,3].copy();f=g[ankle][:3,3].copy();reachable=h+axis*distance
 rotate_subtree(d,g,knee,rotation_between(f-k,reachable-k))
 rotate_subtree(d,g,ankle,foot_rotation@g[ankle][:3,:3].T)
 if toe is not None and toe_relative is not None:
  desired_toe=g[ankle]@toe_relative;delta_toe=desired_toe@np.linalg.inv(g[toe])
  for j in subtree(d,toe):g[j]=delta_toe@g[j]
 return reach_error,float(np.linalg.norm(g[ankle][:3,3]-desired))
def foot_masks(d,b):
 masks=[]
 for node in d['nodes']:
  if 'mesh'not in node:continue
  for p in d['meshes'][node['mesh']]['primitives']:
   sk=d['skins'][node['skin']];names=[d['nodes'][j].get('name','').replace('_',' ')for j in sk['joints']];js=accessor(d,b,p['attributes']['JOINTS_0']).astype(int);ws=accessor(d,b,p['attributes']['WEIGHTS_0'])
   masks.append({s:(np.isin(js,[i for i,n in enumerate(names)if ' '+s+' 'in n and ('Foot'in n or'Toe'in n)])*ws).sum(1)>.15 for s in ['L','R']})
 return masks
def foot_min(parts,masks,side):
 x=[p['positions'][m[side]]for p,m in zip(parts,masks)if m[side].any()];pts=np.concatenate(x);index=np.argmin(pts[:,1]);return float(pts[index,1]),pts[index]
def animation_nodes(d):
 seen=set();todo=list(d['scenes'][d.get('scene',0)]['nodes'])
 while todo:
  k=todo.pop()
  if k in seen:continue
  seen.add(k);todo.extend(d['nodes'][k].get('children',[]))
 joints={j for skin in d.get('skins',[])for j in skin['joints']}
 if not joints<=seen:raise ValueError('Skin joint is not reachable from the glTF scene')
 return seen

def serialize_action(d,b,name,times,global_frames):
 nd=copy.deepcopy(d);binary=bytearray(b);parents={j:i for i,n in enumerate(d['nodes'])for j in n.get('children',[])}
 for i in d['scenes'][d.get('scene',0)]['nodes']:parents.pop(i,None)
 def add(x,atype):
  arr=np.asarray(x,np.float32);binary.extend(bytes((-len(binary))%4));vi=len(nd['bufferViews']);nd['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':arr.nbytes});binary.extend(arr.tobytes());ai=len(nd['accessors']);nd['accessors'].append({'bufferView':vi,'componentType':5126,'count':len(arr),'type':atype});return ai
 ti=add(np.asarray(times).reshape(-1,1),'SCALAR');nd['accessors'][ti].update(min=[float(times[0])],max=[float(times[-1])])
 ac={'name':name,'channels':[],'samplers':[]}
 for i in sorted(animation_nodes(d)):
  ls=np.stack([np.linalg.inv(g[parents[i]])@g[i]if i in parents else g[i]for g in global_frames])
  scale=np.linalg.norm(ls[:,:3,:3],axis=1);rots=ls[:,:3,:3]/scale[:,None,:];quat=Rotation.from_matrix(rots).as_quat()
  # Avoid sign flips across keys; UE and glTF receive the same rotations.
  for j in range(1,len(quat)):
   if quat[j]@quat[j-1]<0:quat[j]*=-1
  for path,x,atype in [('translation',ls[:,:3,3],'VEC3'),('rotation',quat,'VEC4'),('scale',scale,'VEC3')]:
   oi=add(x,atype);si=len(ac['samplers']);ac['samplers'].append({'input':ti,'output':oi,'interpolation':'LINEAR'});ac['channels'].append({'sampler':si,'target':{'node':i,'path':path}})
 nd.setdefault('animations',[]).append(ac);nd['buffers'][0]['byteLength']=len(binary);return nd,bytes(binary)
class FootPlant:
 def __init__(self,d,b,action,times,positions,yaws,play_rate,source_start,source_duration,floor,basis=None,chains=None,masks=None,emitter_bone=None,canonical_flip=True,backward_threshold=.2):
  self.d=d;self.b=b;self.action=action;self.times=np.asarray(times);self.pos=np.asarray(positions);self.yaw=np.asarray(yaws);self.basis=C if basis is None else basis;self.canonical_flip=canonical_flip;self.emitter_bone=emitter_bone;self.masks=foot_masks(d,b) if masks is None else masks;self.floor=floor;self.source=[];self.events=[];self.state={s:None for s in (chains or {'L':None,'R':None})};self.max_reach=0;self.max_ik_error=0
  self.chains=chains or {s:[find_bone(d,' '+s+' '+part)for part in ['Thigh','Calf','Foot','Toe0']]for s in ['L','R']}
  for i,t in enumerate(times):
   phase=source_start+(float(t)*play_rate)%source_duration;raw=globals_at(d,b,action,phase);parts,ev=pose(d,b,action,phase,canonical_flip=canonical_flip,emitter_bone=emitter_bone);g={j:self.basis@m for j,m in raw.items()}
   for m in g.values():m[1,3]+=ev['ground_correction_m']
   heights={s:foot_min(parts,self.masks,s)[0]for s in self.chains}
   self.source.append((g,phase,heights,ev))
  self.active={}
  for side,chain in self.chains.items():
   bone=chain[2];backward=[]
   for _,phase,_,_ in self.source:
    lo=max(source_start,phase-.004);hi=min(source_start+source_duration,phase+.004);ga=globals_at(d,b,action,lo);gb=globals_at(d,b,action,hi);backward.append(float((self.basis@gb[bone])[2,3]-(self.basis@ga[bone])[2,3])/max(hi-lo,1e-6))
   self.active[side]=np.array([rec[2][side]<.025 for rec in self.source])&(np.asarray(backward)>backward_threshold)
 def frame(self,i):
  d=self.d;original=self.source[i][0];g=copy.deepcopy(original);rot=yaw_matrix(self.yaw[i]);root=self.pos[i];phase=self.source[i][1];info={};raw_parts,_=pose(d,self.b,self.action,phase,canonical_flip=self.canonical_flip,emitter_bone=self.emitter_bone)
  def support(parts,side):
   choices=[]
   for pi,(part,mask) in enumerate(zip(parts,self.masks)):
    ids=np.flatnonzero(mask[side])
    if len(ids):
     points=part['positions'][ids];world=points@rot.T+root;floors=self.floor.many(world) if hasattr(self.floor,'many') else np.array([self.floor(p)for p in world]);scores=world[:,1]-floors;pick=int(np.argmin(scores));vi=int(ids[pick]);choices.append((float(scores[pick]),pi,vi))
   _,pi,vi=min(choices);return pi,vi,parts[pi]['positions'][vi].copy()
  targets={};states={}
  for side,chain in self.chains.items():
   hip,knee,ankle,toe=chain;world=root+rot@original[ankle][:3,3];worldr=rot@original[ankle][:3,:3];state=self.state[side];active=bool(self.active[side][i])
   if active:
    if state is None or not state['active']:
     pi,vi,p=support(raw_parts,side);wp=root+rot@p;wp[1]=self.floor(wp)
     state={'active':True,'support_world':wp,'support_id':[pi,vi],'start':i,'switches':0};self.state[side]=state
    pi,vi=state['support_id'];local=np.linalg.inv(original[ankle])@np.r_[raw_parts[pi]['positions'][vi],1]
    target=state['support_world']-worldr@local[:3]
   else:
    if state is not None and state['active']:
     pi,vi=state['support_id'];local=np.linalg.inv(original[ankle])@np.r_[raw_parts[pi]['positions'][vi],1];last_ankle=state['support_world']-worldr@local[:3]
     state={'active':False,'world':last_ankle,'swing_start':i,'initial_source_world':world.copy()};self.state[side]=state
    target=world.copy()
    if state is not None:
     nexts=np.flatnonzero(self.active[side][i:]);end=i+int(nexts[0])if len(nexts)else len(self.times)-1;start=state['swing_start'];u=np.clip((self.times[i]-self.times[start])/max(self.times[end]-self.times[start],1e-6),0,1);smooth=u*u*u*(10+u*(-15+6*u))
     target+=(1-smooth)*(state['world']-state['initial_source_world']);target[1]+=.035*math.sin(math.pi*u)
   targets[side]=[target,worldr];states[side]=(state,active)
  lower=0.
  for side,chain in self.chains.items():
   hip,knee,ankle,_=chain;h=root+rot@g[hip][:3,3];target=targets[side][0];total=np.linalg.norm(g[knee][:3,3]-g[hip][:3,3])+np.linalg.norm(g[ankle][:3,3]-g[knee][:3,3]);horizontal=np.linalg.norm((h-target)[[0,2]])
   need=max(0,float(h[1]-target[1]-math.sqrt(max(0,(total*.99)**2-horizontal**2))));lower=max(lower,need)
  for m in g.values():m[1,3]-=lower
  for side,chain in self.chains.items():
   hip,knee,ankle,toe=chain;target,worldr=targets[side]
   reach,error=two_bone(d,g,hip,knee,ankle,rot.T@(target-root),rot.T@worldr)
   self.max_reach=max(self.max_reach,reach);self.max_ik_error=max(self.max_ik_error,error)
  for iteration in range(4):
   raw={j:self.basis@m for j,m in g.items()};parts,ev=pose(d,self.b,None,phase,globals_override=raw,ground=False,canonical_flip=self.canonical_flip,emitter_bone=self.emitter_bone)
   for side,chain in self.chains.items():
    hip,knee,ankle,toe=chain;state,active=states[side];target,worldr=targets[side];pi,vi,p=support(parts,side);wp=root+rot@p;floor=self.floor(wp);delta=np.zeros(3)
    if active:
     spi,svi=state['support_id'];actual=root+rot@parts[spi]['positions'][svi]
     # A new part of the rolling sole becomes the support only when it is lower.
     # Roll onto the actual lower sole earlier in the private phase-two
     # adapter, leaving precision headroom inside the unchanged native margin.
     switch_depth=.0008 if getattr(self.floor,'original_ray',None) is not None else .001
     if wp[1]<floor-switch_depth and (pi,vi)!=(spi,svi):
      state['support_world']=wp.copy();state['support_world'][1]=floor;state['support_id']=[pi,vi];state['switches']+=1;state['start']=i;actual=root+rot@parts[pi]['positions'][vi]
     delta=state['support_world']-actual
    elif wp[1]<floor+.003:delta[1]=floor+.003-wp[1]
    target+=delta;targets[side][0]=target
    reach,error=two_bone(d,g,hip,knee,ankle,rot.T@(target-root),rot.T@worldr)
    self.max_reach=max(self.max_reach,reach);self.max_ik_error=max(self.max_ik_error,error)
  raw={j:self.basis@m for j,m in g.items()};parts,ev=pose(d,self.b,None,phase,globals_override=raw,ground=False,canonical_flip=self.canonical_flip,emitter_bone=self.emitter_bone);feet={}
  for side in self.chains:
   state,active=states[side];pi,vi,p=support(parts,side);world=root+rot@p;level=self.floor(world);ankle=self.chains[side][2]
   feet[side]={'in_contact':active,'min_world_y_m':float(world[1]),'floor_y_m':float(level),'floor_delta_m':float(world[1]-level),'sole_local_m':p.tolist(),'sole_world_m':world.tolist(),'ankle_world_m':(root+rot@g[ankle][:3,3]).tolist(),'target_world_m':targets[side][0].tolist(),'ik_error_m':float(np.linalg.norm(root+rot@g[ankle][:3,3]-targets[side][0]))}
   if active:
    spi,svi=state['support_id'];actual=root+rot@parts[spi]['positions'][svi];feet[side].update(support_vertex=[spi,svi],support_world_m=actual.tolist(),support_target_world_m=state['support_world'].tolist(),support_lock_error_m=float(np.linalg.norm(actual-state['support_world'])),support_segment_start_frame=state['start'],support_switches=state['switches'])
  ev.update(source_phase_s=float(phase),time_s=float(self.times[i]),world_root_m=root.tolist(),yaw_rad=float(self.yaw[i]),feet=feet,mouth_world_m=(root+rot@np.asarray(ev['mouth_m'])).tolist(),root_motion_grounded_in_clip=True,pelvis_lowering_m=lower)
  return raw,parts,ev
