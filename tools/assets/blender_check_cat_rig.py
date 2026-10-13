#!/usr/bin/env python3
"""Independent GLB skin readback and nonadjacent triangle intersection QA."""
import argparse,json,pathlib,sys,struct,math,time,runpy
import numpy as np
from mathutils import Quaternion,Vector
from mathutils.bvhtree import BVHTree

DT={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32}
WIDTH={'SCALAR':1,'VEC3':3,'VEC4':4,'MAT4':16}
PAWS=('FL','FR','HL','HR')

def glb(path):
 b=path.read_bytes();n=struct.unpack_from('<I',b,12)[0];return json.loads(b[20:20+n]),b[28+n:]
def accessor(d,b,i):
 a=d['accessors'][i];v=d['bufferViews'][a['bufferView']];dt=np.dtype(DT[a['componentType']]);w=WIDTH[a['type']]
 x=np.ndarray((a['count'],w),dt,buffer=b,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',w*dt.itemsize),dt.itemsize)).copy()
 if a.get('normalized'):x=x.astype(float)/np.iinfo(dt).max
 return x

def globals_at(d,b,action,t):
 ns=[dict(x) for x in d['nodes']]
 for c in action['channels']:
  sam=action['samplers'][c['sampler']];ts=accessor(d,b,sam['input']).ravel();vs=accessor(d,b,sam['output']);i=np.clip(np.searchsorted(ts,t,side='right')-1,0,len(ts)-1);j=min(i+1,len(ts)-1);u=np.clip((t-ts[i])/(ts[j]-ts[i]),0,1) if j!=i else 0
  if c['target']['path']=='rotation':
   x,y,z,w=vs[i];q=Quaternion((w,x,y,z));x,y,z,w=vs[j];r=q.slerp(Quaternion((w,x,y,z)),float(u));value=[r.x,r.y,r.z,r.w]
  else:value=(vs[i]*(1-u)+vs[j]*u).tolist()
  ns[c['target']['node']][c['target']['path']]=value
 result={}
 def visit(i,parent):
  n=ns[i]
  if 'matrix' in n:m=np.array(n['matrix']).reshape(4,4).T
  else:
   m=np.eye(4);x,y,z,w=n.get('rotation',[0,0,0,1]);m[:3,:3]=np.array(Quaternion((w,x,y,z)).to_matrix())@np.diag(n.get('scale',[1,1,1]));m[:3,3]=n.get('translation',[0,0,0])
  result[i]=parent@m
  for j in n.get('children',[]):visit(j,result[i])
 for i in d['scenes'][d.get('scene',0)]['nodes']:visit(i,np.eye(4))
 return result

def load_parts(d,b):
 parts=[]
 for node in d['nodes']:
  if 'mesh' not in node:continue
  s=d['skins'][node['skin']];ibm=accessor(d,b,s['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
  for p in d['meshes'][node['mesh']]['primitives']:
   att=p['attributes'];v=accessor(d,b,att['POSITION']).astype(float);sets=sorted(int(k.split('_')[1]) for k in att if k.startswith('JOINTS_'));js=np.concatenate([accessor(d,b,att[f'JOINTS_{k}']).astype(int) for k in sets],axis=1);ws=np.concatenate([accessor(d,b,att[f'WEIGHTS_{k}']).astype(float) for k in sets],axis=1);ws/=ws.sum(1,keepdims=True)
   parts.append({'positions':v,'joints':js,'weights':ws,'skin_joints':s['joints'],'ibm':ibm,'faces':accessor(d,b,p['indices']).reshape(-1,3).astype(int)})
 return parts

def skin(parts,g):
 posed=[]
 for p in parts:
  mats=np.array([g[j]@m for j,m in zip(p['skin_joints'],p['ibm'])]);v=np.c_[p['positions'],np.ones(len(p['positions']))];result=np.zeros((len(v),4))
  for k in range(p['joints'].shape[1]):result+=np.einsum('nij,nj->ni',mats[p['joints'][:,k]],v)*p['weights'][:,k,None]
  posed.append(result[:,:3])
 return posed

def crossings(vertices,faces):
 tree=BVHTree.FromPolygons([Vector(x) for x in vertices],faces.tolist(),all_triangles=True,epsilon=1e-8);pairs=np.array([(i,j) for i,j in tree.overlap(tree) if i<j],int)
 if not len(pairs):return set()
 aa=vertices[faces[pairs[:,0]]];bb=vertices[faces[pairs[:,1]]]
 # UV seams can duplicate one shared geometric vertex with a different index.
 shared=np.min(np.linalg.norm(aa[:,:,None,:]-bb[:,None,:,:],axis=-1),axis=(1,2))<1e-8
 pairs=pairs[~shared];aa=aa[~shared];bb=bb[~shared];hit=np.zeros(len(pairs),bool)
 def segments(a,c):
  nonlocal hit
  edge1=c[:,1]-c[:,0];edge2=c[:,2]-c[:,0]
  for i in range(3):
   origin=a[:,i];delta=a[:,(i+1)%3]-origin;h=np.cross(delta,edge2);det=np.sum(edge1*h,axis=1);valid=np.abs(det)>1e-14;inv=np.divide(1.,det,out=np.zeros_like(det),where=valid);s=origin-c[:,0];u=np.sum(s*h,axis=1)*inv;q=np.cross(s,edge1);v=np.sum(delta*q,axis=1)*inv;t=np.sum(edge2*q,axis=1)*inv
   hit|=valid&(u>1e-5)&(v>1e-5)&(u+v<1-1e-5)&(t>1e-5)&(t<1-1e-5)
 segments(aa,bb);segments(bb,aa)
 normals_a=np.cross(aa[:,1]-aa[:,0],aa[:,2]-aa[:,0]);normals_b=np.cross(bb[:,1]-bb[:,0],bb[:,2]-bb[:,0]);na=np.linalg.norm(normals_a,axis=1);nb=np.linalg.norm(normals_b,axis=1)
 unit=np.divide(normals_a,na[:,None],out=np.zeros_like(normals_a),where=na[:,None]>1e-20)
 coplanar=(na>1e-14)&(nb>1e-14)&(np.linalg.norm(np.cross(normals_a,normals_b),axis=1)<na*nb*1e-6)&(np.max(np.abs(np.sum((bb-aa[:,0,None,:])*unit[:,None,:],axis=2)),axis=1)<1e-8)
 def cross2(x,y):return x[:,0]*y[:,1]-x[:,1]*y[:,0]
 for drop in range(3):
  ids=np.flatnonzero(coplanar&(np.argmax(np.abs(normals_a),axis=1)==drop));keep=[i for i in range(3) if i!=drop];x=aa[ids][:,:,keep];y=bb[ids][:,:,keep];ok=np.zeros(len(ids),bool)
  for one,other in ((x,y),(y,x)):
   for vi in range(3):
    signs=[cross2(other[:,(k+1)%3]-other[:,k],one[:,vi]-other[:,k]) for k in range(3)];signs=np.stack(signs,axis=1);ok|=(signs.min(1)>1e-12)|(signs.max(1)<-1e-12)
  for i in range(3):
   for j in range(3):
    a0=x[:,i];a1=x[:,(i+1)%3];b0=y[:,j];b1=y[:,(j+1)%3];s1=cross2(a1-a0,b0-a0);s2=cross2(a1-a0,b1-a0);s3=cross2(b1-b0,a0-b0);s4=cross2(b1-b0,a1-b0);ok|=(s1*s2<-1e-24)&(s3*s4<-1e-24)
  hit[ids]|=ok
 return set(map(tuple,pairs[hit].tolist()))

def new_disjoint_contact_pairs(pairs, vertices, faces, gap=.0001):
 """New physical contacts, excluding mere pair changes inside old overlaps.

A previously disjoint surface gap over 0.1 mm defines a new contact patch.
This is smaller than native floor/foot tolerances and cannot hide a leg or
 tail entering the torso. Counts are recorded only; acceptance uses area and depth.
 """
 if not pairs:return set()
 ids=np.asarray(list(pairs),int);a=vertices[faces[ids[:,0]]];b=vertices[faces[ids[:,1]]];distance=np.full(len(ids),np.inf)
 def point_triangle(pt,tr):
  ab=tr[:,1]-tr[:,0];ac=tr[:,2]-tr[:,0];n=np.cross(ab,ac);nn=(n*n).sum(1);delta=pt-tr[:,0];signed=(delta*n).sum(1);projection=delta-n*np.divide(signed,nn,out=np.zeros_like(signed),where=nn>1e-30)[:,None]
  aa=(ab*ab).sum(1);bb=(ab*ac).sum(1);cc=(ac*ac).sum(1);pp=(projection*ab).sum(1);qq=(projection*ac).sum(1);den=aa*cc-bb*bb
  u=np.divide(cc*pp-bb*qq,den,out=np.zeros_like(den),where=den>1e-30);v=np.divide(aa*qq-bb*pp,den,out=np.zeros_like(den),where=den>1e-30);inside=(den>1e-30)&(u>=0)&(v>=0)&(u+v<=1)
  dist=np.where(inside,np.abs(signed)/np.sqrt(np.maximum(nn,1e-30)),np.inf)
  for i in range(3):
   edge=tr[:,(i+1)%3]-tr[:,i];dd=pt-tr[:,i];t=np.clip((dd*edge).sum(1)/np.maximum((edge*edge).sum(1),1e-30),0,1);dist=np.minimum(dist,np.linalg.norm(dd-edge*t[:,None],axis=1))
  return dist
 for i in range(3):distance=np.minimum(distance,point_triangle(a[:,i],b));distance=np.minimum(distance,point_triangle(b[:,i],a))
 for i in range(3):
  for j in range(3):
   u=a[:,(i+1)%3]-a[:,i];v=b[:,(j+1)%3]-b[:,j];w=a[:,i]-b[:,j];aa=(u*u).sum(1);bb=(u*v).sum(1);cc=(v*v).sum(1);dd=(u*w).sum(1);ee=(v*w).sum(1);den=aa*cc-bb*bb
   ss=np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-30);tt=np.divide(aa*ee-bb*dd,den,out=np.zeros_like(den),where=den>1e-30);inside=(den>1e-30)&(ss>=0)&(ss<=1)&(tt>=0)&(tt<=1);d=np.linalg.norm(w+ss[:,None]*u-tt[:,None]*v,axis=1);distance=np.minimum(distance,np.where(inside,d,np.inf))
 # A segment can enter the interior of the other triangle without touching
 # its boundary edges: those already intersecting rest surfaces have zero gap.
 touching=np.zeros(len(ids),bool)
 for one,other in ((a,b),(b,a)):
  e1=other[:,1]-other[:,0];e2=other[:,2]-other[:,0]
  for k in range(3):
   origin=one[:,k];delta=one[:,(k+1)%3]-origin;hh=np.cross(delta,e2);det=(e1*hh).sum(1);inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-20);q=origin-other[:,0];u=(q*hh).sum(1)*inv;qq=np.cross(q,e1);v=(delta*qq).sum(1)*inv;t=(e2*qq).sum(1)*inv
   touching|=(np.abs(det)>1e-20)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>=-1e-8)&(t<=1+1e-8)
 distance[touching]=0.
 return set(map(tuple,ids[distance>gap].tolist()))

def novel_patch_area(pairs, faces, welded, areas):
 """Area of connected new crossing patches on the unchanged rest surface."""
 if not pairs:return 0.,0
 active=np.unique(np.array(list(pairs),int).ravel());parent={int(i):int(i) for i in active}
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def join(i,j):
  x,y=root(i),root(j)
  if x!=y:parent[x]=y
 owners={}
 for f in active:
  f=int(f)
  for vertex in welded[faces[f]]:
   v=int(vertex)
   if v in owners:join(f,owners[v])
   else:owners[v]=f
 for i,j in pairs:join(int(i),int(j))
 totals={}
 for f in active:
  k=root(int(f));totals[k]=totals.get(k,0.)+float(areas[f])
 # Both contacting surfaces contribute; half their sum is the patch area.
 return max(totals.values())*.5,len(totals)

def main():
 argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
 p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False)
 start=time.process_time();receipt=json.loads((a.case/'rigging.json').read_text());d,b=glb(a.case/'motion.glb');parts=load_parts(d,b);names={n.get('name'):i for i,n in enumerate(d['nodes'])};actions={x['name']:x for x in d['animations']};idle=globals_at(d,b,actions['Idle'],0);rest=skin(parts,idle)
 vv=np.concatenate(rest);ff=np.concatenate([p['faces']+sum(len(x) for x in rest[:i]) for i,p in enumerate(parts)]);baseline=crossings(vv,ff);_,welded=np.unique(np.round(vv/1e-7).astype(np.int64),axis=0,return_inverse=True);areas=np.linalg.norm(np.cross(vv[ff[:,1]]-vv[ff[:,0]],vv[ff[:,2]]-vv[ff[:,0]]),axis=1)*.5;surface_helpers=runpy.run_path(str(pathlib.Path(__file__).with_name('cat_surface_checks.py')),run_name='surface_helpers');monitor=surface_helpers['RegionSurfaceChecks'](vv,ff,surface_helpers['anatomical_regions'](parts,d),crossings,new_disjoint_contact_pairs);max_excess=0;max_patch_area=0.;max_pairs=0;max_inter_depth=0.;max_intra_depth=0.;surface_failures=set()
 refs={};edges=[]
 for pi,(part,v) in enumerate(zip(parts,rest)):
  faces=part['faces'];e=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0);edges.append((e,np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1)))
  for s in PAWS:
   j=part['skin_joints'].index(names['Ankle_'+s]);ws=(part['weights']*(part['joints']==j)).sum(1);ids=np.flatnonzero(ws>.999)
   if len(ids):
    vi=int(ids[np.argmin(v[ids,1])]);r=(float(v[vi,1]),pi,vi)
    if s not in refs or r[0]<refs[s][0]:refs[s]=r
 assert set(refs)==set(PAWS),'four_exported_rigid_soles_missing'
 candidate=json.loads((a.case/'candidate.json').read_text());ml=np.r_[candidate['emitter_head_local_m'],1];head=names['Head'];tail=names['TailRoot'];body=names['Body'];offsets=dict(zip(PAWS,(.25,.75,0,.5)));anchors={};frames=[];max_drift=0.;max_gap=0.;min_floor=1.;max_stretch=1.;new_pairs=set();max_new=0;max_nose_error=0.
 emitter_rest=(idle[head]@ml)[:3];near=[]
 for pi,(part,v) in enumerate(zip(parts,rest)):
  j=part['skin_joints'].index(head);mask=((part['weights']*(part['joints']==j)).sum(1)>.999)&(np.linalg.norm(v-emitter_rest,axis=1)<.012)
  ids=np.flatnonzero(mask)
  if len(ids):near.append((pi,ids,(np.linalg.inv(idle[head])@np.c_[v[ids],np.ones(len(ids))].T).T))
 assert near,'no_actual_mouth_skin_bound_to_head'
 for frame in range(receipt['frame_count']):
  t=frame/30;g=globals_at(d,b,actions['Walking'],t);posed=skin(parts,g);v=np.concatenate(posed);pairs=crossings(v,ff);new=pairs-baseline;new_pairs|=new;max_new=max(max_new,len(new));max_pairs=max(max_pairs,len(pairs));max_excess=max(max_excess,len(pairs)-len(baseline));surface=monitor.inspect(v,pairs);patch_area=surface['new_interregion_visible_area_m2'];patch_count=len(surface['region_contacts']);max_patch_area=max(max_patch_area,patch_area);max_inter_depth=max(max_inter_depth,surface['max_new_interregion_depth_m']);max_intra_depth=max(max_intra_depth,surface['max_new_intraregion_depth_m']);surface_failures.update(surface['surface_failures']);min_floor=min(min_floor,float(v[:,1].min()));feet={}
  for s in PAWS:
   _,pi,vi=refs[s];pt=posed[pi][vi];q=t/receipt.get('gait_period_s',2.4)+offsets[s];cycle=math.floor(q);phase=q-cycle;active=phase>=.25
   if active:
    key=(s,cycle)
    if key not in anchors:anchors[key]=pt.copy()
    max_drift=max(max_drift,float(np.linalg.norm(pt-anchors[key])));max_gap=max(max_gap,abs(float(pt[1])))
   feet[s]={'in_contact':active,'support_segment':cycle if active else None,'support_world_m':pt.tolist(),'support_vertex':[pi,vi],'floor_delta_m':float(pt[1])}
  for part,(e,lengths) in zip(posed,edges):
   good=lengths>1e-5;ratios=np.linalg.norm(part[e[:,0]]-part[e[:,1]],axis=1)[good]/lengths[good];max_stretch=max(max_stretch,float(ratios.max()))
  for pi,ids,local in near:max_nose_error=max(max_nose_error,float(np.max(np.linalg.norm(posed[pi][ids]-(g[head]@local.T).T[:,:3],axis=1))))
  frames.append({'time_s':t,'root_world_m':g[body][:3,3].tolist(),'head_world_m':g[head][:3,3].tolist(),'tail_world_m':g[tail][:3,3].tolist(),'mouth_world_m':(g[head]@ml)[:3].tolist(),'feet':feet,'new_self_intersection_pairs':len(new),'self_intersection_pairs':len(pairs),'new_crossing_patch_area_m2':patch_area,'new_crossing_patch_count':patch_count,'surface_check':surface,'body_floor_min_m':float(v[:,1].min())})
  if frame%30==0:print('QA_FRAME',frame,'new_crossings',len(new),'stretch',max_stretch,flush=True)
 roots=np.array([f['root_world_m'] for f in frames]);heads=np.array([f['head_world_m'] for f in frames]);tails=np.array([f['tail_world_m'] for f in frames]);vel=np.diff(roots[:,[0,2]],axis=0)*30
 def direction(h,ta):
  axes=(h-ta)[:,[0,2]];angles=[]
  for i,speed in enumerate(np.linalg.norm(vel,axis=1)):
   if speed>.005:
    for j in (i,i+1):angles.append(math.degrees(math.acos(np.clip(axes[j]@vel[i]/(np.linalg.norm(axes[j])*speed),-1,1))))
  return {'status':'pass' if angles and max(angles)<=15 else 'fail','max_angle_deg':max(angles) if angles else None}
 direct=direction(heads,tails);negative=[]
 for deg in (180,90):
  theta=math.radians(deg);rot=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]]);rh=(heads-roots)@rot.T+roots;rt=(tails-roots)@rot.T+roots
  # Physical mesh positions are rotated about their measured root as well.
  mesh=(skin(parts,globals_at(d,b,actions['Walking'],2.4))[0]-roots[72])@rot.T+roots[72]
  negative.append({'body_rotation_error_deg':deg,'check':direction(rh,rt),'physically_rotated_exported_skin':True,'rotated_skin_bounds_m':[mesh.min(0).tolist(),mesh.max(0).tolist()]})
 failures=[]
 if direct['status']!='pass':failures.append('heading')
 if not all(n['check']['status']=='fail' for n in negative):failures.append('negative_control')
 if max_drift>.003:failures.append('stance_drift')
 if max_gap>.012:failures.append('contact_gap')
 if min_floor<-.00101:failures.append('skin_penetration')
 if max_stretch>1.5:failures.append('mesh_stretch')
 failures.extend(sorted(surface_failures))
 if max_nose_error>.0001:failures.append('mouth_skin_attachment')
 result={'asset_id':receipt['asset_id'],'status':'pass' if not failures else 'fail','failures':failures,'frame_count':len(frames),'direction':direct,'direction_negative_controls':negative,'max_stance_drift_m':max_drift,'max_contact_gap_m':max_gap,'min_full_skin_floor_m':min_floor,'max_edge_stretch_ratio':max_stretch,'max_mouth_skin_attachment_error_m':max_nose_error,'surface_check_rule':monitor.provenance(),'counts_participate_in_acceptance':False,'max_new_interregion_depth_m':max_inter_depth,'max_new_intraregion_depth_m':max_intra_depth,'max_new_crossing_patch_area_m2':max_patch_area,'max_self_intersection_count_excess':max_excess,'max_transverse_self_intersection_pairs':max_pairs,'initial_transverse_self_intersection_pairs':len(baseline),'max_new_transverse_self_intersections':max_new,'new_intersection_pair_examples':list(sorted(new_pairs))[:20],'self_intersection_method':'BVH candidate pairs; bidirectional triangle-edge crossing and coplanar 2D interior overlap; shared geometric vertices excluded','coplanar_self_overlap_status':'checked','native_habitat_status':'not_run','frames':frames,'cpu_s':time.process_time()-start}
 (a.output/'selfcheck.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='frames'},flush=True)
 if failures:raise SystemExit(2)
if __name__=='__main__':main()
