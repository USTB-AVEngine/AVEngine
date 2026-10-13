#!/usr/bin/env python3
"""Fit a target-owned quadruped, skin in Blender and retarget CC0 Cat motion.

No donor vertices, joints positions, weights or materials are reused. A constant
four-paw support-plane transform precedes target-native rigging. Generated
artifacts remain candidates until the independent Habitat/RLR checks pass.
"""
from __future__ import annotations
import argparse,copy,json,math,pathlib,sys,time,struct
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"src"))
import bpy,numpy as np
from mathutils import Matrix,Vector,Quaternion
from mathutils.bvhtree import BVHTree

PAWS=('FL','FR','HL','HR')
ANCHORS=('paw_front_left','paw_front_right','paw_hind_left','paw_hind_right')
DONORS={'FL':('Bone.017','Bone.018','Bone.019'),'FR':('Bone.014','Bone.015','Bone.016'),'HL':('Bone.008','Bone.009','Bone.010'),'HR':('Bone.011','Bone.012','Bone.013')}
# Habitat +Y up / -Z forward <-> Blender +Z up / +Y forward.
HB=np.array([[1.,0,0],[0,0,-1],[0,1.,0]])
# Quaternius +X forward / +Z up -> target Blender +Y forward / +Z up.
DC=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])

def vec(p):return Vector(HB@np.asarray(p,float))
def hvec(p):return HB.T@np.asarray(p,float)
def smooth5(x):return x*x*x*(10+x*(-15+6*x))
def between(a,b):return Vector(a).rotation_difference(Vector(b)).to_matrix()
def quaternion_scaled(matrix,amount):
 q=Matrix(matrix).to_quaternion();axis,angle=q.to_axis_angle();return Quaternion(axis,float(angle)*amount)
def snapshot_meshes(meshes):
 deps=bpy.context.evaluated_depsgraph_get();parts=[]
 for ob in meshes:
  eo=ob.evaluated_get(deps);me=eo.to_mesh();coords=np.empty(len(me.vertices)*3);me.vertices.foreach_get('co',coords)
  coords=coords.reshape(-1,3);mat=np.array(ob.matrix_world);coords=coords@mat[:3,:3].T+mat[:3,3]
  parts.append(coords@HB);eo.to_mesh_clear()
 return parts

def main():
 argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
 p=argparse.ArgumentParser();p.add_argument('--roster',type=pathlib.Path,required=True);p.add_argument('--inventory',type=pathlib.Path,required=True);p.add_argument('--template-motion',type=pathlib.Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--weight-method',choices=('heat','envelope'),default='heat');p.add_argument('--speed',type=float,default=.03);p.add_argument('--period',type=float,default=2.4);a=p.parse_args(argv)
 a.output.mkdir(parents=True,exist_ok=False);started=time.process_time()
 row=next(r for r in json.loads(a.roster.read_text())['assets'] if r['asset_id']==a.asset)
 record=copy.deepcopy(next(r for r in json.loads(a.inventory.read_text())['assets'] if r['asset_id']==a.asset))
 template=json.loads(a.template_motion.read_text());assert template['license']=='CC0-1.0'
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
 bpy.ops.import_scene.gltf(filepath=row['runtime_glb']);meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];assert meshes
 # Apply source node transforms before fitting; retain target topology/materials.
 for ob in meshes:
  world=ob.matrix_world.copy();ob.parent=None;ob.matrix_world=world;ob.select_set(True)
 bpy.context.view_layer.objects.active=meshes[0];bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 original=snapshot_meshes(meshes);verts=np.concatenate(original);height=float(np.ptp(verts[:,1]));lm=row['landmarks_local_m']
 feet=np.array([lm[f'foot_{i}'] for i in range(4)])
 order=np.argsort(feet[:,2]);order=np.r_[order[:2][np.argsort(feet[order[:2],0])],order[2:][np.argsort(feet[order[2:],0])]];feet=feet[order]
 sep=np.linalg.norm(feet[:,None,[0,2]]-feet[None,:,[0,2]],axis=-1);np.fill_diagonal(sep,np.inf);radius=min(.05,float(sep.min())*.42)
 foot_probe=[]
 for foot in feet:
  mask=(np.linalg.norm(verts[:,[0,2]]-foot[[0,2]],axis=1)<radius)&(verts[:,1]<foot[1]+height*.18)
  assert mask.sum()>5,('missing_target_foot',foot)
  ids=np.flatnonzero(mask);foot_probe.append(verts[ids[np.argmin(verts[ids,1])]])
 foot_probe=np.array(foot_probe);coeff=np.linalg.lstsq(np.c_[foot_probe[:,0],foot_probe[:,2],np.ones(4)],foot_probe[:,1],rcond=None)[0]
 normal=np.array([-coeff[0],1,-coeff[1]]);normal/=np.linalg.norm(normal);level=np.array(between(normal,[0,1,0]));tilt=math.degrees(math.acos(np.clip(level[1,1],-1,1)));assert tilt<=10,('support_plane_tilt',tilt)
 leveled=foot_probe@level.T;shift=np.array([0,-leveled[:,1].mean()+.0004,0]);trans=np.eye(4);trans[:3,:3]=level;trans[:3,3]=shift
 plane_residual=float(np.ptp(leveled[:,1]));assert plane_residual<=.012,('nonplanar_four_paws',plane_residual)
 bmat=np.eye(4);bmat[:3,:3]=HB@level@HB.T;bmat[:3,3]=HB@shift
 for ob in meshes:ob.matrix_world=Matrix(bmat)@ob.matrix_world
 bpy.context.view_layer.objects.active=meshes[0];bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 verts=verts@level.T+shift;feet=feet@level.T+shift;foot_probe=foot_probe@level.T+shift
 head=np.asarray(row['head_center_local_m'])@level.T+shift;mouth=np.asarray(row['emit_point_local_m'])@level.T+shift
 front=feet[:2].mean(0);hind=feet[2:].mean(0);body=(front+hind)/2;body[1]=height*.56
 neck=head.copy();neck[1]=min(head[1],height*.62);neck[2]+=.015
 rear=verts[verts[:,2]>np.quantile(verts[:,2],.99)];tip=rear.mean(0);tailroot=hind.copy();tailroot[0]=body[0];tailroot[1]=height*.56;tailroot[2]=max(hind[2],body[2]+.02)
 if np.linalg.norm(tip-tailroot)<.03:raise ValueError('target_tail_geometric_landmark_degenerate')
 points={'Body':body,'Spine':np.array([body[0],height*.60,front[2]*.6+body[2]*.4]),'Neck':neck,'Head':head,'TailRoot':tailroot,'Tail01':tailroot*.65+tip*.35,'Tail02':tailroot*.3+tip*.7}
 parents={'Body':None,'Spine':'Body','Neck':'Spine','Head':'Neck','TailRoot':'Body','Tail01':'TailRoot','Tail02':'Tail01'};chains={}
 for s,f in zip(PAWS,feet):
  hip=f.copy();hip[1]=height*(.53 if s.startswith('F') else .56);hip[0]*=.9
  ankle=foot_probe[PAWS.index(s)].copy();ankle[1]+=.018
  knee=(hip+ankle)*.5;knee[2]+=.025 if s.startswith('F') else -.028
  toe=ankle.copy();toe[2]-=.015
  chains[s]=[f'{part}_{s}' for part in ('Hip','Knee','Ankle','Toe')]
  for i,(name,pt) in enumerate(zip(chains[s],(hip,knee,ankle,toe))):points[name]=pt;parents[name]=chains[s][i-1] if i else ('Spine' if s.startswith('F') else 'Body')
 bpy.ops.object.select_all(action='DESELECT');ad=bpy.data.armatures.new('TargetNativeCatSkeleton');arm=bpy.data.objects.new('TargetNativeCatRig',ad);bpy.context.collection.objects.link(arm);arm.select_set(True);bpy.context.view_layer.objects.active=arm;bpy.ops.object.mode_set(mode='EDIT')
 for name,pt in points.items():
  bone=ad.edit_bones.new(name);bone.head=vec(pt)
  child=next((n for n in points if parents[n]==name),None)
  end=points[child] if child else (tip if name=='Tail02' else pt+([0,0,-.02] if name.startswith('Toe') else [0,.025,-.01]))
  if np.linalg.norm(end-pt)<.005:end=pt+[0,.01,0]
  bone.tail=vec(end)
 for name,parent in parents.items():
  if parent:ad.edit_bones[name].parent=ad.edit_bones[parent]
 bpy.ops.object.mode_set(mode='OBJECT');bpy.context.scene.render.fps=30
 # Heat diffusion is attempted on the new target rig, with a recorded fallback.
 heat_error=None
 if a.weight_method=='heat':
  for ob in meshes:ob.select_set(True)
  arm.select_set(True);bpy.context.view_layer.objects.active=arm
  try:bpy.ops.object.parent_set(type='ARMATURE_AUTO')
  except RuntimeError as e:heat_error=str(e)
 weight_summaries=[];sole_refs={};global_offset=0
 for pi,ob in enumerate(meshes):
  if not any(m.type=='ARMATURE' for m in ob.modifiers):m=ob.modifiers.new('OwnTargetSkin','ARMATURE');m.object=arm
  ob.parent=arm;ob.matrix_parent_inverse=arm.matrix_world.inverted()
  groups={name:(ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)) for name in points}
  coords=np.array([hvec(ob.matrix_world@v.co) for v in ob.data.vertices]);dists=np.linalg.norm(coords[:,None,[0,2]]-feet[None,:,[0,2]],axis=-1);nearest=dists.argmin(1)
  complete_heat=a.weight_method=='heat' and not heat_error and all(len(v.groups)>0 for v in ob.data.vertices)
  original_group={g.index:g.name for g in ob.vertex_groups};sole_count={s:0 for s in PAWS}
  for vi,v in enumerate(ob.data.vertices):
   pt=coords[vi];side=PAWS[nearest[vi]];hip,knee,ankle,toe=chains[side];fy=foot_probe[nearest[vi],1];legtop=points[hip][1];u=np.clip((pt[1]-fy)/(legtop-fy),0,1)
   weights={original_group[g.group]:g.weight for g in v.groups if original_group[g.group] in points} if complete_heat else {}
   # Distal sole ownership is rigid: the same material vertex stays planted.
   is_sole=pt[1]<=fy+.013 and dists[vi,nearest[vi]]<radius*1.5
   if is_sole:weights={ankle:1.};sole_count[side]+=1
   elif not complete_heat:
    radial=float(dists[vi,nearest[vi]]/(radius*(1+u*.5)));radial_weight=1-smooth5(float(np.clip((radial-1.6)/1.2,0,1)));vertical_weight=1-smooth5(float(np.clip((u-.4)/.65,0,1)));second=float(np.partition(dists[vi],1)[1]);dominance=smooth5(float(np.clip((second-dists[vi,nearest[vi]])/(float(sep.min())*.5),0,1)));strength=radial_weight*vertical_weight*dominance
    if u<.26:limb={ankle:1.}
    elif u<.65:
     w=(u-.26)/.39;limb={ankle:1-w,knee:w}
    else:
     w=np.clip((u-.65)/.35,0,1);limb={knee:1-w,hip:w}
    spine_weight=smooth5(float(np.clip((body[2]+height*.18-pt[2])/(height*.4),0,1)));weights={'Spine':(1-strength)*spine_weight,'Body':(1-strength)*(1-spine_weight)};weights.update({n:w*strength for n,w in limb.items()})
    hgate=max(0,min(1,(head[2]+height*.06-pt[2])/(height*.12)))*max(0,min(1,(pt[1]-height*.35)/(height*.16)))
    if hgate>0:weights={n:w*(1-hgate) for n,w in weights.items()};weights['Head']=hgate
    tail_axis=tip-tailroot;tail_u=float(np.clip(np.dot(pt-tailroot,tail_axis)/np.dot(tail_axis,tail_axis),0,1));tail_distance=float(np.linalg.norm(pt-(tailroot+tail_u*tail_axis)));tail_tube=1-smooth5(float(np.clip((tail_distance-height*.06)/(height*.18),0,1)));tail_weight=tail_tube*smooth5(float(np.clip((pt[2]-tailroot[2])/(height*.18),0,1)))*smooth5(float(np.clip((pt[1]-height*.04)/(height*.16),0,1)))
    if tail_weight>0:
     path_length=np.linalg.norm(tip-tailroot);u_tail=float(np.clip(np.dot(pt-tailroot,tip-tailroot)/(path_length**2),0,1));knots=np.array([0.,.35,.7]);k=min(1,int(np.searchsorted(knots,u_tail,side='right')-1));mix=float(np.clip((u_tail-knots[k])/(knots[k+1]-knots[k]),0,1));tail_names=('TailRoot','Tail01','Tail02');weights={n:w*(1-tail_weight) for n,w in weights.items()};weights[tail_names[k]]=tail_weight*(1-mix);weights[tail_names[k+1]]=tail_weight*mix
   # Attach the actual emitter neighbourhood to the head, including nose shells.
   if np.linalg.norm(pt-mouth)<.012:weights={'Head':1.}
   weights=sorted(((n,w) for n,w in weights.items() if w>1e-7),key=lambda z:-z[1])[:4]
   if not weights:weights=[('Body',1.)]
   total=sum(w for n,w in weights)
   for vg in ob.vertex_groups:vg.remove([vi])
   for n,w in weights:groups[n].add([vi],float(w/total),'REPLACE')
  # Harmonic regularization on spatially welded target topology. UV seams
  # remain separate in the mesh; they share only the computed skin field.
  bone_names=list(points);bone_index={n:i for i,n in enumerate(bone_names)};w=np.zeros((len(coords),len(bone_names)),float)
  for vi,v in enumerate(ob.data.vertices):
   for g in v.groups:
    name=original_group.get(g.group,ob.vertex_groups[g.group].name)
    if name in bone_index:w[vi,bone_index[name]]=g.weight
  _,first,weld=np.unique(np.round(coords/1e-7).astype(np.int64),axis=0,return_index=True,return_inverse=True);wc=coords[first];nunique=len(first);w0=np.zeros((nunique,len(bone_names)));counts=np.bincount(weld,minlength=nunique)
  for j in range(len(bone_names)):w0[:,j]=np.bincount(weld,weights=w[:,j],minlength=nunique)/counts
  ee=np.array([list(e.vertices) for e in ob.data.edges],int);ee=np.unique(np.sort(weld[ee],axis=1),axis=0);ee=ee[ee[:,0]!=ee[:,1]];ew=1/np.maximum(np.linalg.norm(wc[ee[:,0]]-wc[ee[:,1]],axis=1),.0001)
  degree=np.bincount(ee[:,0],weights=ew,minlength=nunique)+np.bincount(ee[:,1],weights=ew,minlength=nunique);degree=np.maximum(degree,1e-20)
  protected=np.zeros(nunique,bool)
  for si,s in enumerate(PAWS):protected|=(wc[:,1]<=foot_probe[si,1]+.013)&(np.linalg.norm(wc[:,[0,2]]-feet[si,[0,2]],axis=1)<radius*1.5)&(w0[:,bone_index[chains[s][2]]]>.999)
  protected|=np.linalg.norm(wc-mouth,axis=1)<.012
  current=w0.copy()
  for iteration in range(256):
   neighbour=np.empty_like(current)
   for j in range(len(bone_names)):neighbour[:,j]=(np.bincount(ee[:,0],weights=current[ee[:,1],j]*ew,minlength=nunique)+np.bincount(ee[:,1],weights=current[ee[:,0],j]*ew,minlength=nunique))/degree
   current=.25*current+.75*(.0005*w0+.9995*neighbour);current[protected]=w0[protected]
  selected=np.tile(np.arange(len(bone_names)),(nunique,1));values=current/current.sum(1,keepdims=True)
  for vi in range(len(coords)):
   wi=weld[vi]
   for vg in ob.vertex_groups:vg.remove([vi])
   for ni,value in zip(selected[wi],values[wi]):
    if value>1e-7:groups[bone_names[ni]].add([vi],float(value),'REPLACE')
  print('TARGET_WEIGHT_DIFFUSION',len(coords),'welded',nunique,'iterations',256,'protected',int(protected.sum()),flush=True)
  for si,s in enumerate(PAWS):
   ids=np.array([vi for vi,v in enumerate(ob.data.vertices) if any(g.group==groups[chains[s][2]].index and g.weight>.99 for g in v.groups)],int)
   if len(ids):
    vi=int(ids[np.argmin(coords[ids,1])]);candidate=(float(coords[vi,1]),pi,vi,coords[vi].copy())
    if s not in sole_refs or candidate[0]<sole_refs[s][0]:sole_refs[s]=candidate
  weight_summaries.append({'part':pi,'vertices':len(coords),'triangles':sum(len(f.vertices)-2 for f in ob.data.polygons),'heat_diffusion_used':complete_heat,'sole_vertex_counts':sole_count})
 assert set(sole_refs)==set(PAWS),('missing_rigid_soles',sole_refs)
 rest={n:arm.data.bones[n].matrix_local.copy() for n in points};rest_parts=snapshot_meshes(meshes)
 rest_error=max(float(np.max(np.abs(p-(o@level.T+shift)))) for p,o in zip(rest_parts,original));assert rest_error<1e-5,('rest_geometry_changed',rest_error)
 targets={};offsets=dict(zip(PAWS,(.25,.75,0,.5)))
 for s,c in chains.items():
  ob=bpy.data.objects.new('TargetFoot_'+s,None);bpy.context.collection.objects.link(ob);targets[s]=ob
  con=arm.pose.bones[c[1]].constraints.new('IK');con.target=ob;con.chain_count=2;con.use_tail=True;con.iterations=200
  # The sole retains the neutral world orientation independently of toe curl.
  rot=arm.pose.bones[c[2]].constraints.new('COPY_ROTATION');rot.target=ob;rot.target_space='WORLD';rot.owner_space='WORLD';rot.mix_mode='REPLACE'
  ob.rotation_mode='QUATERNION';ob.rotation_quaternion=rest[c[2]].to_quaternion()
 donor_idle={n:np.array(v['matrix'])[:3,:3] for n,v in template['sampled_actions']['Idle'][0]['bones'].items()}
 source_samples=template['sampled_actions']['Walking'];mapping={'Body':'Bone','Spine':'Bone.001','Neck':'Bone.002','Head':'Bone.003','TailRoot':'Bone.004','Tail01':'Bone.005','Tail02':'Bone.006'}
 mapping.update({chains[s][0]:c[0] for s,c in DONORS.items()})
 actions={};frames=[];nf=round(a.period*2*30)
 for action_name in ('Idle','Walking'):
  action=bpy.data.actions.new(action_name);action.use_fake_user=True;arm.animation_data_create();arm.animation_data.action=action;actions[action_name]=action
  for frame in range(nf+1 if action_name=='Walking' else 2):
   t=frame/30.;bpy.context.scene.frame_set(frame)
   for pb in arm.pose.bones:pb.matrix_basis=Matrix.Identity(4)
   for pb in arm.pose.bones:
    for con in pb.constraints:con.mute=(action_name=='Idle')
   if action_name=='Walking':
    index=min(len(source_samples)-1,round((t/a.period%1)*(len(source_samples)-1)));sample=source_samples[index]['bones']
    for n,don in mapping.items():
     source_rot=np.array(sample[don]['matrix'])[:3,:3];delta=DC@source_rot@np.linalg.inv(donor_idle[don])@DC.T
     q=quaternion_scaled(delta,.18 if n in ('Body','Spine','Neck') else .35)
     own=rest[n].to_quaternion();arm.pose.bones[n].rotation_mode='QUATERNION';arm.pose.bones[n].rotation_quaternion=own.inverted()@q@own
    # Root and foot route have identical +Y Blender / -Z Habitat travel.
    root_world=Vector((0,a.speed*t,0));body_pb=arm.pose.bones['Body'];local=rest['Body'].to_3x3().inverted()@root_world;body_pb.location=local
    for s,c in chains.items():
     q=t/a.period+offsets[s];cycle=math.floor(q);phase=q-cycle;plant=(cycle+.25-offsets[s])*a.period
     anchor=sole_refs[s][3].copy();anchor[2]-=a.speed*(plant+.375*a.period)
     if phase<.25:anchor[2]+=a.speed*a.period*(1-smooth5(phase/.25))
     anchor[1]=.0004+(.008*math.sin(math.pi*phase/.25)**2 if phase<.25 else 0)
     offset=points[c[2]]-sole_refs[s][3];targets[s].location=vec(anchor+offset)
   bpy.context.view_layer.update()
   if action_name=='Walking':
    evaluated=arm.evaluated_get(bpy.context.evaluated_depsgraph_get());lower=0.
    for side,chain in chains.items():
     hip_world=np.array(evaluated.pose.bones[chain[0]].head);target_world=np.array(targets[side].location);length=ad.bones[chain[0]].length+ad.bones[chain[1]].length;horizontal=np.linalg.norm((hip_world-target_world)[:2]);lower=max(lower,float(hip_world[2]-target_world[2]-math.sqrt(max(0,(length*.999)**2-horizontal**2))))
    if lower>.08:raise ValueError('pelvis_lowering_exceeds_80mm')
    arm.pose.bones['Body'].location-=rest['Body'].to_3x3().inverted()@Vector((0,0,lower))
    bpy.context.view_layer.update()
   eval_arm=arm.evaluated_get(bpy.context.evaluated_depsgraph_get());matrices={pb.name:pb.matrix.copy() for pb in eval_arm.pose.bones}
   # Bake evaluated IK into regular local TRS channels; no controls required.
   for pb in arm.pose.bones:
    for con in pb.constraints:con.mute=True
   for name in points:
    pb=arm.pose.bones[name];pb.matrix=matrices[name];pb.rotation_mode='QUATERNION'
    pb.keyframe_insert('location',frame=frame,group=name);pb.keyframe_insert('rotation_quaternion',frame=frame,group=name);pb.keyframe_insert('scale',frame=frame,group=name)
   bpy.context.view_layer.update()
  for curve in action.fcurves:
   for key in curve.keyframe_points:key.interpolation='LINEAR'
 for pb in arm.pose.bones:
  for con in list(pb.constraints):pb.constraints.remove(con)
 for target in targets.values():bpy.data.objects.remove(target,do_unlink=True)
 # Check every baked frame against the fixed support material vertex.
 arm.animation_data.action=actions['Walking'];edges=[]
 for ob,base in zip(meshes,rest_parts):
  e=np.array([list(x.vertices) for x in ob.data.edges],int);edges.append((e,np.linalg.norm(base[e[:,0]]-base[e[:,1]],axis=1)))
 max_stretch=1.;max_drift=0.;anchors={};max_gap=0.;min_floor=1.;max_pitch=0.;max_heading=0.;mouth_error=0.
 for frame in range(nf+1):
  t=frame/30.;bpy.context.scene.frame_set(frame);bpy.context.view_layer.update();parts=snapshot_meshes(meshes)
  marks={n:hvec(arm.matrix_world@arm.pose.bones[n].head) for n in ('Body','Head','TailRoot')}
  mouth_local=rest['Head'].inverted()@vec(mouth);mouth_now=hvec(arm.matrix_world@arm.pose.bones['Head'].matrix@mouth_local)
  faxis=marks['Head']-marks['TailRoot'];heading=math.degrees(math.acos(np.clip(-faxis[2]/np.linalg.norm(faxis[[0,2]]),-1,1)));pitch=math.degrees(math.atan2(abs(faxis[1]),np.linalg.norm(faxis[[0,2]])))
  ff={}
  for s in PAWS:
   q=t/a.period+offsets[s];cycle=math.floor(q);phase=q-cycle;active=phase>=.25;_,pi,vi,_=sole_refs[s];pt=parts[pi][vi]
   if active:
    key=(s,cycle)
    if key not in anchors:anchors[key]=pt.copy()
    max_drift=max(max_drift,float(np.linalg.norm(pt-anchors[key])))
    max_gap=max(max_gap,abs(float(pt[1])))
   ff[s]={'in_contact':active,'support_segment':cycle if active else None,'support_vertex':[pi,vi],'support_world_m':pt.tolist(),'floor_delta_m':float(pt[1])}
  min_floor=min(min_floor,min(float(x[:,1].min()) for x in parts));max_heading=max(max_heading,heading);max_pitch=max(max_pitch,pitch)
  for part,(e,base) in zip(parts,edges):
   valid=base>1e-5;stretch=np.linalg.norm(part[e[:,0]]-part[e[:,1]],axis=1)[valid]/base[valid];max_stretch=max(max_stretch,float(stretch.max()))
  frames.append({'time_s':t,'root_world_m':marks['Body'].tolist(),'head_world_m':marks['Head'].tolist(),'tail_world_m':marks['TailRoot'].tolist(),'mouth_world_m':mouth_now.tolist(),'feet':ff,'body_floor_min_m':min(float(x[:,1].min()) for x in parts),'body_head_tail_pitch_deg':pitch,'heading_angle_deg':heading})
 # Preserve target-owned mesh, PBR and complete baked actions in a derived GLB.
 arm.animation_data.action=actions['Idle'];bpy.context.scene.frame_set(0);bpy.context.view_layer.update();bpy.ops.object.select_all(action='DESELECT')
 arm.select_set(True)
 for ob in meshes:ob.select_set(True)
 bpy.context.view_layer.objects.active=arm
 bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'rig.blend'))
 bpy.ops.export_scene.gltf(filepath=str(a.output/'motion.glb'),export_format='GLB',use_selection=True,export_animations=True,export_animation_mode='ACTIONS',export_frame_range=False,export_force_sampling=True,export_sampling_interpolation_fallback='LINEAR',export_skins=True,export_yup=True,export_all_influences=True)
 # Derive the emitter offset from the actual exported head basis, not a guess.
 data=(a.output/'motion.glb').read_bytes();length=struct.unpack_from('<I',data,12)[0];d=json.loads(data[20:20+length]);binary=data[28+length:];nodes=copy.deepcopy(d['nodes'])
 def ac(i):
  x=d['accessors'][i];v=d['bufferViews'][x['bufferView']];width={'SCALAR':1,'VEC3':3,'VEC4':4}[x['type']];assert x['componentType']==5126
  return np.ndarray((x['count'],width),np.float32,buffer=binary,offset=v.get('byteOffset',0)+x.get('byteOffset',0),strides=(v.get('byteStride',width*4),4))
 idle=next(x for x in d['animations'] if x['name']=='Idle')
 for c in idle['channels']:nodes[c['target']['node']][c['target']['path']]=ac(idle['samplers'][c['sampler']]['output'])[0].tolist()
 g={}
 def visit(i,parent):
  n=nodes[i]
  if 'matrix' in n:m=np.array(n['matrix']).reshape(4,4).T
  else:
   m=np.eye(4);x,y,z,w=n.get('rotation',[0,0,0,1]);m[:3,:3]=np.array(Quaternion((w,x,y,z)).to_matrix())@np.diag(n.get('scale',[1,1,1]));m[:3,3]=n.get('translation',[0,0,0])
  g[i]=parent@m
  for j in n.get('children',[]):visit(j,g[i])
 for i in d['scenes'][d.get('scene',0)]['nodes']:visit(i,np.eye(4))
 names={n.get('name'):i for i,n in enumerate(d['nodes'])};head_local=(np.linalg.inv(g[names['Head']])@np.r_[mouth,1])[:3]
 manifest={'schema':'owned_target_cat_rig_candidate','anchors':[{'anchor_id':'body','joint_id':'Body'},{'anchor_id':'head','joint_id':'Head'},*[{'anchor_id':n,'joint_id':c[-1]} for n,c in zip(ANCHORS,chains.values())]],'target_native_joint_positions_m':{n:pt.tolist() for n,pt in points.items()},'motion_license':'CC0-1.0','target_support_plane':{'matrix':trans.tolist(),'tilt_deg':tilt,'residual_range_m':plane_residual,'source_paw_points':foot_probe.tolist()}}
 mp=a.output/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');cp=a.output/'candidate.json';cp.write_text(json.dumps({'rotation':np.eye(3).tolist(),'translation_m':[0,0,0],'scale_xyz':[1,1,1],'emitter_head_local_m':head_local.tolist(),'source_package_manifest':str(mp)},indent=2)+'\n')
 record['has_skeleton']=True;record['has_walking']=True;record['walking_candidate']=str(cp);record['files']['motion_runtime_glb']={'path':str(a.output/'motion.glb'),'skins':len(d['skins']),'animations':[x['name'] for x in d['animations']]}
 record['files']['motion_master_glb']={'path':str(a.output/'motion.glb')}
 (a.output/'inventory.json').write_text(json.dumps({'assets':[record]},indent=2)+'\n')
 metric={'max_heading_deg':max_heading,'max_stance_drift_m':max_drift,'max_contact_gap_m':max_gap,'min_full_skin_floor_m':min_floor,'max_body_pitch_deg':max_pitch,'max_edge_stretch_ratio':max_stretch}
 errors=[]
 if max_heading>15:errors.append('heading')
 if max_drift>.003:errors.append('stance_drift')
 if max_gap>.012:errors.append('contact_gap')
 if min_floor<-.00001:errors.append('full_skin_penetration')
 if max_pitch>45:errors.append('body_upright')
 if max_stretch>1.5:errors.append('mesh_stretch')
 result={'asset_id':a.asset,'status':'baked_candidate_needs_independent_validation' if not errors else 'failed_baked_geometry','errors':errors,'source_glb':row['runtime_glb'],'template':template['source_blend'],'license':'CC0-1.0','weight_method_requested':a.weight_method,'heat_error':heat_error,'joints':len(points),'gait_period_s':a.period,'root_speed_m_s':a.speed,'weight_regularization':'256 Jacobi iterations on welded own mesh, fixed rigid soles and mouth, no donor weights','target_rest_max_error_m':rest_error,'support_plane':manifest['target_support_plane'],'weights':weight_summaries,'metrics':metric,'frame_count':len(frames),'frames':frames,'cpu_s':time.process_time()-started,'self_intersection_status':'not_run','native_habitat_status':'not_run'}
 (a.output/'rigging.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('frames','weights','support_plane')},flush=True)
 if errors:raise SystemExit(2)
if __name__=='__main__':main()
