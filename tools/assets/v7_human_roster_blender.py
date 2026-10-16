"""Task-scoped Rocketbox UV extraction and native animation builds.

The supplied legacy tool directory is read-only; all outputs go to --task-root.
No source FBX, texture, archive, checkpoint or shared project is overwritten.
"""
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True
import bpy
import numpy as np
from mathutils import Vector,Matrix
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['extract','build'],required=True);p.add_argument('--task-root',type=Path,required=True);p.add_argument('--legacy-root',type=Path,required=True);p.add_argument('--asset-id');a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
T=a.task_root.resolve();sys.path.insert(0,str(a.legacy_root));sys.path.insert(0,str(a.legacy_root/'tools'));sys.path.insert(0,str(a.legacy_root/'tools/spike_rlr'))
from tools import blender_build_rocketbox_batch_runtime as batch
from tools import blender_retarget_rocketbox_walk as rt
from blender_render_rocketbox_source_review import rebuild_surface_material,rebuild_opacity_material
selection=json.loads((T/'registrations/selection.json').read_text());rows=selection['avatars']
if a.asset_id:rows=[r for r in rows if r['asset_id']==a.asset_id]
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def uv_islands(mesh):
 uv=mesh.data.uv_layers.active.data;polys=[q for q in mesh.data.polygons if q.material_index==0];parent={q.index:q.index for q in polys}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 edges={}
 for q in polys:
  ls=list(q.loop_indices)
  for i,l in enumerate(ls):
   n=ls[(i+1)%len(ls)];u=mesh.data.loops[l].vertex_index;v=mesh.data.loops[n].vertex_index;edge=tuple(sorted((u,v)));mapping={u:tuple(round(float(x),6) for x in uv[l].uv),v:tuple(round(float(x),6) for x in uv[n].uv)}
   for other,m in edges.get(edge,[]):
    if m==mapping:parent[find(q.index)]=find(other)
   edges.setdefault(edge,[]).append((q.index,mapping))
 groups={}
 for q in polys:groups.setdefault(find(q.index),[]).append(q)
 out=[]
 for idx,g in enumerate(sorted(groups.values(),key=lambda z:min(q.index for q in z))):
  verts=sorted({i for q in g for i in q.vertices});points=np.array([list(mesh.matrix_world@mesh.data.vertices[i].co) for i in verts]);uvs=[[[float(v) for v in uv[l].uv] for l in q.loop_indices] for q in g];uvflat=np.array([v for q in uvs for v in q]);out.append({'island_id':idx,'face_indices':[q.index for q in g],'face_count':len(g),'xyz_min':points.min(0).tolist(),'xyz_max':points.max(0).tolist(),'xyz_mean':points.mean(0).tolist(),'uv_min':uvflat.min(0).tolist(),'uv_max':uvflat.max(0).tolist(),'uv_polygons':uvs})
 return out
for row in rows:
 if a.mode=='extract' and not row['controlled_color']:continue
 dest=T/('uv' if a.mode=='extract' else 'assets/rocketbox')/row['asset_id'];dest.mkdir(parents=True,exist_ok=True)
 if (dest/('uv_islands.json' if a.mode=='extract' else 'build.json')).exists():continue
 avatar=batch._import_avatar(row['legacy_record']);mesh=avatar.mesh;arm=avatar.armature
 if a.mode=='extract':
  write(dest/'uv_islands.json',{'source_fbx':row['fbx'],'source_sha256':sha(Path(row['fbx'])),'islands':uv_islands(mesh)})
  print('UV_DONE',row['asset_id'],flush=True);continue
 # Build logic is task scoped, with the proven Rocketbox animation helpers.
 batch.sanitize_non_surface_loose_vertices(mesh,row['legacy_record'])
 for mat in mesh.data.materials:
  raw_stem=mat.name.split('.')[0];tag='_a' if '_A_'in raw_stem else '_b' if '_B_'in raw_stem else '';stem=raw_stem.replace('_A_','_').replace('_B_','_');td=Path(row['textures']);color=td/(stem+'_color'+tag+'.tga');normal=td/(stem+'_normal'+tag+'.tga');spec=td/(stem+'_specular'+tag+'.tga')
  if not color.exists():raise FileNotFoundError(str(color))
  if stem.endswith('_opacity'):rebuild_opacity_material(mat,color)
  else:rebuild_surface_material(mat,color,normal,spec)
 for mat in mesh.data.materials:
  for n in mat.node_tree.nodes:
   if n.type=='BSDF_PRINCIPLED':n.inputs['Metallic'].default_value=0.0;n.inputs['Roughness'].default_value=max(.72,float(n.inputs['Roughness'].default_value))
 source_metrics=rt.mesh_metrics(mesh,arm)
 actions=batch._bake_actions(avatar,row['legacy_record'],batch._motion_contract(row['gender']))
 act={'Walking':actions['walk_action'],'Standing_Idle':actions['idle_action']}
 # Bake the sex-matched official seated neutral motion using the same retargeter.
 rename=batch._canonicalize_bip02(avatar) if row['legacy_record']['blender_audit']['skeleton_family']=='Bip02' else {}
 code='f' if row['gender']=='female' else 'm';motion=Path('/data/datasets/rocketbox/Microsoft-Rocketbox/Assets/Animations/all_animations_max_motextr_static')/f'{code}_sit_chair_idle_neutral_01.max.fbx'
 src=rt.import_source_motion(motion)
 try:
  rt.validate_mapping(src.armature,arm);start,end=rt.integer_frame_range(src.action);cached,helper=rt.cache_source_frames(src,start,end);order=rt.parent_first_names(arm)
 finally:rt.remove_source_import(src)
 sit=rt.create_target_action(arm,row['asset_id']+'_sit');rt.bake_target_action(arm,cached,helper,order,sit);sit.name='Sitting_Idle'
 if rename:batch._restore_bip02(avatar,rename,[sit])
 act['Sitting_Idle']=sit
 assert source_metrics==rt.mesh_metrics(mesh,arm),'animation changed source mesh/rig'
 for tr in list(arm.animation_data.nla_tracks):arm.animation_data.nla_tracks.remove(tr)
 arm.data.pose_position='POSE';arm.animation_data.action=act['Standing_Idle'];bpy.context.scene.frame_set(1);bpy.context.view_layer.update()
 # Keep authored stature in metres. A common wrapper grounds the standing pose.
 dep=bpy.context.evaluated_depsgraph_get();ev=mesh.evaluated_get(dep);em=ev.to_mesh();pts=np.array([list(ev.matrix_world@v.co) for v in em.vertices]);ev.to_mesh_clear();base_floor=float(pts[:,2].min())
 root=bpy.data.objects.new('HumanRosterRoot',None);bpy.context.collection.objects.link(root)
 for obj in [arm,mesh]:
  if obj.parent is None:world=obj.matrix_world.copy();obj.parent=root;obj.matrix_world=world
 root.location.z=-base_floor
 # Remove horizontal root trajectories from the wrapper object's action curves.
 # Pelvis motion is retained; actor translation is the route authority.
 reports={}
 for name,action in act.items():
  for curve in action.fcurves:
   if curve.data_path=='location' and curve.array_index<2:
    first=curve.keyframe_points[0].co.y
    for k in curve.keyframe_points:k.co.y=first
  arm.animation_data.action=action;arm.data.pose_position='POSE';start,end=rt.integer_frame_range(action);samples=[]
  for frame in sorted(set([start,end,(start+end)//2]+list(range(start,end+1,max(1,(end-start)//24))))):
   bpy.context.scene.frame_set(frame);bpy.context.view_layer.update();ev=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get());em=ev.to_mesh();pts=np.array([list(ev.matrix_world@v.co) for v in em.vertices]);ev.to_mesh_clear();fam=row['legacy_record']['blender_audit']['skeleton_family'];lip=arm.pose.bones[fam+' MUpperLip'];mouth=arm.matrix_world@lip.head;foot=[]
   for side in ['L','R']:foot.append(list(arm.matrix_world@arm.pose.bones[fam+' '+side+' Foot'].head))
   samples.append({'frame':frame,'bounds_min_m':pts.min(0).tolist(),'bounds_max_m':pts.max(0).tolist(),'mouth_m':list(mouth),'ankles_m':foot})
  reports[name]={'frame_range':[start,end],'samples':samples,'source_motion':str(motion) if name=='Sitting_Idle' else str(batch._motion_contract(row['gender'])[name]['path'])}
 arm.animation_data.action=None
 for name,action in act.items():
  tr=arm.animation_data.nla_tracks.new();tr.name=name;strip=tr.strips.new(name,int(action.frame_range[0]),action);strip.action_frame_start=action.frame_range[0];strip.action_frame_end=action.frame_range[1];tr.mute=True
 bpy.context.scene.render.fps=30
 used={n.image for mat in mesh.data.materials for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image is not None}
 for img in used:
  if img.source=='FILE':img.pack()
 bpy.ops.object.select_all(action='DESELECT');arm.select_set(True);mesh.select_set(True);root.select_set(True);bpy.context.view_layer.objects.active=arm
 bpy.ops.export_scene.gltf(filepath=str(dest/'runtime.glb'),export_format='GLB',use_selection=True,export_animations=True,export_animation_mode='NLA_TRACKS',export_extra_animations=True,export_frame_range=False,export_force_sampling=True,export_materials='EXPORT',export_image_format='AUTO',export_yup=True)
 bpy.ops.wm.save_as_mainfile(filepath=str(dest/'runtime.blend'),check_existing=False)
 write(dest/'build.json',{'schema':'avengine_v7_rocketbox_runtime_v1','asset_id':row['asset_id'],'source':row['fbx'],'source_sha256':sha(Path(row['fbx'])),'source_mesh_metrics':source_metrics,'post_mesh_metrics':rt.mesh_metrics(mesh,arm),'standing_floor_wrapper_shift_m':-base_floor,'mouth_bone':row['legacy_record']['blender_audit']['skeleton_family']+' MUpperLip','actions':reports,'texture_source_paths':[str(x) for x in sorted(Path(row['textures']).glob('*.tga'))],'metallic':0,'roughness_min':.72,'runtime_sha256':sha(dest/'runtime.glb')})
 print('BUILD_DONE',row['asset_id'],flush=True)
