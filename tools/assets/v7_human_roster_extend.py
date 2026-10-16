"""Bake official Rocketbox gesture/turn clips into task-owned derivatives."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import bpy
p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--legacy-root',type=Path,required=True);p.add_argument('--asset-id',required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);T=a.task_root;R=T/'round2'
sys.dont_write_bytecode=True;sys.path[:0]=[str(a.legacy_root),str(a.legacy_root/'tools'),str(a.legacy_root/'tools/spike_rlr')]
from tools import blender_build_rocketbox_batch_runtime as batch
from tools import blender_retarget_rocketbox_walk as rt
row=next(x for x in json.loads((T/'registrations/selection.json').read_text())['avatars']if x['asset_id']==a.asset_id)
dest=R/'assets/humans'/a.asset_id;dest.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(T/'assets/rocketbox'/a.asset_id/'runtime.blend'))
arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');mesh=next(o for o in bpy.context.scene.objects if o.type=='MESH');avatar=SimpleNamespace(armature=arm,mesh=mesh)
metrics=rt.mesh_metrics(mesh,arm)
for tr in list(arm.animation_data.nla_tracks):arm.animation_data.nla_tracks.remove(tr)
arm.animation_data.action=None
existing={x.name:x for x in bpy.data.actions if x.name in ['Walking','Standing_Idle','Sitting_Idle']}
code='f'if row['gender']=='female'else'm'
paths={
 'Standing_Talk':'all_animations_max_motextr_static/'+code+'_gestic_talk_neutral_01.max.fbx',
 'Sitting_Gesture':'all_animations_max_motextr_static/'+code+'_sit_chair_gestic_thoughtful.max.fbx',
 'Turning_Left90':'all_animations_max_motextr_xyz/'+code+'_turn_left_90.max.fbx',
}
base_matrix=arm.matrix_world.copy();base_scale=arm.scale.copy();new={};reports={}
rename=batch._canonicalize_bip02(avatar)if row['legacy_record']['blender_audit']['skeleton_family']=='Bip02'else{}
for name,rel in paths.items():
 arm.animation_data.action=None;arm.matrix_world=base_matrix;arm.scale=base_scale
 motion=Path('/data/datasets/rocketbox/Microsoft-Rocketbox/Assets/Animations')/rel
 source=rt.import_source_motion(motion)
 try:
  rt.validate_mapping(source.armature,arm);start,end=rt.integer_frame_range(source.action);cached,helper=rt.cache_source_frames(source,start,end);order=rt.parent_first_names(arm)
 finally:rt.remove_source_import(source)
 arm.matrix_world=base_matrix;arm.scale=base_scale
 ac=rt.create_target_action(arm,a.asset_id+'_'+name);rt.bake_target_action(arm,cached,helper,order,ac);ac.name=name;new[name]=ac
 for c in ac.fcurves:
  if c.data_path=='location'and c.array_index<2:
   v=c.keyframe_points[0].co.y
   for k in c.keyframe_points:k.co.y=v
 reports[name]={'source_motion':str(motion),'frame_range':[start,end],'audio_driven_visemes':False}
if rename:batch._restore_bip02(avatar,rename,new.values())
arm.animation_data.action=None;arm.matrix_world=base_matrix;arm.scale=base_scale
assert metrics==rt.mesh_metrics(mesh,arm),'retarget changed base mesh/rig'
actions={**existing,**new};arm.animation_data.action=None
for name,ac in actions.items():
 tr=arm.animation_data.nla_tracks.new();tr.name=name;st=tr.strips.new(name,int(ac.frame_range[0]),ac);st.action_frame_start=ac.frame_range[0];st.action_frame_end=ac.frame_range[1];tr.mute=True
bpy.context.scene.render.fps=30
bpy.ops.object.select_all(action='DESELECT')
for o in [arm,mesh,*([arm.parent]if arm.parent else[])]:o.select_set(True)
bpy.context.view_layer.objects.active=arm
bpy.ops.export_scene.gltf(filepath=str(dest/'source_units.glb'),export_format='GLB',use_selection=True,export_animations=True,export_animation_mode='NLA_TRACKS',export_extra_animations=True,export_frame_range=False,export_force_sampling=True,export_materials='EXPORT',export_image_format='AUTO',export_yup=True)
(dest/'bake.json').write_text(json.dumps({'asset_id':a.asset_id,'source_blend':str(T/'assets/rocketbox'/a.asset_id/'runtime.blend'),'actions':list(actions),'new_actions':reports,'mesh_unchanged':True},indent=2))
print('EXTEND_DONE',a.asset_id,flush=True)
