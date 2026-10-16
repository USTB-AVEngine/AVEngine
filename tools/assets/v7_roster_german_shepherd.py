"""Retarget the retained Beagle motion onto the original German Shepherd rig."""
from pathlib import Path
import argparse,json,sys,math,hashlib
import bpy
from mathutils import Matrix,Vector
p=argparse.ArgumentParser();p.add_argument('--breed',choices=['german_shepherd','beagle'],default='german_shepherd');p.add_argument('--task-root',type=Path,required=True);p.add_argument('--legacy-root',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
T=a.task_root;R=T/'round2';D=R/'assets/dogs'/a.breed;D.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True;sys.path[:0]=[str(a.legacy_root),str(a.legacy_root/'tools')]
from tools import blender_retarget_quaternius_to_rocketbox_animal as rt
donor=Path('/data/jzy/code/AVEngine-lead-a/tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/visual.glb')
name='Dog_Beagle_01'if a.breed=='beagle'else'Dog_GermanShepard_01';stem='beagle'if a.breed=='beagle'else'shepherd_dog';root=Path('/data/datasets/rocketbox/Microsoft-Rocketbox/Assets/Animals')/name
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.scene.render.fps=30
source,objs,acts=rt.import_source(donor);target,mesh,_=rt.import_target(root/'Export'/(name+'.fbx'))
def token(s):
 s=s.replace('beagle ','').replace('dog ','').replace('EarR','Ear R')
 return s
source_names={token(b.name):b.name for b in source.data.bones}
rt.BONE_MAP={b.name:source_names[token(b.name)]for b in target.data.bones}
rt.LOCOMOTION_LIMB_BONES={b.name for b in target.data.bones if any(t in b.name for t in ['Thigh','Calf','Foot','UpperArm','Forearm','Hand'])}
rt.validate_mapping(source,target);before=rt.mesh_skin_sha256(mesh);records=[];out=[]
for name in ['Walking','Idle']:
 ac,rec=rt.bake_action(source,target,acts[name],name,limb_lateral_damping=1.0);out.append(ac);records.append(rec)
rt.remove_source(objs,acts.values());rt.remove_export_extras(target,mesh)
rt.install_pbr(mesh,root/'Textures'/(stem+'_color.tga'),root/'Textures'/(stem+'_bump.tga'),root/'Textures'/(stem+'_specular.tga'))
assert before==rt.mesh_skin_sha256(mesh)
rt.export_target(target,mesh,out,D/'source_units.glb')
(D/'retarget.json').write_text(json.dumps({'asset_id':'v7_rocketbox_'+a.breed+'_01','target_fbx':str(root/'Export'/(name+'.fbx')),'source_motion_glb':str(donor),'target_native_skin_unchanged':True,'target_bone_count':len(target.data.bones),'bone_map':rt.BONE_MAP,'actions':records,'mouth_bone':('beagle'if a.breed=='beagle'else'dog')+' Xtra Mouth','raw_runtime_front':'+X','material_metallic':0,'material_roughness':.72,'motion_provenance':'Beagle retained compatible motion; original Rocketbox dog FBX has no animations','geometry_provenance':'Microsoft Rocketbox Dog_GermanShepard_01, original dog mesh and native weights'},indent=2))
print('GERMAN_SHEPHERD_BUILD_DONE')
