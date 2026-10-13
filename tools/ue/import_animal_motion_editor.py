"""Import only owned repaired clips and read every native skeletal pose."""
import os,json,math,time,traceback
from pathlib import Path
import unreal
ROOT=Path(os.environ['V7_ANIMAL_MOTION_ROOT']).resolve();REQUEST=Path(os.environ['V7_ANIMAL_MOTION_IMPORT_REQUEST']).resolve()
if ROOT not in REQUEST.parents:raise RuntimeError('owned UE request required')
request=json.loads(REQUEST.read_text())
def xyz(v):return [float(v.x),float(v.y),float(v.z)]
def norm(s):return str(s).lower().replace('_','').replace(' ','').replace('-','')
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))
def transform(t):return {'translation_cm':xyz(t.translation),'rotation_xyzw':[float(t.rotation.x),float(t.rotation.y),float(t.rotation.z),float(t.rotation.w)],'scale_xyz':xyz(t.scale3d)}
results=[]
for row in request['assets']:
 result={'asset_id':row['asset_id'],'status':'failed','frames':[],'request':str(REQUEST)}
 try:
  dest=row['destination']
  if not dest.startswith('/Game/V7AnimalMotion20261013/'):raise RuntimeError('private destination required')
  if row.get('reuse_objects'):
   objects=[unreal.load_asset(x['path']) for x in row['reuse_objects']]
  else:
   task=unreal.AssetImportTask();task.filename=row['glb'];task.destination_path=dest;task.automated=True;task.save=True;task.replace_existing=False;task.async_=True
   unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task]);objects=task.get_objects()
  items=[{'name':x.get_name(),'path':x.get_path_name(),'class':x.get_class().get_name()} for x in objects];result['objects']=items
  mesh=next(x for x in objects if x.get_class().get_name()=='SkeletalMesh');clips=[x for x in objects if x.get_class().get_name()=='AnimSequence']
  if len(clips)!=1:raise RuntimeError('owned file must import exactly its single repaired Walking sequence')
  clip=clips[0]
  unreal.EditorAssetLibrary.save_directory(dest,only_if_is_dirty=True,recursive=True)
  options=unreal.AnimPoseEvaluationOptions();options.optional_skeletal_mesh=mesh
  initial=unreal.AnimPoseExtensions.get_anim_pose_at_time(clip,0.,options);names={norm(n):str(n) for n in unreal.AnimPoseExtensions.get_bone_names(initial)}
  mapped={n:names[norm(n)] for n in row['bone_names']}
  for frame in row['frames']:
   pose=unreal.AnimPoseExtensions.get_anim_pose_at_time(clip,frame['time_s'],options)
   bones={name:transform(unreal.AnimPoseExtensions.get_bone_pose(pose,native,unreal.AnimPoseSpaces.WORLD)) for name,native in mapped.items()}
   result['frames'].append({'time_s':frame['time_s'],'bones':bones})
  result.update(status='native_clip_imported_pose_readback_pending_geometry',skeletal_mesh=mesh.get_path_name(),animation=clip.get_path_name(),bone_mapping=mapped,animation_length_s=float(clip.get_editor_property('sequence_length')))
 except Exception:result['error']=traceback.format_exc()
 dump(ROOT/'ue/imports'/request['revision']/(row['case']+'.json'),result);results.append(result);unreal.log('ANIMAL_MOTION_IMPORT '+row['asset_id']+' '+result['status'])
unreal.SystemLibrary.execute_console_command(None,'Editor.AsyncAssetCompilationFinishAll')
dump(ROOT/'ue/imports'/request['revision']/'complete.json',{'results':results,'status':'pass' if all(x['status']!='failed' for x in results) else 'failed'})
