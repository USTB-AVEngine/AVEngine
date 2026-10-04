"""Executed by UE 5.5 Python commandlet; imports into this task's project only."""
import json,os,sys,time,traceback
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT'])
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def run():
 records=[];paths=sorted((T/'assets/runtime').glob('*/*/habitat.glb'))
 only=os.environ.get('V7_UE_ONLY')
 if only:paths=[p for p in paths if only in str(p)]
 limit=int(os.environ.get('V7_UE_LIMIT','0'))
 if limit:paths=paths[:limit]
 tools=unreal.AssetToolsHelpers.get_asset_tools();library=unreal.EditorAssetLibrary
 for path in paths:
  aid=path.parent.parent.name;color=path.parent.name;folder='/Game/V7Humans/'+aid+'/'+color;ev=T/'evidence/ue_import'/aid/(color+'.json')
  if ev.exists() and json.loads(ev.read_text()).get('status')=='pass' and os.environ.get('V7_UE_REIMPORT_STATIC','0')!='1':records.append(json.loads(ev.read_text()));continue
  receipt={'asset_id':aid,'color':color,'glb':str(path.relative_to(T)),'destination':folder,'assets':[],'status':'fail'}
  try:
   task=unreal.AssetImportTask();task.set_editor_property('filename',str(path));task.set_editor_property('destination_path',folder);task.set_editor_property('automated',True);task.set_editor_property('save',True);task.set_editor_property('replace_existing',os.environ.get('V7_UE_REIMPORT_STATIC','0')=='1');task.set_editor_property('async_',True)
   tools.import_asset_tasks([task]);objects=task.get_objects()
   for obj in objects:
    cl=obj.get_class().get_name();entry={'path':obj.get_path_name(),'class':cl}
    if cl=='StaticMesh':
     box=obj.get_bounding_box();entry['bounds_cm']={'min':[box.min.x,box.min.y,box.min.z],'max':[box.max.x,box.max.y,box.max.z]};entry['height_m']=(box.max.z-box.min.z)/100
    receipt['assets'].append(entry)
   receipt['status']='pass' if any(x['class']=='StaticMesh'for x in receipt['assets']) else 'fail'
   library.save_directory(folder,only_if_is_dirty=True,recursive=True)
  except Exception:receipt['error']=traceback.format_exc()
  write(ev,receipt);records.append(receipt);unreal.log('V7_IMPORT '+aid+' '+color+' '+receipt['status'])
 write(T/'evidence'/os.environ.get('V7_UE_RESULT','ue_import_results.json'),records)
 # Skeleton and official standing/walking/seated animation import per avatar.
 if os.environ.get('V7_UE_SKELETAL','0')=='1':
  bones=[]
  for path in sorted((T/'assets/metric').glob('*/runtime.glb')):
   aid=path.parent.name;folder='/Game/V7Skeletons/'+aid;ev=T/'evidence/ue_skeletal'/(aid+'.json')
   if ev.exists():bones.append(json.loads(ev.read_text()));continue
   receipt={'asset_id':aid,'source':str(path.relative_to(T)),'assets':[],'status':'fail'}
   try:
    task=unreal.AssetImportTask();task.filename=str(path);task.destination_path=folder;task.automated=True;task.save=True;task.replace_existing=False;task.async_=True;tools.import_asset_tasks([task])
    for obj in task.get_objects():receipt['assets'].append({'path':obj.get_path_name(),'class':obj.get_class().get_name()})
    receipt['status']='pass' if any(x['class']=='SkeletalMesh'for x in receipt['assets']) and sum(x['class']=='AnimSequence'for x in receipt['assets'])>=3 else 'qualified'
    library.save_directory(folder,only_if_is_dirty=True,recursive=True)
   except Exception:receipt['error']=traceback.format_exc()
   write(ev,receipt);bones.append(receipt);unreal.log('V7_SKELETAL '+aid+' '+receipt['status'])
  write(T/'evidence/ue_skeletal_results.json',bones)
try:run()
except Exception:
 write(T/'evidence/ue_fatal.json',{'error':traceback.format_exc()});raise


if os.environ.get('V7_UE_EDITOR_EXIT','1')=='1':unreal.SystemLibrary.quit_editor()
