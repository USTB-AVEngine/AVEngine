"""Import new human action masters, silent chairs and static dog candidates privately."""
import os,json,traceback,sys,hashlib
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT']);sys.path.insert(0,str(T/'worktree/tools/assets'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def run():
 rows=[]
 entries=[('human',p.parent.name,p,15)for p in sorted((T/'round2/assets/humans').glob('*/final_ue_root.glb'))]+[('chair',p.stem,p,0)for p in sorted((T/'round3/assets/chairs').glob('*.glb'))]+[('dog',p.parent.name,p,0)for p in sorted((T/'round3/assets/dogs').glob('*/idle.glb'))]+[('dog_motion',p.parent.name,p,8)for p in sorted((T/'round2/assets/dogs').glob('*/motion_runtime.glb'))]+[('bird','v7_rocketbox_bird_sparrow_fallback_01',T/'round3/assets/bird_sparrow_fallback/runtime.glb',0)]
 for kind,aid,path,expected in entries:
  sha=hashlib.sha256(path.read_bytes()).hexdigest();ev=T/'round3/evidence/ue_import'/kind/(aid+'.json')
  if ev.exists():
   cached=json.loads(ev.read_text())
   if cached.get('status')=='pass'and cached.get('source_sha256')==sha:rows.append(cached);continue
  folder='/Game/V7RosterMotionV3/'+aid if kind=='human'else('/Game/V7RosterChairs/'+aid if kind=='chair'else('/Game/V7RosterAnimalMotion/'+aid if kind=='dog_motion'else('/Game/V7RosterBirdsV3/'+aid if kind=='bird'else'/Game/V7RosterAnimals/'+aid)))
  rec={'source_sha256':sha,'asset_id':aid,'kind':kind,'source_glb':str(path),'destination':folder,'assets':[],'status':'fail','source_animation_count':expected}
  try:
   task=unreal.AssetImportTask();task.filename=str(path);task.destination_path=folder;task.automated=True;task.save=True;task.async_=True;task.replace_existing=True;unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
   for obj in task.get_objects():rec['assets'].append({'path':obj.get_path_name(),'name':obj.get_name(),'class':obj.get_class().get_name()})
   rec['imported_animation_count']=sum(x['class']=='AnimSequence'for x in rec['assets'])
   passed=(any(x['class']=='SkeletalMesh'for x in rec['assets'])and rec['imported_animation_count']>=expected)if kind in ['human','dog_motion']else any(x['class']=='StaticMesh'for x in rec['assets']);rec['status']='pass'if passed else'fail'
   unreal.EditorAssetLibrary.save_directory(folder,only_if_is_dirty=True,recursive=True)
  except Exception:rec['error']=traceback.format_exc()
  write(ev,rec);rows.append(rec);unreal.log('V7_ROUND3_IMPORT '+kind+' '+aid+' '+rec['status'])
 write(T/'round3/evidence/ue_import_results.json',rows)
try:run()
except Exception:write(T/'round3/evidence/ue_import_fatal.json',{'error':traceback.format_exc()});raise
p=T/'worktree/tools/assets/v7_roster_ue_motion_readback.py'
exec(compile(p.read_text(),str(p),'exec'),{'__name__':'v7_motion_import_readback'})
p=T/'worktree/tools/assets/v7_roster_ue_dog_readback.py'
exec(compile(p.read_text(),str(p),'exec'),{'__name__':'v7_dog_readback'})
p=T/'worktree/tools/assets/v7_roster_ue_extra_readback.py'
exec(compile(p.read_text(),str(p),'exec'),{'__name__':'v7_extra_readback'})
unreal.SystemLibrary.quit_editor()
