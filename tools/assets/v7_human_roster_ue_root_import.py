"""Import derived root-aware Rocketbox rigs into this task's private UE project."""
import os,json,traceback
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT'])
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def run():
 rows=[];only=os.environ.get('V7_UE_ROOT_ONLY')
 for p in sorted((T/'assets/ue_joint_root').glob('*/runtime.glb')):
  aid=p.parent.name
  if only and aid!=only:continue
  proof=T/'evidence/ue_skeletal_root'/(aid+'.json')
  if proof.exists()and json.loads(proof.read_text()).get('status')=='pass':rows.append(json.loads(proof.read_text()));continue
  row={'asset_id':aid,'source':str(p.relative_to(T)),'assets':[],'status':'fail'}
  try:
   task=unreal.AssetImportTask();task.filename=str(p);task.destination_path='/Game/V7SkeletonsRoot/'+aid;task.automated=True;task.save=True;task.async_=True;unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
   for obj in task.get_objects():row['assets'].append({'path':obj.get_path_name(),'class':obj.get_class().get_name()})
   row['status']='pass'if any(x['class']=='SkeletalMesh'for x in row['assets'])and sum(x['class']=='AnimSequence'for x in row['assets'])==3 else'fail'
   unreal.EditorAssetLibrary.save_directory(task.destination_path,only_if_is_dirty=True,recursive=True)
  except Exception:row['error']=traceback.format_exc()
  write(proof,row);rows.append(row);unreal.log('V7_ROOT_RIG '+aid+' '+row['status'])
 write(T/'evidence/ue_skeletal_root_results.json',rows)
run()
# Existing readback helpers are loaded without the scene-capture runner.
scope={'__name__':'v7_root_readback'};source=T/'worktree/tools/assets/v7_human_roster_ue_review.py';s=source.read_text().split('class Review:')[0].replace("T/'evidence/ue_skeletal'","T/'evidence/ue_skeletal_root'").replace("T/'evidence/ue_animation_readback.json'","T/'evidence/ue_animation_root_readback.json'")
exec(compile(s,str(source),'exec'),scope);scope['native_pose_readback']()


if not os.environ.get('V7_UE_ROOT_ONLY'):
 material_source=source.read_text().split('class Review:')[0].replace("T/'evidence/ue_skeletal_results.json'","T/'evidence/ue_skeletal_root_results.json'")
 exec(compile(material_source,str(source),'exec'),scope);scope['native_material_overrides']()
