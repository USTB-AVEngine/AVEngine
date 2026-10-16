"""Three representative GLB imports into an owned UE project; CPU NullRHI smoke only."""
import os,json,traceback
from pathlib import Path
import unreal
T=Path(os.environ['V7_ANIMAL_TASK_ROOT']);selected=json.loads((T/'evidence/ue_smoke_selection.json').read_text());records=[]
try:
    for item in selected:
        task=unreal.AssetImportTask();task.filename=item['runtime_glb'];task.destination_path='/Game/V7AnimalVariants20261012/'+item['asset_id'];task.automated=True;task.save=True;task.replace_existing=False;task.async_=False;unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task]);objects=task.get_objects();assets=[{'path':o.get_path_name(),'class':o.get_class().get_name(),'name':o.get_name()} for o in objects];meshes=[o for o in objects if isinstance(o,unreal.StaticMesh)];bounds=[]
        for mesh in meshes:
            b=mesh.get_bounding_box();bounds.append({'asset':mesh.get_path_name(),'min_cm':[b.min.x,b.min.y,b.min.z],'max_cm':[b.max.x,b.max.y,b.max.z]})
        rec={'asset_id':item['asset_id'],'category':item['category'],'source_glb':task.filename,'assets':assets,'static_mesh_count':len(meshes),'bounds':bounds,'status':'pass_import_smoke' if meshes else 'fail_no_mesh','renderer':'NullRHI CPU-only; no visual rendering or animation admission claimed'};records.append(rec);(T/'evidence/ue_import_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    (T/'evidence/ue_smoke_complete.json').write_text(json.dumps({'status':'pass' if len(records)==3 and all(x['status']=='pass_import_smoke' for x in records) else 'fail','import_count':len(records),'records':records},ensure_ascii=False,indent=2)+'\n')
except BaseException:
    (T/'evidence/ue_smoke_fatal.json').write_text(json.dumps({'error':traceback.format_exc(),'completed_imports':records},indent=2)+'\n')
    raise
finally:unreal.SystemLibrary.quit_editor()
