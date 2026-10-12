"""Persist skeletal usage on task-owned imported dog materials before QA."""
import unreal,json,os,traceback
from pathlib import Path
ROOT=Path(os.environ['V7_KUJIALE_ASSET_ROOT'])
inputs=json.loads(Path(os.environ['V7_KUJIALE_IMPORT_INPUTS']).read_text())
rows=[]
for row in inputs['assets']:
    if not row['walk_allowed']:continue
    result={'asset_id':row['asset_id'],'status':'fail','materials':[]}
    try:
        receipt=json.loads((ROOT/'evidence/import/final'/(row['asset_id']+'.json')).read_text())
        path=next(v['path'] for v in receipt['motion'] if v['class']=='SkeletalMesh')
        mesh=unreal.load_asset(path)
        slots=list(mesh.get_editor_property('materials'))
        for index,slot in enumerate(slots):
            original=slot.material_interface
            base=original if isinstance(original,unreal.Material) else original.get_base_material()
            package='/Game/V7KujialeAssets20261012/final/'+row['asset_id']+'/skeletal_materials/'
            material=unreal.EditorAssetLibrary.duplicate_asset(base.get_path_name(),package+'Base_'+str(index))
            if material is None:raise RuntimeError('cannot derive task-owned skeletal base material')
            before=bool(material.get_editor_property('used_with_skeletal_mesh'))
            unreal.MaterialEditingLibrary.set_material_usage(material,unreal.MaterialUsage.MATUSAGE_SKELETAL_MESH)
            unreal.MaterialEditingLibrary.recompile_material(material)
            okay=unreal.EditorAssetLibrary.save_loaded_asset(material,only_if_is_dirty=False)
            if isinstance(original,unreal.MaterialInstanceConstant):
                child=unreal.EditorAssetLibrary.duplicate_asset(original.get_path_name(),package+'Instance_'+str(index))
                unreal.MaterialEditingLibrary.set_material_instance_parent(child,material)
                unreal.MaterialEditingLibrary.update_material_instance(child)
                okay=unreal.EditorAssetLibrary.save_loaded_asset(child,only_if_is_dirty=False) and okay
            else:child=material
            slot.material_interface=child
            after=bool(material.get_editor_property('used_with_skeletal_mesh'))
            result['materials'].append({'source_interface':original.get_path_name(),'source_readonly_base':base.get_path_name(),'path':material.get_path_name(),'derived_interface':child.get_path_name(),'previous_used_with_skeletal_mesh':before,'used_with_skeletal_mesh':after,'saved':okay,'color_and_roughness_unchanged':True})
        mesh.set_editor_property('materials',slots)
        unreal.EditorAssetLibrary.save_loaded_asset(mesh,only_if_is_dirty=False)
        result['status']='pass' if all(v['used_with_skeletal_mesh'] and v['saved'] for v in result['materials']) else 'fail'
    except Exception:result['error']=traceback.format_exc()
    rows.append(result)
    unreal.log('DOG_MATERIAL_USAGE '+row['asset_id']+' '+result['status'])
(ROOT/'evidence/native_dog_material_usage.json').write_text(json.dumps(rows,indent=2)+'\n')
