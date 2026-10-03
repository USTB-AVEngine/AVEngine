"""Audit native asset material parameters, changing only an isolated project."""
import os,json,hashlib,time
from pathlib import Path
import unreal
start=time.monotonic()
r=json.loads(Path(os.environ['AVENGINE_NATIVE_ASSET_AUDIT_REQUEST']).read_text());output=Path(r['output'])
if output.exists():raise FileExistsError(output)
registry=json.loads(Path(r['registry']).read_text());inventory=json.loads(Path(r['inventory']).read_text())
profiles={a['asset_id']:a.get('material_receipt') for a in inventory['assets']}
assets=unreal.get_editor_subsystem(unreal.EditorAssetSubsystem);actor_system=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
result={'schema':'kujiale_isolated_native_asset_material_audit_v1','records':[],'material_repairs':[],'status':'started'}
seen=set()
try:
    for record in registry['assets']:
        binding=record['runtime_backends'].get('spear_unreal')
        if not binding:continue
        row={'asset_id':record['asset_id'],'entity_class':record['entity_class'],'materials':[],'status':'started'}
        actor=None
        try:
            if binding.get('static_mesh_object_path'):
                mesh=unreal.load_asset(binding['static_mesh_object_path']);materials=[slot.material_interface for slot in mesh.static_materials]
            elif binding.get('skeletal_mesh_path'):
                mesh=unreal.load_asset(binding['skeletal_mesh_path']);materials=[slot.material_interface for slot in mesh.materials]
            else:
                cls=unreal.load_class(None,binding['blueprint_class_path'])
                actor=actor_system.spawn_actor_from_class(cls,unreal.Vector(0,0,0),unreal.Rotator(0,0,0),transient=True)
                components=actor.get_components_by_class(unreal.SkeletalMeshComponent)
                materials=[c.get_material(i) for c in components for i in range(c.get_num_materials())]
            for material in materials:
                if material is None:
                    row['materials'].append({'path':None,'status':'missing_native_material'});continue
                names=[str(n) for n in unreal.MaterialEditingLibrary.get_scalar_parameter_names(material)]
                textures=[str(n) for n in unreal.MaterialEditingLibrary.get_texture_parameter_names(material)]
                values={n:float(unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(material,n)) for n in names} if isinstance(material,unreal.MaterialInstanceConstant) else {}
                row['materials'].append({'path':material.get_path_name(),'class':type(material).__name__,'scalar_parameter_names':names,'scalar_values':values,'texture_parameter_names':textures})
                parent=material
                while isinstance(parent,unreal.MaterialInstance):parent=parent.get_editor_property('parent')
                if parent and isinstance(parent,unreal.Material):
                    path=parent.get_path_name();usage=bool(parent.get_editor_property('used_with_skeletal_mesh'))
                    row['materials'][-1].update(master_material=path,used_with_skeletal_mesh=usage)
                    if record['entity_class']!='rigid_object' and not usage and r.get('repair_skeletal_usage'):
                        if not path.startswith('/Game/'):raise ValueError('material usage repair outside own /Game content refused: '+path)
                        if path not in seen:
                            seen.add(path);unreal.MaterialEditingLibrary.set_material_usage(parent,unreal.MaterialUsage.MATUSAGE_SKELETAL_MESH)
                            after=bool(parent.get_editor_property('used_with_skeletal_mesh'));saved=assets.save_loaded_asset(parent,only_if_is_dirty=False)
                            result['material_repairs'].append({'master':path,'change':'used_with_skeletal_mesh','before':usage,'after':after,'saved':bool(saved)})
            row['status']='audited'
        except Exception as e:row.update(status='fail',error=repr(e))
        finally:
            if actor is not None:actor_system.destroy_actor(actor)
        result['records'].append(row)
finally:
    result.update(status='completed',elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2)+'\n')
print('NATIVE_ASSET_AUDIT',json.dumps({'assets':len(result['records']),'usage_repairs':len(result['material_repairs']),'seconds':result['elapsed_seconds']}))
