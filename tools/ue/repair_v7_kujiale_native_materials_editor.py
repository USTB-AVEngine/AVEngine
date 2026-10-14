"""Apply reviewed GLB PBR changes only to MICs in the task's isolated project."""
from __future__ import annotations
import hashlib,json,os,shutil,time
from pathlib import Path
import unreal


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    r=json.loads(Path(os.environ['AVENGINE_NATIVE_MATERIAL_REPAIR_REQUEST']).read_text())
    request_path=Path(r['request']);request=json.loads(request_path.read_text())
    project=Path(unreal.Paths.project_dir()).resolve();expected=Path(r['project_directory']).resolve()
    if project!=expected or expected.parent.name!='runtime' or expected.name!='SpearSim':
        raise ValueError('material writes outside the isolated project refused')
    output=Path(r['output']);backup=Path(r['backup_directory'])
    if output.exists() or backup.exists():raise FileExistsError('repair evidence must be new')
    backup.mkdir(parents=True)
    root=expected/'Content';destination=r['destination_path']
    if not destination.startswith('/Game/AVEngine/V7SourceMaterials'):raise ValueError('texture destination is not task-owned')
    assets=unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
    tools=unreal.AssetToolsHelpers.get_asset_tools()
    lib=unreal.MaterialEditingLibrary
    result={'schema':'kujiale_isolated_native_material_repair_v1','status':'started','request':str(request_path),
        'request_sha256':sha(request_path),'project_directory':str(project),'repairs':[],
        'unresolved':list(request['unresolved']),'basecolor_uv_normals_geometry_modified':False}
    start=time.monotonic();seen={};textures={}
    try:
        for repair in request['repairs']:
            path=repair['native_material']
            package=path.split('.')[0]
            if not package.startswith('/Game/'):raise ValueError('shared material write refused')
            file=root/(package.removeprefix('/Game/')+'.uasset')
            if file.is_symlink() or root not in file.resolve().parents:raise ValueError('material path escapes isolated project')
            signature=(repair['metallic_factor'],repair['roughness_factor'],repair['packed_mr_sha256'])
            if path in seen:
                if seen[path]!=signature:raise ValueError('conflicting PBR recipes for shared source material')
                continue
            seen[path]=signature
            material=unreal.load_asset(path)
            if not isinstance(material,unreal.MaterialInstanceConstant):raise ValueError('target is not a MIC')
            saved=file.read_bytes();before_sha=hashlib.sha256(saved).hexdigest()
            target=backup/file.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(saved)
            names=[str(n) for n in lib.get_texture_parameter_names(material)]
            unchanged={n:(lib.get_material_instance_texture_parameter_value(material,n).get_path_name() if lib.get_material_instance_texture_parameter_value(material,n) else None) for n in names if n!='MetallicRoughnessTexture'}
            before={n:float(lib.get_material_instance_scalar_parameter_value(material,n)) for n in ('MetallicFactor','RoughnessFactor')}
            old_mr=lib.get_material_instance_texture_parameter_value(material,'MetallicRoughnessTexture')
            row=dict(repair,before_sha256=before_sha,backup_path=str(target),before=before,
                     before_mr_texture=old_mr.get_path_name() if old_mr else None)
            for name,value in (('MetallicFactor',repair['metallic_factor']),('RoughnessFactor',repair['roughness_factor'])):
                # UE 5.5 MaterialEditingLibrary.cpp always returns false from
                # this setter. Actual parameter readback below is authoritative.
                lib.set_material_instance_scalar_parameter_value(material,name,value)
            if repair['packed_mr_png']:
                image=Path(repair['packed_mr_png'])
                if sha(image)!=repair['packed_mr_sha256']:raise ValueError('derived MR input hash changed')
                compression=old_mr.get_editor_property('compression_settings') if old_mr else unreal.TextureCompressionSettings.TC_DEFAULT
                key=(repair['packed_mr_sha256'],str(compression))
                if key not in textures:
                    token=hashlib.sha256((key[0]+key[1]).encode()).hexdigest()[:24]
                    task=unreal.AssetImportTask();task.set_editor_property('filename',str(image));task.set_editor_property('destination_path',destination)
                    task.set_editor_property('destination_name','MR_'+token);task.set_editor_property('automated',True)
                    task.set_editor_property('replace_existing',False);task.set_editor_property('save',False)
                    tools.import_asset_tasks([task]);imported=task.get_editor_property('imported_object_paths')
                    if len(imported)!=1:raise ValueError('MR import did not create exactly one texture')
                    texture=unreal.load_asset(imported[0]);texture.set_editor_property('srgb',False)
                    texture.set_editor_property('compression_settings',compression)
                    if not assets.save_loaded_asset(texture,only_if_is_dirty=False):raise ValueError('MR texture save failed')
                    textures[key]=texture
                texture=textures[key]
                lib.set_material_instance_texture_parameter_value(material,'MetallicRoughnessTexture',texture)
                readback_texture=lib.get_material_instance_texture_parameter_value(material,'MetallicRoughnessTexture')
                if not readback_texture or readback_texture.get_path_name()!=texture.get_path_name():raise ValueError('native MR texture readback failed')
                switches=[str(n) for n in lib.get_static_switch_parameter_names(material)]
                switch='bHasMetallicRoughnessTexture'
                if switch in switches:
                    row['mr_static_switch_before']=bool(lib.get_material_instance_static_switch_parameter_value(material,switch))
                    lib.set_material_instance_static_switch_parameter_value(material,switch,True)
                    row['mr_static_switch_after']=bool(lib.get_material_instance_static_switch_parameter_value(material,switch))
                    if not row['mr_static_switch_after']:raise ValueError('native MR presence switch readback failed')
                row['mr_static_switch_names']=switches
                row['after_mr_texture']=texture.get_path_name();row['mr_srgb']=bool(texture.get_editor_property('srgb'));row['mr_compression']=str(compression)
            lib.update_material_instance(material)
            after={n:float(lib.get_material_instance_scalar_parameter_value(material,n)) for n in ('MetallicFactor','RoughnessFactor')}
            for name,value in (('MetallicFactor',repair['metallic_factor']),('RoughnessFactor',repair['roughness_factor'])):
                if abs(after[name]-value)>1e-6:raise ValueError('native PBR readback disagrees')
            for n,old in unchanged.items():
                value=lib.get_material_instance_texture_parameter_value(material,n)
                if (value.get_path_name() if value else None)!=old:raise ValueError('basecolor/normal/other texture unexpectedly changed')
            if not assets.save_loaded_asset(material,only_if_is_dirty=False):raise ValueError('native MIC save failed')
            row.update(after=after,after_sha256=sha(file),status='pass');result['repairs'].append(row)
            output.write_text(json.dumps(result,indent=2)+'\n')
        result.update(status='pass' if not result['unresolved'] else 'unresolved',unique_material_count=len(result['repairs']),
                      imported_mr_texture_count=len(textures))
    except Exception as e:
        result.update(status='fail',error=f'{type(e).__name__}: {e}');raise
    finally:
        result['elapsed_seconds']=time.monotonic()-start;output.write_text(json.dumps(result,indent=2)+'\n')
    print('NATIVE_MATERIAL_REPAIR',json.dumps({k:result[k] for k in ('status','unique_material_count','imported_mr_texture_count','elapsed_seconds')}))


main()
