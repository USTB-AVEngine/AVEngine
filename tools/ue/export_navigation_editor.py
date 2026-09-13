"""Export current UE navigation input geometry, settings and native mesh bounds.

Run inside UE Editor with AVENGINE_UE_NAVIGATION_EXPORT_REQUEST naming a JSON
request containing map_path, output_dir and result_path. Set -UserDir to the
fresh output_dir so Unreal's ExportNavigation command writes its timestamped
OBJ there. This exports the navigation octree geometry and generator settings;
it does not claim to serialize the already-built Detour navigation mesh.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import unreal


def vec(v):
    return [float(v.x),float(v.y),float(v.z)]


def main():
    request_path=Path(os.environ['AVENGINE_UE_NAVIGATION_EXPORT_REQUEST']).resolve()
    request=json.loads(request_path.read_text())
    out=Path(request['output_dir']).resolve()
    result_path=Path(request['result_path']).resolve()
    if result_path.exists():raise FileExistsError(result_path)
    saved=Path(unreal.Paths.project_saved_dir()).resolve()
    saved.relative_to(out)
    before=set(saved.glob('*NavDataSet*.obj'))
    world=unreal.EditorLoadingAndSavingUtils.load_map(request['map_path'])
    if world is None:raise RuntimeError('Requested map did not load')
    world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if not world.get_path_name().startswith(request['map_path']+'.'):
        raise RuntimeError('Editor loaded a different map')
    actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    nav=[]; meshes=[]
    for actor in actors:
        if actor.get_class().get_name()=='RecastNavMesh':
            record={'actor_path':actor.get_path_name(),'properties':{},'unavailable_properties':{}}
            for key in ['agent_radius','agent_height','agent_max_step_height','cell_size','cell_height','tile_size_uu','runtime_generation','max_simplification_error','min_region_area','merge_region_size','nav_data_config']:
                try:record['properties'][key]=str(actor.get_editor_property(key))
                except Exception as exc:record['unavailable_properties'][key]=str(exc)
            nav.append(record)
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=component.get_editor_property('static_mesh')
            if mesh is None:continue
            low,high=component.get_local_bounds()
            transform=component.get_world_transform()
            rotation=transform.rotation
            meshes.append({'actor_path':actor.get_path_name(),'actor_label':actor.get_actor_label(),
                'component_path':component.get_path_name(),'mesh_asset':mesh.get_path_name(),
                'local_min_cm':vec(low),'local_max_cm':vec(high),
                'world_translation_ue_cm':vec(transform.translation),
                'world_rotation_xyzw':[float(rotation.x),float(rotation.y),float(rotation.z),float(rotation.w)],
                'world_scale_xyz':vec(transform.scale3d),
                'collision_enabled':str(component.get_collision_enabled()),
                'collision_profile':str(component.get_collision_profile_name()),
                'source':'Loaded UE StaticMeshComponent local bounds and world transform'})
    if not nav:raise RuntimeError('Loaded map contains no RecastNavMesh actor')
    unreal.SystemLibrary.execute_console_command(world,'ExportNavigation')
    files=sorted(set(saved.glob('*NavDataSet*.obj'))-before)
    result={'status':'exported_inputs' if files else 'fail','source_map':request['map_path'],
        'loaded_world':world.get_path_name(),'actual_project_dir':str(Path(unreal.Paths.project_dir()).resolve()),
        'engine_version':unreal.SystemLibrary.get_engine_version(),'project_saved_dir':str(saved),
        'navigation_actors':nav,'navigation_geometry_exports':[str(p) for p in files],
        'mesh_component_count':len(meshes),'mesh_components':meshes,
        'claim_boundary':'Native navigation octree geometry, generator settings and real component bounds only. Path equivalence and collision checks have not run.'}
    result_path.parent.mkdir(parents=True,exist_ok=True)
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    if not files:raise RuntimeError('ExportNavigation did not create geometry; inspect the native log')
    unreal.log_warning('AVENGINE_UE_NAVIGATION_EXPORT_OK '+str(result_path))


main()
