"""Prepare an audited UE render-surface export for the shared room compiler.

Map, room, measured floor and rig/source positions are explicit inputs. The
legacy Apartment entry point remains available with its historical defaults.
This tool does not render UE pixels or claim a navigation mesh was generated.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from prepare_legacy_apartment import load_json_object, sha256_file, validate_real_surface_inputs, write_json
from avengine.rooms.contracts import validate_room_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene-glb','ue-manifest','mesh-audit','output-dir','source-root','project-dir','map','room-id','floor-reference'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--camera-listener-m', type=float, nargs=3, required=True)
    parser.add_argument('--source-m', type=float, nargs=3, action='append', required=True)
    parser.add_argument('--agent-height-m', type=float, default=1.5)
    parser.add_argument('--agent-radius-m', type=float, default=0.2)
    return parser.parse_args()


def source_snapshot(root, project, map_asset):
    if not map_asset.startswith('/Game/') or '..' in map_asset.split('/'):
        raise ValueError('Map must be inside /Game without parent traversal')
    if not (root/'native/spear').is_dir():
        raise FileNotFoundError('Integrated SPEAR source is missing')
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if git('status','--porcelain','--untracked-files=no','--','native/spear'):
        raise ValueError('Tracked integrated SPEAR source is dirty')
    package=(project/'Content'/(map_asset[6:]+'.umap')).resolve()
    package.relative_to(project)
    return {'schema':'avengine_integrated_spear_source_snapshot_v1',
            'capture_phase':'before_ue_gltf_export', 'repository_root':str(root),
            'commit':git('rev-parse','HEAD'),'tracked_source_scope':'native/spear',
            'tracked_source_scope_dirty':False,'actual_project_dir':str(project),
            'map_asset':map_asset,'map_package_path':str(package),
            'map_package_sha256':sha256_file(package)}


def generate_package(args):
    out=Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Package output must be fresh: {out}')
    scene=Path(args.scene_glb).resolve()
    export_path=Path(args.ue_manifest).resolve()
    audit_path=Path(args.mesh_audit).resolve()
    project=Path(args.project_dir).resolve()
    export=load_json_object(export_path,'UE export')
    audit=load_json_object(audit_path,'Mesh audit')
    snapshot=source_snapshot(Path(args.source_root).resolve(),project,args.map)
    digest=validate_real_surface_inputs(scene,export_path,export,audit_path,audit,snapshot,project_dir=project)
    floor_path=Path(args.floor_reference).resolve()
    floor=load_json_object(floor_path,'Measured floor reference')
    if floor.get('status') not in {'measured','pass','depth_readback_fallback'}:
        raise ValueError('A measured or explicitly labelled depth-fallback floor is required')
    height=float(floor['floor_height_m'])
    room=args.room_id
    base=f'visual/{room}'
    stage=f'visual/stages/{room}.stage_config.json'
    dataset=base+'.scene_dataset_config.json'
    instance=f'visual/scenes/{room}.scene_instance.json'
    lighting=f'visual/lighting/{room}.lighting_config.json'
    navmesh=f'visual/navmeshes/{room}.navmesh'
    assets=[{'role':role,'path':str(path)} for role,path in [
        ('render_surface_mesh',scene),('ue_export_manifest',export_path),
        ('real_surface_mesh_audit',audit_path),('legacy_source_map_package',snapshot['map_package_path']),
        ('floor_reference',floor_path),('scene_dataset_config',dataset),('stage_config',stage),
        ('scene_instance',instance),('lighting_config',lighting),('navmesh',navmesh)]]
    rig=list(args.camera_listener_m)
    pairs=[{'pair_id':f'rig_to_source_{i}','start_m':[rig[0],height,rig[2]],
            'end_m':[point[0],height,point[2]]} for i,point in enumerate(args.source_m)]
    manifest={'schema':'avengine_room_package_v1','room_id':room,
        'room_kind':'legacy_ue_real_surface_export','geometry_representation':'real_surface_mesh',
        'coordinate_system':{'handedness':'right','up_axis':'+Y','forward_axis':'-Z','linear_unit':'meter','quaternion_order':'xyzw'},
        'scene':{'scene_id_kind':'path','scene_id':str(scene),'dataset_config_path':dataset,
                 'navmesh_path':navmesh,'navmesh_policy':'recompute_if_missing','load_semantic_mesh':False,'enable_physics':True},
        'assets':assets,'semantics':{'interpretation':'UE render material slots are interpreted by the shared semantic material rules.'},
        'navigation':{'agent_height_m':args.agent_height_m,'agent_radius_m':args.agent_radius_m,'include_static_objects':False},
        'openings':[],'connectivity_pairs':pairs,'ray_checks':[],
        'acoustics':{'status':'deferred_to_m3','reason':'The shared visual-slot compiler assigns research acoustic materials after this surface audit.'},
        'provenance':{'source':args.map,'source_revision':snapshot['commit'],
            'ue_source_snapshot':snapshot,'floor_reference':str(floor_path),'floor_height_m':height,
            'camera_listener_m':rig,'source_positions_m':args.source_m,
            'navigation_status':'not_run','navigation_note':'The registered QA room keeps its measured navigation input; this acoustic compiler input does not generate a new navmesh.'},
        'surface_audit':{'aabb_proxy':False,'method':'UE StaticMesh render LOD0 and Blender evaluated mesh audit.',
            'triangle_count':audit['triangles'],'real_surface_gate_status':audit['real_surface_gate']['status'],
            'mesh_sha256':digest,'bounds':audit.get('bounds')}}
    errors=validate_room_manifest(manifest)
    if errors:
        raise ValueError('; '.join(errors))
    out.mkdir(parents=True)
    write_json(out/stage,{'render_asset':str(scene),'collision_asset':str(scene),'up':[0,1,0],'front':[0,0,-1],'units_to_meters':1.0})
    write_json(out/dataset,{'stages':{'paths':{'.json':['stages']}},'scene_instances':{'paths':{'.json':['scenes']}}})
    write_json(out/instance,{'stage_instance':{'template_name':room},'object_instances':[]})
    write_json(out/lighting,{'lights':{}})
    write_json(out/'room_manifest.json',manifest)
    return {'status':'pass','room_manifest':str(out/'room_manifest.json'),'navigation_status':'not_run','scene_sha256':digest,'triangle_count':audit['triangles']}


def main():
    print(json.dumps(generate_package(parse_args()),indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
