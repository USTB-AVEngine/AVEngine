#!/usr/bin/env python3
"""Inventory and native before/after acceptance for inserted source materials.

Use inventory first, then run render-one serially on explicitly selected GPUs.
The caller supplies installed runtime, real room manifest, capture request and
output roots. This tool never selects a GPU, imports a Basis source, edits an
original asset, or counts a diagnostic view as a production Episode.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image

from avengine.assets.source_materials import (
    SourceMaterialError, inspect_source_glb, prepare_source_glb,
    configure_source_pbr, measure_source_appearance, prepare_texture_reference_glb, prepare_source_ao_config,
)


def inventory(args):
    assets = {}; sources = {}; usage = {}; registries = []
    # Later explicitly selected registries take precedence over default rows.
    for path in args.registry:
        path = path.resolve(); registries.append(str(path))
        for a in json.loads(path.read_text())['assets']:
            sources.setdefault(a['asset_id'],[]).append(str(path))
            if a.get('runtime_backends',{}).get('habitat'): assets[a['asset_id']] = a
    plans = []
    if args.plan_root:
        for p in sorted(args.plan_root.glob('*/episode/plan/episode_plan.json')):
            x = json.loads(p.read_text()); plans.append(str(p))
            for a in x.get('entity_instances',[]):
                if a.get('asset_id'): usage.setdefault(a['asset_id'],[]).append(str(p))
    library = []
    if args.external_index:
        index = json.loads(args.external_index.read_text())
        for a in index.get('assets',[]):
            geometry = a.get('geometry',{}); raw = geometry.get('finalized_glb')
            p = args.external_index.parent/a.get('path','')/raw if raw else None
            library.append({'asset_id':a['asset_id'],'category':a.get('category'),
                            'raw_glb_path':str(p) if p else None,
                            'registered_habitat':a['asset_id'] in assets})
    records = []
    for asset_id,a in assets.items():
        h = a['runtime_backends']['habitat']; source = Path(h['glb_path']).resolve()
        item = {'asset_id':asset_id,'entity_class':a['entity_class'],'habitat':h,
                'registry_paths':sources[asset_id],'used_plan_count':len(usage.get(asset_id,[]))}
        try:
            manifest_path = h.get('asset_manifest_path')
            if manifest_path:
                manifest_path = Path(manifest_path)
                manifest = json.loads(manifest_path.read_text())
                config_file = next(f for f in manifest['files'] if f['role']=='habitat_ao_config')
                config_path = manifest_path.parent/config_file['path']
                item['runtime_materials'] = {'config_path':str(config_path),'original_ao_config':json.loads(config_path.read_text())}
            else:
                item['runtime_materials'] = {'shader_type':'Habitat rigid-template default; verified native readback in render result'}
            item['source'] = inspect_source_glb(source)
            derived,receipt = prepare_source_glb(source,asset_id=asset_id,entity_class=a['entity_class'],output_directory=args.asset_output)
            item['derived_path'] = str(derived); item['material_receipt'] = receipt
            item['derived'] = inspect_source_glb(derived); item['status'] = 'prepared'
        except (SourceMaterialError,OSError,KeyError,ValueError) as exc:
            item.update(status='blocked',error=str(exc))
        records.append(item)
    result = {'schema':'avengine_source_material_inventory_v1','registries':registries,
              'external_index':str(args.external_index) if args.external_index else None,
              'library':library,'plan_count':len(plans),'used_asset_count':len(usage),
              'missing_used_assets':sorted(set(usage)-set(assets)),
              'asset_count':len(records),'assets':records}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'asset_count':len(records),'used_asset_count':len(usage),'blocked':[r['asset_id'] for r in records if r['status']=='blocked']}))
    return int(any(r['status']=='blocked' for r in records) or bool(result['missing_used_assets']))


def render_one(args):
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
    from avengine.rooms.contracts import load_and_validate_inputs
    from avengine.assets.contracts import load_and_validate_inputs as load_asset
    from avengine.assets.habitat_capture import load_runtime_asset_bundle
    from avengine.capture.mp3d_multi_actor import _base_template_handle,_camera_agent_state,_instantiate_rigid_object
    from avengine.timeline.visual import _instantiate_actor_with_semantic_template
    row = next(a for a in json.loads(args.inventory.read_text())['assets'] if a['asset_id']==args.asset_id)
    if row['status']!='prepared': raise SourceMaterialError(row.get('error','source not prepared'))
    r = prepare_installed_habitat_runtime(runtime_prefix=args.runtime_prefix,magnum_python_site=args.magnum_python_site,
        mp3d_root=args.data_root,rlr_sdk_root=args.rlr_sdk_root)
    inputs = load_and_validate_inputs(args.room_manifest,args.capture_request)
    inputs.request['primary_camera_rig']['shared_calibration']['resolution_hw'] = [480,640]
    # One fixed camera and one fixed floor location per real house. The one-metre
    # display normalization is diagnostic only and identical in all phases.
    pose = args.camera_position; rotation = args.camera_rotation
    inputs.request['primary_camera_rig']['world_from_rig'] = {'translation_m':pose,'rotation_xyzw':rotation}
    inputs.request['primary_camera_rig']['shared_calibration']['hfov_degrees'] = 70.0
    h = row['habitat']; bundle = None
    if row['entity_class'].startswith('articulated'):
        bundle = load_runtime_asset_bundle(load_asset(h['asset_manifest_path'],h['base_m2_request_path'],allow_research_candidate=True))
    import trimesh
    bounds = np.asarray(trimesh.load(row['source']['path'],force='scene',process=False).bounds)
    lo,hi = bounds
    scale = 1.0/max(float((hi-lo).max()),.001)
    placement = {'scale':scale,'translation':(np.asarray(args.object_position)-scale*np.array([(hi[0]+lo[0])/2,lo[1],(hi[2]+lo[2])/2])).tolist(),
                 'original_local_bounds':[lo.tolist(),hi.tolist()], 'bounds_source':'original_glb_scene_rest_geometry'}
    if bundle is not None:
        original_config = Path(bundle.paths_by_role['habitat_ao_config'])
        diagnostic_config = json.loads(original_config.read_text())
        diagnostic_config['uniform_scale'] = placement['scale']
        diagnostic_config['render_asset'] = row['source']['path']
        diagnostic_config['urdf_filepath'] = str(bundle.paths_by_role['habitat_urdf'])
        args.output.mkdir(parents=True,exist_ok=True)
        diagnostic_path = args.output/'diagnostic.ao_config.json'
        diagnostic_path.write_text(json.dumps(diagnostic_config,indent=2))
        bundle = replace(bundle,paths_by_role={**bundle.paths_by_role,'habitat_ao_config':diagnostic_path})
    phases = {}
    reference_path = prepare_texture_reference_glb(row['source']['path'],args.output/'visual.texture_reference_v1.glb')
    # Isolate all phases: native shader/material caches affect flat reference
    # colors, and rigid MATERIAL shaders can use the default PBR configuration.
    # The unlit color-only reference needs no scan or illumination.
    for phase_group in [('reference',),('before',),('after',)]:
        config,uuids,_,resolved = _make_configuration(inputs,None,args.output/'scene_scratch',
            mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path)
        config.sim_cfg.enable_physics = True; config.sim_cfg.gpu_device_id = args.gpu_device_id
        # Guard even an innocently named .glb that contains Basis extensions.
        from avengine.assets.source_materials import read_glb
        read_glb(resolved['scene_id'])
        if phase_group == ('reference',):
            config.sim_cfg.scene_id = 'NONE'; config.sim_cfg.load_semantic_mesh = False
        with r.habitat_sim.Simulator(config) as sim:
            if phase_group == ('after',):
                configure_source_pbr(config,r.habitat_sim,metadata_mediator=sim.metadata_mediator)
            for phase_index,phase in enumerate(phase_group):
                semantic = 2100001
                path = reference_path if phase=='reference' else (row['derived_path'] if phase=='after' else row['source']['path'])
                shader = 'pbr' if phase=='after' else ('flat' if phase=='reference' else None)
                if bundle is not None:
                    phase_config = prepare_source_ao_config(bundle.paths_by_role['habitat_ao_config'],
                        bundle.paths_by_role['habitat_urdf'],path,args.output/phase)
                    handle = _base_template_handle(sim,bundle,cache={},ao_config_path=phase_config)
                    actor,binding = _instantiate_actor_with_semantic_template(sim,bundle=bundle,habitat_sim=r.habitat_sim,
                        semantic_id=semantic,actor_index=phase_index,base_handle=handle,shader_type=shader,render_asset_path=path,
                        light_setup_key='no_lights' if phase=='reference' else None)
                    idle = bundle.action_sets_by_role['idle_poses'].action('idle')
                    actor.joint_positions = np.asarray(binding.map_pose(idle.rotations_xyzw[0]))
                else:
                    actor = _instantiate_rigid_object(sim,binding={'glb_path':str(path)},habitat_sim=r.habitat_sim,
                        semantic_id=semantic,object_index=phase_index,shader_type=shader,uniform_scale=placement['scale'],
                        light_setup_key='no_lights' if phase=='reference' else None)
                node = actor.root_scene_node
                actor.translation = r.magnum.Vector3(placement['translation'])
                sim.initialize_agent(0,_camera_agent_state(r,inputs.request))
                obs = sim.get_sensor_observations(); rgb = np.asarray(obs[uuids['rgb']])[...,:3].copy(); mask = np.asarray(obs[uuids['semantic']])==semantic
                folder = args.output/phase; folder.mkdir(parents=True,exist_ok=True)
                Image.fromarray(rgb).save(folder/'rgb.png');Image.fromarray(mask.astype('uint8')*255).save(folder/'mask.png')
                yy,xx = np.where(mask)
                if len(yy): Image.fromarray(rgb[max(0,yy.min()-8):min(480,yy.max()+9),max(0,xx.min()-8):min(640,xx.max()+9)]).save(folder/'crop.png')
                phases[phase] = {'rgb':rgb,'mask':mask,'shader':str(actor.creation_attributes.shader_type),'pixels':int(mask.sum()),
                                 'native_render_asset_fullpath':actor.creation_attributes.get('__renderAssetFullPath')}
                manager = sim.get_articulated_object_manager() if bundle is not None else sim.get_rigid_object_manager()
                manager.remove_object_by_id(actor.object_id)
    reference = phases['reference']; metrics = {}
    threshold = max(p['max_highlight_ratio'] for p in row['material_receipt']['profiles'])
    for phase in ['before','after']:
        # Erode one pixel to exclude anti-alias/semantic boundary differences.
        mask = phases[phase]['mask'] & reference['mask']
        mask = mask & np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1)
        metrics[phase] = measure_source_appearance(phases[phase]['rgb'],mask,
            palette=np.asarray(row['material_receipt']['source_palette_srgb']),max_highlight_ratio=threshold,
            metallic_surface=any(p['metallic']>0 for p in row['material_receipt']['profiles']),
            reference_rgb=reference['rgb'],strict=False)
        metrics[phase]['runtime_palette_check'] = measure_source_appearance(phases[phase]['rgb'],mask,
            palette=np.asarray(row['material_receipt']['source_palette_srgb']),max_highlight_ratio=threshold,strict=False)
    foreground = phases['before']['mask']|phases['after']['mask']
    dilated = foreground.copy()
    for dy in range(-2,3):
        for dx in range(-2,3):dilated |= np.roll(np.roll(foreground,dy,0),dx,1)
    background = ~dilated
    result = {'asset_id':args.asset_id,'house':args.house,'placement':placement,'camera_position':pose,'camera_rotation':rotation,
              'object_position':args.object_position,'reference_kind':'unlit_original_texture','reference_scene':'NONE',
              'runtime_prefix':str(args.runtime_prefix),'room_manifest':str(args.room_manifest),'source':row['source']['path'],
              'derived':row['derived_path'],'metrics':metrics,'phase_readbacks':{k:{'shader':v['shader'],'visible_pixels':v['pixels'],
                    'native_render_asset_fullpath':v['native_render_asset_fullpath']} for k,v in phases.items()},
              'scanned_background_changed_pixels':int(np.any(phases['before']['rgb']!=phases['after']['rgb'],axis=-1)[background].sum()),
              'status':'pass' if metrics['after']['status']=='pass' and metrics['after']['runtime_palette_check']['status']=='pass' else 'fail',
              'diagnostic_size_normalization':True,'production_episode_counted':False}
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return int(result['status']!='pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('inventory');p.add_argument('--registry',type=Path,action='append',required=True)
    p.add_argument('--external-index',type=Path);p.add_argument('--plan-root',type=Path)
    p.add_argument('--asset-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('render-one');p.add_argument('--inventory',type=Path,required=True);p.add_argument('--asset-id',required=True)
    for flag in ['room-manifest','capture-request','runtime-prefix','magnum-python-site','data-root','rlr-sdk-root','output']:
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--house',required=True);p.add_argument('--gpu-device-id',type=int,required=True)
    p.add_argument('--camera-position',type=float,nargs=3,required=True);p.add_argument('--camera-rotation',type=float,nargs=4,required=True)
    p.add_argument('--object-position',type=float,nargs=3,required=True)
    args = parser.parse_args()
    return inventory(args) if args.command=='inventory' else render_one(args)


if __name__=='__main__': raise SystemExit(main())
