"""Renderer boundaries for neutral QA plans and retained room resources."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

import numpy as np

from avengine.qa.answerability import MeshHandle
from avengine.rooms.walkable_space import RasterWalkableSpace, NativeRouteWalkableSpace, HabitatWalkableSpace


def _read(path):
    return json.loads(Path(path).expanduser().read_text())


def _resolved(raw, base=None, runtime=None):
    text=str(raw)
    if runtime and runtime.get('mp3d_root'):
        text=text.replace('${AVENGINE_MP3D_ROOT}',str(runtime['mp3d_root']))
    path=Path(os.path.expandvars(text)).expanduser()
    if not path.is_absolute() and base is not None:path=Path(base)/path
    return path.resolve()


def _package_mesh(package):
    geometry=package.get('static_geometry') or {}
    vertices=geometry.get('vertices',geometry.get('vertices_path'))
    triangles=geometry.get('triangles',geometry.get('triangles_path'))
    if vertices and triangles:
        mesh=MeshHandle.from_paths(vertices,triangles)
        coordinate=geometry.get('coordinate_frame',{'linear_unit':'meter','up_axis':'+Y','handedness':'right'})
        if coordinate.get('linear_unit')!='meter' or coordinate.get('up_axis')!='+Y' or coordinate.get('handedness')!='right':
            raise ValueError('static geometry must declare the shared meter/+Y frame')
        return mesh
    return None


def load_ue_walkable_grid(path, *, floor_height_m, clearance_m=.38):
    """Convert the retained UE raster once at the renderer boundary."""
    from avengine.routes.raster_pathfinder import RasterPathfinder
    path=Path(path);path=path/'walkable_grid.json' if path.is_dir() else path
    metadata=_read(path)
    if metadata.get('schema')!='qa_v3_walkable_grid_v1' or metadata.get('axes')!={'rows':'ue_y_cm','cols':'ue_x_cm'}:
        raise ValueError('unsupported retained walkable-grid axes/schema')
    with np.load(path.parent/metadata['arrays']['path'],allow_pickle=False) as data:
        binary=np.asarray(data['walkable'],dtype=bool)
        if 'clearance_cm' in data:binary=binary&(data['clearance_cm']/100.>=clearance_m)
        elif clearance_m>0:raise ValueError('retained grid has no clearance values')
    step=float(metadata['cell_cm'])/100.;origin=np.asarray(metadata['origin_xy_cm'],dtype=float)/100.
    bounds=np.array([[origin[0],floor_height_m-1.,origin[1]],
                     [origin[0]+binary.shape[1]*step,floor_height_m+1.,origin[1]+binary.shape[0]*step]])
    pf=RasterPathfinder(binary,bounds_m=bounds,floor_height_m=floor_height_m)
    nav={'authority':'retained_walkable_grid_with_raster_astar','floor_height_m':float(floor_height_m),
         'resolution_m':step,'bounds_habitat_m':bounds.tolist(),'source_manifest':str(path.resolve()),
         'free_area_m2':int(binary.sum())*step*step,'clearance_m':clearance_m}
    return RasterWalkableSpace(pf,nav)


def _floor_value(package, room):
    if room.get('floor_height_m') is not None:return float(room['floor_height_m'])
    ref=package.get('floor_reference')
    path=ref.get('path') if isinstance(ref,dict) else ref
    if not path:raise ValueError('measured floor_reference is required for this adapter')
    value=_read(path)
    for key in ('floor_height_m','floor_y_m','measured_floor_height_m'):
        if value.get(key) is not None:return float(value[key])
    # Existing UE floor artifacts retain their explicit engine units.
    for key in ('floor_z_cm','ground_z_cm','measured_floor_z_cm'):
        if value.get(key) is not None:return float(value[key])/100.
    raise ValueError('floor_reference has no supported measured height field')


def load_planning_resources(room, request):
    """Load once per request; return an existing solver plus shared static mesh."""
    from avengine.rooms import native_qa_room as nq
    from avengine.rooms.qa_episode import build_room_navigation
    from avengine.rooms.furniture_layout import load_room_layout
    from avengine.rooms.furnished_episode import _load_static_triangle_geometry
    package=room.get('room_package',{});runtime=request.get('runtime',{})
    kind=package.get('walkable_space',{}).get('kind')
    if room.get('native_room_adapter')==nq.SCHEMA or kind=='route_bank':
        resources=nq.discover_native_apartment_resources(repository=Path.cwd(),source_root=room['native_input_root'],
                    route_bank=room['route_bank'],room_profile_path=room.get('native_room_profile'))
        layout=nq.build_native_apartment_layout(resources)
        pf,nav=build_room_navigation(layout,floor_height_m=float(layout['native_floor_height_m']))
        bank=_read(resources.route_bank);seconds=float(bank['clip_seconds']);count=int(bank['frame_count']);rate=bank.get('frame_rate_hz',bank.get('frame_rate',count/seconds))
        routes=[]
        for raw in bank['routes']:
            try:points=nq._route_points(raw)
            except nq.NativeQAResourceError:continue
            if len(points)!=count:continue
            length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
            if length>=2. and .6<=length/seconds<=1.5:routes.append({'route_id':raw['route_id'],'points_m':points})
        if not routes:raise ValueError('native route bank has no legal retained paths')
        nav.update(route_authority='native_spear_ue_recast_route_bank',native_route_bank=str(resources.route_bank),native_route_count=len(routes))
        space=NativeRouteWalkableSpace(pf,nav,routes,float(rate))
    elif kind=='walkable_grid':
        space=load_ue_walkable_grid(package['walkable_space']['path'],floor_height_m=_floor_value(package,room),
                                    clearance_m=float(request.get('body_clearance_m',.38)))
        layout={'room_id':room['room_id'],'scene_id':room.get('scene_id',room['room_id']),
                'backend_route':'spear_unreal','visual_lighting':{},'manifest_path':package['walkable_space']['path']}
        mesh=_package_mesh(package)
        if mesh is None:raise ValueError('walkable-grid room needs shared static triangles')
        return space,mesh,layout
    elif room.get('backend')=='habitat' or package.get('renderer')=='habitat':
        from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
        rt=prepare_installed_habitat_runtime(**{k:runtime[k] for k in ('runtime_prefix','mp3d_root','magnum_python_site','rlr_sdk_root') if runtime.get(k)})
        manifest_path=room.get('room_manifest');manifest=_read(manifest_path) if manifest_path else {}
        navpath=package.get('walkable_space',{}).get('path') or manifest.get('scene',{}).get('navmesh_path')
        navpath=_resolved(navpath,Path(manifest_path).parent if manifest_path else None,runtime)
        pf=rt.habitat_sim.PathFinder()
        if not pf.load_nav_mesh(str(navpath)):raise ValueError('native Habitat navmesh did not load')
        # floor_reference comes from P3 native measurement, never a guessed ground height.
        floor=_floor_value(package,room);bounds=np.asarray(pf.get_bounds())
        nav={'authority':'habitat_native_pathfinder','floor_height_m':floor,'resolution_m':.08,
             'bounds_habitat_m':bounds.tolist(),'source_manifest':str(navpath),'runtime_prefix':str(rt.prefix)}
        space=HabitatWalkableSpace(pf,nav);mesh=_package_mesh(package)
        if mesh is None:raise ValueError('Habitat room needs declared shared static triangles')
        layout={'room_id':room['room_id'],'scene_id':manifest.get('room_id',room['room_id']),
                'manifest_path':manifest_path,'backend_route':'habitat','visual_lighting':{},
                'capture_resolution_hw':_read(room['m1_request'])['primary_camera_rig']['shared_calibration']['resolution_hw'] if room.get('m1_request') else [240,320]}
        return space,mesh,layout
    else:
        layout=load_room_layout(room['manifest'],asset_root=room.get('asset_root'),require_seats=False)
        pf,nav=build_room_navigation(layout,clearance_m=float(request.get('body_clearance_m',.38)),floor_height_m=_floor_value(package,{}) if package.get('floor_reference') else room.get('floor_height_m'))
        space=RasterWalkableSpace(pf,nav)
    mesh=_package_mesh(package)
    if mesh is None:
        raw=_load_static_triangle_geometry(layout)
        if raw is None:raise ValueError('room has no retained static mesh for LOS')
        vertices=np.asarray(raw['vertices'],dtype=float)
        # Retained furniture/native room mesh is in authoring Z-up meters.
        vertices=np.column_stack((vertices[:,0],vertices[:,2],-vertices[:,1]))
        mesh=MeshHandle(vertices,raw['triangles'],{'path':raw['source'],'transform':'authoring_xyz_m_to_xz_negative_y_m'})
    return space,mesh,layout


def materialize_ue_episode_plan(plan, registry):
    """Add UE driving fields in the UE executor, without changing the neutral plan."""
    if plan.get('plan_coordinates')!='renderer_neutral':return deepcopy(plan)
    from avengine.rooms.qa_episode import source_declaration
    from avengine.rooms.furniture_layout import habitat_to_ue_cm
    from avengine.runtime_profiles import resolve_source_asset_runtime_profile
    result=deepcopy(plan);result['execution_coordinates']='ue_spear';result['renderer_backend']='spear_unreal_native'
    package=result['resources'].get('room_package',{})
    result['scene']['map_path']=package.get('visual_scene',{}).get('map_path',result['resources'].get('map_path'))
    result['scene']['backend']='spear_unreal'
    actors=[]
    for neutral in plan['visual_plan']['actors']:
        if neutral['entity_class']=='rigid_object':
            record=resolve_source_asset_runtime_profile(registry,neutral['asset_id']);backend=record.get('runtime_backends',{}).get('spear_unreal')
            if not backend:raise ValueError(neutral['asset_id']+' has no UE renderer binding')
            actor={**deepcopy(neutral),**deepcopy(backend)}
            offset=neutral['emitter_binding']['emitter_offset_m'];actor['emitter_local_ue_cm']=[100*offset[0],100*offset[2],100*offset[1]]
        else:
            actor=source_declaration(registry,neutral['asset_id'],neutral['actor_id']);actor.update(entity_class=neutral['entity_class'])
        actors.append(actor)
    result['visual_plan']['actors']=actors;by_id={a['actor_id']:a for a in actors}
    def camera(native):
        point=native['position_m'];forward=native['basis']['forward'];fue=[forward[0],forward[2],forward[1]]
        yaw=math.degrees(math.atan2(fue[1],fue[0]));pitch=math.degrees(math.atan2(fue[2],math.hypot(fue[0],fue[1])))
        return {**deepcopy(native),'position_habitat_m':deepcopy(point),'position_authoring_m':[point[0],-point[2],point[1]],
                'position_ue_cm':habitat_to_ue_cm(point),'ue_position_cm':habitat_to_ue_cm(point),
                'ue_yaw_deg':yaw,'ue_pitch_deg':pitch,'ue_roll_deg':0.,'yaw_deg':-yaw,'pitch_deg':pitch,
                'forward_ue':fue,'forward_blender':[forward[0],-forward[2],forward[1]],'roll_deg':0.,
                **({'exposure_bias_ev':result['resources']['exposure_bias_ev']} if result['resources'].get('exposure_bias_ev') is not None else {})}
    result['visual_plan']['camera']=camera(plan['visual_plan']['camera'])
    for frame in result['visual_plan']['frames']:
        frame['camera_state']=camera(frame['camera_state'])
        frame['camera_state']['frame_index']=int(frame['frame_index'])
        for state in frame['actor_states']:
            actor=by_id[state['actor_id']];transform=state['root_transform'];q=transform['rotation_xyzw'];yaw=-math.degrees(2*math.atan2(q[1],q[3]))
            state.update(translation_m=deepcopy(transform['translation_m']),translation_ue_cm=habitat_to_ue_cm(transform['translation_m']),
                         rotation_xyzw=deepcopy(q),actor_yaw_ue_deg=yaw)
            if actor.get('animation_paths_by_action_id'):state['ue_animation']=actor['animation_paths_by_action_id'][state['action_id']]
    result.setdefault('visual_lighting',{})
    if result['resources'].get('review_lights'):
        result['review_lights']=deepcopy(result['resources']['review_lights'])
    return result


def materialize_habitat_room_manifest(package, source_manifest, output):
    """Make the package's actual scene/dataset config authoritative for capture."""
    manifest=deepcopy(_read(source_manifest));visual=package['visual_scene'];scene=manifest['scene']
    scene.update(scene_id=str(_resolved(visual['scene_glb'])),
                 dataset_config_path=str(_resolved(visual['dataset_config'])),
                 navmesh_path=str(_resolved(visual['navmesh'])))
    semantics=package.get('semantics',{})
    roles={'render_surface_mesh':scene['scene_id'],'scene_dataset_config':scene['dataset_config_path'],
           'navmesh':scene['navmesh_path'],'semantic_surface_mesh':semantics.get('source'),
           'semantic_descriptor':semantics.get('descriptor')}
    for asset in manifest.get('assets',[]):
        if roles.get(asset.get('role')):asset['path']=str(_resolved(roles[asset['role']]))
    manifest.setdefault('provenance',{})['source_room_manifest']=str(_resolved(source_manifest))
    manifest['provenance']['adaptation']='room_package_visual_scene_and_semantics'
    with Path(output).open('x') as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2)
    return manifest
