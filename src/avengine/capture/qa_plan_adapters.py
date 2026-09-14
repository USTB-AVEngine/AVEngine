"""Renderer boundaries for neutral QA plans and retained room resources."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from pathlib import Path
from string import Template
from typing import Mapping

import numpy as np

from avengine.qa.answerability import MeshHandle
from avengine.rooms.room_package import (
    REPOSITORY_ROOT,
    canonical_runtime,
    configured_path_bindings,
    load_host_runtime_config,
    resolve_room_package_paths,
)
from avengine.rooms.walkable_space import RasterWalkableSpace, NativeRouteWalkableSpace, HabitatWalkableSpace


def _read(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _resolved(raw, base=None, runtime=None):
    text = Template(str(raw)).safe_substitute(configured_path_bindings(runtime))
    missing = re.findall(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", text)
    if missing:
        raise ValueError(
            "QA plan path is missing configured roots: "
            + ", ".join(sorted(set(missing)))
        )
    path = Path(text).expanduser()
    if not path.is_absolute():
        root = Path(base).expanduser() if base is not None else REPOSITORY_ROOT
        if not root.is_absolute():
            root = REPOSITORY_ROOT / root
        path = root / path
    return path.resolve()


def _package_mesh(package, *, runtime=None, base=None):
    geometry = package.get("static_geometry") or {}
    vertices = geometry.get("vertices", geometry.get("vertices_path"))
    triangles = geometry.get("triangles", geometry.get("triangles_path"))
    if vertices and triangles:
        if isinstance(vertices, (str, Path)):
            vertices = _resolved(vertices, base=base, runtime=runtime)
        if isinstance(triangles, (str, Path)):
            triangles = _resolved(triangles, base=base, runtime=runtime)
        mesh = MeshHandle.from_paths(vertices, triangles)
        coordinate = geometry.get(
            "coordinate_frame",
            {"linear_unit": "meter", "up_axis": "+Y", "handedness": "right"},
        )
        if (
            coordinate.get("linear_unit") != "meter"
            or coordinate.get("up_axis") != "+Y"
            or coordinate.get("handedness") != "right"
        ):
            raise ValueError("static geometry must declare the shared meter/+Y frame")
        return mesh
    return None


def _declared_planning_floors(package):
    """Carry a navigation package's interior floors into the planning space.

    ``prepare_render_surface_navigation`` writes ``planning_floors_m`` when it
    was given the room's static triangles, and the sampler reads that key off
    the space's metadata. A package built before this existed simply says
    nothing, and the sampler measures the levels itself.
    """
    declared = (package.get("walkable_space") or {}).get("planning_floors_m")
    if not isinstance(declared, (list, tuple)) or not len(declared):
        return {}
    values = [float(value) for value in declared]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("declared planning floors must be finite heights in meters")
    return {"planning_floors_m": values}


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


def _floor_value(package, room, *, runtime=None, base=None):
    if room.get("floor_height_m") is not None:
        return float(room["floor_height_m"])
    ref = package.get("floor_reference")
    path = ref.get("path") if isinstance(ref, dict) else ref
    if not path:
        raise ValueError("measured floor_reference is required for this adapter")
    path = _resolved(path, base=base, runtime=runtime)
    value = _read(path)
    for key in ("floor_height_m", "floor_y_m", "measured_floor_height_m"):
        if value.get(key) is not None:
            return float(value[key])
    # Existing UE floor artifacts retain their explicit engine units.
    for key in ("floor_z_cm", "ground_z_cm", "measured_floor_z_cm"):
        if value.get(key) is not None:
            return float(value[key]) / 100.
    raise ValueError("floor_reference has no supported measured height field")


def _expanded_room(room, request):
    """Accept either a raw catalog row or an already-expanded room mapping.

    A catalog row names its RoomPackage by path, so a caller that iterates a
    catalog used to reach the adapters with ``room_package`` still a string.
    Expanding it here gives every caller the route the unified entry gives,
    and a row that carries no package at all is returned untouched.
    """
    from avengine.rooms.room_package import package_from_catalog_entry
    from avengine.rooms.room_providers import catalog_runtime, load_room_catalog, planning_room_mapping

    package = room.get("room_package")
    if package is None or isinstance(package, Mapping):
        return room
    if not isinstance(package, (str, Path)):
        raise ValueError(
            "room_package must be a RoomPackage mapping or a path to one, got "
            f"{type(package).__name__}"
        )
    runtime = canonical_runtime(
        request.get("runtime") if isinstance(request.get("runtime"), dict) else {}
    )
    catalog_path = request.get("room_catalog")
    if catalog_path is not None:
        catalog_path = _resolved(catalog_path, runtime=runtime)
        runtime = catalog_runtime(load_room_catalog(catalog_path), runtime)
    expanded = package_from_catalog_entry(
        room, runtime=runtime, catalog_path=catalog_path
    )
    normalized = planning_room_mapping(expanded)
    for key, value in room.items():
        if key != "room_package" and key not in normalized:
            normalized[key] = value
    return normalized


def _load_native_apartment_recast_navmesh(
    package, room, planning_inputs, request, runtime, resource_base, habitat_runtime
):
    """Load Apartment Recast navigation while retaining its UE visual route."""
    from avengine.rooms import native_qa_room as nq

    source_root = room.get("native_input_root") or planning_inputs.get("native_input_root")
    route_bank = room.get("route_bank") or planning_inputs.get("route_bank")
    profile_path = room.get("native_room_profile") or planning_inputs.get("native_room_profile")
    if not source_root:
        raise ValueError("native Apartment free navigation needs native_input_root")
    if not route_bank:
        raise ValueError("native Apartment route_bank declaration is required as optional fallback")
    resources = nq.discover_native_apartment_resources(
        repository=REPOSITORY_ROOT,
        source_root=_resolved(source_root, base=resource_base, runtime=runtime),
        route_bank=_resolved(route_bank, base=resource_base, runtime=runtime),
        room_profile_path=(
            _resolved(profile_path, base=resource_base, runtime=runtime)
            if profile_path else None
        ),
    )
    package_navpath = (package.get("walkable_space") or {}).get("path")
    if not package_navpath:
        raise ValueError("native Apartment free navigation needs walkable_space.path")
    navpath = _resolved(package_navpath, base=resource_base, runtime=runtime)
    if not navpath.is_file():
        raise ValueError(
            "native Apartment package navmesh is not a readable declared file: "
            f"{navpath}"
        )
    floor = _floor_value(package, room, runtime=runtime, base=resource_base)
    pf = habitat_runtime.habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(navpath)):
        raise ValueError("native Apartment Recast navmesh did not load")
    bounds = np.asarray(pf.get_bounds(), dtype=float)
    nav = {
        "authority": "native_apartment_recast_navmesh",
        "floor_height_m": float(floor),
        "resolution_m": 0.08,
        "bounds_habitat_m": bounds.tolist(),
        "source_manifest": str(navpath),
        "runtime_prefix": str(habitat_runtime.prefix),
        "legacy_native_navmesh": str(resources.navmesh),
        "navmesh_source_match": navpath == resources.navmesh.resolve(),
        "route_bank": str(resources.route_bank),
        "route_bank_optional": True,
    }
    nav.update(_declared_planning_floors(package))
    space = HabitatWalkableSpace(pf, nav)
    layout = nq.build_native_apartment_layout(resources)
    layout["native_floor_height_m"] = float(floor)
    layout["native_navigation_mode"] = "habitat_navmesh"
    layout["native_navigation_authority"] = "native_apartment_recast_navmesh"
    layout["native_route_bank_optional"] = str(resources.route_bank)
    layout["backend_route"] = "spear_unreal"
    mesh = _package_mesh(package, runtime=runtime, base=resource_base)
    if mesh is None:
        raise ValueError("native Apartment free navigation needs shared static triangles")
    return space, mesh, layout


def load_planning_resources(room, request):
    """Load once per request; return an existing solver plus shared static mesh."""
    from avengine.rooms import native_qa_room as nq
    from avengine.rooms.qa_episode import build_room_navigation
    from avengine.rooms.furniture_layout import load_room_layout
    from avengine.rooms.furnished_episode import _load_static_triangle_geometry

    room = _expanded_room(room, request)
    package = room.get("room_package") or {}
    runtime = canonical_runtime(
        request.get("runtime") if isinstance(request.get("runtime"), dict) else {}
    )
    catalog_path = request.get("room_catalog")
    if catalog_path:
        catalog_file = _resolved(catalog_path, runtime=runtime)
        resource_base = catalog_file.parent
        package = resolve_room_package_paths(
            package, runtime=runtime, relative_roots=[resource_base]
        )
    else:
        resource_base = REPOSITORY_ROOT
    planning_inputs = package.get("planning_inputs") or {}
    adapter = planning_adapter_for_room(room, package)

    if adapter == "native_spear_route_bank":
        source_root = room.get("native_input_root") or planning_inputs.get("native_input_root")
        route_bank = room.get("route_bank") or planning_inputs.get("route_bank")
        profile_path = room.get("native_room_profile") or planning_inputs.get("native_room_profile")
        resources = nq.discover_native_apartment_resources(
            repository=REPOSITORY_ROOT,
            source_root=_resolved(source_root, base=resource_base, runtime=runtime),
            route_bank=_resolved(route_bank, base=resource_base, runtime=runtime),
            room_profile_path=(
                _resolved(profile_path, base=resource_base, runtime=runtime)
                if profile_path else None
            ),
        )
        layout = nq.build_native_apartment_layout(resources)
        pf, nav = build_room_navigation(
            layout, floor_height_m=float(layout["native_floor_height_m"])
        )
        bank = _read(resources.route_bank)
        seconds = float(bank["clip_seconds"])
        count = int(bank["frame_count"])
        rate = bank.get("frame_rate_hz", bank.get("frame_rate", count / seconds))
        routes = []
        for raw in bank["routes"]:
            try:
                points = nq._route_points(raw)
            except nq.NativeQAResourceError:
                continue
            if len(points) != count:
                continue
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length >= 2. and .6 <= length / seconds <= 1.5:
                motion_request = request.get("binding_motion") or request.get("binding_identity") or {}
                speed_range = motion_request.get("walk_speed_range_mps")
                if speed_range is not None:
                    if (len(speed_range) != 2 or not all(np.isfinite(speed_range))
                            or not 0 < speed_range[0] <= speed_range[1]):
                        raise ValueError("binding motion requires two finite positive ordered walk speeds")
                    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1)*float(rate)
                    moving = speeds[speeds > .05]
                    if (not len(moving) or moving.min() < speed_range[0]-1e-5
                            or moving.max() > speed_range[1]+1e-5):
                        continue
                routes.append({"route_id": raw["route_id"], "points_m": points})
        if not routes:
            raise ValueError("native route bank has no legal retained paths")
        nav.update(
            route_authority="native_spear_ue_recast_route_bank",
            native_route_bank=str(resources.route_bank),
            native_route_count=len(routes),
        )
        space = NativeRouteWalkableSpace(pf, nav, routes, float(rate))
    elif adapter == "retained_ue_walkable_grid":
        floor_height = _floor_value(
            package, room, runtime=runtime, base=resource_base
        )
        grid_path = _resolved(
            package["walkable_space"]["path"], base=resource_base, runtime=runtime
        )
        space = load_ue_walkable_grid(
            grid_path,
            floor_height_m=floor_height,
            clearance_m=float(request.get("body_clearance_m", .38)),
        )
        layout = {
            "room_id": room["room_id"],
            "scene_id": room.get("scene_id", room["room_id"]),
            "backend_route": "spear_unreal",
            "visual_lighting": {},
            "manifest_path": str(grid_path),
        }
        mesh = _package_mesh(package, runtime=runtime, base=resource_base)
        if mesh is None:
            raise ValueError("walkable-grid room needs shared static triangles")
        return space, mesh, layout
    elif adapter == "habitat_native_navmesh":
        from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

        rt = prepare_installed_habitat_runtime(
            **{
                key: runtime[key]
                for key in ("runtime_prefix", "mp3d_root", "magnum_python_site", "rlr_sdk_root")
                if runtime.get(key)
            }
        )
        manifest_raw = room.get("room_manifest") or planning_inputs.get("room_manifest")
        if (
            package.get("family") == "apartment"
            and package.get("renderer") == "ue_spear"
        ):
            return _load_native_apartment_recast_navmesh(
                package, room, planning_inputs, request, runtime, resource_base, rt
            )

        manifest_path = (
            _resolved(manifest_raw, base=resource_base, runtime=runtime)
            if manifest_raw else None
        )
        manifest = _read(manifest_path) if manifest_path else {}
        package_navpath = package.get("walkable_space", {}).get("path")
        navpath = package_navpath or manifest.get("scene", {}).get("navmesh_path")
        nav_base = (
            resource_base
            if package_navpath
            else (manifest_path.parent if manifest_path else resource_base)
        )
        navpath = _resolved(navpath, base=nav_base, runtime=runtime)
        pf = rt.habitat_sim.PathFinder()
        if not pf.load_nav_mesh(str(navpath)):
            raise ValueError("native Habitat navmesh did not load")
        floor = _floor_value(package, room, runtime=runtime, base=resource_base)
        bounds = np.asarray(pf.get_bounds())
        nav = {
            "authority": "habitat_native_pathfinder",
            "floor_height_m": floor,
            "resolution_m": .08,
            "bounds_habitat_m": bounds.tolist(),
            "source_manifest": str(navpath),
            "runtime_prefix": str(rt.prefix),
        }
        nav.update(_declared_planning_floors(package))
        space = HabitatWalkableSpace(pf, nav)
        mesh = _package_mesh(package, runtime=runtime, base=resource_base)
        if mesh is None:
            raise ValueError("Habitat room needs declared shared static triangles")
        m1_path = room.get("m1_request") or planning_inputs.get("m1_request")
        layout = {
            "room_id": room["room_id"],
            "scene_id": manifest.get("room_id", room["room_id"]),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "backend_route": ("spear_unreal" if package.get("renderer") == "ue_spear" else "habitat"),
            "visual_lighting": {},
            "capture_resolution_hw": (
                _read(_resolved(m1_path, base=resource_base, runtime=runtime))
                ["primary_camera_rig"]["shared_calibration"]["resolution_hw"]
                if m1_path else [240, 320]
            ),
        }
        return space, mesh, layout
    else:
        manifest_raw = room.get("manifest") or planning_inputs.get("manifest")
        if not manifest_raw:
            raise ValueError("furnished room requires a declared manifest path")
        manifest_path = _resolved(manifest_raw, base=resource_base, runtime=runtime)
        asset_root = room.get("asset_root") or planning_inputs.get("asset_root")
        if asset_root:
            asset_root = _resolved(asset_root, base=resource_base, runtime=runtime)
        layout = load_room_layout(
            manifest_path, asset_root=asset_root, require_seats=False
        )
        floor_height = (
            _floor_value(package, {}, runtime=runtime, base=resource_base)
            if package.get("floor_reference") else room.get("floor_height_m")
        )
        pf, nav = build_room_navigation(
            layout,
            clearance_m=float(request.get("body_clearance_m", .38)),
            floor_height_m=floor_height,
        )
        space = RasterWalkableSpace(pf, nav)

    mesh = _package_mesh(package, runtime=runtime, base=resource_base)
    if mesh is None:
        raw = _load_static_triangle_geometry(layout)
        if raw is None:
            raise ValueError("room has no retained static mesh for LOS")
        vertices = np.asarray(raw["vertices"], dtype=float)
        # Retained furniture/native room mesh is in authoring Z-up meters.
        vertices = np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1]))
        mesh = MeshHandle(
            vertices,
            raw["triangles"],
            {"path": raw["source"], "transform": "authoring_xyz_m_to_xz_negative_y_m"},
        )
    return space, mesh, layout


NEUTRAL_UE_RUNTIME_BINDING_MODE = "renderer_neutral_asset_frame_v2"


def materialize_ue_episode_plan(plan, registry):
    """Add UE driving fields in the UE executor, without changing the neutral plan."""
    if plan.get('plan_coordinates')!='renderer_neutral':return deepcopy(plan)
    from avengine.rooms.qa_episode import source_declaration
    from avengine.rooms.furniture_layout import habitat_to_ue_cm
    from avengine.runtime_profiles import resolve_source_asset_runtime_profile
    result=deepcopy(plan);result['execution_coordinates']='ue_spear';result['renderer_backend']='spear_unreal_native'
    result['visual_plan']['ue_neutral_runtime_binding'] = {
        'schema': 'avengine_spear_neutral_runtime_binding_v1',
        'mode': NEUTRAL_UE_RUNTIME_BINDING_MODE,
        'source': 'materialize_ue_episode_plan',
        'actor_root_preserved': True,
    }
    package=result['resources'].get('room_package',{})
    result['scene']['map_path']=package.get('visual_scene',{}).get('map_path',result['resources'].get('map_path'))
    result['scene']['backend']='spear_unreal'
    actors=[]
    for neutral in plan['visual_plan']['actors']:
        actor=source_declaration(registry,neutral['asset_id'],neutral['actor_id'])
        actor.update(entity_class=neutral['entity_class'])
        if isinstance(neutral.get('static_placement'), Mapping):
            actor['static_placement'] = deepcopy(neutral['static_placement'])
        timeline = resolve_source_asset_runtime_profile(
            registry, neutral['asset_id']
        ).get('timeline')
        if isinstance(timeline, dict) and actor.get(
            'ue_anatomical_forward_yaw_deg'
        ) is not None:
            axis = timeline.get('local_anatomical_forward_axis')
            if (
                isinstance(axis, (list, tuple))
                and len(axis) == 3
                and all(isinstance(value, (int, float)) for value in axis)
                and math.isfinite(float(axis[0]))
                and math.isfinite(float(axis[2]))
                and math.hypot(float(axis[0]), float(axis[2])) > 1.0e-12
            ):
                timeline_yaw = math.degrees(
                    math.atan2(float(axis[2]), float(axis[0]))
                )
                ue_yaw = float(actor['ue_anatomical_forward_yaw_deg'])
                correction_yaw = (
                    (timeline_yaw - ue_yaw + 180.0) % 360.0
                ) - 180.0
                actor['ue_neutral_visual_frame_correction'] = {
                    'schema': 'avengine_spear_component_frame_delta_v1',
                    'rotation_deg': [0.0, 0.0, correction_yaw],
                    'translation_cm': [0.0, 0.0, 0.0],
                    'composition': 'add_relative_preserving_blueprint_transform',
                    'reason': (
                        'renderer_neutral_timeline_axis_to_'
                        'ue_anatomical_forward'
                    ),
                    'timeline_forward_yaw_deg': timeline_yaw,
                    'ue_anatomical_forward_yaw_deg': ue_yaw,
                }
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
            actor=by_id[state['actor_id']]
            transform=state['root_transform']
            resting_pose = actor.get('resting_pose') or {}
            attachment_surface = str(resting_pose.get('attachment_surface', 'floor'))
            base_plane_offset = float(resting_pose.get('base_plane_offset_m', 0.0))
            needs_static_placement = (
                attachment_surface in {'wall', 'ceiling'}
                or abs(base_plane_offset) > 1.0e-9
            )
            if (
                actor.get('motion_model') == 'rigid_static'
                and needs_static_placement
                and not isinstance(actor.get('static_placement'), Mapping)
            ):
                raise ValueError(
                    f"{actor['actor_id']} {attachment_surface} static source "
                    "requires the shared static placement plan"
                )
            q=transform['rotation_xyzw']
            if actor.get('motion_model') == 'rigid_static':
                # Keep this optional-backend import lazy: articulated neutral
                # plans remain usable with lightweight test registries and do
                # not initialize the SPEAR Apartment registry on import.
                from avengine.optional_backends.spear_apartment import (
                    habitat_root_transform_to_ue,
                )
                ue_root = habitat_root_transform_to_ue(transform)
                ue_translation = ue_root['translation_cm']
                yaw=float(ue_root['rotation_deg'][2])
            else:
                yaw=-math.degrees(2*math.atan2(q[1],q[3]))
                ue_translation=habitat_to_ue_cm(transform['translation_m'])
                ue_root = None
            state.update(translation_m=deepcopy(transform['translation_m']),
                         translation_ue_cm=deepcopy(ue_translation),
                         rotation_xyzw=deepcopy(q),actor_yaw_ue_deg=yaw)
            if ue_root is not None:
                state['ue_root_transform'] = ue_root
            emitter_transform = state.get('emitter_transform')
            emitter_position = None
            if isinstance(emitter_transform, Mapping):
                emitter_position = emitter_transform.get('position_m')
            if emitter_position is None:
                emitter_position = state.get('planned_emitter_m')
            if emitter_position is not None:
                emitter_ue_cm = habitat_to_ue_cm(emitter_position)
                state['planned_emitter_ue_cm'] = deepcopy(emitter_ue_cm)
                if isinstance(emitter_transform, Mapping):
                    state['emitter_transform'] = {
                        **deepcopy(dict(emitter_transform)),
                        'ue_position_cm': deepcopy(emitter_ue_cm),
                    }
            correction = actor.get('ue_neutral_visual_frame_correction')
            if isinstance(correction, dict):
                timeline_yaw = float(correction.get('timeline_forward_yaw_deg', 0.0))
                expected_yaw = math.radians(yaw + timeline_yaw)
                state['anatomical_forward_ue_world'] = [
                    math.cos(expected_yaw),
                    math.sin(expected_yaw),
                    0.0,
                ]
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


# ---------------------------------------------------------------------------
# Route boundary: which adapter plans a room, which executor renders it
# ---------------------------------------------------------------------------


def planning_adapter_for_room(room, package):
    """Name the planning adapter this room routes to, from declared facts.

    These are the branches ``load_planning_resources`` implements, in its
    order. Naming them keeps the route reportable, and keeps the decision on
    walkable_space.kind and renderer rather than on a room_id.
    """
    from avengine.rooms import native_qa_room as nq

    kind = (package.get("walkable_space") or {}).get("kind")
    if kind == "route_bank":
        return "native_spear_route_bank"
    if kind == "walkable_grid":
        return "retained_ue_walkable_grid"
    if kind == "habitat_navmesh":
        return "habitat_native_navmesh"
    if room.get("native_room_adapter") == nq.SCHEMA:
        return "native_spear_route_bank"
    if room.get("backend") == "habitat" or package.get("renderer") == "habitat":
        return "habitat_native_navmesh"
    return "furnished_manifest_raster"


def selected_scene_reference(package):
    """The scene identity a readback must match for this room.

    Execution readback has to be checked against the scene this run selected,
    not against whatever the executor happened to load, so the identity
    travels with the route.
    """
    from avengine.rooms.room_package import renderer_for_room

    renderer = renderer_for_room(package)
    visual = package.get("visual_scene") or {}
    keys = (("map_path", "uproject") if renderer == "ue_spear"
            else ("scene_glb", "dataset_config", "navmesh"))
    return {
        "renderer": renderer,
        "room_id": package.get("room_id"),
        "family": package.get("family"),
        **{key: visual.get(key) for key in keys},
    }


def capture_adapter_binding(package, runtime=None, *, repository=None,
                            host_config=None, room_id=None, environment=None,
                            allow_environment=False):
    """Bind the selected room to its executor and that executor's parameters.

    This is the capture-side half of the route. It reports rather than runs,
    so a missing ``uproject`` or ``runtime_prefix`` comes back as a named
    blocker instead of an executor failing later with a partial command line.
    ``host_config`` is the run-local host runtime; it is a separate axis from
    the room's own resources, and its per-room scope is how MP3D and HM3D
    reach their different Habitat prefixes.
    """
    from avengine.rooms.room_providers import CAPTURE_ENTRYPOINTS
    from avengine.rooms.room_package import renderer_for_room, resolve_room_runtime

    renderer = renderer_for_room(package)
    if renderer not in CAPTURE_ENTRYPOINTS:
        raise ValueError(
            f"renderer {renderer!r} has no capture adapter; supported renderers "
            f"are {sorted(CAPTURE_ENTRYPOINTS)}"
        )
    report = resolve_room_runtime(
        package, runtime, host_config=host_config, room_id=room_id,
        environment=environment, allow_environment=allow_environment)
    root = Path(repository) if repository is not None else REPOSITORY_ROOT
    relative = CAPTURE_ENTRYPOINTS[renderer]
    entrypoint = root / relative
    blockers = list(report["missing"])
    if not entrypoint.is_file():
        blockers.append(f"capture entrypoint is absent: {entrypoint}")
    return {
        "schema": "avengine_qa_capture_adapter_binding_v1",
        "renderer": renderer,
        "planning_adapter": planning_adapter_for_room(
            {"room_package": package}, package
        ),
        "entrypoint": str(entrypoint),
        "entrypoint_repository_relative": relative,
        "selected_scene": selected_scene_reference(package),
        "runtime": report,
        "status": "pass" if not blockers else "blocked",
        "reason": None if not blockers else (
            "capture cannot be launched for this room: " + "; ".join(
                str(item) for item in blockers)
        ),
        "native_execution": "not_run",
    }


def request_host_runtime_config(request, *, runtime=None):
    """Load the host runtime config a request names, if it names one.

    Server paths live in that file rather than in the request or the
    distributable examples, so a request stays portable between machines.
    """
    declared = (request or {}).get("host_runtime")
    if declared is None:
        return None
    if isinstance(declared, Mapping):
        return dict(declared)
    return load_host_runtime_config(_resolved(declared, runtime=runtime or {}))


def load_planning_resources_for_room(room_id, request, *, catalog=None,
                                     profile_registry=None, host_config=None):
    """Resolve a registered room and load its planning resources in one call.

    The single entry a sampler should use: it takes a ``room_id`` against the
    request's ``room_catalog`` and returns the navigation space, the shared
    static mesh, the layout and the resolution that produced them. Runtime
    executor parameters are not required here - planning reads the room's
    declared resources, and capture is where ``uproject`` or
    ``runtime_prefix`` becomes mandatory.
    """
    from avengine.rooms.room_providers import load_room_catalog, require_catalog_room

    catalog_path = request.get("room_catalog")
    if catalog_path is None:
        raise ValueError(
            "load_planning_resources_for_room needs request['room_catalog']"
        )
    runtime = canonical_runtime(
        request.get("runtime") if isinstance(request.get("runtime"), dict) else {}
    )
    resolved_catalog_path = _resolved(catalog_path, runtime=runtime)
    if catalog is None:
        catalog = load_room_catalog(resolved_catalog_path)
    if host_config is None:
        host_config = request_host_runtime_config(request, runtime=runtime)
    resolution = require_catalog_room(
        catalog, room_id, catalog_path=resolved_catalog_path, runtime=runtime,
        require_runtime=False, profile_registry=profile_registry,
        host_config=host_config, request=request,
    )
    space, mesh, layout = load_planning_resources(dict(resolution.planning_room), request)
    return space, mesh, layout, resolution
