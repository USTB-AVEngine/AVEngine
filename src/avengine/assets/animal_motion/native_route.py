"""Fit admitted animal routes without moving a certified plan unnecessarily.

The room floor polygon is a proposal outline. Tiny GEOS line/boundary errors
must not discard a route whose complete species map and native roots pass.
No map, NavMesh, physical clearance or answer tolerance is changed here.
"""
from copy import copy
import numpy as np
from shapely.geometry import LineString, mapping, shape

FLOOR_OUTLINE_NUMERICAL_MARGIN_M = 1e-6


def ordered_native_animal_routes(video, entry, scene, local_forward, request):
    if request['motion_quartet_route'].get('joint_exact_route'):
        from types import SimpleNamespace
        from avengine.rooms.walkable_map import load_room_map
        from avengine.construction.joint.routes import validate_track
        from avengine.construction.joint import CameraPose
        template = request['motion_quartet_route']
        points = np.asarray(template['positions_xz_m'], float)
        roots = np.column_stack((points[:, 0], np.full(len(points), entry['applied_root_input'][1,3]), points[:, 1]))
        walkable = load_room_map(template['walkable_map_room_id'], path=template['walkable_map_path'])
        context = SimpleNamespace(walkable_map=walkable, pathfinder=video.pathfinder)
        camera = CameraPose(tuple(scene['listener_m']), float(scene['camera_yaw_knots'][0]['yaw_deg']), width=1280, height=720)
        _, report = validate_track(context, roots, template['species'], camera=camera, snap=False)
        if report['max_snap_horizontal_m'] > .002:
            raise ValueError('solved animal route exceeds the original 2mm native NavMesh tolerance')
        local_angle = np.arctan2(local_forward[2], local_forward[0])
        yaws = np.unwrap(local_angle - np.asarray(template['travel_yaws_rad'], float))
        yield SimpleNamespace(positions=roots, yaws=yaws, receipt={
            'route_kind':'joint solved original animal route', 'maximum_planned_xz_displacement_m':0.,
            'joint_execution_search':False, 'root_samples':len(roots), 'species':template['species'],
            'animal_stationary_turn_intervals_s':template.get('animal_stationary_turn_intervals_s',[]),
            'native_navigation':report})
        return
    from avengine.assets.roster_motion.reflect_navmesh import reflected_route_proposals
    proposed_scene = copy(scene)
    proposed_scene['production_scenario'] = dict(scene['production_scenario'])
    outline = shape(scene['production_scenario']['motion_floor_polygon'])
    proposed_scene['production_scenario']['motion_floor_polygon'] = mapping(
        outline.buffer(FLOOR_OUTLINE_NUMERICAL_MARGIN_M))
    planned = np.asarray(request['motion_quartet_route']['positions_xz_m'], float)
    routes = list(reflected_route_proposals(video, entry, proposed_scene, local_forward, request))
    # The original route has already been certified with this fixed observer.
    # A clearance-only ordering could rotate it 180 degrees or shift it metres.
    # Keep every independently valid fallback, after the least altered plan.
    def displacement(route):
        return float(np.linalg.norm(np.asarray(route.positions)[:, [0, 2]] - planned, axis=1).max())
    routes.sort(key=displacement)
    for route in routes:
        points = np.asarray(route.positions)[:, [0, 2]]
        route.receipt.update(
            animal_route_order='smallest maximum displacement from the certified plan first',
            maximum_planned_xz_displacement_m=displacement(route),
            floor_outline_numerical_margin_m=FLOOR_OUTLINE_NUMERICAL_MARGIN_M,
            original_outline_uncovered_route_length_m=float(LineString(points).difference(outline).length),
            species_map_and_native_clearance_unchanged=True)
        yield route
