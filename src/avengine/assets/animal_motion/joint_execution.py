"""Prepare original animal poses inside solve; execute the same geometry later.

A geometry preparation uses the real room stage/NavMesh and original admitted
meshes. It does not claim native actor/root or pixel readback. The retained
NativePhase2Video performs those independent checks before rendering.
"""
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import math
import numpy as np


def load_prepared_geometry(entry, scene, house, request):
    path = Path(request['joint_animal_prepared_geometry'])
    payload = json.loads(path.read_text())
    authority = request['animal_motion_admission']
    record = payload['record']
    from .admitted_reconstruction import digest
    for key, source_path in [('native_motion_sha256', authority['native_motion']),
                             ('source_master_sha256', authority['source_master'])]:
        if authority.get(key) and digest(source_path) != authority[key]:
            raise ValueError('qualified animal source changed: ' + key)
    if (payload['asset_id'] != entry['declaration']['asset_id'] or payload['house'] != house['house']
            or payload['room_id'] != house['room']['room_label']):
        raise ValueError('prepared animal geometry belongs to another entity or room')
    if (record['motion_mode'] != authority['mode']
            or record['lineage']['admitted_native_motion'] != authority['native_motion']
            or record['lineage']['master'] != authority['source_master']):
        raise ValueError('prepared animal geometry changed its admitted body or locomotion mode')
    roots = np.asarray(record['world_roots_m'], float)
    planned = np.asarray(request['motion_quartet_route']['positions_xz_m'], float)
    if roots.shape != (901, 3) or planned.shape != (901, 2) or not np.isfinite(roots).all():
        raise ValueError('prepared animal geometry lacks the complete planned root clock')
    if float(np.max(np.abs(roots[:, [0, 2]] - planned))) > 2e-6:
        raise ValueError('prepared animal geometry no longer follows the solved horizontal route')
    if not np.allclose(scene['listener_m'], payload['listener_m'], atol=1e-7, rtol=0):
        raise ValueError('prepared animal observer position changed after solve')
    yaw = float(payload['camera_yaw_deg'])
    if any(abs((float(knot['yaw_deg']) - yaw + 180.) % 360. - 180.) > 1e-7
           for knot in scene['camera_yaw_knots']):
        raise ValueError('prepared animal observer orientation changed after solve')
    if len(record['pose_glbs']) != 901 or any(not Path(p).is_file() for p in set(record['pose_glbs'])):
        raise ValueError('prepared original animal pose buffers are missing')
    result = deepcopy(record)
    result.update(joint_geometry_executed_without_replanning=True,
                  joint_prepared_geometry_path=str(path), native_pose_readback_status='pending_actual_execution')
    return result


def prepare_route_geometry(context, house, runtime, asset, authority, actor, route, directory):
    """Construct full source pose/terrain placement before the core accepts a plan."""
    from avengine.assets.roster_motion.contact import yaw_matrix
    from avengine.assets.roster_motion.episode import PhysicalFloor
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    from .production_episode import compile_admitted_animal
    stage = getattr(context, '_owned_video', None)
    if stage is None:
        raise ValueError('animal geometry preparation needs the core original native room stage')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    # The adapter exposes the genuine stage APIs. No animal has been inserted
    # into the native renderer yet; the placeholder ID only excludes no stage.
    video = copy(stage)
    video.motion_house = house
    video.motion_inputs = {**runtime, 'phase2': True}
    native = json.loads(Path(authority['native_motion']).read_text())
    source = native['frames'][0]
    forward = np.asarray(source['head_m'], float) - np.asarray(source['tail_m'], float)
    if np.linalg.norm(forward[[0, 2]]) < 1e-8:
        raise ValueError('admitted animal has no physical forward axis')
    native_yaw = math.atan2(forward[2], forward[0]) - float(actor.headings_rad[0])
    mesh = indexed_static_mesh(asset['habitat_glb'])
    floor = PhysicalFloor(video, context.room['measured_floor_y_m'], (), .08)
    first = np.asarray(actor.positions_m[0], float).copy()
    first[1] = floor(first) - float(np.asarray(mesh.vertices)[:, 1].min()) + .0004
    root = np.eye(4);root[:3,:3] = yaw_matrix(native_yaw);root[:3,3] = first
    entry = {'declaration': asset, 'applied_root_input': root, 'resolved_yaw_rad': native_yaw,
             'prepared_glb': asset['habitat_glb'], 'original_mesh': mesh,
             'object': SimpleNamespace(object_id=2147483000)}
    video.actors = [entry]
    exact = deepcopy(route)
    exact['joint_exact_route'] = True
    request = {'action': 'walk', 'speed_m_s': route['speed_m_s'],
        'scene_speed_range_m_s': route['speed_range_m_s'], 'animal_motion_admission': authority,
        'motion_quartet_route': exact, 'maximum_native_route_attempts': 1}
    scene = {'listener_m': list(route['listener_m']),
        'camera_yaw_knots': [{'time_s':t,'yaw_deg':route['camera_yaw_deg']}for t in (0.,30.)],
        'production_scenario': {'motion_floor_polygon': context.room['floor_polygon']},
        'native_placement_candidates': []}
    selected_house = deepcopy(house);selected_house['room'] = deepcopy(context.room)
    record = compile_admitted_animal(video, entry, scene, selected_house, video.motion_inputs,
                                     directory/'geometry', request)
    record.update(asset_id=asset['asset_id'], frame_count=901, sample_rate_hz=30., geometry_preparation_only=True,
                  native_pose_readback_status='not_run_before_native_preparation')
    actor.positions_m = np.asarray(record['world_roots_m'], float)
    actor.metadata['native_yaw_rad'] = float(record['yaw_rad_per_frame'][0])
    payload = {'schema': 'joint-prepared-admitted-animal-geometry-v1', 'asset_id': asset['asset_id'],
        'house': house['house'], 'room_id': context.room['room_label'],
        'listener_m': list(route['listener_m']), 'camera_yaw_deg': route['camera_yaw_deg'], 'record': record}
    path = directory/'prepared_animal_geometry.json'
    path.write_text(json.dumps(payload, indent=2)+'\n')
    actor.metadata['joint_animal_prepared_geometry'] = str(path)
    return path, record
