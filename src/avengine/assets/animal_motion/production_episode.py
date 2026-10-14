"""Replay final admitted animal poses; independently admit the new 30s route.

The final motion-only manifest does not replace production sound, contact,
collision, direction or question gates. Slide is explicitly marked and has no
claim of joint animation or a planted-paw gait lock.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
from avengine.assets.roster_motion.contact import yaw_matrix


def source_yaw(frame):
    local = np.asarray(frame['head_m']) - np.asarray(frame['tail_m'])
    world = np.asarray(frame['head_world_m']) - np.asarray(frame['tail_world_m'])
    if min(np.linalg.norm(local[[0, 2]]), np.linalg.norm(world[[0, 2]])) < 1e-6:
        raise ValueError('admitted pose lacks a usable head/tail frame')
    return math.atan2(local[2], local[0]) - math.atan2(world[2], world[0])


def local_paw(frame, foot):
    return yaw_matrix(source_yaw(frame)).T @ (np.asarray(foot['sole_world_m']) - np.asarray(frame['root_world_m']))



def replay_paw_support(frame, foot, root, yaw):
    """Keep the observed support vertex distinct from the region's sole centre.

    New native rig receipts provide an actual vertex and its separate planted
    target. Older receipts store only the target and use the sole as the actual
    support point. Neither representation changes the original 3mm gate.
    """
    source_inverse = yaw_matrix(source_yaw(frame)).T
    origin = np.asarray(frame['root_world_m'])
    if foot.get('support_target_world_m') is not None:
        observed = np.asarray(foot['support_world_m'])
        target = np.asarray(foot['support_target_world_m'])
        representation = 'native_support_vertex'
    else:
        observed = np.asarray(foot['sole_world_m'])
        target = np.asarray(foot['support_world_m'])
        representation = 'legacy_sole_support'
    actual = yaw_matrix(yaw) @ (source_inverse @ (observed - origin)) + root
    target = yaw_matrix(yaw) @ (source_inverse @ (target - origin)) + root
    return {'support_world_m': actual.tolist(), 'support_target_world_m': target.tolist(),
            'support_lock_error_m': float(np.linalg.norm((actual - target)[[0, 2]])),
            'support_representation': representation, 'support_vertex': foot.get('support_vertex')}


def compile_admitted_animal(video, entry, scene, house, inputs, directory, request):
    from shapely.geometry import shape, LineString, Point, box
    from avengine.assets.roster_motion.episode import PhysicalFloor, TriangleClearance
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    from avengine.timeline.continuous_pilot import write_json
    authority = request['animal_motion_admission']
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    admission = json.loads(Path(authority['admission_path']).read_text())
    original = next(a for a in admission['assets'] if a['asset_id'] == entry['declaration']['asset_id'])
    selected = original['selected']
    if selected.get('status') != 'pass' or original['motion_mode'] != authority['mode'] or selected['native_motion'] != authority['native_motion']:
        raise ValueError('final animal motion admission mismatch')
    native = json.loads(Path(authority['native_motion']).read_text())
    from avengine.assets.animal_motion.admitted_reconstruction import reconstruct_frames, digest
    for key, path in [('native_motion_sha256', authority['native_motion']), ('source_master_sha256', authority['source_master'])]:
        if authority.get(key) and digest(path) != authority[key]:
            raise ValueError('qualified animal source changed: ' + key)
    source, reconstruction_receipt = reconstruct_frames(native, entry['declaration']['species_id'], directory / 'reconstructed_qualified_poses')
    if len(source) < 2 or any(not Path(f['pose_glb']).is_file() for f in source):
        raise ValueError('complete retained admitted pose stream missing')
    mode = authority['mode']
    period = float(source[-1]['time_s']) - float(source[0]['time_s'])
    if period <= 0:
        raise ValueError('admitted pose clock invalid')
    old_origin = np.asarray(source[0]['root_world_m'], float)
    old_end = np.asarray(source[-1]['root_world_m'], float)
    source_speed = float(np.linalg.norm((old_end - old_origin)[[0, 2]]) / period)
    speed = float(request.get('speed_m_s', source_speed))
    if not 0 < speed <= .08 or source_speed <= 1e-6:
        raise ValueError('unsupported admitted animal route speed')
    # Re-time actual saved joint poses with the translated distance. No new
    # deformation, inferred skeleton, or gait is invented for a slide.
    times = np.arange(901) / 30.
    phase = np.mod(times * speed / source_speed, period)
    indices = np.minimum(len(source) - 2, np.rint(phase / period * (len(source) - 1)).astype(int))
    if mode == 'slide_no_animation':
        if len({f['pose_glb'] for f in source}) != 1:
            raise ValueError('slide admission must retain one original static mesh')
        indices[:] = 0
    poses = [source[int(i)]['pose_glb'] for i in indices]
    meshes = {p: indexed_static_mesh(p) for p in set(poses)}
    old_floor = float(np.median([float(f['sole_world_m'][1]) - float(f['floor_delta_m']) for f in source[0]['feet'].values()]))
    local_forward = np.asarray(source[0]['head_m']) - np.asarray(source[0]['tail_m'])
    local_forward[1] = 0.
    local_forward /= np.linalg.norm(local_forward)
    poly = shape(scene['production_scenario']['motion_floor_polygon']).buffer(-.02)
    start = np.asarray(entry['applied_root_input'][:3, 3])
    candidates = [start]
    candidates.extend(np.asarray([p[0], start[1], p[2]]) for p in scene.get('native_placement_candidates', [])[:128])
    bodies = []
    for a in video.actors:
        if a is entry or a.get('phase2_layout_qualified') is False:
            continue
        vertices = np.asarray(a['original_mesh'].vertices) @ a['applied_root_input'][:3, :3].T + a['applied_root_input'][:3, 3]
        bodies.append(box(vertices[:, 0].min() - .03, vertices[:, 2].min() - .03, vertices[:, 0].max() + .03, vertices[:, 2].max() + .03))
    routes = []
    for p0 in candidates:
        for yaw in [entry['resolved_yaw_rad'], 0., math.pi / 2, math.pi, -math.pi / 2]:
            direction = yaw_matrix(yaw) @ local_forward
            end = p0 + times[-1] * speed * direction
            line = LineString([p0[[0, 2]], end[[0, 2]]])
            if not poly.covers(line) or any(line.intersects(b) for b in bodies):
                continue
            score = min([line.distance(poly.boundary), *[line.distance(b) for b in bodies]])
            routes.append((float(score), p0.copy(), yaw, direction))
    routes.sort(key=lambda r: (-r[0], float(np.linalg.norm(r[1][[0, 2]] - start[[0, 2]]))))
    failures = []
    begun = time.monotonic()
    ignore = {int(a['object'].object_id) for a in video.actors}
    # Retain the already qualified initial full-skin root offset in this house.
    # An old validation-house sole offset can bury the actual initial skin.
    placed_floor = PhysicalFloor(video, house['room']['measured_floor_y_m'], ignore)(start)
    qualified_initial_root_offset = float(start[1] - placed_floor)
    for _, p0, yaw, direction in routes[:8]:
        if time.monotonic() - begun > 1800:
            break
        try:
            actual_floor = PhysicalFloor(video, house['room']['measured_floor_y_m'], ignore)(p0)
            p0[1] = actual_floor + qualified_initial_root_offset
            roots = p0 + times[:, None] * speed * direction
            roots[:, 1] += [source[int(i)]['root_world_m'][1] - old_origin[1] for i in indices]
            snaps = np.asarray([video.pathfinder.snap_point(p) for p in roots])
            nav = float(np.max(np.linalg.norm((snaps - roots)[:, [0, 2]], axis=1)))
            if not math.isfinite(nav) or nav > .002:
                raise ValueError('admitted animal route exceeds unchanged 2mm NavMesh drift')
            floor = PhysicalFloor(video, actual_floor, ignore)
            clearance = TriangleClearance(video, house, actual_floor, entry)
            frames, bounds, minimum = [], [], math.inf
            for i, (path, root) in enumerate(zip(poses, roots)):
                if time.monotonic() - begun > 1800:
                    raise ValueError('bounded animal route search expired')
                old = source[int(indices[i])]
                mesh = meshes[path]
                dist, world = clearance.check(np.asarray(mesh.vertices), np.asarray(mesh.faces), root, yaw)
                minimum = min(minimum, dist)
                feet = {}
                for side, old_foot in old['feet'].items():
                    local = local_paw(old, old_foot)
                    sole = yaw_matrix(yaw) @ local + root
                    delta = float(sole[1] - floor(sole))
                    planted = bool(old_foot['in_contact'])
                    foot = {'sole_local_m': local.tolist(), 'sole_world_m': sole.tolist(),
                            'floor_delta_m': delta, 'in_contact': planted}
                    if mode == 'walk_animation' and planted:
                        if old_foot.get('support_world_m') is None:
                            raise ValueError('animated admission lacks its actual planted-paw target')
                        foot.update(replay_paw_support(old, old_foot, root, yaw))
                    elif mode == 'slide_no_animation':
                        foot.update(intentional_slide=True, planted_gait_lock_applicable=False)
                    limit = .012 if entry['declaration']['species_id'] == 'cat' else .008
                    lower = -.00101 if entry['declaration']['species_id'] == 'cat' else -.008
                    if delta < lower or (planted and abs(delta) >= limit) or foot.get('support_lock_error_m', 0) >= .003:
                        raise ValueError('admitted animal pose violates unchanged production foot/contact gate: ' + json.dumps({
                            'frame_index': i, 'source_frame_index': int(indices[i]), 'paw': side,
                            'floor_delta_m': delta, 'in_contact': planted,
                            'support_lock_error_m': foot.get('support_lock_error_m', 0),
                            'required_lower_m': lower, 'required_planted_abs_less_than_m': limit,
                            'required_support_lock_less_than_m': .003,
                            'qualified_initial_root_floor_offset_m': qualified_initial_root_offset,
                            'root_m': root.tolist(), 'sole_local_m': local.tolist()}, sort_keys=True))
                    feet[side] = foot
                if not any(f['in_contact'] for f in feet.values()):
                    raise ValueError('admitted animal pose has no actual support')
                frames.append({'time_s': float(times[i]), 'source_phase_s': float(phase[i]),
                               'source_frame_index': int(indices[i]), 'mouth_m': old['mouth_m'],
                               'head_m': old['head_m'], 'tail_m': old['tail_m'], 'feet': feet,
                               'moving': True, 'motion_mode': mode,
                               'joint_animation_present': mode == 'walk_animation',
                               'minimum_nonfloor_clearance_m': dist})
                bounds.append([world.min(0).tolist(), world.max(0).tolist()])
            lineage = {'roster': authority['admission_path'], 'roster_sha256': hashlib.sha256(Path(authority['admission_path']).read_bytes()).hexdigest(),
                       'master': authority['source_master'], 'master_sha256': hashlib.sha256(Path(authority['source_master']).read_bytes()).hexdigest(),
                       'admitted_native_motion': authority['native_motion'],
                       'source_pose_reconstruction_receipt': reconstruction_receipt,
                       'new_house_admission_required': True, 'motion_only_does_not_admit_audio': True}
            evidence = {'status': 'pass', 'revision': 'final-admitted-animal-poses-new-house-v2-qualified-initial-root',
                        'lineage': lineage, 'source_frames': frames, 'pose_glbs': poses,
                        'world_roots_m': roots.tolist(), 'yaw_rad': yaw, 'physical_ground_y_m': actual_floor,
                        'original_full_skin_floor_collision': clearance.floor_certificate(),
                        'qualified_initial_root_floor_offset_m': qualified_initial_root_offset,
                        'maximum_navmesh_horizontal_error_m': nav,
                        'minimum_certified_triangle_nonfloor_clearance_m': minimum,
                        'species_id': entry['declaration']['species_id'],
                        'motion_kind': 'animal_animated_walk' if mode == 'walk_animation' else 'animal_intentional_slide',
                        'motion_mode': mode, 'motion_marker': authority['motion_marker'],
                        'full_skin_bounds_m': [np.asarray(bounds)[:, 0].min(0).tolist(), np.asarray(bounds)[:, 1].max(0).tolist()],
                        'unique_native_pose_meshes': len(meshes), 'failed_candidates': failures,
                        'native_gates_changed': False}
            write_json(directory / 'motion_geometry.json', evidence)
            return evidence
        except ValueError as error:
            failures.append({'start_m': p0.tolist(), 'yaw_rad': yaw, 'reason': str(error)})
    write_json(directory / 'motion_rejected.json', {'status': 'blocked', 'mode': mode,
               'reason': 'no newly admitted original-gate route; no silent fallback', 'failed_candidates': failures})
    raise ValueError('final animal admission has no route passing unchanged production gates')
