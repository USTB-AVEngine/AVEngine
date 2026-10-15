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

# Explicit owner-authorized visual quality limits; native truth/audio checks
# remain independent. This is an admission policy, never synthetic evidence.
OWNER_PRODUCTION_QUALITY = {
    'policy_name': 'owner_visual_quality_20261014',
    'contact_gap_m': .012,
    'support_lock_error_m': .015,
    'quality_thresholds_changed': True,
    'truth_and_answer_gates_changed': False,
    'original_floor_and_scene_collision_checks_unchanged': True,
}


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
    support point. Both representations preserve the admitted vertex and target; the caller
    applies the current owner contact policy.
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


def rigid_route_placement(poses, meshes, roots, source, indices, yaw, floor, species):
    """Place the complete admitted stream with one rigid vertical translation.

    Static and animated buffers need not share a local root-to-paw offset.
    Intersect the original skin/floor and planted-paw placement intervals over
    the whole route. This never changes a saved joint pose or corrects a paw
    per frame; the ordinary native gates still check every resulting pose.
    """
    rotation = yaw_matrix(yaw)
    lower = -math.inf
    upper = math.inf
    minimum_skin_delta = math.inf
    preferred = 0.
    paw_lower = -.00101 if species == 'cat' else -.008
    paw_limit = OWNER_PRODUCTION_QUALITY['contact_gap_m']
    for path, root, index in zip(poses, roots, indices):
        vertices = np.asarray(meshes[path].vertices)
        low = vertices[vertices[:, 1] + root[1] <= floor.reference + .04]
        if len(low):
            world = low @ rotation.T + root
            delta = float(np.min(world[:, 1] - floor.many(world)))
            minimum_skin_delta = min(minimum_skin_delta, delta)
            # A positive placement target leaves numerical headroom inside the
            # existing 1mm original-skin penetration criterion.
            lower = max(lower, -.001 - delta)
            preferred = max(preferred, .0004 - delta)
        frame = source[int(index)]
        feet = list(frame['feet'].values())
        soles = np.asarray([rotation @ local_paw(frame, foot) + root for foot in feet])
        deltas = soles[:, 1] - floor.many(soles)
        for foot, delta in zip(feet, deltas):
            lower = max(lower, paw_lower + .00001 - float(delta))
            if foot['in_contact']:
                lower = max(lower, -paw_limit + .00001 - float(delta))
                upper = min(upper, paw_limit - .00001 - float(delta))
    if not math.isfinite(lower) or lower > upper:
        raise ValueError('no constant rigid animal route height satisfies original skin and paw gates: '
                         + json.dumps({'minimum_shift_m': lower, 'maximum_shift_m': upper}))
    shift = min(max(preferred, lower), upper)
    placed = roots.copy()
    placed[:, 1] += shift
    return placed, {'method': 'one rigid vertical translation of complete admitted pose stream',
                    'vertical_translation_m': float(shift),
                    'minimum_translation_m': float(lower), 'maximum_translation_m': float(upper),
                    'minimum_original_skin_delta_before_translation_m': minimum_skin_delta,
                    'per_frame_correction': False, 'joint_poses_changed': False,
                    'production_gates_changed': True,
                    'production_visual_quality_policy': dict(OWNER_PRODUCTION_QUALITY),
                    'truth_and_answer_gates_changed': False}


def route_preview_visible(scene, start, end, yaw, mouth):
    """Proposal filter matching the existing observed camera-path frustum.

    The complete native semantic pixel gate still runs after placement. This
    prevents route scoring from moving a visible animal under the camera.
    """
    listener = np.asarray(scene['listener_m'], float)
    offset = yaw_matrix(yaw) @ np.asarray(mouth, float)
    directions = []
    for knot in scene['camera_yaw_knots']:
        angle = math.radians(float(knot['yaw_deg']))
        forward = np.array([math.sin(angle), 0., -math.cos(angle)])
        directions.append((forward, np.cross(forward, [0., 1., 0.])))
    for fraction in (0., .5, 1.):
        delta = np.asarray(start) * (1. - fraction) + np.asarray(end) * fraction + offset - listener
        visible = False
        for forward, right in directions:
            depth = float(delta @ forward)
            if (depth > 0. and abs(math.degrees(math.atan2(float(delta @ right), depth))) <= 37.
                    and abs(math.degrees(math.atan2(float(delta[1]), depth))) <= 24.):
                visible = True
                break
        if not visible:
            return False
    return True


def motion_route_attempts(route_tiers, requested_mode, stationary):
    """Try shorter/slower physical routes while preserving the requested motion.

    Owner 2026-10-15: construct missing conditions instead of degrading a walk
    to a slide or a stationary pose. ``stationary`` is a retained call argument
    for existing callers; those proposals cannot satisfy a motion request.
    """
    return [(route, requested_mode) for tier in route_tiers for route in tier[:2]]


def realized_animal_motion(scene, visual):
    """Carry actual native motion into the scene labels used by the questions."""
    for ep, record in visual.get('native_motion_admissions', {}).items():
        if record.get('species_id') not in {'dog', 'cat'}:
            continue
        spec = scene['sources'][ep]
        spec['motion_marker'] = record['motion_marker']
        spec['motion_metadata'] = {**spec.get('motion_metadata', {}),
            'requested_mode': record.get('requested_motion_mode', record['motion_mode']),
            'realized_mode': record['motion_mode'],
            'joint_animation_present': record['motion_mode'] == 'walk_animation',
            'actual_speed_m_s': record.get('actual_speed_m_s'),
            'motion_degradation': record.get('motion_degradation')}


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
    requested_mode = mode
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
    route_tiers = []
    for route_speed in [speed, speed / 2., speed / 4., speed / 10.]:
        tier = []
        for p0 in candidates:
            for yaw in [entry['resolved_yaw_rad'], 0., math.pi / 2, math.pi, -math.pi / 2]:
                direction = yaw_matrix(yaw) @ local_forward
                end = p0 + times[-1] * route_speed * direction
                line = LineString([p0[[0, 2]], end[[0, 2]]])
                if (not poly.covers(line) or any(line.intersects(b) for b in bodies)
                        or not route_preview_visible(scene, p0, end, yaw, source[0]['mouth_m'])):
                    continue
                score = min([line.distance(poly.boundary), *[line.distance(b) for b in bodies]])
                tier.append((float(score), p0.copy(), yaw, direction, route_speed))
        tier = list({(tuple(r[1]), r[2]): r for r in tier}.values())
        tier.sort(key=lambda r: (float(np.linalg.norm(r[1][[0, 2]] - start[[0, 2]])), -r[0]))
        route_tiers.append(tier)
    # Keep eight proposals, while giving shorter/slower motion an actual turn.
    stationary = [(0., p0.copy(), entry['resolved_yaw_rad'],
                   yaw_matrix(entry['resolved_yaw_rad']) @ local_forward, 0.) for p0 in candidates[:4]
                  if poly.covers(Point(p0[0], p0[2])) and route_preview_visible(
                      scene, p0, p0, entry['resolved_yaw_rad'], source[0]['mouth_m'])]
    routes = motion_route_attempts(route_tiers, requested_mode, stationary)
    failures = []
    begun = time.monotonic()
    ignore = {int(a['object'].object_id) for a in video.actors}
    # Retain the already qualified initial full-skin root offset in this house.
    # An old validation-house sole offset can bury the actual initial skin.
    placed_floor = PhysicalFloor(video, house['room']['measured_floor_y_m'], ignore)(start)
    qualified_initial_root_offset = float(start[1] - placed_floor)
    for (_, p0, yaw, direction, route_speed), mode in routes:
        if time.monotonic() - begun > 1800:
            break
        try:
            phase = np.mod(times * route_speed / source_speed, period)
            indices = np.minimum(len(source) - 2, np.rint(phase / period * (len(source) - 1)).astype(int))
            if mode != 'walk_animation':
                indices[:] = 0
            poses = [source[int(i)]['pose_glb'] for i in indices]
            for path in set(poses):
                if path not in meshes:
                    meshes[path] = indexed_static_mesh(path)
            actual_floor = PhysicalFloor(video, house['room']['measured_floor_y_m'], ignore)(p0)
            p0[1] = actual_floor + qualified_initial_root_offset
            roots = p0 + times[:, None] * route_speed * direction
            roots[:, 1] += [source[int(i)]['root_world_m'][1] - old_origin[1] for i in indices]
            snaps = np.asarray([video.pathfinder.snap_point(p) for p in roots])
            nav = float(np.max(np.linalg.norm((snaps - roots)[:, [0, 2]], axis=1)))
            if not math.isfinite(nav) or nav > .002:
                raise ValueError('admitted animal route exceeds unchanged 2mm NavMesh drift')
            floor = PhysicalFloor(video, actual_floor, ignore)
            roots, rigid_placement = rigid_route_placement(
                poses, meshes, roots, source, indices, yaw, floor, entry['declaration']['species_id'])
            p0 = roots[0].copy()
            snaps = np.asarray([video.pathfinder.snap_point(p) for p in roots])
            nav = float(np.max(np.linalg.norm((snaps - roots)[:, [0, 2]], axis=1)))
            if not math.isfinite(nav) or nav > .002:
                raise ValueError('rigidly placed animal route exceeds unchanged 2mm NavMesh drift')
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
                    planted = True if mode != 'walk_animation' else bool(old_foot['in_contact'])
                    foot = {'sole_local_m': local.tolist(), 'sole_world_m': sole.tolist(),
                            'floor_delta_m': delta, 'in_contact': planted}
                    if mode == 'walk_animation' and planted:
                        if old_foot.get('support_world_m') is None:
                            raise ValueError('animated admission lacks its actual planted-paw target')
                        foot.update(replay_paw_support(old, old_foot, root, yaw))
                    elif mode != 'walk_animation':
                        foot.update(intentional_slide=mode == 'slide_no_animation', planted_gait_lock_applicable=False, static_supported_pose=mode == 'static_after_failed_route')
                    limit = OWNER_PRODUCTION_QUALITY['contact_gap_m']
                    lower = -.00101 if entry['declaration']['species_id'] == 'cat' else -.008
                    if delta < lower or (planted and abs(delta) > limit + 1e-12) or foot.get('support_lock_error_m', 0) > OWNER_PRODUCTION_QUALITY['support_lock_error_m'] + 1e-12:
                        raise ValueError('admitted animal pose violates owner visual foot/contact quality gate: ' + json.dumps({
                            'frame_index': i, 'source_frame_index': int(indices[i]), 'paw': side,
                            'floor_delta_m': delta, 'in_contact': planted,
                            'support_lock_error_m': foot.get('support_lock_error_m', 0),
                            'required_lower_m': lower, 'required_planted_abs_max_m': limit,
                            'required_support_lock_max_m': OWNER_PRODUCTION_QUALITY['support_lock_error_m'],
                            'qualified_initial_root_floor_offset_m': qualified_initial_root_offset,
                            'root_m': root.tolist(), 'sole_local_m': local.tolist()}, sort_keys=True))
                    feet[side] = foot
                if not any(f['in_contact'] for f in feet.values()):
                    raise ValueError('admitted animal pose has no actual support')
                frames.append({'time_s': float(times[i]), 'source_phase_s': float(phase[i]),
                               'source_frame_index': int(indices[i]), 'mouth_m': old['mouth_m'],
                               'head_m': old['head_m'], 'tail_m': old['tail_m'], 'feet': feet,
                               'moving': mode != 'static_after_failed_route', 'motion_mode': mode,
                               'joint_animation_present': mode == 'walk_animation',
                               'minimum_nonfloor_clearance_m': dist})
                bounds.append([world.min(0).tolist(), world.max(0).tolist()])
            lineage = {'roster': authority['admission_path'], 'roster_sha256': hashlib.sha256(Path(authority['admission_path']).read_bytes()).hexdigest(),
                       'master': authority['source_master'], 'master_sha256': hashlib.sha256(Path(authority['source_master']).read_bytes()).hexdigest(),
                       'admitted_native_motion': authority['native_motion'],
                       'source_pose_reconstruction_receipt': reconstruction_receipt,
                       'new_house_admission_required': True, 'motion_only_does_not_admit_audio': True}
            evidence = {'status': 'pass', 'revision': 'final-admitted-animal-poses-new-house-v3-rigid-stream-placement',
                        'lineage': lineage, 'source_frames': frames, 'pose_glbs': poses,
                        'world_roots_m': roots.tolist(), 'yaw_rad': yaw, 'physical_ground_y_m': actual_floor,
                        'original_full_skin_floor_collision': clearance.floor_certificate(),
                        'qualified_initial_root_floor_offset_m': qualified_initial_root_offset,
                        'rigid_route_placement': rigid_placement,
                        'requested_speed_m_s': speed, 'actual_speed_m_s': route_speed,
                        'requested_motion_mode': requested_mode,
                        'motion_degradation': ('static_after_failed_route' if mode == 'static_after_failed_route'
                            else 'intentional_slide_after_failed_walk_route' if mode != requested_mode
                            else 'slower_shorter_admitted_motion' if route_speed < speed else None),
                        'owner_quality_policy': {'authority': 'owner 2026-10-14: ability to generate questions first',
                            'paw_contact_m': .012, 'support_lock_m': .015, 'full_skin_floor_margin_changed': False},
                        'maximum_navmesh_horizontal_error_m': nav,
                        'minimum_certified_triangle_nonfloor_clearance_m': minimum,
                        'species_id': entry['declaration']['species_id'],
                        'motion_kind': ('animal_animated_walk' if mode == 'walk_animation'
                            else 'animal_intentional_slide' if mode == 'slide_no_animation' else 'animal_static_fallback'),
                        'motion_mode': mode, 'motion_marker': ('animated_walk' if mode == 'walk_animation'
                            else 'intentional_slide_no_animation' if mode == 'slide_no_animation' else 'static_after_failed_route'),
                        'full_skin_bounds_m': [np.asarray(bounds)[:, 0].min(0).tolist(), np.asarray(bounds)[:, 1].max(0).tolist()],
                        'unique_native_pose_meshes': len(meshes), 'failed_candidates': failures,
                        'native_gates_changed': True, 'quality_gates_changed': True,'truth_or_scorer_changed':False,
                        'production_visual_quality_policy': dict(OWNER_PRODUCTION_QUALITY),
                        'truth_and_answer_gates_changed': False}
            write_json(directory / 'motion_geometry.json', evidence)
            return evidence
        except ValueError as error:
            failures.append({'start_m': p0.tolist(), 'yaw_rad': yaw, 'speed_m_s': route_speed, 'mode': mode, 'reason': str(error)})
    write_json(directory / 'motion_rejected.json', {'status': 'blocked', 'mode': mode,
               'reason': 'walk/shorter route, explicit retained-pose slide and static placement all failed physical gates', 'failed_candidates': failures})
    raise ValueError('final animal admission has no route passing retained answer gates and owner visual quality policy')
