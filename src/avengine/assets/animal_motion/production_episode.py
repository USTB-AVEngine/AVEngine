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
    yaw_values=np.full(len(roots),float(yaw)) if np.ndim(yaw)==0 else np.asarray(yaw,float)
    lower = -math.inf
    upper = math.inf
    minimum_skin_delta = math.inf
    preferred = 0.
    paw_lower = -.00101 if species == 'cat' else -.008
    paw_limit = OWNER_PRODUCTION_QUALITY['contact_gap_m']
    for frame_index,(path, root, index) in enumerate(zip(poses, roots, indices)):
        rotation=yaw_matrix(float(yaw_values[frame_index]))
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


def route_navmesh_preview(pathfinder,start,end):
    # Reject cheaply at five exact sample indices, then check all 901 original
    # root samples. The full native admission still repeats the same 2mm gate.
    start=np.asarray(start);end=np.asarray(end)
    for indices in [(0,225,450,675,900),range(901)]:
        for index in indices:
            p=start*(1.-index/900.)+end*(index/900.)
            snapped=np.asarray(pathfinder.snap_point(p))
            if not np.isfinite(snapped).all() or np.linalg.norm((snapped-p)[[0,2]])>.002:return False
    return True


def native_walkable_starts(video,scene,poly,start,previous,mouth,yaw):
    """Construct at least 200 visible NavMesh proposals before full skin work."""
    from shapely.geometry import Point
    points={};x0,z0,x1,z1=poly.bounds;attempted=0
    def add(raw):
        nonlocal attempted
        p=np.asarray(raw,float);attempted+=1;snap=np.asarray(video.pathfinder.snap_point(p))
        if not np.isfinite(snap).all() or np.linalg.norm((snap-p)[[0,2]])>.002:return
        p=p.copy();p[[0,2]]=snap[[0,2]]
        if not poly.covers(Point(p[0],p[2])) or not route_preview_visible(scene,p,p,yaw,mouth):return
        points.setdefault((float(p[0]),float(p[2])),p)
    for p in previous:add(p)
    for cells in (33,65,129):
        for x in np.linspace(x0,x1,cells):
            for z in np.linspace(z0,z1,cells):add([x,start[1],z])
        if len(points)>=200:break
    selected=sorted(points.values(),key=lambda p:float(np.linalg.norm((p-start)[[0,2]])))
    return selected,{'method':'deterministic dense floor grid, visible native NavMesh starts; no RNG or seed substitution',
        'grid_points_tested':attempted,'visible_walkable_starts':len(selected),
        'headings_per_start':16,'complete_segment_samples':901,'maximum_horizontal_snap_error_m':.002,
        'original_complete_native_admission_unchanged':True}


def require_moving_animal_mode(mode):
    if mode not in {'walk_animation','slide_no_animation'}:
        raise ValueError('requested moving animal requires admitted walking or intentional sliding; no static fallback')
    return mode


def motion_route_attempts(route_tiers, requested_mode, stationary):
    """Try shorter/slower physical routes while preserving the requested motion.

    Owner 2026-10-15: construct missing conditions instead of degrading a walk
    to a slide or a stationary pose. ``stationary`` is a retained call argument
    for existing callers; those proposals cannot satisfy a motion request.
    """
    require_moving_animal_mode(requested_mode)
    return [(route, requested_mode) for tier in route_tiers for route in tier]


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




def animal_reflected_route_proposals(video,entry,scene,local_forward,request):
    """Reuse the common physical fit, refining rigid placement after height failures.

    A coarse 25cm proposal grid can leave only a few routes whose rigid height
    interval misses by fractions of a millimetre on a scanned floor. Smaller
    translations/rotations construct other routes; every one goes through the
    exact same shared NavMesh, camera, polygon, body and radial prefilter.
    """
    from avengine.assets.roster_motion.reflect_navmesh import reflected_route_proposals
    original=request['motion_quartet_route'];xz=np.asarray(original['positions_xz_m'],float)
    heading=np.asarray(original['travel_yaws_rad'],float);centre=xz.mean(0)
    needs_refinement=[False];seen=set()
    seeds=[(0.,0.,0.)]+[(x,z,0.)for x,z in ((.025,0.),(-.025,0.),(0.,.025),(0.,-.025),(.025,.025),(-.025,-.025),(.05,0.),(-.05,0.),(0.,.05),(0.,-.05))]+[(0.,0.,a)for a in (math.pi/72,-math.pi/72,math.pi/36,-math.pi/36)]
    for number,(dx,dz,angle)in enumerate(seeds):
        if number and not needs_refinement[0]:return
        proposal_request=deepcopy(request)
        rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        proposal_request['motion_quartet_route']['positions_xz_m']=((xz-centre)@rotation.T+centre+[dx,dz]).tolist()
        proposal_request['motion_quartet_route']['travel_yaws_rad']=(heading+angle).tolist()
        if number and request.get('_reflection_diagnostics_path'):
            path=Path(request['_reflection_diagnostics_path'])
            proposal_request['_reflection_diagnostics_path']=str(path.parent/('fine_fit_'+str(number))/path.name)
        for route in reflected_route_proposals(video,entry,scene,local_forward,proposal_request):
            key=(np.asarray(route.positions).tobytes(),np.asarray(route.yaws).tobytes())
            if key in seen:continue
            seen.add(key)
            route.receipt={**route.receipt,'animal_rigid_fitting_round':number,
                'animal_seed_translation_xz_m':[dx,dz],'animal_seed_rotation_rad':angle,
                'original_physical_gates_unchanged':True}
            previous=getattr(route,'report_failure',None)
            def report_failure(reason,point=None,previous=previous):
                if 'no constant rigid animal route height' in reason:needs_refinement[0]=True
                if previous is not None:previous(reason,point)
            route.report_failure=report_failure
            yield route


def animal_route_samples(times, origin, yaw, direction, speed, curved_route=None):
    """Execute a supplied native route and drive its gait by translated distance.

    Rigid fitting preserves the template path. Stationary samples keep their
    last heading and accumulate no distance, so their admitted pose also stays
    fixed. The original paw, skin, NavMesh and full-scene checks run afterwards.
    """
    times=np.asarray(times,float)
    if curved_route is None:
        roots=np.asarray(origin,float)+times[:,None]*float(speed)*np.asarray(direction,float)
        yaws=np.full(len(times),float(yaw))
        distance=times*float(speed)
        return roots,yaws,distance,np.full(len(times),float(speed))
    roots=np.asarray(curved_route.positions,float).copy()
    yaws=np.asarray(curved_route.yaws,float).copy()
    if roots.shape!=(len(times),3) or yaws.shape!=(len(times),) or not np.isfinite(roots).all() or not np.isfinite(yaws).all():
        raise ValueError('animal native route needs one finite position and heading per frame')
    steps=np.linalg.norm(np.diff(roots[:,[0,2]],axis=0),axis=1)
    distance=np.r_[0.,np.cumsum(steps)]
    velocity=np.r_[steps[0],steps]/(times[1]-times[0])
    for i in range(1,len(yaws)):
        if steps[i-1]<=1e-10:yaws[i]=yaws[i-1]
    return roots,np.unwrap(yaws),distance,velocity


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
    mode = require_moving_animal_mode(authority['mode'])
    requested_mode = mode
    period = float(source[-1]['time_s']) - float(source[0]['time_s'])
    if period <= 0:
        raise ValueError('admitted pose clock invalid')
    old_origin = np.asarray(source[0]['root_world_m'], float)
    old_end = np.asarray(source[-1]['root_world_m'], float)
    source_speed = float(np.linalg.norm((old_end - old_origin)[[0, 2]]) / period)
    from avengine.assets.roster_motion.navmesh_execution import REVISION as EXECUTION_REVISION,curved_route_proposals
    template=request.get('motion_quartet_route')
    reflected_motion=bool(template and template.get('reflect_walk'))
    curved_motion=reflected_motion or request.get('motion_execution_revision')==EXECUTION_REVISION
    speed = float(request.get('speed_m_s', source_speed))
    from avengine.timeline.v7_motion_speed import SPEED_RANGES
    speed_kind='slide' if mode=='slide_no_animation' else entry['declaration']['species_id']
    minimum_scene_speed,maximum_scene_speed=SPEED_RANGES[speed_kind]
    if not minimum_scene_speed <= speed <= maximum_scene_speed or source_speed <= 1e-6:
        raise ValueError('unsupported admitted animal route speed')
    # Re-time actual saved joint poses with the translated distance. No new
    # deformation, inferred skeleton, or gait is invented for a slide.
    times = np.arange(901) / 30.
    distance=times*speed;velocity=np.full(len(times),speed)
    template=request.get('motion_quartet_route')
    if template:
        from avengine.assets.roster_motion.timed_straight_clock import single_pass_clock,timed_straight_clock
        if template.get('reflect_walk'):
            from avengine.assets.roster_motion.reflect_navmesh import reflected_distance_clock
            distance,velocity,_=reflected_distance_clock(template,source_speed)
        elif template.get('single_pass'):distance,velocity=single_pass_clock(times,speed,template['turn_speed_m_s'],template.get('minimum_stop_s',7.9))
        else:distance,velocity,_=timed_straight_clock(times,speed,straight_start_s=template['straight_start_s'],straight_end_s=template['straight_end_s'],turn_speed_m_s=template['turn_speed_m_s'])
    phase = np.mod(distance / source_speed, period)
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
    search_floor=shape(scene['production_scenario']['motion_floor_polygon'])
    poly=(search_floor.convex_hull if request.get('motion_quartet_route') else search_floor).buffer(-.02)
    start = np.asarray(entry['applied_root_input'][:3, 3])
    candidates = [start]
    candidates.extend(np.asarray([p[0], start[1], p[2]]) for p in scene.get('native_placement_candidates', [])[:128])
    bodies = []
    for a in video.actors:
        if a is entry or a.get('phase2_layout_qualified') is False:
            continue
        vertices = np.asarray(a['original_mesh'].vertices) @ a['applied_root_input'][:3, :3].T + a['applied_root_input'][:3, 3]
        bodies.append(box(vertices[:, 0].min() - .03, vertices[:, 2].min() - .03, vertices[:, 0].max() + .03, vertices[:, 2].max() + .03))
    if curved_motion:
        if request.get('motion_quartet_route',{}).get('reflect_walk'):request['_reflection_diagnostics_path']=str(directory/'reflection_native_prefilter.json')
        if reflected_motion:
            curve_stream=animal_reflected_route_proposals(video,entry,scene,local_forward,request)
        else:
            curve_stream=curved_route_proposals(video,entry,scene,distance,local_forward,request)
        routes=(((0.,route.positions[0].copy(),float(route.yaws[0]),np.zeros(3),speed),requested_mode,route)for route in curve_stream)
    else:
        candidates,construction=native_walkable_starts(video,scene,poly,start,candidates,source[0]['mouth_m'],entry['resolved_yaw_rad'])
        headings=list(dict.fromkeys([entry['resolved_yaw_rad'],*[i*math.pi/8 for i in range(16)]]))
        construction.update(actual_headings_per_start=len(headings),segments_tested=0,complete_navmesh_segments=0)
        routes = []
        route_tiers = []
        for route_speed in [speed]:
            tier = []
            for p0 in candidates:
                for yaw in headings:
                    direction = yaw_matrix(yaw) @ local_forward
                    end = p0 + times[-1] * route_speed * direction
                    line = LineString([p0[[0, 2]], end[[0, 2]]])
                    if (not poly.covers(line) or any(line.intersects(b) for b in bodies)
                            or not route_preview_visible(scene, p0, end, yaw, source[0]['mouth_m'])):
                        continue
                    construction['segments_tested']+=1
                    if not route_navmesh_preview(video.pathfinder,p0,end):continue
                    construction['complete_navmesh_segments']+=1
                    score = min([line.distance(poly.boundary), *[line.distance(b) for b in bodies]])
                    tier.append((float(score), p0.copy(), yaw, direction, route_speed))
            tier = list({(tuple(r[1]), r[2]): r for r in tier}.values())
            tier.sort(key=lambda r: (float(np.linalg.norm(r[1][[0, 2]] - start[[0, 2]])), -r[0]))
            route_tiers.append(tier)
        write_json(directory/'route_construction.json',construction)
        # Keep eight proposals, while giving shorter/slower motion an actual turn.
        stationary = []
        routes = motion_route_attempts(route_tiers, requested_mode, stationary)
        routes=[(route,mode,None)for route,mode in routes]
    failures = []
    begun = time.monotonic()
    ignore = {int(a['object'].object_id) for a in video.actors}
    # Retain the already qualified initial full-skin root offset in this house.
    # An old validation-house sole offset can bury the actual initial skin.
    placed_floor = PhysicalFloor(video, house['room']['measured_floor_y_m'], ignore)(start)
    qualified_initial_root_offset = float(start[1] - placed_floor)
    maximum_attempts=int(request.get('maximum_native_route_attempts',64))
    if not 1<=maximum_attempts<=256:raise ValueError('invalid native route retry budget')
    for attempt_number,((_, p0, yaw, direction, route_speed), mode, curved_route) in enumerate(routes,1):
        if attempt_number>maximum_attempts:
            failures.append({'reason':'native route retry budget exhausted; planner must construct another room or grid cell','maximum_attempts':maximum_attempts})
            break
        if not curved_motion and time.monotonic() - begun > 1800:
            break
        try:
            roots,frame_yaws,route_distance,velocity=animal_route_samples(
                times,p0,yaw,direction,route_speed,curved_route)
            phase = np.mod(route_distance / source_speed, period)
            indices = np.minimum(len(source) - 2, np.rint(phase / period * (len(source) - 1)).astype(int))
            if mode != 'walk_animation':
                indices[:] = 0
            poses = [source[int(i)]['pose_glb'] for i in indices]
            for path in set(poses):
                if path not in meshes:
                    meshes[path] = indexed_static_mesh(path)
            actual_floor = PhysicalFloor(video, house['room']['measured_floor_y_m'], ignore)(p0)
            p0[1] = actual_floor + qualified_initial_root_offset
            roots[:,1]+=p0[1]-roots[0,1]
            roots[:, 1] += [source[int(i)]['root_world_m'][1] - old_origin[1] for i in indices]
            snaps = np.asarray([video.pathfinder.snap_point(p) for p in roots])
            nav = float(np.max(np.linalg.norm((snaps - roots)[:, [0, 2]], axis=1)))
            if not math.isfinite(nav) or nav > .002:
                raise ValueError('admitted animal route exceeds unchanged 2mm NavMesh drift')
            floor = PhysicalFloor(video, actual_floor, ignore)
            roots, rigid_placement = rigid_route_placement(
                poses, meshes, roots, source, indices, frame_yaws, floor, entry['declaration']['species_id'])
            p0 = roots[0].copy()
            snaps = np.asarray([video.pathfinder.snap_point(p) for p in roots])
            nav = float(np.max(np.linalg.norm((snaps - roots)[:, [0, 2]], axis=1)))
            if not math.isfinite(nav) or nav > .002:
                raise ValueError('rigidly placed animal route exceeds unchanged 2mm NavMesh drift')
            clearance = TriangleClearance(video, house, actual_floor, entry)
            frames, bounds, minimum = [], [], math.inf
            for i, (path, root) in enumerate(zip(poses, roots)):
                if not curved_motion and time.monotonic() - begun > 1800:
                    raise ValueError('bounded animal route search expired')
                old = source[int(indices[i])]
                frame_yaw=float(frame_yaws[i])
                mesh = meshes[path]
                dist, world = clearance.check(np.asarray(mesh.vertices), np.asarray(mesh.faces), root, frame_yaw)
                minimum = min(minimum, dist)
                feet = {}
                for side, old_foot in old['feet'].items():
                    local = local_paw(old, old_foot)
                    sole = yaw_matrix(frame_yaw) @ local + root
                    delta = float(sole[1] - floor(sole))
                    planted = True if mode != 'walk_animation' else bool(old_foot['in_contact'])
                    foot = {'sole_local_m': local.tolist(), 'sole_world_m': sole.tolist(),
                            'floor_delta_m': delta, 'in_contact': planted}
                    if mode == 'walk_animation' and planted:
                        if old_foot.get('support_world_m') is None:
                            raise ValueError('animated admission lacks its actual planted-paw target')
                        foot.update(replay_paw_support(old, old_foot, root, frame_yaw))
                    elif mode != 'walk_animation':
                        foot.update(intentional_slide=mode == 'slide_no_animation', planted_gait_lock_applicable=False, static_supported_pose=False)
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
                               'moving': bool(velocity[i]>1e-8), 'motion_mode': mode,
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
                        'requested_speed_m_s': speed, 'actual_speed_m_s': float(np.median(velocity[velocity>1e-8])) if np.any(velocity>1e-8) else 0.,
                        'requested_motion_mode': requested_mode,
                        'motion_degradation': None,
                        'owner_quality_policy': {'authority': 'owner 2026-10-14: ability to generate questions first',
                            'paw_contact_m': .012, 'support_lock_m': .015, 'full_skin_floor_margin_changed': False},
                        'maximum_navmesh_horizontal_error_m': nav,
                        'minimum_certified_triangle_nonfloor_clearance_m': minimum,
                        'species_id': entry['declaration']['species_id'],
                        'motion_kind': 'animal_animated_walk' if mode == 'walk_animation' else 'animal_intentional_slide',
                        'motion_mode': mode, 'motion_marker': 'animated_walk' if mode == 'walk_animation' else 'intentional_slide_no_animation',
                        'full_skin_bounds_m': [np.asarray(bounds)[:, 0].min(0).tolist(), np.asarray(bounds)[:, 1].max(0).tolist()],
                        'unique_native_pose_meshes': len(meshes), 'failed_candidates': failures,
                        'native_gates_changed': True, 'quality_gates_changed': True,'truth_or_scorer_changed':False,
                        'production_visual_quality_policy': dict(OWNER_PRODUCTION_QUALITY),
                        'truth_and_answer_gates_changed': False}
            if curved_route is not None:
                evidence.update(yaw_rad_per_frame=frame_yaws.tolist(),native_swept_pose_bounds_m=bounds,
                    scene_motion_execution=curved_route.receipt,animation_playback_multiplier=route_speed/source_speed,
                    source_clip_root_speed_m_s=source_speed,
                    animation_distance_clock={'method':'actual cumulative horizontal native route distance / admitted source speed; pauses freeze phase',
                        'travelled_distance_m':float(route_distance[-1]),'maximum_scene_speed_m_s':float(velocity.max()),
                        'stationary_frame_count':int(np.count_nonzero(velocity<=1e-8)),
                        'native_route_kind':'reflected template' if reflected_motion else 'curved native proposal'},
                    animation_playback_multiplier_per_frame=(velocity/source_speed).tolist())
            write_json(directory / 'motion_geometry.json', evidence)
            return evidence
        except ValueError as error:
            if curved_route is not None and hasattr(curved_route,'report_failure'):curved_route.report_failure(str(error),p0)
            failures.append({'start_m': p0.tolist(), 'yaw_rad': yaw, 'speed_m_s': route_speed, 'mode': mode, 'reason': str(error)})
            with(directory/'motion_route_search.jsonl').open('a')as stream:
                stream.write(json.dumps({'candidate':len(failures),'wall_seconds':time.monotonic()-begun,**failures[-1]})+'\n')
    write_json(directory / 'motion_rejected.json', {'status': 'blocked', 'mode': mode,
               'reason': 'requested motion routes failed original physical gates; no animal removed or mode substituted', 'failed_candidates': failures})
    raise ValueError('final animal admission has no route passing retained answer gates and owner visual quality policy')
