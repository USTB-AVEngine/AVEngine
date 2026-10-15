"""Animal motion obligations on the joint core room and route services.

Search all cameras for the requested four-family corridor before accepting a
shorter partial-family fallback. Species maps and the original verifier remain
authoritative; no separate navigation or geometric engine is implemented here.
"""
from copy import deepcopy
import numpy as np
from shapely.geometry import Point, shape


def delivery_speed_reports(route, species):
    """Use the production ruler on all five planned 15 Hz delivery crops."""
    from avengine.timeline.v7_export_scene_policy import measured_actor_metrics
    xz = np.asarray(route['positions_xz_m'], float)
    roots = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
    asset = 'planned_animal'
    category = 'cat' if species == 'slide' else species
    reports = []
    for offset in (0, 5, 10, 15, 20):
        samples = roots[offset * 30:offset * 30 + 300:2]
        actor = {asset: {'asset_id': asset, 'species_id': category, 'root_positions_m': samples}}
        metrics = measured_actor_metrics(actor, 15., {asset: {'intentional_slide': species == 'slide'}})
        reports.append({'offset_s': offset, **metrics[0]})
    return reports


def delivery_safe_timing(route, species):
    """Shift a complete bout, rather than expose only a slow ramp in a crop."""
    times = np.arange(901) / 30.
    source = np.asarray(route['positions_xz_m'], float)
    headings = np.unwrap(np.asarray(route['travel_yaws_rad'], float))
    offset = float(route['emission_slot']['offset_s'])
    window = route['short_window_walk']
    # A finite time-domain construction. Sound and geometry move together;
    # speeds, endpoints and the complete distance-driven gait are preserved.
    shifts = sorted((k / 10. for k in range(-7, 8)), key=lambda d: (abs(d), d))
    for delta in shifts:
        if window['walk_start_s'] + delta < offset + .05 or window['walk_end_s'] + delta > offset + 9.5:
            continue
        candidate = deepcopy(route)
        candidate['positions_xz_m'] = np.column_stack([
            np.interp(times - delta, times, source[:, axis]) for axis in range(2)]).tolist()
        candidate['travel_yaws_rad'] = np.interp(times - delta, times, headings).tolist()
        slot = candidate['emission_slot']
        for key in ('start_s', 'end_s', 'post_until_s'):
            if slot.get(key) is not None:
                slot[key] = float(slot[key]) + delta
        for key in ('leg', 'window_frames'):
            if slot.get(key) is not None:
                slot[key] = [int(value) + round(delta * 30) for value in slot[key]]
        actual = candidate['short_window_walk']
        for key in ('walk_start_s', 'walk_end_s', 'silent_query_until_s'):
            actual[key] = float(actual[key]) + delta
        actual['emission_s'] = [slot['start_s'], slot['end_s']]
        reports = delivery_speed_reports(candidate, species)
        if all(item['status'] == 'pass' for item in reports):
            candidate['animal_delivery_timing'] = {
                'shift_s': delta, 'reason': 'every crop retains legal moving-root median speed; sound and path shifted together',
                'speed_thresholds_changed': False, 'window_checks': reports}
            return candidate
    raise ValueError('no crop-safe animal bout timing within the complete 10s window')


def find_animal_route(room, pathfinder, seed, attempt=0, *, species='dog', utterance_s=2.3,
                      focus='M1', speed_range=None, source_asset=None):
    """Solve a radial corridor for a predetermined answer and fixed observer.

    Geometry, candidate cameras, complete map segments and the distance clock
    come from joint_core. The family chooses its radial obligation and sound
    timing. A missing finite-domain solution is returned before rendering.
    """
    import math
    import time
    from collections import Counter
    from dataclasses import asdict
    from avengine.construction.joint import BodySpec, CameraPose, ScenePlan
    from avengine.construction.joint.context import build_room_context, camera_legal
    from avengine.construction.joint.geometry import framing
    from avengine.construction.joint.routes import construct_route, SPEED_BANDS, ConstructionError
    from avengine.construction.joint.certification import certify_plan
    from avengine.assets.roster_motion.straight_route import _radial_flags

    if species not in {'dog', 'cat', 'slide'}:
        raise ValueError('animal route requires dog, cat or slide')
    lo, hi = SPEED_BANDS[species]
    duration = math.ceil(float(utterance_s) * 10 - 1e-9) / 10
    # Both emission directions must retain 0.5m in the middle 60 percent,
    # with headroom for actual muzzle height and audio activity boundaries.
    speed = max(lo + .025, .58 / (.6 * duration * .9))
    if speed > hi - .01:
        raise ValueError('animal complete call is too short for a robust radial question at the owner speed band')
    start, wet_tail, post, lead, ramp = 1.3, .5, 1.8, .5, .3
    end = start + duration
    tail = end + wet_tail + post
    walk_start, walk_end = start - lead, tail + lead
    length = speed * (walk_end - walk_start - ramp)
    if walk_end > 9.4:
        raise ValueError('animal complete call and silent query do not fit one delivery window')
    # Fix the desired answer before any geometry search.
    branch = 'nearer' if (int(seed) + int(attempt)) % 2 == 0 else 'farther'
    body = BodySpec.for_species(species)
    dimensions = (source_asset or {}).get('dimensions_m')
    if isinstance(dimensions, (list, tuple)) and len(dimensions) == 3:
        dimensions = np.asarray(dimensions, float)
        if np.isfinite(dimensions).all() and np.min(dimensions) > 0:
            body = BodySpec.for_species(species,
                radius_m=max(body.radius_m, float(min(dimensions[0], dimensions[2])) / 2),
                height_m=max(body.height_m, float(dimensions[1])))
    rejected = Counter()
    tested = 0
    begun = time.process_time()
    floor = float(room['measured_floor_y_m'])
    with build_room_context(room, pathfinder=pathfinder, camera_limit=36, grid_step_m=.25,
                            camera_heights_m=(1.1,), width=1280, height=720) as context:
        available = context.walkable_map.contains_many(context.points_m[:, [0, 2]], species)
        points = context.points_m[available]
        for anchor_camera in context.cameras:
            origin = np.asarray(anchor_camera.position_m)
            distances = np.linalg.norm(points[:, [0, 2]] - origin[[0, 2]], axis=1)
            indices = np.flatnonzero((distances >= 2.6) & (distances <= 4.5))
            indices = sorted(indices, key=lambda index: (float(distances[index]), float(points[index, 0]), float(points[index, 2])))
            for index in indices:
                if time.process_time() - begun > 120.:
                    raise ValueError('animal joint route construction CPU budget exhausted: ' + str(dict(rejected)))
                near = points[index].copy()
                near[1] = floor
                direction = near[[0, 2]] - origin[[0, 2]]
                direction /= np.linalg.norm(direction)
                far = near.copy()
                far[[0, 2]] += length * direction
                if not context.walkable_map.contains_many(np.array([near[[0, 2]], far[[0, 2]]]), species).all():
                    rejected['species_endpoints'] += 1
                    continue
                if not context.walkable_map.segment_free(near[[0, 2]], far[[0, 2]], species):
                    rejected['species_full_segment'] += 1
                    continue
                camera = CameraPose(tuple(origin), math.degrees(math.atan2(direction[0], -direction[1])),
                                    width=1280, height=720)
                if not camera_legal(room, camera)[0]:
                    rejected['camera_inward'] += 1
                    continue
                if any(framing(camera, point, body)[0] != 'inside' for point in (near, far)):
                    rejected['complete_rest_body_frame'] += 1
                    continue
                waypoints = [far, near] if branch == 'nearer' else [near, far]
                tested += 1
                try:
                    actor = construct_route(context, waypoints, species=species, speed_m_s=speed,
                        start_s=walk_start, camera=camera, entity_id='planned_animal', body=body)
                    if actor.metadata['validation']['max_snap_horizontal_m'] > .002:
                        raise ConstructionError('original_2mm_native_snap')
                    certificate = certify_plan(context, ScenePlan(camera, {'planned_animal': actor}))
                    if not certificate.passed:
                        rejected.update(check.name for check in certificate.checks if check.status != 'pass')
                        continue
                except ConstructionError as error:
                    rejected[error.reason] += 1
                    continue
                xz = actor.positions_m[:, [0, 2]]
                k, e, q = (round(value * 30) for value in (start, end, tail))
                forward = _radial_flags(xz, origin[[0, 2]], k, e, q)
                reverse = _radial_flags(xz[0] + xz[-1] - xz, origin[[0, 2]], k, e, q)
                if not all([forward[0], forward[1], reverse[0], reverse[1]]):
                    rejected['both_radial_answer_directions'] += 1
                    continue
                frames = [round(walk_start * 30), round(walk_end * 30)]
                slot = {'start_s': start, 'end_s': end, 'post_until_s': tail, 'offset_s': 0.,
                    'leg': frames, 'window_frames': frames, 'branch': branch,
                    'radial_change_m': forward[2], 'radial_cos': forward[3],
                    'qa06': True, 'qa15': True, 'qa16': True, 'qa17': True,
                    'constructed_families': 'all4', 'kind': 'radial_emission'}
                window = {'walk_start_s': walk_start, 'walk_end_s': walk_end, 'ramp_s': ramp,
                    'speed_m_s': speed, 'length_m': length, 'emission_s': [start, end],
                    'silent_query_until_s': tail, 'heading_rad': float(actor.headings_rad[0]),
                    'straight_line': True, 'shape': None, 'reason': 'joint core radial corridor for the requested answer'}
                route = {'reflect_walk': True, 'positions_xz_m': xz.tolist(),
                    'travel_yaws_rad': actor.headings_rad.tolist(), 'listener_m': list(camera.position_m),
                    'camera_yaw_deg': camera.yaw_deg, 'species': species, 'template': focus,
                    'speed_m_s': speed, 'speed_range_m_s': [lo, hi], 'emission_slot': slot,
                    'short_window_walk': window, 'walkable_map_path': str(context.walkable_map.path),
                    'walkable_map_version': context.walkable_map.metadata['version'],
                    'walkable_map_room_id': context.walkable_map.metadata['room_id'], 'walkable_map_bypassed': False,
                    'candidate_count': tested, 'candidate_trials': [{'rejections': dict(rejected), 'construction': 'joint_core_radial_obligation'}],
                    'selected_attempt': 'animal-joint-' + str(seed) + '-' + str(attempt),
                    'candidate_native_root_samples': 901, 'require_rest_poses_in_frame': True,
                    'minimum_camera_distance_m': 2.6, 'animal_native_preflight_required': True,
                    'animal_required_families': ['QA-06', 'QA-15', 'QA-16', 'QA-17'],
                    'joint_core_certificate': asdict(certificate), 'animal_body_spec': asdict(body),
                    'construction': 'joint_core_radial_animal_corridor',
                    'geometry_status': 'joint map/NavMesh/body-framing certified; original event verifier and native pixels still required'}
                return delivery_safe_timing(route, species)
    raise ValueError('animal joint radial candidate domain exhausted: ' + str(dict(rejected)))
