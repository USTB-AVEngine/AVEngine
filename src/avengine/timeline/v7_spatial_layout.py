"""Automatic spacing, dispersion and orientation proposals for registered bodies."""
import math
import numpy as np


def is_person(declaration):
    return declaration.get('kind') in {'human', 'performer'}


def minimum_separation(first, second):
    if is_person(first) and is_person(second):
        return 1.
    if (is_person(first) and second.get('kind') == 'animal') or (is_person(second) and first.get('kind') == 'animal'):
        return .8
    if first.get('kind') == second.get('kind') == 'animal':
        return .65
    return 0.


def entry_roots(entry):
    roots = entry.get('motion_episode', {}).get('world_roots_m')
    if roots is not None:
        return np.asarray(roots, float)
    root=entry['resolved_root_m'] if 'resolved_root_m' in entry else np.asarray(entry['applied_root_input'])[:3,3]
    return np.asarray(root,float)[None,:]


def spacing_rejection(declaration, roots, others):
    """Check actual proposed centres over the full route, including parked bodies."""
    roots = np.asarray(roots, float).reshape(-1, 3)
    for other in others:
        if other.get('phase2_layout_qualified') is False:
            continue
        required = minimum_separation(declaration, other['declaration'])
        if not required:
            continue
        observed = entry_roots(other)
        if len(roots) != len(observed) and len(roots) != 1 and len(observed) != 1:
            raise ValueError('layout routes need the same native sampling clock')
        minimum = float(np.min(np.linalg.norm((roots-observed)[:, [0, 2]], axis=1)))
        if minimum < required-1e-6:
            return {'reason': 'registered_body_centre_spacing', 'other_endpoint_id': other['endpoint_id'],
                    'minimum_centre_distance_m': minimum, 'required_distance_m': required}
    return None


def noncollinear_rejection(declaration, root, others):
    if not is_person(declaration):
        return False
    people = [entry_roots(e)[0, [0, 2]] for e in others if is_person(e['declaration'])]
    if len(people) < 2:
        return False
    points = np.vstack([people, np.asarray(root)[[0, 2]]])
    _, singular, _ = np.linalg.svd(points-points.mean(0), full_matrices=False)
    # A third person establishes depth rather than extending the same row.
    return bool(singular[1] < .25)


def body_yaw(base_yaw, declaration, person_index, retry=0):
    if not is_person(declaration):
        return float(base_yaw)
    offset = (0., -25., 75., 155., -95.)[person_index % 5] + (0., -15., 15.)[retry % 3]
    return float(base_yaw)+math.radians(offset)


def separated_layout(layout, count):
    """Use all room regions as proposals; physical admission is still required."""
    from copy import deepcopy
    from shapely.geometry import shape, Point
    valid = deepcopy(layout['valid'])
    if not valid:
        raise ValueError('layout has no measured floor-height proposal')
    polygon = shape(layout['room_floor_polygon']).buffer(-.35)
    floor = float(np.median([p['position_m'][1] for p in valid]))
    x0, z0, x1, z1 = polygon.bounds
    pool = valid + [{'position_m':[float(x),floor,float(z)]} for x in np.arange(x0,x1,.4)
                  for z in np.arange(z0,z1,.4) if polygon.covers(Point(x,z))]
    selected = []
    listener = np.asarray(layout['listener_m'])
    while pool and len(selected) < count:
        def score(p):
            root = np.asarray(p['position_m'])
            separation = min((np.linalg.norm((root-np.asarray(q['position_m']))[[0,2]]) for q in selected),default=1.)
            return (min(separation,2.), -abs(np.linalg.norm((root-listener)[[0,2]])-3.))
        pool.sort(key=score,reverse=True)
        candidate = pool.pop(0);root=np.asarray(candidate['position_m'])
        if person_camera_rejection({'kind':'human'},root,listener):continue
        if any(np.linalg.norm((root-np.asarray(p['position_m']))[[0,2]])<1. for p in selected):
            continue
        if len(selected)>=2:
            points=np.asarray([p['position_m'] for p in selected]+[candidate['position_m']])[:,[0,2]]
            if np.linalg.svd(points-points.mean(0),compute_uv=False)[1]<.25:continue
        delta=listener-root;base=math.atan2(-delta[0],-delta[2])
        candidate['body_yaw_deg']=math.degrees(body_yaw(base,{'kind':'human'},len(selected)))
        candidate['status']='candidate_requires_original_native_admission'
        selected.append(candidate)
    if len(selected)<count:
        raise ValueError('room cannot supply requested count with >=1m spacing and dispersed centres; try another room region')
    return selected


def person_camera_rejection(declaration,roots,listener,minimum=1.5):
    if not is_person(declaration):return None
    values=np.asarray(roots,float).reshape(-1,3);origin=np.asarray(listener,float)
    distance=np.linalg.norm((values-origin)[:,[0,2]],axis=1);index=int(np.argmin(distance))
    if distance[index]<float(minimum):
        return {'reason':'person_too_close_to_camera','minimum_horizontal_distance_m':float(distance[index]),
                'required_distance_m':float(minimum),'sample_index':index}
    return None


def native_person_camera_clearance(visual,assets,minimum=1.2):
    people={a['asset_id']:a for a in assets if is_person(a)}
    if not people:return {'status':'pass','people':{},'minimum_required_m':minimum}
    frames=visual.get('frames',[]);cameras=visual.get('camera_frame_readbacks',[])
    expected=int(visual.get('frame_count',len(frames)))
    if not frames or len(frames)!=expected or len(cameras)!=expected:
        return {'status':'fail','reason':'missing complete native person/camera frames','minimum_required_m':minimum}
    camera_by_frame={c['frame_index']:c for c in cameras};measured={aid:float('inf')for aid in people};errors=[]
    for frame in frames:
        camera=camera_by_frame.get(frame['frame_index'])
        if camera is None:return {'status':'fail','reason':'missing matching native camera frame'}
        origin=np.asarray(camera['position_m'],float);seen=set()
        for actor in frame['actors']:
            aid=actor['asset_id']
            if aid not in people:continue
            seen.add(aid);root=np.asarray(actor['body_to_world_m'],float)[:3,3]
            if not np.isfinite(root).all() or not np.isfinite(origin).all():return {'status':'fail','reason':'nonfinite native person/camera position'}
            distance=float(np.linalg.norm((root-origin)[[0,2]]))
            if actor.get('native_emitter_present') and actor.get('emitter_position_m') is not None:
                distance=min(distance,float(np.linalg.norm((np.asarray(actor['emitter_position_m'])-origin)[[0,2]])))
            measured[aid]=min(measured[aid],distance)
        if seen!=set(people):return {'status':'fail','reason':'a required person lacks native frame readback'}
    for aid,distance in measured.items():
        if distance<minimum:errors.append({'asset_id':aid,'minimum_horizontal_distance_m':distance})
    return {'status':'fail' if errors else 'pass','people':measured,'errors':errors,
            'minimum_required_m':minimum,'native_frame_count':expected,
            'measurement':'every native frame; horizontal body-root and present native mouth distance from the fixed camera'}
