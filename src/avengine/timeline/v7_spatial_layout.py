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


def spacing_rejection(declaration, roots, others, camera_position=None):
    """Check actual proposed centres over the full route, including parked bodies."""
    roots = np.asarray(roots, float).reshape(-1, 3)
    if is_person(declaration)and camera_position is not None:
        clearance=float(np.linalg.norm((roots-np.asarray(camera_position))[:,[0,2]],axis=1).min())
        if clearance<1.5-1e-6:return {'reason':'person_camera_clearance_below_construction_1_5m','minimum_camera_distance_m':clearance,'required_distance_m':1.5}
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


def body_yaw(base_yaw,declaration,person_index=0,retry=0,*,seed=0,root=None,listener=None,room_center=None,other_people=(),furniture=()):
    if not is_person(declaration):return float(base_yaw)
    from avengine.timeline.v7_natural_orientation import sample_initial_orientation
    point=[0.,0.,0.]if root is None else root
    origin=[0.,0.,0.]if listener is None else listener
    return sample_initial_orientation(declaration,seed,point,origin,room_center,other_people,furniture,retry)['yaw_rad']


def separated_layout(layout, count, *, seed=None):
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
    if seed is None:
        import hashlib,json
        seed=int(hashlib.blake2b(json.dumps([layout['listener_m'],layout['room_floor_polygon']],sort_keys=True).encode(),digest_size=8).hexdigest(),16)
    centre=polygon.representative_point();room_center=[centre.x,floor,centre.y]
    while pool and len(selected) < count:
        def score(p):
            root = np.asarray(p['position_m'])
            separation = min((np.linalg.norm((root-np.asarray(q['position_m']))[[0,2]]) for q in selected),default=1.)
            return (min(separation,2.), -abs(np.linalg.norm((root-listener)[[0,2]])-3.))
        pool.sort(key=score,reverse=True)
        candidate = pool.pop(0);root=np.asarray(candidate['position_m'])
        if np.linalg.norm((root-listener)[[0,2]])<1.5-1e-6:continue
        if any(np.linalg.norm((root-np.asarray(p['position_m']))[[0,2]])<1. for p in selected):
            continue
        if len(selected)>=2:
            points=np.asarray([p['position_m'] for p in selected]+[candidate['position_m']])[:,[0,2]]
            if np.linalg.svd(points-points.mean(0),compute_uv=False)[1]<.25:continue
        delta=listener-root;base=math.atan2(-delta[0],-delta[2])
        from avengine.timeline.v7_natural_orientation import sample_initial_orientation,all_people_face_camera
        for retry in range(12):
            orientation=sample_initial_orientation({'kind':'human'},seed,root,listener,room_center,[{'position_m':p['position_m'],'yaw_rad':math.radians(p['body_yaw_deg'])}for p in selected],retry=retry)
            proposed=[{'position_m':p['position_m'],'yaw_rad':math.radians(p['body_yaw_deg'])}for p in selected]+[{'position_m':root,'yaw_rad':orientation['yaw_rad']}]
            if len(selected)+1<count or not all_people_face_camera(proposed,listener):break
        else:continue
        candidate['body_yaw_deg']=math.degrees(orientation['yaw_rad'])
        candidate['natural_orientation']=orientation
        candidate['status']='candidate_requires_original_native_admission'
        selected.append(candidate)
    if len(selected)<count:
        raise ValueError('room cannot supply requested count with >=1m spacing and dispersed centres; try another room region')
    return selected
