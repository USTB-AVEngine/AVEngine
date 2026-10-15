"""Independent, certified human speech bouts on one reusable free corridor.

The route search remains the existing direct room/NavMesh constructor.  Only
its temporal use is changed: three complete walks alternate along the same
corridor, with slow turns during the intervening rests.  No truth is authored.
"""
from copy import deepcopy
import math

import numpy as np
from shapely.geometry import Point, shape

FPS = 30
EVENT_OFFSETS_S = (0., 10., 20.)


def find_human_seed_route(room, pathfinder, seed, attempt=0, *, utterance_s=1.5, speed_range=None, local_start_s=None, candidate_validator=None, cpu_deadline=None, post_motion_s=None):
    """Require all four families before selecting a camera/corridor.

    In the old constructor the short QA06/15 fallback at camera 0 prevented a
    later camera from solving QA16.  Here a failed short line cannot satisfy
    the declared post-sound request.
    """
    from .straight_route import _find_straight_route, _rooms_with_cameras
    errors = []
    polygon = shape(room['floor_polygon'])
    for index, candidate in enumerate(_rooms_with_cameras(room, pathfinder, 'human')):
        if cpu_deadline is not None:
            import time
            if time.process_time()>=cpu_deadline:raise TimeoutError('joint human corridor construction CPU budget exhausted')
        candidate = deepcopy(candidate)
        candidate['placement']['camera_m'][1] = float(room['measured_floor_y_m']) + 1.1
        camera = np.asarray(candidate['placement']['camera_m'], float)
        if Point(*camera[[0, 2]]).distance(polygon.boundary) < .5 - 1e-6:
            continue
        for use_map in (True, False):
            try:
                route = _find_straight_route(candidate, pathfinder, seed, attempt, species='human',
                    utterance_s=utterance_s, focus='M1', families='all4', speed_range=speed_range,
                    corners=False, use_map=use_map, minimum_camera_distance_m=2.6, require_reverse_radial=True, require_rest_in_frame=True, local_start_s=local_start_s,candidate_validator=candidate_validator,cpu_deadline=cpu_deadline,post_motion_s=post_motion_s)
            except ValueError as error:
                errors.append(str(error))
                continue
            if index:
                route['camera_relocation'] = {'reason': 'solve all four requested human motion families',
                    'room_camera_m': list(room['placement']['camera_m']),
                    'constructed_camera_m': route['listener_m'], 'candidate_index': index}
            route['minimum_camera_distance_m'] = 2.6
            route['human_motion_required_families'] = ['QA-06', 'QA-15', 'QA-16', 'QA-17']
            return route
    raise ValueError('no certified all-four human corridor: ' + ' | '.join(errors[:4]))


def repeat_human_bouts(route, offsets_s=EVENT_OFFSETS_S):
    """Reuse a straight corridor with a separate complete emission per bout."""
    result = deepcopy(route)
    slot = route['emission_slot']
    if not all(slot.get(key) for key in ('qa06', 'qa15', 'qa16', 'qa17')):
        raise ValueError('repeated human bouts require all four motion conditions')
    if not route['short_window_walk'].get('straight_line'):
        raise ValueError('reversing a corridor requires a straight seed route')
    original_offset = float(slot['offset_s'])
    t = np.arange(901) / FPS
    local_t = np.arange(301) / FPS
    xz = np.asarray(route['positions_xz_m'], float)
    segment = np.column_stack([np.interp(local_t + original_offset, t, xz[:, k]) for k in (0, 1)])
    heading = float(route['short_window_walk']['heading_rad'])
    offsets = [float(v) for v in offsets_s]
    if offsets != [0., 10., 20.]:
        raise ValueError('human bouts use the three complete nonoverlapping delivery windows')
    positions = np.empty((901, 2))
    yaws = np.empty(901)
    slots = []
    windows = []
    for index, offset in enumerate(offsets):
        reverse = bool(index % 2)
        points = segment[0] + segment[-1] - segment if reverse else segment
        first = round(offset * FPS)
        positions[first:first + 301] = points
        yaws[first:first + 301] = heading + index * math.pi
        event = deepcopy(slot)
        for key in ('start_s', 'end_s', 'post_until_s'):
            if event.get(key) is not None:
                event[key] = float(event[key]) - original_offset + offset
        event['offset_s'] = offset
        event['event_index'] = index
        event['branch'] = ('farther' if slot['branch'] == 'nearer' else 'nearer') if reverse else slot['branch']
        event['radial_change_m'] = (-1 if reverse else 1) * float(slot['radial_change_m'])
        for key in ('leg', 'window_frames'):
            event[key] = [round((v / FPS - original_offset + offset) * FPS) for v in slot[key]]
        slots.append(event)
        window = deepcopy(route['short_window_walk'])
        for key in ('walk_start_s', 'walk_end_s', 'silent_query_until_s'):
            window[key] = float(window[key]) - original_offset + offset
        window['emission_s'] = [event['start_s'], event['end_s']]
        windows.append(window)
    # Turn gradually while stopped; root translation and the distance-driven
    # gait clock are held.  No heading change contaminates a sounding interval.
    turns = []
    for index in (0, 1):
        begin = windows[index]['walk_end_s'] + .15
        end = windows[index + 1]['walk_start_s'] - .15
        if end - begin < 2.:
            raise ValueError('independent bouts need at least two seconds for a resting turn')
        mask = (t >= begin) & (t <= end)
        u = (t[mask] - begin) / (end - begin)
        yaws[mask] = heading + index * math.pi + math.pi * u * u * (3 - 2 * u)
        turns.append({'interval_s': [begin, end], 'turn_deg': 180., 'root_held': True,
                      'maximum_yaw_rate_deg_s': 270. / (end - begin)})
    result.update(positions_xz_m=positions.tolist(), travel_yaws_rad=yaws.tolist(),
        emission_slots=slots, emission_slot=slots[0], human_event_windows=windows,
        human_rest_turns=turns, require_rest_poses_in_frame=True,
        human_motion_event_count=3, prefer_planned_placement=True, construction='three_independent_human_corridor_bouts')
    result['short_window_walk'] = windows[0]
    result['candidate_trials'] = deepcopy(route['candidate_trials'])
    result['candidate_trials'].append({'construction': 'repeat_certified_corridor', 'independent_events': 3})
    return result
