"""Multi-event plans must pass the existing judge separately at every offset."""
from copy import deepcopy
import math

import numpy as np
import pytest

from avengine.assets.roster_motion.human_event_route import repeat_human_bouts
from avengine.assets.roster_motion.straight_route import stand_walk_stand_distance
from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row, verify_planned_trajectories

pytestmark = pytest.mark.fast_unit


def seed_route():
    s = stand_walk_stand_distance(1., 5.7, .9, .3)
    points = np.column_stack((np.zeros(901), -2. - s))
    slot = {'start_s': 1.5, 'end_s': 3.1, 'post_until_s': 5.2, 'offset_s': 0.,
        'branch': 'farther', 'radial_change_m': 1., 'qa06': True, 'qa15': True,
        'qa16': True, 'qa17': True, 'leg': [30, 171], 'window_frames': [30, 171]}
    return {'positions_xz_m': points.tolist(), 'travel_yaws_rad': [-math.pi/2]*901,
        'short_window_walk': {'walk_start_s': 1., 'walk_end_s': 5.7, 'silent_query_until_s': 5.2,
            'heading_rad': -math.pi/2, 'straight_line': True},
        'emission_slot': slot, 'speed_m_s': .9, 'speed_range_m_s': [.8, 1.4],
        'listener_m': [0., 1.25, 0.], 'candidate_trials': []}


def test_three_corridor_bouts_are_continuous_with_real_rest_turns():
    original = seed_route()
    route = repeat_human_bouts(original)
    p = np.asarray(route['positions_xz_m'])
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1)*30
    assert speed.max() <= .9 + 1e-9
    assert [s['branch'] for s in route['emission_slots']] == ['farther', 'nearer', 'farther']
    assert [s['start_s'] for s in route['emission_slots']] == [1.5, 11.5, 21.5]
    assert original == seed_route()
    for turn in route['human_rest_turns']:
        a, b = [math.ceil(x*30) for x in turn['interval_s']]
        assert np.max(np.linalg.norm(p[a:b] - p[a], axis=1)) < 1e-10
        assert turn['maximum_yaw_rate_deg_s'] < 70


def certification_row(route):
    assets = [{'asset_id': aid, 'kind': 'human', 'walk_allowed': True} for aid in ('walker', 'reply')]
    records = [{'sound_asset_id': sid, 'header': {'sample_count': n, 'sample_rate_hz': 16000},
        'path': '/read-only/' + sid + '.wav', 'screens': {'automatic_activity': {
            'activity_intervals_samples': [[100, n-100]]}}} for sid, n in [('speech', 25600), ('response', 24000)]]
    requests = []
    for slot in route['emission_slots']:
        for qa, branch in [('QA-06', 'moving'), ('QA-15', slot['branch']), ('QA-16', None), ('QA-17', 'yes')]:
            requests.append({'qa_id': qa, 'branch': branch, 'target_asset_id': 'walker', 'speed_m_s': .9,
                'speed_range_m_s': [.8, 1.4], 'minimum_motion_s': 1.5, 'end_hold_s': 4.4,
                'offsets_s': [slot['offset_s']], 'reserve_tail_s': .5, 'wet_tail_planning_bound_s': None,
                'minimum_distance_change_m': .5})
    row = {'asset_ids': ['walker', 'reply'], 'motion_requests': {'walker': {'action': 'walk'}},
        'proposed_sound_asset_ids': {'walker': 'speech', 'reply': 'response'},
        'motion_source_start_s': {'walker': 1.5, 'reply': 7.5}, 'camera_yaw_deg': 0.,
        'motion_quartet_template_revision': 'multi-test', 'motion_during_condition_requests': requests}
    return prepare_motion_during_row(row, assets, records)


def test_each_independent_bout_passes_original_four_family_verifier():
    route = repeat_human_bouts(seed_route())
    row = certification_row(route)
    p = np.asarray(route['positions_xz_m'])
    track = np.column_stack((p[:, 0], np.full(901, 1.5), p[:, 1]))
    tracks = {'walker': track, 'reply': np.tile([10., 1.5, -10.], (2, 1))}
    verdict = verify_planned_trajectories(row['motion_during_condition_plan'], tracks, [0., 1.25, 0.])
    assert verdict == {'QA-06': True, 'QA-15': True, 'QA-16': True, 'QA-17': True}
    # A later event cannot borrow the movement of the earlier one.
    tracks['walker'] = track.copy()
    tracks['walker'][600:] = track[600]
    bad = verify_planned_trajectories(row['motion_during_condition_plan'], tracks, [0., 1.25, 0.])
    assert not bad['QA-06'] and not bad['QA-15'] and not bad['QA-16'] and not bad['QA-17']


def test_short_during_sound_recipe_cannot_claim_post_sound_production():
    route = seed_route()
    route['emission_slot']['qa16'] = False
    with pytest.raises(ValueError, match='all four'):
        repeat_human_bouts(route)


def test_solved_three_bout_geometry_has_a_native_prefilter_placement():
    from types import SimpleNamespace
    from shapely.geometry import Point, box
    from avengine.assets.roster_motion.human_event_route import find_human_seed_route
    from avengine.assets.roster_motion.reflect_navmesh import reflected_route_proposals
    poly = box(0., 0., 8., 7.)
    class Pathfinder:
        def snap_point(self, p):return np.asarray(p, float)
        def distance_to_closest_obstacle(self, p, maximum):
            return min(maximum, poly.exterior.distance(Point(p[0], p[2])))
    pf = Pathfinder()
    room = {'floor_polygon': poly.__geo_interface__, 'measured_floor_y_m': 0.,
            'placement': {'camera_m': [.5, 1.25, .5]}}
    seed = find_human_seed_route(room, pf, 42, utterance_s=1.6)
    route = repeat_human_bouts(seed)
    assert route['listener_m'][1] == pytest.approx(1.1)
    points = np.asarray(route['positions_xz_m'])
    radii = np.linalg.norm(points - np.asarray(route['listener_m'])[[0, 2]], axis=1)
    assert radii.min() >= 2.6
    # The real native prefilter trims to the middle 60% for an audibility
    # margin.  Certifying only the original verifier's full span missed the
    # reverse bout in the first native canary (0.325m despite 0.551m full span).
    for slot in route['emission_slots']:
        a, b = round(slot['start_s']*30), round(slot['end_s']*30)
        trim = round(.2*(b-a));v = radii[a+trim:b-trim+1]
        assert abs(v[-1]-v[0]) >= .5
    entry = {'applied_root_input': np.eye(4), 'declaration': {'asset_id': 'walker', 'kind': 'human'}}
    video = SimpleNamespace(actors=[entry], pathfinder=pf)
    scene = {'listener_m': route['listener_m'], 'camera_yaw_knots': [{'yaw_deg': route['camera_yaw_deg']}],
             'production_scenario': {'motion_floor_polygon': poly.__geo_interface__}}
    proposal = next(reflected_route_proposals(video, entry, scene, [0., 0., -1.], {'motion_quartet_route': route}))
    assert proposal.receipt['native_admissible_route_pool'] > 0


def test_micrometre_outline_slit_does_not_move_a_certified_native_route():
    from types import SimpleNamespace
    from shapely.geometry import Point, box, LineString
    from avengine.assets.roster_motion.human_event_route import find_human_seed_route
    from avengine.assets.roster_motion.reflect_navmesh import reflected_route_proposals
    poly = box(0., 0., 8., 7.)
    class Pathfinder:
        def snap_point(self, p):return np.asarray(p, float)
        def distance_to_closest_obstacle(self, p, maximum):
            return min(maximum, poly.exterior.distance(Point(p[0], p[2])))
    pf = Pathfinder();room = {'floor_polygon': poly.__geo_interface__, 'measured_floor_y_m': 0.,
                             'placement': {'camera_m': [.5, 1.25, .5]}}
    route = repeat_human_bouts(find_human_seed_route(room, pf, 42, utterance_s=1.6))
    p = np.asarray(route['positions_xz_m']);k = round(route['emission_slots'][0]['start_s']*30)+10
    centre=(p[k]+p[k+1])/2
    outline = poly.difference(box(centre[0]-1e-7,centre[1]-1e-7,centre[0]+1e-7,centre[1]+1e-7))
    assert not outline.covers(LineString(p))
    entry={'applied_root_input':np.eye(4),'declaration':{'asset_id':'walker','kind':'human'}}
    video=SimpleNamespace(actors=[entry],pathfinder=pf)
    scene={'listener_m':route['listener_m'],'camera_yaw_knots':[{'yaw_deg':route['camera_yaw_deg']}],
           'production_scenario':{'motion_floor_polygon':outline.__geo_interface__}}
    first=next(reflected_route_proposals(video,entry,scene,[0.,0.,-1.],{'motion_quartet_route':route}))
    assert first.receipt['rigid_route_rotation_rad']==0.
    assert first.receipt['rigid_route_translation_xz_m']==[0.,0.]
    assert entry['reflection_native_prefilter']['certified_planned_placement_preserved']
    # An actual wider floor gap stays outside the numerical outline tolerance.
    wide=poly.difference(box(centre[0]-.01,centre[1]-.01,centre[0]+.01,centre[1]+.01))
    assert not wide.buffer(.002).covers(LineString(p))
