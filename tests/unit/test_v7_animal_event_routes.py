from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.assets.animal_motion.route_solver import delivery_safe_timing, delivery_speed_reports, find_animal_route
from avengine.assets.animal_motion.production_episode import animal_route_samples
from avengine.assets.roster_motion.straight_route import stand_walk_stand_distance

pytestmark = pytest.mark.fast_unit


def speed_sliver_route():
    s = stand_walk_stand_distance(11.2, 15.1, .9377326052, .3)
    points = np.column_stack((np.zeros(901), -3. - s))
    return {'positions_xz_m': points.tolist(), 'travel_yaws_rad': [-np.pi/2] * 901,
            'emission_slot': {'offset_s': 10., 'start_s': 11.7, 'end_s': 14., 'post_until_s': 14.6,
                              'leg': [336,453], 'window_frames': [336,453]},
            'short_window_walk': {'walk_start_s': 11.2, 'walk_end_s': 15.1,
                                 'silent_query_until_s': 14.6, 'emission_s': [11.7,14.]}}


def test_retime_complete_sound_and_walk_avoids_00560_slow_tail_crop():
    route = speed_sliver_route()
    assert delivery_speed_reports(route, 'dog')[3]['status'] == 'fail'
    fixed = delivery_safe_timing(route, 'dog')
    assert all(r['status'] == 'pass' for r in delivery_speed_reports(fixed, 'dog'))
    delta = fixed['animal_delivery_timing']['shift_s']
    assert delta != 0.
    assert fixed['emission_slot']['start_s'] == pytest.approx(11.7 + delta)
    assert fixed['short_window_walk']['walk_end_s'] == pytest.approx(15.1 + delta)
    np.testing.assert_allclose(fixed['positions_xz_m'][0], route['positions_xz_m'][0])
    np.testing.assert_allclose(fixed['positions_xz_m'][-1], route['positions_xz_m'][-1])
    assert route['emission_slot']['start_s'] == 11.7


def test_declared_root_held_animal_turn_keeps_gait_frozen_and_executes_heading():
    t = np.arange(901) / 30.
    roots = np.tile([0.,0.,-3.],(901,1))
    yaw = np.pi * np.clip((t - 3.) / 3., 0., 1.)
    route = SimpleNamespace(positions=roots,yaws=yaw,
        receipt={'animal_stationary_turn_intervals_s': [[3.,6.]]})
    actual, headings, distance, speed = animal_route_samples(t, roots[0], 0., [1,0,0], .6, route)
    np.testing.assert_array_equal(actual, roots)
    assert np.all(distance == 0.) and np.all(speed == 0.)
    assert headings[150] > .5 and headings[-1] == pytest.approx(np.pi)
    # Undeclared rest rotations remain frozen, preserving single-bout behavior.
    route.receipt = {}
    _, frozen, _, _ = animal_route_samples(t, roots[0], 0., [1,0,0], .6, route)
    assert np.all(frozen == 0.)


def test_stationary_turn_declaration_cannot_hide_a_translated_root():
    t = np.arange(901) / 30.
    roots = np.column_stack((t, np.zeros(901), np.zeros(901)))
    route = SimpleNamespace(positions=roots,yaws=np.zeros(901),
        receipt={'animal_stationary_turn_intervals_s': [[3.,6.]]})
    with pytest.raises(ValueError, match='translates'):
        animal_route_samples(t, roots[0], 0., [1,0,0], .6, route)
