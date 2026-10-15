"""Claude 10-15: the executed straight walk is constructed directly instead of found by chance."""
import math

import numpy as np
import pytest
from shapely.geometry import Point, box

from avengine.assets.roster_motion.straight_route import find_straight_route, RADIAL_COS


class FakePathfinder:
    def __init__(self, poly):
        self.poly = poly

    def snap_point(self, p):
        return np.asarray(p, float)

    def distance_to_closest_obstacle(self, p, maximum):
        return min(float(maximum), float(self.poly.exterior.distance(Point(p[0], p[2]))))


def _room(w, d):
    poly = box(0., 0., w, d)
    return {'floor_polygon': poly.__geo_interface__, 'measured_floor_y_m': 0.,
            'placement': {'camera_m': [0.4, 1.25, 0.4]}}, FakePathfinder(poly)


@pytest.mark.parametrize('species,focus', [('human', 'M1'), ('human', 'M2'), ('dog', 'M1'), ('cat', 'M2')])
def test_direct_straight_line_serves_the_families(species, focus):
    room, pf = _room(6., 5.)
    route = find_straight_route(room, pf, 7, species=species, utterance_s=1.8, focus=focus)
    xz = np.asarray(route['positions_xz_m']); slot = route['emission_slot']; win = route['short_window_walk']
    d = np.diff(xz, axis=0); v = np.linalg.norm(d, axis=1) * 30
    heads = np.arctan2(d[v > 1e-9, 1], d[v > 1e-9, 0])
    assert np.ptp(np.unwrap(heads)) < 1e-9                                   # one straight line
    lo, hi = {'human': (.8, 1.4), 'dog': (.5, 1.2), 'cat': (.3, .8)}[species]
    assert lo <= float(v.max()) <= hi                                        # species speed band
    assert slot['qa06'] and (slot['qa17'] or slot['constructed_families'] == 'M2')
    assert slot['qa15'] or focus == 'M1'
    poly = box(0., 0., 6., 5.).buffer(-.35)
    assert all(poly.covers(Point(*p)) and np.hypot(p[0] - .4, p[1] - .4) >= 1.5 for p in xz)
    t = np.arange(len(xz)) / 30
    moving = v[(t[1:] >= slot['start_s']) & (t[1:] <= (slot['post_until_s'] or slot['end_s']))]
    assert moving.min() > .9 * lo                                            # moving through emission and silent query
    assert slot['offset_s'] <= slot['start_s'] - .5 and (slot['post_until_s'] or slot['end_s']) <= slot['offset_s'] + 9.8
    if focus == 'M2':
        k, e = round(slot['start_s'] * 30), round(slot['end_s'] * 30)
        r = np.linalg.norm(xz[k:e + 1] - np.array([.4, .4]), axis=1)
        assert abs(r[-1] - r[0]) >= .5 and slot['radial_cos'] >= RADIAL_COS


def test_too_small_room_raises_for_the_fallback():
    room, pf = _room(2.2, 2.2)
    with pytest.raises(ValueError, match='no free straight'):
        find_straight_route(room, pf, 1, species='human', utterance_s=1.8, focus='M1')


def test_shallow_room_falls_back_to_the_short_radial_pair():
    room, pf = _room(4.2, 4.2)
    route = find_straight_route(room, pf, 3, species='human', utterance_s=1.2, focus='M2')
    slot = route['emission_slot']
    assert slot['qa15'] and slot['qa06']
    assert slot['constructed_families'] in ('all4', 'M2')
