from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import Polygon, LineString, mapping, shape
from avengine.assets.animal_motion.native_route import ordered_native_animal_routes

pytestmark = pytest.mark.fast_unit


def fixture_scene():
    # A sub-micrometre polygon seam between two emitted root samples. The
    # independent native/map route remains valid; only GEOS covers differs.
    outline = Polygon([(-1,-1),(3,-1),(3,1),(-1,1)],
                      [[(.50000005,-.1),(.50000009,-.1),(.50000009,.1),(.50000005,.1)]])
    points = np.column_stack((np.linspace(0.,2.,901),np.zeros(901),np.zeros(901)))
    request = {'motion_quartet_route': {'positions_xz_m': points[:,[0,2]].tolist()}}
    scene = {'production_scenario': {'motion_floor_polygon': mapping(outline)}}
    return outline, points, request, scene


def test_numeric_outline_seam_does_not_discard_the_native_certified_plan(monkeypatch):
    from avengine.assets.roster_motion import reflect_navmesh
    outline, points, request, scene = fixture_scene()
    line = LineString(points[:,[0,2]])
    assert not outline.covers(line)
    def native_pool(video, entry, actual_scene, forward, actual_request):
        assert shape(actual_scene['production_scenario']['motion_floor_polygon']).covers(line)
        assert actual_request is request  # map/NavMesh policies are not replaced
        yield SimpleNamespace(positions=points,yaws=np.zeros(901),receipt={})
    monkeypatch.setattr(reflect_navmesh,'reflected_route_proposals',native_pool)
    route, = ordered_native_animal_routes(None,{},scene,[1,0,0],request)
    np.testing.assert_array_equal(route.positions,points)
    assert shape(scene['production_scenario']['motion_floor_polygon']).equals(outline)
    assert route.receipt['maximum_planned_xz_displacement_m'] == 0.
    assert 0. < route.receipt['original_outline_uncovered_route_length_m'] < 1e-6


def test_real_floor_gap_is_preserved(monkeypatch):
    from avengine.assets.roster_motion import reflect_navmesh
    _, points, request, scene = fixture_scene()
    outline = Polygon([(-1,-1),(3,-1),(3,1),(-1,1)],
                      [[(.49,-.1),(.51,-.1),(.51,.1),(.49,.1)]])
    scene['production_scenario']['motion_floor_polygon']=mapping(outline)
    def native_pool(video,entry,actual_scene,forward,actual_request):
        if shape(actual_scene['production_scenario']['motion_floor_polygon']).covers(LineString(points[:,[0,2]])):
            yield SimpleNamespace(positions=points,yaws=np.zeros(901),receipt={})
    monkeypatch.setattr(reflect_navmesh,'reflected_route_proposals',native_pool)
    assert list(ordered_native_animal_routes(None,{},scene,[1,0,0],request)) == []


def test_certified_route_precedes_a_more_clear_but_transformed_route(monkeypatch):
    from avengine.assets.roster_motion import reflect_navmesh
    _, points, request, scene = fixture_scene()
    changed=SimpleNamespace(positions=points+[0.,0.,.5],yaws=np.zeros(901),receipt={'native_obstacle_clearance_score_m':1.})
    original=SimpleNamespace(positions=points.copy(),yaws=np.zeros(901),receipt={'native_obstacle_clearance_score_m':.3})
    monkeypatch.setattr(reflect_navmesh,'reflected_route_proposals',lambda *args:iter([changed,original]))
    routes=list(ordered_native_animal_routes(None,{},scene,[1,0,0],request))
    assert routes == [original,changed]
    assert routes[1].receipt['maximum_planned_xz_displacement_m'] == pytest.approx(.5)
