"""Polygon membership and frozen escape-measurement regression checks."""
from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import Polygon, MultiPolygon, mapping
import shapely

from tools.acoustics.check_split_room_escape import exterior_scope, band, origins_for
from tools.acoustics.room_split_escape import ray_checks
from tools.rooms.room_selection.navigation import sample_navigation

pytestmark = pytest.mark.fast_unit


def test_noise_holes_filled_but_concavity_and_separate_parts_preserved():
    outer = Polygon([(0,0),(5,0),(5,1),(1,1),(1,5),(0,5)], [[(.2,.2),(.8,.2),(.8,.8),(.2,.8)]])
    other = Polygon([(7,0),(8,0),(8,1),(7,1)])
    scope, record = exterior_scope(mapping(MultiPolygon([outer, other])))
    assert record['internal_hole_count'] == 1
    assert shapely.contains_xy(scope,.5,.5)
    assert not shapely.contains_xy(scope,3,3)  # concave gap is still outside
    assert not shapely.contains_xy(scope,6,.5)  # components are not hull-joined
    assert shapely.contains_xy(scope,7.5,.5)
    assert record['added_area_m2'] == pytest.approx(.36)


def test_filled_hole_still_requires_native_navmesh_support():
    geo = Polygon([(0,0),(2,0),(2,2),(0,2)], [[(.5,.5),(1.5,.5),(1.5,1.5),(.5,1.5)]])
    scope, _ = exterior_scope(mapping(geo))
    class PathFinder:
        def snap_point(self,p):
            # Navmesh lacks support in a tiny furniture obstruction in a hole.
            return [np.nan]*3 if p[0]==.875 and p[2]==.875 else p
        def distance_to_closest_obstacle(self,p,r): return 1.
        def get_island(self,p): return 0
        def find_path(self,sp):
            sp.geodesic_distance=np.linalg.norm(sp.requested_end-sp.requested_start)
            return True
    p=dict(grid_step_m=.25,floor_height_separation_m=.3,nav_snap_horizontal_max_m=.12,clearance_query_radius_m=2.,nav_edge_max_geodesic_factor=1.5,clearance_min_m=.5)
    _,points,_,_,_=sample_navigation(PathFinder(),SimpleNamespace(ShortestPath=SimpleNamespace),scope,0.,p)
    assert len(points)==63
    assert any(.5<x<1.5 and .5<z<1.5 for x,y,z in points)
    assert not any(x==.875 and z==.875 for x,y,z in points)


def test_original_four_origin_roles_and_listener_height_retained():
    p=dict(camera_m=[0,1.5,0],source_1_m=[3,1.2,0],source_2_m=[3,1.2,2])
    origins=origins_for('room',p)
    assert [role for _,role,_ in origins]==['camera_m','source_1_m','source_2_m','camera_alt_m']
    np.testing.assert_array_equal(origins[-1][2],[.4,1.5,0])


def test_original_ray_origin_near_distance_range_and_unhit_count():
    calls=[]
    class Context:
        def trace_ray_first_hit(self,origin,direction,minimum,maximum):
            calls.append((origin,direction,minimum,maximum))
            return SimpleNamespace(hit=len(calls)%8!=0,distance=.2)
    result=ray_checks(Context(),[('r','source_1_m',[1,1.2,1])],np.array([[0,0,0],[2,3,4]]))
    assert len(calls)==512
    assert all(c[2]==.01 and c[3]==30. for c in calls)
    np.testing.assert_allclose(np.linalg.norm([c[1] for c in calls],axis=1),1.,atol=1e-15)
    assert result[0]['escape_count']==64
    assert result[0]['escape_fraction']==.125
    assert [r['direction_index'] for r in result[0]['escaped_rays']]==list(range(7,512,8))


def test_frozen_thresholds_include_exact_boundaries():
    assert band(.05)=='测试'
    assert band(np.nextafter(.05,1.))=='只训练'
    assert band(.15)=='只训练'
    assert band(np.nextafter(.15,1.))=='不用'
    with pytest.raises(ValueError):band(float('nan'))
