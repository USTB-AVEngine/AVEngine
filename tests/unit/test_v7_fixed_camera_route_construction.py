from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import box,mapping,LineString
from avengine.capture.v7_camera_path_proposals import fixed_camera_route_proposals,matches_camera_path
pytestmark=pytest.mark.fast_unit


class Pathfinder:
    def snap_point(self,p):return np.asarray(p,float)
    def find_path(self,path):path.geodesic_distance=float(np.linalg.norm(path.requested_end-path.requested_start));return True
    def distance_to_closest_obstacle(self,p,r):return 1.


def test_entry_search_constructs_many_straight_native_routes_with_an_unchanged_fixed_camera():
    root=np.eye(4);root[:3,3]=[3,0,-1]
    entry={'declaration':{'kind':'human','asset_id':'walker'},'applied_root_input':root}
    scene={'listener_m':[0,1.3,0],'production_scenario':{'motion_floor_polygon':mapping(box(-5,-7,5,4))}}
    video=SimpleNamespace(actors=[entry],pathfinder=Pathfinder(),hs=SimpleNamespace(ShortestPath=SimpleNamespace))
    request={'minimum_listener_horizontal_distance_m':1.5,'route_ranking_candidate_budget':24,
        'camera_path_condition':{'kind':'enter','camera_forward':[0,0,-1],'required_entry_side':1,
        'initial_outside_minimum_abs_deg':60.,'final_inside_maximum_abs_deg':28.}}
    distance=np.linspace(0,4,151);routes=list(fixed_camera_route_proposals(video,entry,scene,distance,np.array([0,0,-1]),request))
    assert len(routes)==24
    for route in routes:
        assert matches_camera_path(route.positions,np.arange(151)/30,scene['listener_m'],request)
        assert box(-5,-7,5,4).covers(LineString(route.positions[:,[0,2]]))
        np.testing.assert_allclose(np.linalg.norm(np.diff(route.positions,axis=0),axis=1),np.diff(distance))
        assert route.receipt['camera_changed']is False
    assert scene['listener_m']==[0,1.3,0]


def test_no_route_proposal_never_becomes_a_static_actor():
    root=np.eye(4);entry={'declaration':{'kind':'human','asset_id':'walker'},'applied_root_input':root}
    scene={'listener_m':[0,1.3,0],'production_scenario':{'motion_floor_polygon':mapping(box(-.2,-.2,.2,.2))}}
    video=SimpleNamespace(actors=[entry],pathfinder=Pathfinder(),hs=SimpleNamespace(ShortestPath=SimpleNamespace))
    assert list(fixed_camera_route_proposals(video,entry,scene,np.linspace(0,4,151),np.array([0,0,-1]),{'camera_path_condition':{'kind':'enter'}}))==[]
