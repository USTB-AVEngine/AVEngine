import numpy as np
import pytest
from avengine.assets.roster_motion.episode import endpoint_route_from_plan,endpoint_route_proposals
from avengine.construction.joint import RoomContext,CameraPose,BodySpec
from avengine.rooms.walkable_map import WalkableMap,VERSION,SPECIES_RADII_M
from avengine.assets.roster_motion.demand import matches_distance_request

pytestmark=pytest.mark.fast_unit


def context():
    mask=np.ones((160,160),bool)
    metadata={'version':VERSION,'species_radii_m':SPECIES_RADII_M,'footprint_policy':'independent-radius-coverage-v2'}
    m=WalkableMap([-4.,-8.],.05,0.,{k:mask.copy()for k in SPECIES_RADII_M},metadata)
    room={'room_label':'unit','measured_floor_y_m':0.,
          'floor_polygon':{'type':'Polygon','coordinates':[[[-4,-8],[4,-8],[4,0],[-4,0],[-4,-8]]]},
          'placement':{'camera_m':[0.,1.25,-.5]}}
    c=RoomContext(room,walkable_map=m)
    c.points_m=np.array([[-.5,0,-6.5],[.5,0,-6.5],[-.5,0,-3.5],[.5,0,-3.5]])
    return c


def test_exact_endpoint_plan_does_not_retime_or_rotate():
    t=np.arange(901)/30;positions=np.column_stack((np.minimum(t,3.),np.zeros(901),np.full(901,-4.)))
    headings=np.zeros(901)
    r=endpoint_route_from_plan({'positions_m':positions,'headings_rad':headings})
    np.testing.assert_array_equal(r.positions,positions)
    np.testing.assert_allclose(r.clock,r.distance/1.2)
    np.testing.assert_allclose(r.yaws,-np.pi/2)
    assert not r.moving[-1]


@pytest.mark.parametrize('trend',['approaching','receding'])
def test_endpoints_solve_requested_distance_branch(trend):
    c=context();camera=CameraPose((0.,1.25,-.5),0.)
    request={'speed_m_s':1.,'distance_trend':trend,'minimum_distance_change_m':.5}
    diagnostics={}
    route=next(endpoint_route_proposals(c,camera,BodySpec(radius_m=.2,margin_m=0.),
        request,diagnostics=diagnostics))
    assert matches_distance_request(route.positions,np.arange(901)/30.,camera.position_m,request)
    assert diagnostics['candidate_pairs']>=4 and diagnostics['certified_pairs']==1
    assert route.receipt['route_kind']=='core_endpoint_plan'


def test_endpoint_pairs_do_not_bridge_disconnected_floor():
    c=context()
    for layer in c.walkable_map.layers.values():layer[60:70,:]=False
    # Both endpoints legal, but the entire map is divided by an impassable gap.
    c.points_m=np.array([[0.,0.,-6.5],[0.,0.,-3.5]])
    diagnostics={}
    assert list(endpoint_route_proposals(c,CameraPose((0.,1.25,-.5),0.),
        BodySpec(radius_m=.2,margin_m=0.),{'speed_m_s':1.,'distance_trend':'approaching'},
        diagnostics=diagnostics))==[]
    assert diagnostics['rejections'].get('no_grid_route')==1


def test_multiple_authored_bouts_reuse_endpoint_path_without_a_long_corridor():
    c=context();camera=CameraPose((0.,1.25,-.5),0.);times=np.arange(901)/30.
    distance=sum(np.clip(times-start,0.,3.) for start in (2.,12.,22.))
    route=next(endpoint_route_proposals(c,camera,BodySpec(radius_m=.2,margin_m=0.),
        {'speed_m_s':1.},distance=distance))
    np.testing.assert_allclose(route.positions[150],route.positions[360])
    np.testing.assert_allclose(route.positions[450],route.positions[60],atol=1e-6)
    assert route.distance[-1]>8.
    assert np.linalg.norm(route.positions[-1]-route.positions[0])<4.
