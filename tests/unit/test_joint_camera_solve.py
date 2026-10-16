"""Independent projection witnesses for continuous camera roles (no renderer)."""
from dataclasses import replace
import math
import numpy as np
import pytest
from avengine.rooms.walkable_map import WalkableMap, VERSION, SPECIES_RADII_M
from avengine.construction.joint.context import RoomContext, camera_legal
from avengine.construction.joint.types import BodySpec, CameraPose
from avengine.construction.joint.geometry import body_box, camera_coordinates, framing, MeshRaycaster
from avengine.construction.joint.camera_solve import (
    PointRole, RouteRole, roles_for_endpoints, solve_cameras, role_points)

pytestmark = pytest.mark.fast_unit


class FlatNav:
    def snap_point(self, p):
        return np.asarray(p)


def room_context(*, heights=(1.1, 1.25, 1.6), single=False, ray=None):
    mask = np.ones((240, 240), bool)
    meta = {"version": VERSION, "species_radii_m": SPECIES_RADII_M,
            "footprint_policy": "independent-radius-coverage-v2", "room_id": "camera_fixture"}
    m = WalkableMap([-6., -8.], .05, 0., {k: mask.copy() for k in SPECIES_RADII_M}, meta)
    room = {"room_label": "camera_fixture", "floor_polygon": {"type": "Polygon", "coordinates": [
        [[-6, -8], [6, -8], [6, 4], [-6, 4], [-6, -8]]]},
        "measured_floor_y_m": 0., "placement": {"camera_m": [0., 1.25, 0.]}}
    c = RoomContext(room, walkable_map=m, pathfinder=FlatNav(), raycaster=ray,
                    camera_heights_m=heights, grid_step_m=.5)
    if single:
        c.points_m = np.array([[0., 0., 0.]])
    return c


@pytest.mark.parametrize("side,sign", [("left", -1), ("right", 1)])
def test_entry_has_full_body_outside_correct_side_then_full_inside(side, sign):
    c = room_context(single=True)
    a, b = (sign*4., 0., -4.), (0., 0., -4.)
    roles = roles_for_endpoints("enter_"+side, a, b, pixel_margin_px=3.)
    result = c.solve_cameras(roles, limit=3)
    assert result.cameras
    for solution in result.solutions:
        camera = solution.camera
        xyz = camera_coordinates(camera, body_box(a, BodySpec()))
        # Independent half-plane test, including every body-box corner.
        assert np.all(sign*xyz[:, 0] - xyz[:, 2]*(camera.width/2+3)/camera.focal_px > 0)
        assert framing(camera, b)[0] == "inside"
        assert framing(camera, b)[1] >= 3
        assert solution.yaw_margin_deg > 0
        assert camera_legal(c.room, camera)[0]


@pytest.mark.parametrize("kind,a,b", [
    ("approach", (0,0,-6), (0,0,-3.5)), ("recede", (0,0,-3.5), (0,0,-6))])
def test_distance_motion_and_vertical_framing(kind, a, b):
    c = room_context(single=True)
    result = c.solve_cameras(roles_for_endpoints(kind, a, b, min_distance_change_m=2.))
    assert result.cameras
    for s in result.solutions:
        assert all(framing(s.camera, p)[0] == "inside" for p in (a,b))
        assert s.role_margins["route"]["distance_change_m"] >= 2
        assert s.yaw_margin_deg > 0


def test_height_search_recovers_feet_cropping_and_keeps_default_table():
    c = room_context(heights=(1.1, 1.6), single=True)
    old = c.cameras
    result = c.solve_cameras([PointRole("inside_full_body", (0,0,-3.))])
    assert result.cameras
    assert {s.camera.position_m[1] for s in result.solutions} == {1.1}
    assert c.cameras is old
    assert any("angular_empty" in count for count in result.diagnostics["role_rejections"].values())


def test_position_domain_exceeds_legacy_24_and_limit_is_only_return_limit():
    c = room_context(heights=(1.25,))
    result = c.solve_cameras([PointRole("inside_full_body", (0,0,-3.))], limit=2)
    assert len(c.cameras) == 24
    assert result.diagnostics["legal_positions"] > len(c.cameras)
    assert result.diagnostics["feasible_components"] > len(c.cameras)
    assert len(result.cameras) == 2
    assert result.solutions[0].yaw_margin_deg >= result.solutions[1].yaw_margin_deg
    cached = c._camera_solve_cache["positions"]
    c.solve_cameras([], limit=0)
    assert c._camera_solve_cache["positions"] is cached


def test_distinct_sectors_and_wraparound_continuous_yaw():
    c = room_context(single=True, heights=(1.25,))
    yaw = 179.12345
    camera = CameraPose((0,1.25,0), yaw)
    p = []
    for angle in (-20., 20.):
        v = 5*(math.cos(math.radians(angle))*camera.forward + math.sin(math.radians(angle))*camera.right)
        p.append((v[0], 0., v[2]))
    roles = [PointRole("azimuth_sector", p[0], azimuth_deg=(-21., -19.)),
             PointRole("azimuth_sector", p[1], azimuth_deg=(19., 21.)),
             PointRole("inside_full_body", p[0]), PointRole("inside_full_body", p[1])]
    result = c.solve_cameras(roles)
    assert result.cameras and result.cameras[0].yaw_deg == pytest.approx(yaw)
    for role, point in zip(roles, p):
        xyz = camera_coordinates(result.cameras[0], [point])[0]
        angle = math.degrees(math.atan2(xyz[0], xyz[2]))
        assert role.azimuth_deg[0] <= angle <= role.azimuth_deg[1]
    # A wraparound azimuth sector itself, and whole-body behind camera.
    wrap = c.solve_cameras([PointRole("azimuth_sector", (0,0,-5), azimuth_deg=(170., -170.)),
                           PointRole("outside_full_body", (0,0,-5))])
    assert wrap.cameras and abs(abs(wrap.cameras[0].yaw_deg)-180) < 1e-6


def test_empty_has_per_role_distance_and_angular_rejection_counts():
    c = room_context(single=True)
    fail = c.solve_cameras([PointRole("distance_range", (0,0,-2), name="far", distance_m=(3.,4.))])
    assert not fail.cameras
    assert fail.diagnostics["role_rejections"]["far"]["distance"] == 3
    fail = c.solve_cameras([PointRole("inside_full_body", (0,0,-4), name="in"),
                           PointRole("outside_full_body", (0,0,-4), name="out")])
    assert not fail.cameras
    assert sum(fail.diagnostics["role_rejections"]["out"].values()) == 3
    assert not fail.diagnostics["continuous_room_infeasibility_proved"]


def test_route_standoff_checks_segment_not_just_endpoints():
    c = room_context(single=True)
    roles = [RouteRole(((-3,0,0), (3,0,0)), "outside_full_body", "outside_full_body")]
    result = c.solve_cameras(roles)
    assert not result.cameras
    assert result.diagnostics["role_rejections"]["route"]["route_camera_distance"] == 3


def test_actual_box_occlusion_identity_and_unknown_geometry():
    import trimesh
    mesh = trimesh.creation.box((3., 4., .2))
    mesh.apply_translation([0, 1., -2.])
    ray = MeshRaycaster(mesh, face_ids=["box"]*len(mesh.faces))
    c = room_context(single=True, ray=ray)
    role = PointRole("occluded_by", (0,0,-4), occluder_id="box")
    good = c.solve_cameras([role, PointRole("inside_full_body", role.point_m)])
    assert good.cameras
    assert good.solutions[0].role_margins["0:occluded_by"]["blocked_fraction"] == 1
    assert not c.solve_cameras([replace(role, occluder_id="wrong")]).cameras
    assert not c.solve_cameras([PointRole("line_of_sight_clear", role.point_m)]).cameras
    assert c.solve_cameras([PointRole("line_of_sight_clear", (4,0,-4))]).cameras
    c.raycaster = None
    missing = c.solve_cameras([role])
    assert not missing.cameras
    assert missing.diagnostics["role_rejections"]["0:occluded_by"]["missing_visual_geometry"] == 3


def test_through_occlusion_helper_checks_clear_endpoints_and_hidden_waypoint():
    import trimesh
    mesh = trimesh.creation.box((2., 4., .2)); mesh.apply_translation([0,1.,-2.])
    c = room_context(single=True, ray=MeshRaycaster(mesh, face_ids=["box"]*len(mesh.faces)))
    roles = roles_for_endpoints("through_occlusion", (-4,0,-6), (4,0,-6),
                                hidden_point_m=(0,0,-6), occluder_id="box")
    assert c.solve_cameras(roles).cameras
    with pytest.raises(ValueError, match="hidden waypoint"):
        roles_for_endpoints("through_occlusion", (-4,0,-6), (4,0,-6))


def test_reverse_query_matches_independent_framing_with_requested_margin():
    c = room_context(heights=(1.25,))
    camera = CameraPose((0,1.25,0), 23.456)
    query = c.role_points(camera, PointRole("inside_full_body", (0,0,0), pixel_margin_px=2.))
    expected = np.array([framing(camera,p)[0] == "inside" and framing(camera,p)[1] >= 2.
                         and np.linalg.norm((p-np.asarray(camera.position_m))[[0,2]]) >= 1.5
                         for p in c.points_m])
    np.testing.assert_array_equal(query.mask, expected)


def test_inward_intervals_handle_hole_and_no_heading_quantization():
    c = room_context(single=True, heights=(1.25,))
    # Camera at x=0 has .5m clearance from a hole east of it.
    c.room["floor_polygon"]["coordinates"].append([[.5,-.4], [1.5,-.4], [1.5,.4], [.5,.4], [.5,-.4]])
    result = c.solve_cameras([])
    assert result.cameras
    assert all(camera_legal(c.room, s.camera)[0] for s in result.solutions)
    assert not any(40 < s.camera.yaw_deg < 140 for s in result.solutions)


def test_heading_range_uses_same_supplied_all_heading_body_envelope():
    c = room_context(single=True)
    a = PointRole("inside_full_body", (0,0,-4))
    assert c.solve_cameras([a]).cameras == c.solve_cameras([replace(a, heading_range_deg=(-179,179))]).cameras


def test_interval_domains_agree_with_direct_planes_for_arbitrary_yaws():
    from avengine.construction.joint.camera_solve import _point_intervals, _angular_slack
    rng = np.random.default_rng(426)
    # Numerical regression only: the production solver never samples yaw.
    for _ in range(30):
        position=np.array([rng.uniform(-2,2),rng.uniform(1.1,1.6),rng.uniform(-2,2)])
        point=np.array([rng.uniform(-5,5),0.,rng.uniform(-6,2)])
        camera=CameraPose(tuple(position),rng.uniform(-180,180))
        xyz=camera_coordinates(camera,body_box(point,BodySpec()))
        x,y,z=xyz.T
        th,tv=camera.width/(2*camera.focal_px),camera.height/(2*camera.focal_px)
        planes=np.column_stack((z-.01,th*z-x,th*z+x,tv*z-y,tv*z+y))
        expected={'inside_full_body':bool(np.all(planes>=0)),
                  'outside_full_body':bool(np.any(np.all(planes<0,axis=0))),
                  'outside_left':bool(np.all(planes[:,2]<0) and np.all(x<0)),
                  'outside_right':bool(np.all(planes[:,1]<0) and np.all(x>0))}
        for kind,good in expected.items():
            role=PointRole(kind,tuple(point))
            allowed=_point_intervals(role,position,camera)
            assert (_angular_slack(allowed,math.radians(camera.yaw_deg))>=0)==good


def test_legacy_camera_check_uses_identical_roles_and_rejects_illegal_camera():
    from avengine.construction.joint.camera_solve import check_camera_roles
    c=room_context(single=True)
    roles=roles_for_endpoints('approach',(0,0,-6),(0,0,-4),min_distance_change_m=1.)
    solved=c.solve_cameras(roles,limit=1)
    assert check_camera_roles(c,solved.cameras[0],roles)['passed']
    bad=replace(solved.cameras[0],position_m=(5.9,1.25,0))
    report=check_camera_roles(c,bad,roles)
    assert not report['passed'] and ('camera','camera_legal') in report['failures']


def test_explicit_height_generator_preserves_legacy_and_solver_height_sets():
    c=room_context(heights=(x for x in [1.1,1.25]),single=True)
    assert c.cameras and c.camera_heights_m == (1.1,1.25)
    assert c.solve_cameras([PointRole('inside_full_body',(0,0,-5))]).cameras


def test_rear_right_point_is_not_mislabeled_outside_left():
    from avengine.construction.joint.camera_solve import check_camera_roles
    c=room_context(single=True)
    camera=CameraPose((0,1.25,0),0.)
    p=(2.,0.,5.)
    assert check_camera_roles(c,camera,[PointRole('outside_full_body',p)])['passed']
    assert check_camera_roles(c,camera,[PointRole('outside_right',p)])['passed']
    assert not check_camera_roles(c,camera,[PointRole('outside_left',p)])['passed']


def test_outside_requires_strict_separation_not_edge_touch():
    from avengine.construction.joint.camera_solve import check_camera_roles
    c=room_context(single=True)
    camera=CameraPose((0,1.25,0),0.)
    body=BodySpec(margin_m=0.)
    th=camera.width/(2*camera.focal_px)
    # The nearest-left corner touches the right clip plane exactly.
    root=(4*th+body.radius_m*(1+th),0.,-4.)
    xyz=camera_coordinates(camera,body_box(root,body))
    assert abs(np.min(xyz[:,0]-th*xyz[:,2])) < 1e-12
    report=check_camera_roles(c,camera,[PointRole('outside_right',root,body)])
    assert not report['passed']


def test_route_endpoint_accepts_parameterized_point_role():
    from avengine.construction.joint.camera_solve import check_camera_roles
    c=room_context(single=True)
    roles=[RouteRole(((-1.,0.,-5.),(1.,0.,-5.)),
                     start_role=PointRole('azimuth_sector',(0,0,0),azimuth_deg=(-20.,-10.)))]
    result=c.solve_cameras(roles,limit=1)
    assert result.cameras and check_camera_roles(c,result.cameras[0],roles)['passed']
    xyz=camera_coordinates(result.cameras[0],[roles[0].points_m[0]])[0]
    assert -20. <= math.degrees(math.atan2(xyz[0],xyz[2])) <= -10.


def test_rest_role_is_inside_or_outside_without_forcing_every_actor_visible():
    from avengine.construction.joint.camera_solve import check_camera_roles
    c=room_context(single=True)
    camera=CameraPose((0,1.25,0),0.)
    assert check_camera_roles(c,camera,[PointRole('rest_full_body',(0,0,-4))])['passed']
    assert check_camera_roles(c,camera,[PointRole('rest_full_body',(0,0,4))])['passed']
    assert not check_camera_roles(c,camera,[PointRole('rest_full_body',(0,0,-1.6))])['passed']
    combined=[PointRole('inside_full_body',(0,0,-4)),PointRole('rest_full_body',(0,0,4))]
    assert c.solve_cameras(combined).cameras


def test_two_events_four_endpoints_share_one_camera():
    from avengine.construction.joint.camera_solve import check_camera_roles
    c=room_context(single=True)
    roles=(roles_for_endpoints('enter_left',(-6,0,-4),(-1,0,-5),name='left_event') +
           roles_for_endpoints('enter_right',(6,0,-4),(1,0,-5),name='right_event'))
    result=c.solve_cameras(roles,limit=1)
    assert result.cameras and check_camera_roles(c,result.cameras[0],roles)['passed']
    assert {'left_event.start','left_event.end','right_event.start','right_event.end'} <= set(result.solutions[0].role_margins)
