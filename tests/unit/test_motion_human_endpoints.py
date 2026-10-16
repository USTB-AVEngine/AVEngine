from types import SimpleNamespace
from collections import Counter
import math
import numpy as np
import pytest
from shapely.geometry import Polygon, mapping, LineString
from avengine.rooms.walkable_map import WalkableMap,VERSION,SPECIES_RADII_M,RESOLUTION_M
from avengine.construction.joint import CameraPose,BodySpec
from avengine.construction.joint.geometry import framing,camera_coordinates
from avengine.construction.joint.families.motion_human_endpoints import (
    HumanEndpointDemand,endpoint_pairs,ranked_endpoint_pairs,floor_bounded_context,
    route_for_pair,native_endpoint_route_proposals)
pytestmark=pytest.mark.fast_unit


def context(polygon, points, cameras):
    origin=np.array([-8.,-8.]);size=320
    mask=np.ones((size,size),dtype=bool)
    metadata={'version':VERSION,'species_radii_m':SPECIES_RADII_M}
    m=WalkableMap(origin,RESOLUTION_M,0.,{s:mask.copy()for s in SPECIES_RADII_M},metadata)
    ctx=SimpleNamespace(room={'floor_polygon':mapping(polygon)},
        walkable_map=m,points_m=np.asarray(points,float),cameras=cameras,
        pathfinder=None,allow_native_map_fallback=False)
    def grid(camera,body):
        xyz=camera_coordinates(camera,ctx.points_m)
        return SimpleNamespace(distances_m=np.linalg.norm(xyz[:,[0,2]],axis=1),
             frustum_state=np.asarray([framing(camera,p,body)[0]for p in ctx.points_m]))
    ctx.grid=grid
    return ctx


def test_pair_domain_includes_hundreds_and_respects_each_endpoint_role():
    points=[[x,0.,z]for x in np.linspace(-3,3,13)for z in np.linspace(-7,-2,11)]
    camera=CameraPose((0.,1.1,0.),0.,width=1280,height=720)
    ctx=context(Polygon([[-7,-7.5],[7,-7.5],[7,7],[-7,7]]),points,[camera])
    stats=Counter();pairs=list(endpoint_pairs(ctx,camera,HumanEndpointDemand('a','nearer'),funnel=stats))
    assert len(pairs)>100 and stats['candidate_pair_domain']==len(pairs)
    assert all(a.radial_difference_m>=b.radial_difference_m for a,b in zip(pairs,pairs[1:]))
    for p in pairs:
        from avengine.qa.unified_catalog import POST_SOUND_DISTANCE_MARGIN_M
        assert p.radial_difference_m>=.5+POST_SOUND_DISTANCE_MARGIN_M-1e-10
        assert framing(camera,ctx.points_m[p.start_index],BodySpec())[0]!='edge'
        assert framing(camera,ctx.points_m[p.end_index],BodySpec())[0]!='edge'
    reversed_pairs=list(endpoint_pairs(ctx,camera,HumanEndpointDemand('b','farther')))
    assert {(p.start_index,p.end_index)for p in pairs}=={(p.end_index,p.start_index)for p in reversed_pairs}


def test_all_camera_candidates_are_consumed_without_a_24_pose_cap():
    cameras=[CameraPose((0.,1.1,0.),float(a),width=1280,height=720)for a in range(0,180,4)]
    ctx=context(Polygon([[-7,-7],[7,-7],[7,7],[-7,7]]),[[0.,0.,-6.],[0.,0.,-3.]],cameras)
    stats=Counter();items=ranked_endpoint_pairs(ctx,HumanEndpointDemand('a','nearer'),funnel=stats)
    assert next(items)[1].radial_difference_m>0
    assert stats['camera_candidates']==len(cameras)>24


def test_routing_domain_is_intersected_before_core_search_and_does_not_edit_original():
    poly=Polygon([[-5,-5],[5,-5],[5,-1],[-1,-1],[-1,5],[-5,5]])
    camera=CameraPose((-5.5,1.1,5.5),135.,width=1280,height=720)
    ctx=context(poly,[[-3.,0.,3.],[3.,0.,-3.]], [camera])
    bounded=floor_bounded_context(ctx)
    assert ctx.walkable_map.contains([3.,3.])
    assert not bounded.walkable_map.contains([3.,3.])
    from avengine.construction.joint import route_between
    path=route_between(bounded,ctx.points_m[0],ctx.points_m[1])
    assert len(path)>2 and poly.covers(LineString(path[:,[0,2]]))
    assert ctx.walkable_map.contains([3.,3.])


def test_core_route_can_follow_an_arbitrary_direction_with_exact_endpoints():
    camera=CameraPose((0.,1.1,0.),13.,width=1280,height=720)
    ctx=context(Polygon([[-7,-7],[7,-7],[7,7],[-7,7]]),
                [[-.7,0.,-6.],[.8,0.,-2.8]], [camera])
    p=next(endpoint_pairs(ctx,camera,HumanEndpointDemand('a','nearer')))
    actor=route_for_pair(ctx,camera,p,HumanEndpointDemand('a','nearer'),speed_m_s=.9)
    assert np.allclose(actor.positions_m[[0,-1]],ctx.points_m[[p.start_index,p.end_index]])
    assert not any(abs(math.atan2(3.2,1.5)-a)<1e-6 for a in [0.,math.pi/2,math.pi,3*math.pi/2])


def test_native_endpoint_clock_is_consumed_without_relocation_and_Y_is_observed():
    camera=CameraPose((0.,1.1,0.),0.,width=1280,height=720)
    pos=np.column_stack((np.linspace(-1.,1.,901),np.zeros(901),np.full(901,-4.)))
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pos[:,[0,2]],axis=0),axis=1))]
    solved={'positions_m':pos.tolist(),'planned_distance_m':distance.tolist(),
            'headings_rad':np.zeros(901).tolist(),'camera':camera.__dict__}
    root=np.eye(4);root[:3,3]=pos[0]+[0,.025,0]
    entry={'applied_root_input':root}
    scene={'listener_m':list(camera.position_m),'camera_forward':camera.forward.tolist()}
    proposals=list(native_endpoint_route_proposals(None,entry,scene,distance,[0,0,-1],{'joint_endpoint_route':solved}))
    assert len(proposals)==1 and np.array_equal(proposals[0].positions,pos)
    assert proposals[0].receipt['standing_y_delta_m']==.025
    with pytest.raises(ValueError,match='clock'):
        list(native_endpoint_route_proposals(None,entry,scene,distance+.01,[0,0,-1],{'joint_endpoint_route':solved}))


def test_joint_camera_roles_include_both_people_and_all_legal_sites():
    from avengine.construction.joint.families.motion_human_endpoints import (
        EndpointPair,solve_endpoint_cameras)
    seen={}
    ctx=SimpleNamespace(points_m=np.array([[-3,0,-5],[-2,0,-3],
                                           [2,0,-3],[4,0,-5]],float))
    def solve(roles,**kwargs):
        seen.update(roles=roles,kwargs=kwargs)
        return SimpleNamespace(solutions=(),diagnostics={"position_height_pairs":291})
    ctx.solve_cameras=solve
    counts=Counter()
    demands=[HumanEndpointDemand("a","nearer"),HumanEndpointDemand("b","farther")]
    solve_endpoint_cameras(ctx,[EndpointPair(0,1,2),EndpointPair(2,3,2)],
                           demands,[BodySpec(),BodySpec()],funnel=counts)
    assert seen["kwargs"]=={"limit":None}
    assert len(seen["roles"])==2
    assert {r.name for r in seen["roles"]}=={"a","b"}
    assert {r.distance_change for r in seen["roles"]}=={"nearer","farther"}
    assert all(r.start_role==r.end_role=="rest_full_body"for r in seen["roles"])
    assert counts["camera_position_height_tests"]==291


def test_endpoint_domain_constructs_pairs_within_route_components():
    from avengine.construction.joint.families.motion_human_endpoints import spatial_endpoint_domain
    poly=Polygon([[-7,-7],[7,-7],[7,7],[-7,7]])
    ctx=context(poly,[[-4,0,-3],[-2,0,-3],[2,0,-3],[4,0,-3]],[])
    for layer in ctx.walkable_map.layers.values():
        layer[:,158:162]=False
    counts=Counter()
    pairs=spatial_endpoint_domain(ctx,HumanEndpointDemand("a","nearer"),funnel=counts)
    assert {(p.start_index,p.end_index)for p in pairs}=={(0,1),(1,0),(2,3),(3,2)}
    assert counts["unconnected_endpoint_pairs_avoided"]>0


def test_route_start_is_solved_from_core_duration_for_a_long_late_crop():
    from avengine.construction.joint.families.motion_human_endpoints import EndpointPair
    camera=CameraPose((0.,1.1,0.),0.,width=1280,height=720)
    ctx=context(Polygon([[-7,-7],[7,-7],[7,7],[-7,7]]),
                [[-5,0,-4],[5,0,-4]],[camera])
    actor=route_for_pair(ctx,camera,EndpointPair(0,1,1),
                        HumanEndpointDemand("a","farther"),speed_m_s=.8,start_s=20.)
    assert actor.metadata["walk_intervals_s"][0][0]<20.
    assert actor.metadata["walk_intervals_s"][0][1]==pytest.approx(30.)
    assert np.allclose(actor.positions_m[-1],ctx.points_m[1])


def test_element_route_failure_retains_the_other_real_mover():
    from avengine.construction.joint import ActorPlan
    from avengine.construction.joint.families.motion_human_endpoints import (
        EndpointPair,retained_actor_sets)
    p=np.column_stack((np.linspace(-4,0,901),np.zeros(901),np.full(901,-4)))
    actor=ActorPlan("a",p,headings_rad=np.zeros(901),
                   metadata={"walk_intervals_s":[[0,5]],"ramp_s":.3})
    ctx=SimpleNamespace(points_m=np.array([p[0],p[-1],[4,0,-4],[4,0,-2]]))
    results=list(retained_actor_sets(ctx,[actor,None],
        [EndpointPair(0,1,1),EndpointPair(2,3,1)],[BodySpec(),BodySpec()],
        ["a","b"],[None,"no_grid_route"],7))
    assert results
    assert all(placed[0]is actor for placed in results)
    assert all(placed[1].entity_id=="b"for placed in results)
    assert all(np.all(placed[1].positions_m==placed[1].positions_m[0])for placed in results)
    assert all(placed[1].metadata["element_degradation"]["to"]=="stand_and_speak"
               for placed in results)


def test_both_reverse_events_use_the_original_four_family_verifier():
    from avengine.construction.joint import ActorPlan,ScenePlan
    from avengine.construction.joint.families.motion_human_endpoints import (
        EndpointPair,repeat_first_event)
    from avengine.timeline.v7_human_endpoint_program import (
        attach_routes,speech_slots,verify_actor_families)
    camera=CameraPose((0.,1.2,0.),0.,width=1280,height=720)
    ctx=context(Polygon([[-7,-7],[7,-7],[7,7],[-7,7]]),
                [[0,0,-6],[0,0,-2.5]],[camera])
    actor=route_for_pair(ctx,camera,EndpointPair(0,1,3.5),
                        HumanEndpointDemand("a","nearer"),speed_m_s=1.)
    actor=repeat_first_event(ctx,actor)
    assert actor is not None
    static=ActorPlan("b",np.tile([4.,0.,-5.],(901,1)),headings_rad=np.zeros(901),
                     metadata={"native_yaw_rad":-math.pi/2})
    plan=ScenePlan(camera,{"a":actor,"b":static})
    assets=[{"asset_id":aid,"kind":"human","walk_allowed":True,
             "emitter_local_m":[0,1.55,0]}for aid in("a","b")]
    records=[{"sound_asset_id":aid,"sound_class":"speech_playback",
              "header":{"sample_rate_hz":16000,"sample_count":16000},
              "screens":{"automatic_activity":{"activity_intervals_samples":[[0,16000]]}}}
             for aid in("sa","sb")]
    row={"asset_ids":["a","b"],"scene_seed":1,"camera_yaw_deg":0,
         "motion_quartet_template_revision":"test","motion_requests":{},
         "proposed_sound_asset_ids":{"a":"sa","b":"sb"}}
    row=attach_routes(row,plan,assets)
    for offset,branch in ((0.,"nearer"),(10.,"farther")):
        choices=speech_slots(actor,camera,records[0],branch,crop_offset=offset,
                             companion_duration_s=1.)
        assert choices and all(choices[0]["predicted_families"].values())
        accepted,verdicts,failures=verify_actor_families(
            row,plan,assets,records,"a",branch,choices[0])
        assert all(verdicts.values()),failures
        assert len(accepted)==4
