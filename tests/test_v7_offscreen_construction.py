"""Real-body offscreen contracts and owner camera clearance, not QA labels."""
import itertools
import numpy as np
import pytest
from shapely.geometry import Polygon,mapping
from avengine.capture.v7_offscreen_layout import (
    native_pixels_ok,vertices_outside_camera,coverage_errors,proposal_view_ok)
from avengine.capture.v7_social_layout import spacing_violation
from avengine.timeline.v7_offscreen_construction import plan_offscreen_floor,answer_branch_demands,_azimuth
from avengine.assets.roster_motion.demand import matches_distance_request
pytestmark=pytest.mark.fast_unit


def test_real_offscreen_and_visible_bodies_have_distinct_native_pixel_requirements():
    entries=[{'endpoint_id':'voice1','declaration':{'asset_id':'inside'}},
             {'endpoint_id':'voice2','declaration':{'asset_id':'outside'}}]
    scene={'native_view_roles':{'outside':{'view':'outside'}}}
    assert native_pixels_ok(scene,entries,{'inside':100,'outside':0})
    assert not native_pixels_ok(scene,entries,{'inside':99,'outside':0})
    assert not native_pixels_ok(scene,entries,{'inside':100,'outside':1})
    assert not native_pixels_ok(scene,entries,{'voice1':100,'voice2':0})


def test_zero_pixels_inside_camera_cannot_be_used_as_offscreen_proof():
    inside=np.array(list(itertools.product((-.2,.2),(-.2,.2),(-.2,.2))))
    assert not vertices_outside_camera(inside,np.eye(4),np.eye(4))
    assert vertices_outside_camera(inside+[2.,0.,0.],np.eye(4),np.eye(4))
    assert not vertices_outside_camera(inside+[1.,0.,0.],np.eye(4),np.eye(4))


def test_planned_outside_role_does_not_override_missing_actual_body_or_proof():
    scene={'native_view_roles':{'out':{'view':'outside','static':True}}}
    assets=[{'asset_id':'out'},{'asset_id':'in'}]
    assert coverage_errors(scene,assets,{}, {'out':0,'in':1})
    proof={'asset_id':'out','intentional_offscreen_proof':{'status':'pass',
        'source_is_real_drawable':True,'all_vertices_outside_native_frustum_plane':True,'native_object_id':7}}
    assert not coverage_errors(scene,assets,{'visible_attachment_checks':[proof]}, {'out':0,'in':1})
    assert coverage_errors(scene,assets,{'visible_attachment_checks':[proof]}, {'out':1,'in':1})
    assert coverage_errors(scene,assets,{'visible_attachment_checks':[proof]}, {'out':0,'in':0})
    assert coverage_errors(scene,[{'asset_id':'in'}],{}, {'in':1})


@pytest.mark.parametrize('outside_count',[1,2])
def test_arbitrary_room_floor_proposals_keep_real_people_separated_from_camera(outside_count):
    ctx={'room_floor_polygon':mapping(Polygon([(-8,-8),(8,-8),(8,8),(-8,8)])),
        'listener_m':[0.,1.5,0.]}
    yaw,inside,outside=plan_offscreen_floor(ctx,outside_count,487)
    assert len(inside)==3 and len(outside)==outside_count
    assert all(np.linalg.norm([p[0],p[1]])>=1.5 for p in inside+outside)
    assert all(abs(_azimuth(p,ctx['listener_m'],yaw))>=60 for p in outside)
    assert all(abs(_azimuth(p,ctx['listener_m'],yaw))<=33 for p in inside)
    assert all(np.linalg.norm(np.asarray(a[:2])-np.asarray(b[:2]))>=1.05
        for a,b in itertools.combinations(inside+outside,2))


def test_clearance_applies_to_static_human_and_every_human_route_sample():
    scene={'listener_m':[0,1.5,0]};human={'kind':'human'}
    assert spacing_violation(scene,human,[1.49,0,0],[])['required_m']==1.5
    assert spacing_violation(scene,human,[1.5,0,0],[])is None
    assert not matches_distance_request([[2,0,0],[1.49,0,0],[2,0,0]],[0,1,2],[0,1.5,0],{})
    assert matches_distance_request([[2,0,0],[1.5,0,0],[2,0,0]],[0,1,2],[0,1.5,0],{})
    with pytest.raises(ValueError,match='at least 1.5m'):
        matches_distance_request([[2,0,0],[3,0,0]],[0,1],[0,0,0],{'minimum_listener_horizontal_distance_m':1.2})


def test_answer_proportion_requests_use_actual_counts_not_scene_event_quota():
    state={'answer_counts':{'QA-08':{'out_of_view':1,'visible':9},
        'STARSS-FRUSTUM':{'out_of_frame':8,'in_frame':2}}}
    result=answer_branch_demands(state)
    assert result['QA-08']['actual_share']==.1
    assert result['QA-08']['requested_branch']=='outside'
    assert result['STARSS-FRUSTUM']['requested_branch']=='inside'
    assert result['QA-20']['actual_total']==0
    assert result['QA-24']['actual_share']is None


def test_ordinary_entity_view_requirement_is_retained_without_declared_outside_role():
    assert not proposal_view_ok({},'ordinary',[2,0,0],[0,0,-1],False)
    assert proposal_view_ok({'native_view_roles':{'off':{'view':'outside'}}},'off',[2,0,0],[0,0,-1],False)
