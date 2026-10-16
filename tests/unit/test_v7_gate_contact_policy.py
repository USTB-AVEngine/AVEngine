from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
from avengine.construction.contact_policy import (
    POLICY, construct_body_floor_support, contact_quality_observation,
    nonfloor_contacts_clear, MotionRouteUnavailable, degrade_motion_to_standing,
    optional_placement_failure,
)

pytestmark=pytest.mark.fast_unit


def support():
    return ({'attachment_surface':'floor','kind':'human',
             '_attachment_template':(np.zeros(3),[np.array([-.2,0,0]),np.array([.2,0,0])],{})},
            {'point':[0.,0.,0.],'normal':[0.,1.,0.],'surface':'floor','floor_y':0.})


def test_old_rejected_uneven_support_is_constructed_and_recorded(monkeypatch):
    from avengine.capture import v7_visible_attachment as attachment
    def ray(sim,r,origin,normal,distance):
        return np.array([origin[0],.04 if origin[0]<0 else 0.,origin[2]])
    monkeypatch.setattr(attachment,'ray',ray)
    meta,proposal=support()
    assert attachment.check_support(None,None,meta,proposal) is None
    result=construct_body_floor_support(None,None,meta,proposal)
    assert result['contact_quality_observation']['policy']=='construct_then_record'
    assert result['contact_quality_observation']['max_penetration_m']==pytest.approx(.04)
    assert result['contact_quality_observation']['root_support_verified']


@pytest.mark.parametrize('height',[None,.3])
def test_void_and_furniture_top_still_rejected(monkeypatch,height):
    from avengine.capture import v7_visible_attachment as attachment
    monkeypatch.setattr(attachment,'ray',lambda *args:None if height is None else np.array([0.,height,0.]))
    meta,proposal=support()
    assert construct_body_floor_support(None,None,meta,proposal) is None


def test_missing_peripheral_ray_does_not_fabricate_a_hit(monkeypatch):
    from avengine.capture import v7_visible_attachment as attachment
    monkeypatch.setattr(attachment,'ray',lambda sim,r,p,*args:None if p[0]<0 else np.array([p[0],0.,p[2]]))
    meta,proposal=support();result=construct_body_floor_support(None,None,meta,proposal)
    assert len(result['support_raycast_points_m'])==1
    assert result['contact_quality_observation']['missing_attachment_probe_indices']==[0]


@pytest.mark.parametrize('point,normal,accepted',[
    ([0,0,0],[0,1,0],True),([0,.5,0],[0,1,0],False),([0,0,0],[1,0,0],False)])
def test_floor_residual_vs_real_furniture_and_wall(point,normal,accepted):
    assert nonfloor_contacts_clear([{'stage_contact_point_m':point,'normal_m':normal,
                                     'contact_distance_m':-.05}],0.,.00101) is accepted


def test_contact_summary_keeps_real_residuals_and_no_stance():
    record={'feet':{'L':{'floor_delta_m':-.02,'in_contact':False,'support_lock_error_m':.1}},
            'pelvis_lowering_m':.3,'ik_reach_deficit_so_far_m':.04}
    value=contact_quality_observation([record])
    assert value['max_penetration_m']==.02
    assert value['max_gap_m'] is None
    assert value['max_sliding_m']==.1 and value['max_ik_reach_deficit_m']==.04
    assert value['max_pelvis_lowering_m']==.3 and value['no_planted_support_frames']==1


def test_unsolved_route_preserves_standing_person_and_speech():
    request={'action':'walk','distance_trend':'approaching'}
    entry={'endpoint_id':'voice1','declaration':{'asset_id':'speaker'}}
    scene={'sources':{'voice1':{'sound':'speech'}},'events':[{'endpoint_id':'voice1'}],
           'production_scenario':{'motion_requests':{'voice1':request}}}
    before=deepcopy(scene['sources'])
    row=degrade_motion_to_standing(scene,entry,request,MotionRouteUnavailable('no route'))
    assert row['replacement']=='stand' and row['dropped_families']==['QA-15']
    assert scene['sources']==before and scene['events']==[{'endpoint_id':'voice1'}]
    assert not scene['production_scenario']['motion_requests']


def test_footsteps_are_not_silently_played_by_standing_body():
    entry={'endpoint_id':'walker','declaration':{'asset_id':'person'}}
    scene={'planned_footsteps':[{'endpoint_id':'walker'}]}
    record=degrade_motion_to_standing(scene,entry,{'action':'walk'},'no route')
    assert record['footsteps_cancelled'] and not scene['planned_footsteps']
    assert scene['element_degradation']==[record]


def test_unplaceable_optional_bystander_uses_existing_recipe_rebuild(tmp_path,monkeypatch):
    from avengine.timeline.v7_phase2_fallback import generate_parent_with_element_fallback
    from avengine.timeline import v7_phase2_generation as generation
    target={'asset_id':'target','kind':'human','silent_decor':False}
    extra={'asset_id':'extra','kind':'human','silent_decor':True,'attachment_surface':'floor'}
    entry={'endpoint_id':'v2','declaration':extra}
    scene={'sources':{'v1':{'sound':'speech'},'v2':{}},'events':[{'endpoint_id':'v1'}]}
    row={'scene_id':'parent','asset_ids':['target','extra'],'silent_asset_ids':['extra']}
    original=deepcopy(row);seen=[]
    (tmp_path/'parents'/'parent').mkdir(parents=True)
    def generate(current,*args):
        seen.append(deepcopy(current))
        if 'extra' in current['asset_ids']:
            raise optional_placement_failure(scene,entry,'all alternative positions exhausted')
        return {'status':'pass','remaining_assets':current['asset_ids']}
    monkeypatch.setattr(generation,'generate_phase2_parent',generate)
    result=generate_parent_with_element_fallback(row,{}, {},[target,extra],{}, {},tmp_path)
    assert result['status']=='pass' and result['remaining_assets']==['target']
    assert len(seen)==2 and seen[1]['element_degradations'][0]['asset_id']=='extra'
    assert row==original  # original request is preserved, actual recipe is rebuilt.
    assert optional_placement_failure(scene,{'endpoint_id':'v1','declaration':target},'required') is None


def test_successful_parent_never_retried_for_failed_element():
    from avengine.timeline.v7_construction_retry import spatial_construction_failure
    assert not spatial_construction_failure({'status':'passed','error_type':'ValueError',
        'reason':'no fully qualified new-house walking route'})


def test_registered_native_qualification_continues_after_route_exhaustion(tmp_path,monkeypatch):
    from avengine.capture.v7_registered_native import NativeRegisteredVideo
    from avengine.assets.roster_motion import episode
    from avengine.capture import v7_social_layout,v7_offscreen_layout
    from avengine.timeline import v7_registered_sampling,v7_camera_evidence
    monkeypatch.setattr(v7_social_layout,'spacing_violation',lambda *a:None)
    monkeypatch.setattr(v7_offscreen_layout,'native_pixels_ok',lambda *a:True)
    monkeypatch.setattr(v7_offscreen_layout,'native_projection_proof',lambda *a:None)
    monkeypatch.setattr(v7_offscreen_layout,'refresh_actual_offscreen_proofs',lambda *a:None)
    monkeypatch.setattr(v7_camera_evidence,'recenter_on_actual_roots',lambda *a:None)
    monkeypatch.setattr(v7_registered_sampling,'registered_distance_band',lambda *a:None)
    def no_route(*args,**kwargs):
        raise MotionRouteUnavailable('all endpoint paths exhausted')
    monkeypatch.setattr(episode,'compile_walking',no_route)
    entry={'endpoint_id':'voice1','object':SimpleNamespace(translation=None),
        'declaration':{'asset_id':'speaker','mouth_local_glb_m':[0.,1.5,0.],'entity_class':'articulated_human'},
        'applied_root_input':np.eye(4)}
    scene={'listener_m':[0.,1.25,0.],'sources':{'voice1':{'trajectory':[{'position_m':[2.,1.5,-3.]}]}},
        'events':[{'endpoint_id':'voice1','sound':'speech'}],
        'production_scenario':{'motion_requests':{'voice1':{'action':'walk'}}}}
    video=object.__new__(NativeRegisteredVideo)
    video.actors=[entry];video.environment_nodes={};video.motion_house={};video.motion_inputs={}
    video.mn=SimpleNamespace(Vector3=lambda *args:args)
    video._apply_entry=lambda *args:{'emitter_position_m':[2.,1.5,-3.]}
    video._native_observed_pixels=lambda *args:({'speaker':200},[])
    video._probe_entry=lambda *args:{'status':'pass','endpoint_id':'voice1','known_bad_detected':True,
        'native_full_skin_bounds_m':[[1.,0.,-4.],[2.,2.,-3.]],'contacts':[]}
    checks=video.qualify_standing(scene,tmp_path)
    assert len(checks)==1 and checks[0]['status']=='pass'
    assert 'motion_episode' not in entry
    assert scene['events'][0]['sound']=='speech'
    assert scene['element_degradation'][0]['replacement']=='stand'
    assert (tmp_path/'element_degradation_voice1.json').is_file()


@pytest.mark.parametrize('offset,yaw,observed,passed',[
    ((0.,.03,0.),0.,False,False),((0.,.03,0.),0.,'unverified',False),((0.,.03,0.),0.,True,True),
    ((.003,.03,0.),0.,True,False),((0.,.03,0.),.1,True,False)])
def test_core_grounding_observation_relaxes_only_y(offset,yaw,observed,passed):
    from test_joint_core import fake_native_world
    from avengine.construction.joint import ScenePlan,CameraPose,static_actor,native_static_checks
    plan=ScenePlan(CameraPose((0.,1.25,0.),0.),{'a':static_actor('a',[0,0,-4])})
    video,entry=fake_native_world(plan,offset=offset,yaw=yaw)
    if observed:entry['contact_quality_observation']={**POLICY,'root_support_verified':observed is True}
    certificate=native_static_checks(video,{},plan)
    assert certificate.passed is passed
    exact=next(c for c in certificate.checks if c.name.endswith(':exact_pose'))
    assert exact.evidence['root_error_axes']==('xz'if observed is True else'xyz')


def test_floor_root_over_furniture_or_void_is_not_a_cosmetic_residual():
    from avengine.assets.roster_motion.episode import PhysicalFloor
    floor=object.__new__(PhysicalFloor)
    floor.reference=0.;floor.reference_tolerance_m=.08;floor.current_root_floor_m=None
    floor.original_ray=SimpleNamespace(many=lambda p:np.full(len(p),.3))
    with pytest.raises(ValueError,match='furniture'):
        floor.ground_roots([[0.,0.,0.]])
    def absent(points):raise ValueError('original whole stage has no vertical support for every candidate point')
    floor.original_ray=SimpleNamespace(many=absent)
    with pytest.raises(ValueError,match='no vertical support'):
        floor.ground_roots([[0.,0.,0.]])


def test_only_walker_degradation_names_its_declared_family_without_muting_speech():
    request={'action':'walk','camera_path_condition':{'kind':'cross_view'}}
    entry={'endpoint_id':'v1','declaration':{'asset_id':'target'}}
    scene={'scene_demand':{'target_family':'QA-11'},
           'production_scenario':{'motion_requests':{'v1':request}},
           'sources':{'v1':{'sound':'speech'}}}
    record=degrade_motion_to_standing(scene,entry,request,'no endpoint path')
    assert record['dropped_families']==['QA-11']
    assert scene['sources']['v1']['sound']=='speech'


def test_ground_only_between_feet_does_not_admit_a_body_over_a_void(monkeypatch):
    from avengine.capture import v7_visible_attachment as attachment
    monkeypatch.setattr(attachment,'ray',lambda sim,r,p,*args:
                        np.array([0.,0.,0.]) if p[0]==0 else None)
    meta,proposal=support()
    assert construct_body_floor_support(None,None,meta,proposal) is None
