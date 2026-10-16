from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from test_joint_core import context, fallback_context, fake_native_world
from avengine.construction.joint import (
    BodySpec,CameraPose,ScenePlan,static_actor,FamilyRequest,CheckResult,PreparedCandidate,solve)
from avengine.construction.joint.routes import validate_track,ConstructionError
from avengine.construction.joint.grounding import floor_contact_variants
from avengine.construction.contact_policy import (
    degrade_motion_to_standing,reconcile_standing_plan,reconcile_degraded_program)
from avengine.timeline.v7_motion_camera_planning import endpoint_camera_routes,plan_parent_motion

pytestmark=pytest.mark.fast_unit


def test_core_uneven_contact_is_constructed_and_observed(monkeypatch):
    import trimesh
    from avengine.capture import v7_visible_native
    c=context()
    def hit(origins,directions):
        q=np.asarray(origins);y=np.where(q[:,0]>.1,.06,0.)
        return q[:,1]-y,np.zeros(len(q),int)
    c.floor_raycaster=SimpleNamespace(source='visual_mesh',first_hit=hit)
    monkeypatch.setattr(v7_visible_native,'attachment_template',
        lambda *args:(np.zeros(3),np.array([[-.2,0.,0.],[.2,0.,0.]]),'two_feet'))
    actor=static_actor('a',[0,.05,-4.],facing_rad=-np.pi/2)
    mesh=trimesh.creation.box((.6,1.8,.4))
    result=next(floor_contact_variants(c,actor,mesh,yaw_rad=0.))
    assert result.positions_m[0,1]==pytest.approx(0.)
    observation=result.metadata['contact_quality_observation']
    assert observation['max_penetration_m']==pytest.approx(.06)
    assert observation['policy']=='construct_then_record'
    np.testing.assert_array_equal(actor.positions_m[:,1],np.full(901,.05))


@pytest.mark.parametrize('floor_height',[np.nan,.12])
def test_core_floor_prefilter_still_rejects_void_and_furniture(floor_height):
    import trimesh
    c=context()
    c.floor_raycaster=SimpleNamespace(source='visual_mesh',
        first_hit=lambda p,d:(np.asarray(p)[:,1]-floor_height,np.zeros(len(p),int)))
    with pytest.raises(ConstructionError,match='missing_original_support'):
        next(floor_contact_variants(c,static_actor('a',[0,0,-4]),trimesh.creation.box()))


def test_native_map_fallback_grounds_large_cosmetic_height_then_records():
    c=fallback_context()
    q,evidence=validate_track(c,[[0,.05,-4],[0,.05,-3]])
    np.testing.assert_allclose(q[:,1],0.,atol=1e-8)
    obs=evidence['native_map_fallback']['contact_quality_observation']
    assert obs['max_gap_m']==pytest.approx(.05)
    c.pathfinder.distance_to_closest_obstacle=lambda *a:.2
    with pytest.raises(ConstructionError,match='body_clearance'):
        validate_track(c,[[0,.05,-4]])


def test_camera_and_endpoints_are_solved_without_the_old_camera_table():
    c=context()
    c.points_m=np.array([[-1.,0.,-5.],[1.,0.,-5.],[0.,0.,.5],[3.,0.,.5],[-3.,0.,.5]])
    c.cameras=()  # The old bounded camera table is irrelevant.
    stats={}
    camera,route,roles=next(endpoint_camera_routes(c,BodySpec(radius_m=.2,margin_m=0.),
        {'speed_m_s':1.,'camera_path_condition':{'kind':'cross_view'}},diagnostics=stats))
    assert stats['camera_solves']>=1
    assert route.receipt['joint_camera_roles']['endpoint_first']
    assert route.positions.shape==(901,3)
    assert camera not in c.cameras


@pytest.mark.parametrize('marker',['input_recipe.json','visible_native_completed.json',
                                   'audio_v0_a0_completed.json','rir_cache'])
def test_frozen_parent_never_opens_a_planning_world(tmp_path,marker,monkeypatch):
    from avengine.rooms import walkable_map_builder
    monkeypatch.setattr(walkable_map_builder,'native_static_video',
                        lambda *a:pytest.fail('frozen world was reopened for camera planning'))
    (tmp_path/marker).write_text('{}')
    row={'motion_requests':{'a':{'action':'walk'}},'listener_m':[1,2,3]}
    before=deepcopy(row)
    result=plan_parent_motion(row,{}, {},{},[{'asset_id':'a','kind':'human'}],{},tmp_path)
    assert result['status']=='frozen_world_retained' and row==before


def test_nonmotion_parent_retains_existing_camera(tmp_path):
    row={'listener_m':[1,2,3]}
    assert plan_parent_motion(row,{}, {},{},[],{},tmp_path)['status']=='no_human_motion_request'
    assert row=={'listener_m':[1,2,3]}


def test_standing_program_cancels_only_its_footsteps_and_retains_speech():
    scene={'planned_footsteps':[{'endpoint_id':'walker'}],
           'sources':{'walker':{},'walker_L':{'owner_endpoint_id':'walker','emitter_part':'foot_L'},
                      'speaker':{}},
           'events':[{'event_id':'foot','endpoint_id':'walker_L'},
                     {'event_id':'speech','endpoint_id':'speaker'}]}
    program={'events':[{'event_id':'foot'},{'event_id':'speech'}]}
    row={'source_classes':{'body':['footsteps'],'speaker':['speech']},
         'motion_requests':{'body':{'action':'walk'}}}
    degrade_motion_to_standing(scene,{'endpoint_id':'walker','declaration':{'asset_id':'body'}},
                              {'action':'walk'},'no route')
    reconcile_degraded_program(scene,program,row)
    assert program['events']==[{'event_id':'speech'}]
    assert scene['events']==[{'event_id':'speech','endpoint_id':'speaker'}]
    assert 'walker_L' not in scene['sources']
    assert row['source_classes']['speaker']==['speech']
    assert row['silent_asset_ids']==['body'] and not row['motion_requests']


def test_prepared_standing_uses_real_readback_and_defers_only_family_obligations():
    c=context();actor=static_actor('a',[0,0,-4])
    p=ScenePlan(c.cameras[0],{'a':actor},metadata={'motion_condition_plan':{'family':'QA-15'}})
    from avengine.construction.joint import construct_route
    moving=construct_route(c,[[0,0,-4],[1,0,-4]],speed_m_s=1.,start_s=1.,entity_id='a',camera=c.cameras[0])
    p.actors['a']=moving
    video,entry=fake_native_world(p)
    scene={'element_degradation':[{'asset_id':'a','replacement':'stand'}]}
    effective=reconcile_standing_plan(p,video,scene)
    assert np.all(effective.actors['a'].positions_m==effective.actors['a'].positions_m[0])
    assert 'motion_condition_plan' not in effective.metadata
    assert p.metadata['motion_condition_plan']=={'family':'QA-15'}
    closed=[]
    payload=SimpleNamespace(plan=effective,
        prepared_degradation={'schema':'prepared_element_degradation_v1','requested_family_status':'deferred'},
        bind_solution=lambda record:None)
    plugin=SimpleNamespace(candidates=lambda c,r:iter([replace(p,metadata={})]),
                           verify=lambda *a:[CheckResult('requested_motion','pass')])
    prepared=PreparedCandidate(payload,(CheckResult('native:common','pass'),),lambda:closed.append(True))
    result=solve(c,FamilyRequest('QA-15','approaching',('a',)),plugin,
                 prepare_and_verify=lambda *a:prepared)
    assert result.status=='degraded' and result.prepared is prepared
    assert result.certificate.passed and not closed
    assert result.plan.metadata['joint_solution']['requested_family_status']=='deferred'


@pytest.mark.parametrize('support_ok',[True,False])
def test_native_prepared_adapter_retains_common_world_but_never_hides_physics_failure(
        tmp_path,monkeypatch,support_ok):
    from avengine.construction.joint import native,construct_route
    c=context()
    actor=construct_route(c,[[0,0,-4],[1,0,-4]],speed_m_s=1.,start_s=1.,entity_id='a',camera=c.cameras[0])
    plan=ScenePlan(c.cameras[0],{'a':actor})
    asset={'asset_id':'a','endpoint_id':'voice1','kind':'human'}
    scene={'production_scenario':{'motion_requests':{'voice1':{'action':'walk'}}},
           'sources':{'voice1':{}},'events':[]}
    row={'scene_id':'unit','asset_ids':['a'],'motion_requests':{'a':{'action':'walk'}}}
    program={'events':[]}
    closed=[]
    def video_factory(*args,**kwargs):
        video,entry=fake_native_world(plan)
        entry['endpoint_id']='voice1'
        video.close=lambda:closed.append(True)
        def qualify(scene,directory):
            degrade_motion_to_standing(scene,entry,{'action':'walk'},'no admissible route')
            return [{'asset_id':'a','status':'pass'if support_ok else'fail'}]
        video.qualify_standing=qualify
        return video
    monkeypatch.setattr(native,'strict_native_bindings',lambda *a,**kw:{})
    result=native.prepare_native(c,FamilyRequest('QA-15','approaching',('a',),
        {'require_native_preflight':True}),plan,row_factory=lambda p:row,
        house={},catalog=[asset],records=[],semantic=None,runtime={},output_dir=tmp_path,
        scene_factory=lambda *a:({},deepcopy(scene),deepcopy(program),[asset]),
        video_factory=video_factory,acoustic_preflight=False,
        family_check=lambda *a:[CheckResult('native:distance_trend','fail','standing_has_no_motion')])
    if support_ok:
        assert result.payload is not None and not closed
        assert result.payload.certificate.passed
        assert not result.payload.family_certificate.passed
        assert result.payload.row['prepared_degradation']['deferred_families']==['QA-15']
        assert np.ptp(result.payload.plan.actors['a'].positions_m[:,0])==0.
        result.close()
    else:
        assert result.payload is None and closed
    assert closed


def test_peripheral_furniture_height_is_observed_only_after_real_root_grounding():
    from avengine.assets.roster_motion.episode import PhysicalFloor
    floor=object.__new__(PhysicalFloor)
    floor.reference=0.;floor.reference_tolerance_m=.08
    floor.current_root_floor_m=None
    floor.original_ray=SimpleNamespace(many=lambda points:np.where(np.asarray(points)[:,0]>.1,.3,0.))
    grounded=floor.ground_roots([[0.,0.,0.]])
    assert grounded[0,1]==0.
    floor.current_root_floor_m=float(grounded[0,1])
    np.testing.assert_allclose(floor.many([[.2,0.,0.],[-.2,0.,0.]]),[0.,0.])
    assert floor.peripheral_floor_outliers[0]['measured_support_y_m']==.3
    # The same hit at the root remains an illegal stand, not a residual.
    with pytest.raises(ValueError,match='furniture'):
        floor.ground_roots([[.2,0.,0.]])
