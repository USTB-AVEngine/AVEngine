from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
from avengine.construction.joint.families.motion_human import native_motion_obligations
pytestmark=pytest.mark.fast_unit


def test_native_motion_preflight_uses_actual_complete_mouths_and_original_root_records(monkeypatch):
    from avengine.timeline import v7_motion_during_conditions as judge
    from avengine.construction import joint
    frames=[{'emitter_position_m':[i/30.,1.7,0.]}for i in range(901)]
    record={'native_30Hz_readbacks':frames,'world_roots_m':[[i/30.,.01,0.]for i in range(901)]}
    entry={'endpoint_id':'voice1','declaration':{'asset_id':'human'},'motion_episode':record}
    scene={'sources':{'voice1':{'asset_id':'human','trajectory':[{'time_s':0.,'position_m':[99,99,99]}]}},'listener_m':[0,1.1,3]}
    prior=deepcopy(scene);seen=[]
    def original(row,actual,visual):
        seen.append((row,actual,visual))
        assert actual['sources']['voice1']['trajectory'][400]['position_m']==frames[400]['emitter_position_m']
        assert visual['native_motion_admissions']['voice1']is record
        return {'status':'pass','reports':[{'qa_id':'QA-16'}]}
    monkeypatch.setattr(judge,'verify_motion_during_native',original)
    monkeypatch.setattr(joint,'native_full_mesh_sweep',lambda *a:())
    results=list(native_motion_obligations(SimpleNamespace(receipt={}),scene,[entry],{'plans':['original']}))
    assert results[0].status=='pass'and len(seen)==1 and scene==prior


def test_native_motion_preflight_rejects_incomplete_readbacks_before_judging():
    entry={'endpoint_id':'voice1','declaration':{'asset_id':'human'},'motion_episode':{'native_30Hz_readbacks':[{}]}}
    results=list(native_motion_obligations(SimpleNamespace(receipt={}),{'sources':{}},[entry],{'plans':['original']}))
    assert results[0].status=='fail'
    assert results[0].reason=='missing_actual_901_mouth_samples'



def test_silent_registered_bystander_outside_role_is_explicit():
    from avengine.construction.joint.families.motion_human import native_view_roles_for_motion
    row={'motion_requests':{'a':{},'b':{}},'semantic_human_asset_ids':['a','b'],'motion_wrong_emitter_control':{'asset_id':'quiet','frame_state':'outside'}}
    roles=native_view_roles_for_motion(row)
    assert roles['quiet']['view']=='outside'
    assert roles['quiet']['required_visible_sometime']is False
    assert roles['quiet']['static']is True
    assert roles['a']['required_visible_sometime']is True
    row['motion_wrong_emitter_control']['frame_state']='edge'
    with pytest.raises(ValueError,match='entirely'):native_view_roles_for_motion(row)



def test_following_approaching_pair_keeps_quiet_bystander_from_facing_camera():
    from avengine.construction.joint.families.motion_human import bystander_native_yaw
    from avengine.capture.v7_visible_attachment import rotation
    root=np.array([2.,0.,-3.]);listener=np.array([0.,1.1,0.])
    toward=listener-root;toward[1]=0;toward/=np.linalg.norm(toward)
    for seed in range(1,200,2):
        yaw=bystander_native_yaw({'scene_seed':seed,'human_motion_pair_mode':'following'},root,listener)
        forward=rotation(yaw)@np.array([0.,0.,-1.])
        assert np.dot(forward,toward)<=-.5+1e-12


def test_cached_single_layout_never_reuses_certificates_or_requires_second_walk():
    from avengine.construction.joint.families.motion_human import single_walker_geometry_proposal
    base={'house':'train','room_id':'room','scene_seed':3,'scene_id':'fresh'}
    saved={**base,'scene_id':'prototype','semantic_human_asset_ids':['a','b'],
        'motion_requests':{'a':{'motion_quartet_route':{'emission_slot':{'branch':'nearer'}}},'b':{}},
        'motion_during_condition_requests':[{'target_asset_id':'a'},{'target_asset_id':'b'}],
        'motion_during_condition_plan':{'plans':[{'target_asset_id':'a'},{'target_asset_id':'b'}]},
        'primary_motion_speech_anchors_s':{'a':2.7,'b':17.7},
        'human_motion_event_planned_verification':[{'asset_id':'a'},{'asset_id':'b'}],
        'motion_quartet_construction':{'routes':{'a':{},'b':{}},'revision':'old'},
        'joint_motion_witness_certificate':{'status':'pass'},
        'joint_prepared_scene_required':True,'joint_solver_certificate':{'status':'pass'},
        'asset_ids':['a','b','quiet'],'silent_asset_ids':['quiet'],
        'phase2_source_event_schedule_s':{'a':[2.7,11.,22.],'b':[7.5,17.7,27.5]}}
    original=deepcopy(saved);row=single_walker_geometry_proposal(base,saved)
    assert saved==original and row['scene_id']=='fresh'
    assert set(row['motion_requests'])=={'a'} and row['human_motion_require_two_walkers']is False
    assert row['asset_ids']==saved['asset_ids'] and row['phase2_source_event_schedule_s']==saved['phase2_source_event_schedule_s']
    assert not any(k in row for k in ('joint_motion_witness_certificate','joint_prepared_scene_required','joint_solver_certificate'))
    assert row['human_motion_event_planned_verification']==[]
    with pytest.raises(ValueError,match='scene_seed'):
        single_walker_geometry_proposal({**base,'scene_seed':4},saved)
    saved['motion_requests']['a']['motion_quartet_route']['emission_slot']['branch']='farther'
    with pytest.raises(ValueError,match='branch'):
        single_walker_geometry_proposal(base,saved)


def test_cached_single_view_is_resolved_by_core_without_moving_any_body(monkeypatch):
    from avengine.construction.joint.families.motion_human import reframe_single_geometry
    from avengine.construction.joint.types import CameraPose,ActorPlan,BodySpec,ScenePlan
    from avengine.construction.joint.geometry import framing
    from avengine.construction.joint import paired
    monkeypatch.setattr(paired,'camera_legal',lambda *args:(True,{}))
    camera=CameraPose((0.,1.1,0.),35.,width=1280,height=720)
    actors={key:ActorPlan(key,np.tile(pos,(901,1)),BodySpec.for_species('human'))
            for key,pos in [('a',[-1.,0.,-4.]),('b',[1.,0.,-4.]),('quiet',[-4.,0.,-4.])]}
    original={key:a.positions_m.copy()for key,a in actors.items()}
    assert any(framing(camera,a.positions_m[0],a.body)[0]!='inside'for a in actors.values())
    plan=ScenePlan(camera,actors)
    row={'semantic_human_asset_ids':['a','b'],'motion_requests':{'a':{'motion_quartet_route':{}}},
         'motion_wrong_emitter_control':{'asset_id':'quiet','frame_state':'outside'},
         'motion_quartet_construction':{}}
    assert reframe_single_geometry(SimpleNamespace(room={}),row,plan)
    assert plan.camera.position_m==camera.position_m
    assert all(np.array_equal(a.positions_m,original[key])for key,a in actors.items())
    assert all(framing(plan.camera,actors[key].positions_m[0],actors[key].body)[0]=='inside'for key in ['a','b'])
    assert row['camera_yaw_knots'][0]['yaw_deg']==row['camera_yaw_knots'][1]['yaw_deg']
    assert row['motion_wrong_emitter_control']['frame_state']!='edge'
