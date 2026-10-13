import numpy as np
import pytest
from avengine.assets.animal_motion.checks import direction_check
from avengine.assets.animal_motion.policy import (select_motion,MotionUnavailable,MotionRejected,attach_to_scene,annotate_question_motion)


def test_directions_and_negative_controls_cover_both_interval_ends():
    rng=np.random.default_rng(261013)
    for yaw in rng.uniform(-np.pi,np.pi,32):
        forward=np.array([np.sin(yaw),0.,-np.cos(yaw)])
        roots=np.arange(61)[:,None]*forward/300
        heads=roots+forward*.3+[0,.3,0];tails=roots-forward*.3+[0,.3,0]
        assert direction_check(np.arange(61)/30,roots,heads,tails)['status']=='pass'
        assert direction_check(np.arange(61)/30,roots,tails,heads)['status']=='fail'
        side=forward[[2,1,0]]*[-1,1,1]
        assert direction_check(np.arange(61)/30,roots,roots+side,roots)['status']=='fail'
        heads[-1]=tails[-1]-forward
        assert direction_check(np.arange(61)/30,roots,heads,tails)['status']=='fail'


def test_stationary_sequence_does_not_vacuously_pass_direction():
    points=np.zeros((3,3))
    assert direction_check([0,1,2],points,points+[0,0,-1],points)['status']=='fail'

@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_bad_native_positions_are_errors(bad):
    points=np.array([[0.,0.,0.],[bad,0.,-.1]])
    with pytest.raises(ValueError):direction_check([0,1],points,points+[0,0,-1],points)


def test_animation_is_attempted_first_and_success_does_not_slide():
    calls=[]
    def walk(speed):calls.append(('walk',speed));return 'animated'
    def slide(speed):raise AssertionError('successful animation must not slide')
    result=select_motion(asset_id='dog',requested_speed_m_s=.1,attempt_animation=walk,attempt_slide=slide,qualify=lambda p,m:{'status':'pass'})
    assert result.motion_mode=='walk_animation' and calls==[('walk',.1)]
    assert result.metadata()['gait_animation_present']

@pytest.mark.parametrize('error',[MotionUnavailable('no_skeleton'),MotionRejected('backwards_180deg')])
def test_missing_or_failed_animation_records_reason_and_caps_slide(error):
    calls=[]
    def walk(speed):calls.append('walk');raise error
    def slide(speed):calls.append(('slide',speed));return 'standing_pose_on_native_route'
    result=select_motion(asset_id='cat',requested_speed_m_s=.15,attempt_animation=walk,attempt_slide=slide,qualify=lambda p,m:{'status':'pass'})
    assert calls==['walk',('slide',.08)]
    assert result.motion_mode=='slide_no_animation' and result.reason==str(error)
    assert result.metadata()['animation_attempted'] and not result.metadata()['gait_animation_present']


def test_failed_native_animation_validation_falls_back_and_failed_slide_cannot_move():
    result=select_motion(asset_id='dog',requested_speed_m_s=.06,attempt_animation=lambda s:'walk',attempt_slide=lambda s:'stand',qualify=lambda p,m:{'status':'fail','failures':['floor_penetration']})
    assert result.motion_mode=='cannot_move'
    assert len(result.animation_attempts)==2
    assert result.payload is None


def test_environment_failure_is_not_hidden_as_missing_animation():
    def broken(speed):raise OSError('disk unavailable')
    with pytest.raises(OSError):select_motion(asset_id='dog',requested_speed_m_s=.06,attempt_animation=broken,attempt_slide=lambda s:'stand',qualify=lambda p,m:{'status':'pass'})


def test_slide_metadata_reaches_question_without_changing_answers_or_source():
    def missing(speed):raise MotionUnavailable('no_Walking')
    choice=select_motion(asset_id='cat',requested_speed_m_s=.05,attempt_animation=missing,attempt_slide=lambda s:{},qualify=lambda p,m:{'status':'pass'})
    scene={'sources':{'meow':{'actor_id':3}},'events':[{'start':0,'end':1}]};attach_to_scene(scene,'meow',choice)
    q=[{'question_id':'q','question':'which entity moves?','answer':'cat','metadata':{'old':'retained'}}]
    output=annotate_question_motion(q,scene)
    assert output[0]['metadata']['source_motion']['meow']['motion_mode']=='slide_no_animation'
    assert output[0]['answer']==q[0]['answer'] and output[0]['metadata']['old']=='retained'
    assert 'source_motion' not in q[0]['metadata']


def test_existing_static_question_serialization_is_unchanged():
    q=[{'question':'same'}]
    assert annotate_question_motion(q,{'sources':{}}) is q

def test_actual_motion_service_entrypoint_imports():
    from avengine.assets.animal_motion.service import execute_request
    assert callable(execute_request)

def test_explicit_target_chain_uses_whole_limb_and_supports_terminal_foot():
    from avengine.assets.animal_motion.profiles import explicit_chains
    nodes=[{'name':'body','children':[]}]
    mapping={}
    for side in ('FL','FR','HL','HR'):
        start=len(nodes);nodes[0]['children'].append(start)
        nodes.extend([{'name':side+'_hip','children':[start+1]},
                      {'name':side+'_knee','children':[start+2]},
                      {'name':side+'_ankle','children':[]}])
        mapping[side]=[side+'_hip',side+'_knee',side+'_ankle']
    chains=explicit_chains({'nodes':nodes},mapping)
    assert all(c[-1]==c[-2] and len(set(c[:3]))==3 for c in chains.values())
    mapping['FL']=['body','FL_knee','FL_ankle']
    with pytest.raises(ValueError,match='another_paw_or_spine'):
        explicit_chains({'nodes':nodes},mapping)


def test_explicit_target_chain_rejects_reversed_ancestry():
    from avengine.assets.animal_motion.profiles import explicit_chains
    doc={'nodes':[{'name':'hip','children':[1]},{'name':'knee','children':[2]},{'name':'ankle'}]}
    with pytest.raises(ValueError,match='ancestry'):
        explicit_chains(doc,{s:['ankle','knee','hip'] for s in ('FL','FR','HL','HR')})


def test_reused_motion_must_match_requested_route_duration_and_speed():
    from avengine.assets.animal_motion.service import validate_reused_route
    route={'root_m':[1.,0.,2.],'yaw_rad':0.}
    frames={'frames':[{'time_s':0.,'root_world_m':[1.,0.,2.]},
                      {'time_s':1.,'root_world_m':[1.,0.,1.95]}]}
    validate_reused_route(frames,route,speed=.05,seconds=1.)
    with pytest.raises(ValueError,match='duration'):
        validate_reused_route(frames,route,speed=.05,seconds=2.)
    with pytest.raises(ValueError,match='route_or_speed'):
        validate_reused_route(frames,route,speed=.08,seconds=1.)


def test_real_engine_entry_routes_qualified_animal_stem(monkeypatch):
    from avengine.timeline.v7_registered_motion_audio import moving_stem
    from avengine.assets.animal_motion import render
    seen=[]
    monkeypatch.setattr(render,'render_qualified_stem',lambda *args:seen.append(args) or 'native-animal-stem')
    spec={'animal_motion_audio':{'native':'native.json','audio':'audio.json'}}
    assert moving_stem(spec,{}, {},None,np.zeros(2),None,'dog')=='native-animal-stem'
    assert seen[0][0] is spec and seen[0][-1]=='dog'


def test_qualified_audio_cannot_be_reused_for_another_acoustic_world():
    from copy import deepcopy
    from avengine.assets.animal_motion.render import require_same_acoustic_world
    identity={k:'same' for k in ('package_id','existing_package_content_sha256','package_manifest',
        'listener_position_m','listener_orientation_wxyz','source_radius_m','listener_radius_m','runtime_identity')}
    identity['simulation']={'direct_ray_count':500,'indirect_ray_count':5000,'source_ray_count':500,'sample_rate_hz':16000}
    admitted=deepcopy(identity)
    for k in ('direct_ray_count','indirect_ray_count','source_ray_count'):admitted['simulation'][k]*=9
    require_same_acoustic_world(identity,admitted)
    admitted['package_id']='another'
    with pytest.raises(ValueError,match='world_identity'):
        require_same_acoustic_world(identity,admitted)
    admitted=deepcopy(identity);admitted['simulation']['sample_rate_hz']=48000
    with pytest.raises(ValueError,match='physical_simulation'):
        require_same_acoustic_world(identity,admitted)
