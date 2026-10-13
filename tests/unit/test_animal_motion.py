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
