from types import SimpleNamespace
import numpy as np
from avengine.rooms.conditioned_sampler import _qa16_camera_distance_evidence


def case(points):
    solution=SimpleNamespace(qa_id='QA-16',requirements=[SimpleNamespace(role='target',entity_instance_id='target')],placements=[SimpleNamespace(entity_instance_id='target',end_frame=20)],query_window={'legal_frames':[50,100]})
    actors=[{'entity_instance_id':'target'}]
    clock={'frame_count':100,'frame_rate_hz':10}
    request={'qa_sampling':{'acceptance_policy':{'question_mode':'ordinary_observation','post_sound_distance_query':'integer_timepoint'}}}
    return solution,actors,np.array([points]),clock,request


def test_walking_tangentially_is_not_a_distance_change_guarantee():
    points=np.repeat([[2.,1.6,0.]],100,axis=0);points[30:,2]=.4
    sol,actors,emitters,clock,req=case(points)
    assert not _qa16_camera_distance_evidence(sol,actors,emitters,[0.,1.6,0.],clock,req)['available']
    assert _qa16_camera_distance_evidence(sol,actors,emitters,[2.,1.6,-2.],clock,req)['available']


def test_ordinary_timepoint_does_not_require_a_whole_stable_interval():
    points=np.repeat([[2.,1.6,0.]],100,axis=0);points[80,2]=.4
    sol,actors,emitters,clock,req=case(points)
    row=_qa16_camera_distance_evidence(sol,actors,emitters,[2.,1.6,-2.],clock,req)
    assert row['usable_query_frames']==[80]
    assert row['available']


def test_other_questions_do_not_gain_the_distance_restriction():
    assert _qa16_camera_distance_evidence(None,[],[],[0,0,0],{}, {}) is None
