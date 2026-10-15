from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from avengine.timeline.v7_independent_reference_memo import ExactIndependentReferenceMemo
from avengine.timeline.v7_phase2_motion_audio import wrong_registered_pose
pytestmark=pytest.mark.fast_unit


def test_identical_pose_queries_keep_three_references_but_measure_only_once():
    memo=ExactIndependentReferenceMemo({'scene':'actual','rays':5000});calls=[]
    def measure():calls.append(1);return {'paths':['native_trial0','native_trial1','native_trial2'],'processes':[11,12,13]}
    with ThreadPoolExecutor(max_workers=5)as pool:results=list(pool.map(lambda _:memo.get([1.,2.,3.],measure),range(5)))
    assert len(calls)==1 and sum(reused for _,reused in results)==4
    assert all(value['processes']==[11,12,13]for value,_ in results)


def test_nearby_pose_is_not_rounded_into_a_reference_measurement():
    memo=ExactIndependentReferenceMemo({'scene':'s'});calls=[]
    def measure():calls.append(1);return {'paths':['a','b','c']}
    memo.get([1.,2.,3.],measure);memo.get([np.nextafter(1.,2.),2.,3.],measure)
    assert len(calls)==2


def test_existing_sample_checkpoint_is_not_replaced_with_a_more_favorable_triplet():
    memo=ExactIndependentReferenceMemo({'scene':'s'});first={'paths':['first_a','first_b','first_c']};existing={'paths':['saved_a','saved_b','saved_c']}
    memo.get([1,2,3],lambda:first)
    assert memo.get([1,2,3],lambda:existing,prefer_existing=True)==(existing,False)
    assert memo.get([1,2,3],lambda:pytest.fail('new native measurement'))==(first,True)


def test_wrong_pose_can_use_another_real_emitters_different_recorded_time():
    scene={'listener_m':[0,0,0],'sources':{
        'a':{'trajectory':[{'time_s':0.,'position_m':[-.6,0,2.6]},{'time_s':10.,'position_m':[-.6,0,5.8]}]},
        'b':{'trajectory':[{'time_s':0.,'position_m':[.6,0,5.8]},{'time_s':10.,'position_m':[.6,0,2.6]}]}}}
    endpoint,point,separation=wrong_registered_pose(scene,np.array([-.6,0,2.6]),0.,current_endpoint='a')
    assert endpoint=='b'and np.array_equal(point,[.6,0,2.6])and separation>=20
    scene['sources'].pop('b')
    with pytest.raises(ValueError,match='20deg'):wrong_registered_pose(scene,np.array([-.6,0,2.6]),0.,current_endpoint='a')
