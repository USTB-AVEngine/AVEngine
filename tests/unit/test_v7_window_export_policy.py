from copy import deepcopy
import pytest
from avengine.timeline.v7_export_window_policy import scoped_window_policy,scope_receipt_windows
pytestmark=pytest.mark.fast_unit


def policy():
    return {'revision':'original','status':'pass','reasons':[],
        'quality_observations':['w03:window_measured_root_speed_out_of_range'],
        'native_views':{'windows':{f'w{i:02d}':{'actors':[{'status':'fail'if i==3 else'pass','median_root_speed_m_s':.128 if i==3 else.94,'required_range_m_s':[.5,1.2]}]}for i in range(5)}}}


def test_bad_speed_crop_is_removed_without_changing_values_or_other_crops():
    source=policy();original=deepcopy(source);windows=[{'window_id':f'w{i:02d}'}for i in range(5)]
    result=scoped_window_policy(source,windows)
    assert result['status']=='pass'and result['excluded_window_ids']==['w03']
    assert result['delivered_window_ids']==['w00','w01','w02','w04']
    assert source==original and result['native_views']==source['native_views']
    assert result['speed_ranges_changed']is False and result['answers_changed']is False


def test_window_selection_cannot_hide_a_moving_camera_or_missing_native_truth():
    source=policy();source.update(status='fail',reasons=['v0:camera_moved'])
    assert scoped_window_policy(source,[{'window_id':f'w{i:02d}'}for i in range(5)])['status']=='fail'
    source['reasons']=['missing_original_native_camera_and_body_readbacks']
    assert scoped_window_policy(source,[{'window_id':'w03'}])['status']=='fail'


def test_all_bad_crops_produce_no_model_delivery():
    result=scoped_window_policy(policy(),[{'window_id':'w03'}])
    assert result['status']=='fail'and not result['delivered_window_ids']


def test_receipt_subset_is_derived_without_mutating_source_seal_or_reader():
    receipt={'export':{'source_seal':'immutable.json','windows':[{'window_id':f'w{i:02d}'}for i in range(5)]},'reader':{'status':'pass','actual_decoded_windows':5}}
    original=deepcopy(receipt);derived,report=scope_receipt_windows(receipt,policy())
    assert receipt==original and derived['export']['source_seal']=='immutable.json'
    assert len(derived['export']['windows'])==4 and derived['reader']==receipt['reader']
    assert len(derived['export']['source_windows_before_exclusion'])==5
