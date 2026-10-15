import copy,json
from pathlib import Path
import pytest
from avengine.capture.v7_kujiale_joint_standing import completed_shared_standing
pytestmark=pytest.mark.fast_unit


def fixture(parent):
    root=parent/'view0';root.mkdir();actors=[{'endpoint_id':'voice'+str(i+1),'asset_id':k,'position_m':[i*1.2,0.,-3.],'native_support_floor_y_m':0.}for i,k in enumerate(['blue','green','yellow'])]
    data={'frame_count':450,'native_body_foot_status':'pass','native_source_material_status':'pass','standing_admission':[{'status':'pass','known_bad_detected':True}for _ in actors],'frames':[{'actors':copy.deepcopy(actors)}for _ in range(450)]};path=root/'native_registered_visual_evidence.json';path.write_text(json.dumps(data));(parent/'visual_v0_completed.json').write_text(json.dumps({'attempt':'view0'}));target=[{'endpoint_id':'voice'+str(i+1),'asset_id':k}for i,k in enumerate(['green','blue','yellow'])];return path,data,target


def test_second_appearance_keeps_first_full_native_positions_and_keeps_all_checks(tmp_path):
    path,data,target=fixture(tmp_path);r=completed_shared_standing(tmp_path,target)
    assert r['actors']['voice3']['position_m']==[2.4,0.,-3.]
    assert r['current_native_contact_collision_visibility_and_acoustics_must_rerun']and not r['thresholds_changed']
    assert completed_shared_standing(tmp_path,[{**a,'asset_id':b}for a,b in zip(target,['blue','green','yellow'])])is None


def test_partial_or_contact_failed_reference_cannot_be_reused(tmp_path):
    path,data,target=fixture(tmp_path);data['frames'].pop();path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='complete passed'):completed_shared_standing(tmp_path,target)


def test_a_moving_reference_cannot_be_used_as_one_static_counterfactual_pose(tmp_path):
    path,data,target=fixture(tmp_path);data['frames'][-1]['actors'][2]['position_m'][0]+=.1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='actually static'):completed_shared_standing(tmp_path,target)


def test_highlight_observation_does_not_reject_measured_shared_geometry(tmp_path):
    path,data,target=fixture(tmp_path)
    data['native_source_material_status']='fail'
    path.write_text(json.dumps(data))
    result=completed_shared_standing(tmp_path,target)
    observation=result['source_material_observation']
    assert observation['native_source_material_status']=='fail'
    assert observation['admission_effect']=='record_only_per_owner_20261015'
    assert observation['current_original_material_paths_and_package_identity_must_pass']
    assert result['actors']['voice1']['position_m']==[0.,0.,-3.]


def test_ignoring_highlights_keeps_original_failed_contact_rejected(tmp_path):
    path,data,target=fixture(tmp_path)
    data['native_source_material_status']='fail'
    data['native_body_foot_status']='fail'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='complete passed native geometry'):
        completed_shared_standing(tmp_path,target)


def test_required_physical_dog_is_not_mistaken_for_a_fourth_standing_human(tmp_path):
    path,data,target=fixture(tmp_path)
    dog={'endpoint_id':'environment1','asset_id':'required_native_dog','position_m':[5.,0.,-4.],
         'native_support_floor_y_m':0.,'source_geometry_present':True}
    for frame in data['frames']:frame['actors'].append(copy.deepcopy(dog))
    path.write_text(json.dumps(data));before=path.read_bytes()
    result=completed_shared_standing(tmp_path,target)
    assert set(result['actors'])=={'voice1','voice2','voice3'}
    assert result['current_native_contact_collision_visibility_and_acoustics_must_rerun']
    assert path.read_bytes()==before
    assert all(any(a.get('asset_id')=='required_native_dog'for a in frame['actors'])for frame in json.loads(path.read_text())['frames'])
