from pathlib import Path
import json
import pytest
import numpy as np
from avengine.capture.v7_registered_ue import NativeRegisteredUEVideo
from avengine.capture.v7_kujiale_standing_construction import NativeStandingPlacementExhausted,standing_group_proposals
pytestmark=pytest.mark.fast_unit


def scene():
    points=[[-1.,1.3,-3.],[1.,1.3,-3.],[0.,1.3,-4.5],[2.,1.3,-4.5]]
    assets=[{'endpoint_id':'voice'+str(i+1),'asset_id':'person'+str(i+1)}for i in range(3)]
    return {'listener_m':[0.,1.5,0.],'camera_yaw_knots':[{'yaw_deg':0.}],
        'sources':{a['endpoint_id']:{'trajectory':[{'position_m':p}]}for a,p in zip(assets,points)},
        'native_placement_candidates':points},assets


def test_proposals_backtrack_complete_groups_and_preserve_real_spacing_and_fixed_view():
    s,assets=scene();groups=list(standing_group_proposals(s,assets))
    assert len(groups)>1 and groups[0]!=groups[1]
    for group in groups:
        points=np.asarray(list(group.values()))
        assert len(points)==3
        assert min(np.linalg.norm((points[i]-points[j])[[0,2]])for i in range(3)for j in range(i))>=1.
        assert np.linalg.norm(points[:,[0,2]],axis=1).min()>=1.5
    assert s['camera_yaw_knots']==[{'yaw_deg':0.}]


def test_retry_keeps_original_admission_and_retries_a_whole_group(tmp_path):
    video=NativeRegisteredUEVideo.__new__(NativeRegisteredUEVideo);video.scene,video.assets=scene();video.shared_native_standing=None
    calls=[]
    def admit(directory):
        calls.append(directory)
        if len(calls)==1:raise NativeStandingPlacementExhausted('registered UE actor lacks original qualified placement')
        assert set(video._standing_candidate_group)=={'voice1','voice2','voice3'}
        Path(directory).mkdir(parents=True);(Path(directory)/'standing_admission.json').write_text(json.dumps({'status':'pass','checks':['original native contact/collision/acoustics']}))
        return ['original native contact/collision/acoustics']
    video._qualify_standing_once=admit
    assert video.qualify_standing(tmp_path)==['original native contact/collision/acoustics']
    proof=json.loads((tmp_path/'standing_admission.json').read_text())
    assert proof['automatic_group_backtracking'][-1]['status']=='pass_original_native_gates'
    assert video._standing_candidate_group is None


def test_counterfactual_cannot_independently_move_its_shared_measured_people(tmp_path):
    video=NativeRegisteredUEVideo.__new__(NativeRegisteredUEVideo);video.scene,video.assets=scene();video.shared_native_standing={'status':'pass'}
    def reject(directory):raise NativeStandingPlacementExhausted('cannot move actual shared geometry')
    video._qualify_standing_once=reject
    with pytest.raises(NativeStandingPlacementExhausted):video.qualify_standing(tmp_path)
    assert not hasattr(video,'_standing_candidate_group')


def test_other_native_failures_are_not_reclassified_as_construction_preferences(tmp_path):
    video=NativeRegisteredUEVideo.__new__(NativeRegisteredUEVideo)
    def reject(directory):raise ValueError('original package hash differs')
    video._qualify_standing_once=reject
    with pytest.raises(ValueError,match='package hash'):video.qualify_standing(tmp_path)
