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
