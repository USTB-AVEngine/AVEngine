import json
from types import SimpleNamespace
import pytest
from avengine.capture.v7_solved_pose_execution import solved_support_proposals
from avengine.capture.v7_family_native_preflight import run_family_native_preflights
pytestmark=pytest.mark.fast_unit


def test_solved_support_keeps_exact_yaw_point_and_single_attempt_without_mutating_input():
    support={'joint_solved_pose':True,'yaw_rad':1.23,'original_surface_point_m':[2.,0.,3.],'normal':[0,1,0]}
    result=solved_support_proposals(support)
    assert len(result)==1 and result[0]['yaw_rad']==1.23 and result[0]['point']==[2.,0.,3.]
    assert result[0]['_outward_clearance_m']==0.
    assert 'point'not in support


def test_unconfigured_legacy_scene_does_not_start_a_native_probe(tmp_path):
    video=SimpleNamespace(actors=[])
    assert run_family_native_preflights(video,{'automatic_construction':None},[],tmp_path)=={}


def test_family_preflight_cannot_run_while_a_required_actor_is_still_unplaced(tmp_path):
    entries=[object(),object()];video=SimpleNamespace(actors=entries)
    scene={'automatic_construction':{'qa21_target_asset_ids':['target']}}
    with pytest.raises(ValueError,match='every final placed actor'):
        run_family_native_preflights(video,scene,entries[:1],tmp_path)


def test_qa21_failure_retains_the_native_witness_and_stops_before_encoding(tmp_path,monkeypatch):
    import avengine.construction.joint.families.qa21 as qa21
    entries=[object()];video=SimpleNamespace(actors=entries,_native_pair_pixels=lambda scene,e:{'target':99})
    monkeypatch.setattr(qa21,'native_candidate_check',lambda video,scene,e,pixels:{'status':'fail','reason':'actual_visible_pixels_below_100','pixels':pixels})
    with pytest.raises(ValueError,match='QA-21 final native preflight failed'):
        run_family_native_preflights(video,{'automatic_construction':{'qa21_target_asset_ids':['target']}},entries,tmp_path)
    assert json.loads((tmp_path/'qa_21_native_preflight.json').read_text())['pixels']['target']==99


def test_qa09_negative_branch_uses_complete_window_probe_without_later_clear(monkeypatch):
    import avengine.capture.continuous_native_visibility as visibility
    import avengine.construction.joint.families.qa09 as qa09
    from avengine.capture.v7_dynamic_occlusion_proposals import measure_dynamic_relations
    class Stream:
        def __init__(self,video):self.closed=False
        def close(self):self.closed=True
        def receipt(self):return {'closed':self.closed}
    monkeypatch.setattr(visibility,'NativeVisibilityStream',Stream)
    monkeypatch.setattr(qa09,'probe_native_sequence',lambda *args,**kwargs:{'status':'pass','complete_window':True,'observed_answer':'no'})
    entries=[{'endpoint_id':'t','declaration':{'asset_id':'target','native_instance_id':1}},
             {'endpoint_id':'o','declaration':{'asset_id':'occluder','native_instance_id':2}}]
    video=SimpleNamespace(actors=entries,_native_pair_pixels=lambda *args:{})
    scene={'native_dynamic_occlusion_relations':[{'target_asset_id':'target','occluder_asset_id':'occluder',
        'qa09_demand':{'target_asset_id':'target','occluder_asset_id':'occluder','answer':'no'}}]}
    report=measure_dynamic_relations(video,scene,entries)
    assert report['satisfies_request']and report['relations'][0]['native_complete_window_certificate']['observed_answer']=='no'
    assert report['native_control_restoration']['closed']


def test_direct_preflight_motion_samples_use_the_same_native_residency_bound(monkeypatch):
    import numpy as np
    from avengine.capture.v7_phase2_native import NativePhase2Video
    events=[];video=object.__new__(NativePhase2Video);video._phase2_scene={'sources':{}};video._pose_context_configuration={}
    video._maybe_recycle_pose_context=lambda scene,index:events.append('residency_guard')
    def apply(self,entry,index):events.append('original_apply');return {'body_to_world_m':np.eye(4).tolist()}
    monkeypatch.setattr(NativePhase2Video.__mro__[1],'_apply_motion_entry',apply)
    entry={'endpoint_id':'voice1','declaration':{'silent_decor':False},'motion_episode':{'source_frames':[{'mouth_m':[0,1,0],'feet':{}}],'pose_glbs':['pose0.glb']}}
    result=video._apply_motion_entry(entry,0)
    assert events==['residency_guard','original_apply']
    assert result['attached_emitter_readbacks']==[]
