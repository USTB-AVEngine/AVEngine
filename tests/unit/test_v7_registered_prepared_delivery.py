from types import SimpleNamespace
import json
import numpy as np
import pytest
from avengine.timeline.v7_registered_prepared_delivery import apply_registered_root_pin,registered_scene_inputs,arm_registered_prepared,consume_registered_qualification
pytestmark=pytest.mark.fast_unit


def test_strict_registered_root_ignores_legacy_mouth_and_floor_reconstruction():
    scene={'joint_strict_placement':True,'native_attachment_supports':{'a':{'joint_solved_pose':True,'joint_expected_root_m':[2.,.003,4.],'yaw_rad':.7}}};root=np.eye(4);root[:3,3]=[0,-1.3,0]
    assert apply_registered_root_pin(scene,{'declaration':{'asset_id':'a'}},root)
    assert np.array_equal(root[:3,3],[2.,.003,4.])
    assert np.arctan2(root[0,2],root[0,0])==pytest.approx(.7)
    with pytest.raises(ValueError,match='missing its solved root pin'):apply_registered_root_pin(scene,{'declaration':{'asset_id':'other'}},root)


def test_both_live_registered_worlds_are_required_without_rebuilding():
    row={'scene_id':'p'};p=SimpleNamespace(closed=False,video=object(),certificate=SimpleNamespace(passed=True),row=row,house={},scene={},program={},assets=[])
    other=SimpleNamespace(**vars(p));worlds={'v0':p,'v1':other}
    actual=registered_scene_inputs(worlds,row)
    assert actual[0]is row and actual[2]is p.scene and actual[4]is p.assets
    with pytest.raises(ValueError,match='both independently'):registered_scene_inputs({'v0':p},row)
    other.closed=True
    with pytest.raises(ValueError,match='closed or uncertified'):registered_scene_inputs(worlds,row)


def test_registered_qualification_reuses_exact_original_physics_but_checks_actual_emitter(tmp_path):
    source=tmp_path/'prepared/standing';source.mkdir(parents=True);(source/'standing_admission.json').write_text('{}');scene={};calls=[]
    video=SimpleNamespace(actors=[{'endpoint_id':'voice1','declaration':{}}],standing_checks=[{'status':'pass'}],_apply_entry=lambda *args,**kwargs:{'emitter_position_m':[1.,2.,3.]},acoustic_position_admission=lambda ep,p:calls.append((ep,p.tolist()))or{'status':'pass'})
    payload=SimpleNamespace(video=video,scene=scene,directory=source.parent)
    assert arm_registered_prepared(payload)is video
    assert consume_registered_qualification(video,scene,tmp_path/'encoded/standing')is video.standing_checks
    assert calls==[('voice1',[1.,2.,3.])]
    with pytest.raises(ValueError,match='original world'):consume_registered_qualification(video,{},tmp_path/'bad')
