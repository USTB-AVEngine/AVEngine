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
    other=SimpleNamespace(**vars(p));other.video=object();other.scene={};worlds={'v0':p,'v1':other}
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


def test_registered_generator_encodes_original_prepared_scene_without_constructor_or_qualify(tmp_path,monkeypatch):
    import avengine.timeline.v7_registered_generation as generation
    row={'scene_id':'p','members':['v0_a0','v0_a1','v1_a0','v1_a1']};house={};program={'records':[],'events':[]};assets=[];seen={}
    class StopAfterFirstEncode(Exception):pass
    class Video:
        standing_checks=[{'status':'pass'}]
        def encode(self,scene,path,**kwargs):
            assert scene is worlds['v0'].scene and self._prepared_registered_scene is scene
            seen['encoded']=True;return {'wall_seconds':0.}
        def close(self):seen['closed']=True
    class Payload:
        def __init__(self,number):
            self.closed=False;self.row=row;self.house=house;self.scene={'world':number};self.program=program;self.assets=assets;self.video=Video();self.directory=tmp_path/number;self.certificate=SimpleNamespace(passed=True)
        def close(self):self.closed=True;self.video.close()
    worlds={'v0':Payload('v0'),'v1':Payload('v1')}
    monkeypatch.setattr('avengine.registry.kujiale_room_categories.bind_formal_room_category',lambda row,inputs,*args:(row,inputs,None))
    monkeypatch.setattr('avengine.timeline.v7_production_inputs.attach_runtime_contract',lambda row,inputs:row)
    monkeypatch.setattr('avengine.timeline.v7_production_inputs.validate_registered_sampling_row',lambda *args:None)
    monkeypatch.setattr(generation,'choose_assets',lambda *args:pytest.fail('reselected prepared assets'))
    monkeypatch.setattr(generation,'qualify_licenses',lambda *args:[])
    monkeypatch.setattr(generation,'build_registered_scene',lambda *args,**kwargs:pytest.fail('rebuilt registered scene'))
    monkeypatch.setattr(generation,'parent_dry',lambda *args:({}, {'endpoint_speaker_ids':{}}))
    monkeypatch.setattr(generation,'derive_native_appearance_evidence',lambda *args:{'status':'pass','review':{'actors':{}}})
    monkeypatch.setattr('avengine.capture.v7_registered_native.NativeRegisteredVideo',lambda *args,**kwargs:pytest.fail('created another native world'))
    class Gate:
        def __init__(self,scene,working,inputs,*args):
            assert scene is worlds['v0'].scene and inputs['fresh_rir_workers']==8
        def check(self,*args):return {'status':'pass'}
        def close(self):pass
    monkeypatch.setattr('avengine.timeline.v7_position_admission.PositionAdmission',Gate)
    def stage(root,name,recipe,callback):
        assert name=='visual_v0'and recipe['scene']is worlds['v0'].scene
        d=root/'actual_visual';d.mkdir();callback(d);raise StopAfterFirstEncode()
    monkeypatch.setattr(generation,'stage',stage)
    with pytest.raises(StopAfterFirstEncode):
        generation.generate_registered_parent(row,house,{'render_backend':'cpu_llvmpipe','fresh_rir_workers':8},[],{}, {},str(tmp_path/'output'),prepared_visuals=worlds)
    assert seen['encoded']and seen['closed']and worlds['v0'].closed
