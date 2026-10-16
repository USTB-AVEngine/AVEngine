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


@pytest.mark.parametrize("encode_fails",[False,True])
def test_registered_generator_borrows_both_live_worlds_until_bundle_owner_cleanup(tmp_path,monkeypatch,encode_fails):
    import avengine.timeline.v7_registered_generation as generation
    row={'scene_id':'p','members':['v0_a0','v0_a1','v1_a0','v1_a1']};house={};program={'records':[],'events':[]};assets=[];seen={}
    class StopAfterFirstEncode(Exception):pass
    class Video:
        standing_checks=[{'status':'pass'}]
        def encode(self,scene,path,**kwargs):
            assert scene is worlds['v0'].scene and self._prepared_registered_scene is scene
            seen['encoded']=True
            assert not any(p.closed for p in worlds.values())
            if encode_fails:raise StopAfterFirstEncode()
            return {'wall_seconds':0.}
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
        def close(self):seen['position_gate_closed']=True
    monkeypatch.setattr('avengine.timeline.v7_position_admission.PositionAdmission',Gate)
    def stage(root,name,recipe,callback):
        assert name=='visual_v0'and recipe['scene']is worlds['v0'].scene
        d=root/'actual_visual';d.mkdir();callback(d);raise StopAfterFirstEncode()
    monkeypatch.setattr(generation,'stage',stage)
    with pytest.raises(StopAfterFirstEncode):
        generation.generate_registered_parent(row,house,{'render_backend':'cpu_llvmpipe','fresh_rir_workers':8},[],{}, {},str(tmp_path/'output'),prepared_visuals=worlds)
    # Closing v0 while v1 is live can destroy their shared GL context. Even
    # after an encode failure, ownership stays with the enclosing bundle.
    assert seen['encoded']and seen['position_gate_closed']
    assert 'closed'not in seen and not any(p.closed for p in worlds.values())


def test_media_only_resume_loads_exact_recorded_world_without_a_live_payload(tmp_path):
    from avengine.timeline.v7_registered_generation import recorded_registered_inputs
    row={'scene_id':'p','joint_prepared_scene_required':True};inputs={'render_backend':'cpu_llvmpipe'}
    scene={'scene_id':'p','listener_m':[1.,1.25,3.],'camera_yaw_knots':[{'time_s':0.,'yaw_deg':42.},{'time_s':30.,'yaw_deg':42.}]}
    original={'seed_row':row,'runtime_inputs':inputs,'house':{'house':'actual'},'recorded_program':{'events':[{'event_id':'original'}]},'assets':[{'asset_id':'real'}]}
    (tmp_path/'input_recipe.json').write_text(json.dumps(original));(tmp_path/'visual_v0_completed.json').write_text(json.dumps({'recipe':{'scene':scene}}))
    before=(tmp_path/'input_recipe.json').read_bytes()
    house,observed,program,assets=recorded_registered_inputs(tmp_path,row,inputs)
    assert observed==scene and program==original['recorded_program']and assets==original['assets']and house==original['house']
    assert(tmp_path/'input_recipe.json').read_bytes()==before
    with pytest.raises(ValueError,match='differ'):recorded_registered_inputs(tmp_path,row,{**inputs,'other':True})


def test_new_joint_generation_still_requires_live_prepared_worlds(tmp_path,monkeypatch):
    import avengine.timeline.v7_registered_generation as generation
    monkeypatch.setattr('avengine.registry.kujiale_room_categories.bind_formal_room_category',lambda row,inputs,*args:(row,inputs,None))
    monkeypatch.setattr('avengine.timeline.v7_production_inputs.attach_runtime_contract',lambda row,inputs:row)
    monkeypatch.setattr(generation,'choose_assets',lambda *args:pytest.fail('native production rebuilt solved actors'))
    with pytest.raises(ValueError,match='live prepared registered worlds required'):
        generation.generate_registered_parent({'scene_id':'p','members':['v0_a0','v0_a1','v1_a0','v1_a1'],'joint_prepared_scene_required':True},{},{},[],{},{},str(tmp_path/'new'))
