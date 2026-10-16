from types import SimpleNamespace
import json
import pytest
from avengine.timeline.v7_prepared_native_delivery import scene_inputs,arm_prepared_encode,consume_prepared_qualification
pytestmark=pytest.mark.fast_unit


def test_prepared_scene_preserves_the_actual_program_house_and_asset_objects():
    row={'scene_id':'p'};scene={};program={};assets=[];house={}
    prepared=SimpleNamespace(closed=False,video=object(),certificate=SimpleNamespace(passed=True),row=row,scene=scene,program=program,assets=assets,house=house)
    actual=scene_inputs(prepared,{'scene_id':'p'})
    assert all(a is b for a,b in zip(actual,(row,house,scene,program,assets)))
    prepared.closed=True
    with pytest.raises(ValueError,match='closed'):scene_inputs(prepared,row)


def test_consumption_uses_existing_support_proofs_and_original_acoustic_check(tmp_path,monkeypatch):
    monkeypatch.setattr('avengine.timeline.v7_cpu_visual_runtime.configure_cpu_visual',lambda *args,**kwargs:{'LP_NUM_THREADS':4})
    scene={};source=tmp_path/'prepared/standing';source.mkdir(parents=True)
    for name in ('standing_admission.json','native_support_survey.json'):(source/name).write_text('{}')
    checks=[{'status':'pass'}];entry={'endpoint_id':'voice','declaration':{'silent_decor':False}};calls=[]
    video=SimpleNamespace(standing_checks=checks,actors=[entry],_phase2_scene=scene,
        _apply_entry=lambda *args,**kwargs:{'emitter_position_m':[1.,2.,3.]},
        acoustic_position_admission=lambda ep,point:calls.append((ep,point.tolist()))or{'status':'pass'})
    prepared=SimpleNamespace(video=video,directory=source.parent)
    assert arm_prepared_encode(prepared)is video
    assert consume_prepared_qualification(video,scene,tmp_path/'encoded/standing')is checks
    assert calls==[('voice',[1.,2.,3.])]
    video.acoustic_position_admission=lambda *args:{'status':'blocked'}
    with pytest.raises(ValueError,match='original native RIR direction gate'):
        consume_prepared_qualification(video,scene,tmp_path/'encoded/standing')


def test_one_video_reuses_one_capture_lease_and_can_reacquire_after_close(monkeypatch):
    from avengine.capture.v7_phase2_native import NativePhase2Video
    import avengine.timeline.v7_native_resource_slots as slots
    calls=[]
    class Lease:
        def __init__(self):self.stream=object()
        def close(self):self.stream=None
    def acquire(kind,expected_incremental_bytes=0):calls.append((kind,expected_incremental_bytes));return Lease()
    monkeypatch.setattr(slots,'acquire_native_resource',acquire)
    video=object.__new__(NativePhase2Video);video.motion_inputs={}
    first=video._ensure_motion_capture_lease()
    assert video._ensure_motion_capture_lease()is first
    assert calls==[('motion_capture',0)]  # Runtime policy owns the actual reservation.
    first.close();assert video._ensure_motion_capture_lease()is not first
    assert len(calls)==2


def test_generator_does_not_rebuild_or_requalify_a_prepared_world(tmp_path,monkeypatch):
    import avengine.timeline.v7_phase2_generation as generation
    import avengine.timeline.v7_production_inputs as production
    import avengine.timeline.v7_family_clock as clock
    import avengine.capture.v7_phase2_native as native
    import avengine.timeline.v7_position_admission as admission
    import avengine.timeline.v7_material_quality_fallback as fallback
    row={'scene_id':'p','members':['v0_a0'],'automatic_construction':{'family':'QA-10'},'joint_prepared_scene_required':True}
    scene={'scene_id':'p'};assets=[{'asset_id':'actual'}];program={'records':[],'events':[]};house={'actual':True};seen={}
    class StopAfterRender(Exception):pass
    class Video:
        standing_checks=[{'status':'pass'}]
        motion_inputs={'fresh_rir_workers':1}
        def encode(self,used,path,**kwargs):
            assert used is scene and self._prepared_native_encode
            seen['encoded']=True;return {'owner_material_quality_fallbacks':[]}
        def close(self):seen['closed']=True
    video=Video()
    class Prepared:
        closed=False;certificate=SimpleNamespace(passed=True);directory=tmp_path/'prepared'
        def close(self):self.closed=True;video.close()
    prepared=Prepared();prepared.row=row;prepared.video=video;prepared.scene=scene;prepared.program=program;prepared.assets=assets;prepared.house=house
    monkeypatch.setattr(production,'attach_runtime_contract',lambda row,inputs:row)
    monkeypatch.setattr(production,'bind_semantic_registry',lambda semantic,assets:semantic)
    monkeypatch.setattr(clock,'build_constructed_scene',lambda *args:pytest.fail('rebuilt scene'))
    monkeypatch.setattr(native,'NativePhase2Video',lambda *args,**kwargs:pytest.fail('rebuilt native video'))
    class Gate:
        def __init__(self,base,working,inputs,*args):assert inputs['fresh_rir_workers']==8;seen['audio_inputs']=inputs
        def check(self,*args):return {'status':'pass'}
        def close(self):pass
    monkeypatch.setattr(admission,'PositionAdmission',Gate)
    monkeypatch.setattr(fallback,'recover_complete_native_attempt',lambda *args:None)
    monkeypatch.setattr(fallback,'qualify_missing_materials',lambda root,assets,visual:visual)
    monkeypatch.setattr(generation,'native_checks',lambda *args,**kwargs:{'status':'pass'})
    def stage(root,name,recipe,callback):
        assert name=='visible_native'and recipe['scene']is scene
        attempt=root/'visual';attempt.mkdir();callback(attempt);raise StopAfterRender()
    monkeypatch.setattr(generation,'stage',stage)
    inputs={'fresh_rir_workers':8,'fresh_rir_cpu_ids':list(range(16,24))}
    with pytest.raises(StopAfterRender):generation.generate_phase2_parent(row,{},inputs,[],{'records':[]},{},str(tmp_path/'native'),prepared_native=prepared)
    assert seen['encoded']and seen['closed']and seen['audio_inputs']is inputs


def test_consumer_preserves_and_checks_contract_metadata_without_rebuilding(monkeypatch):
    row={'scene_id':'p'};scene={'physical':'untouched'};assets=[];program={'events':['same']};house={}
    prepared=SimpleNamespace(closed=False,video=object(),certificate=SimpleNamespace(passed=True),row=row,scene=scene,program=program,assets=assets,house=house)
    receipt={'manifest_path':'caller_contract','manifest_sha256':'hash'};observed=[]
    def validate(actual,catalog):
        observed.append((actual,catalog));return {'status':'pass'}
    monkeypatch.setattr('avengine.timeline.v7_production_inputs.validate_sampling_row',validate)
    actual=scene_inputs(prepared,{'scene_id':'p','production_sampling':receipt})
    assert all(a is b for a,b in zip(actual,(row,house,scene,program,assets)))
    assert observed[0][0]['production_sampling']==receipt and observed[0][1]is assets
    assert scene['physical']=='untouched' and program['events']==['same']
    assert scene['production_sampling_intake']=={'status':'pass'}
    with pytest.raises(ValueError,match='contract differs'):
        scene_inputs(prepared,{'scene_id':'p','production_sampling':{'manifest_path':'other'}})
