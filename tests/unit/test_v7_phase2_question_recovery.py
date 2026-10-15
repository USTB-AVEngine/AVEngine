import json
from pathlib import Path
import pytest
from avengine.timeline.v7_phase2_question_recovery import regenerate_questions
pytestmark=pytest.mark.fast_unit


def test_recovery_preserves_native_bytes_and_only_changes_requested_condition_metadata(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();output=tmp_path/'fresh'
    row={'scene_id':'physical_parent','members':['v0_a0'],'split':'train','scene_demand':{'requested_conditions':['bad_metadata']}}
    scene={'scene_id':'physical_parent','events':[{'actual':'unchanged'}],'scene_demand':row['scene_demand']}
    (source/'input_recipe.json').write_text(json.dumps({'row':row,'assets':[],'program':{},'house':{},'inputs':{}}))
    for name,recipe,files in [('visible_native',{'scene':scene},['scene_video.mp4','native_visible_visual_evidence_phase2.json']),('audio_v0_a0',{'scene':scene},['scene_foa_full_tail.wav']),('dry',{},['dry.wav'])]:
        d=source/name;d.mkdir()
        for f in files:(d/f).write_bytes(b'original native media')
        if name=='visible_native':(d/'native_visible_visual_evidence_phase2.json').write_text(json.dumps({'visible_color_checks':[]}))
        (source/(name+'_completed.json')).write_text(json.dumps({'attempt':name,'recipe':recipe,'result':{},'required_files':files}))
    (source/'resolved_footstep_contacts.json').write_text('{}')
    original={str(p.relative_to(source)):p.read_bytes()for p in source.rglob('*')if p.is_file()}
    monkeypatch.setattr('avengine.assets.v7_visible_color.rebuild_appearance_review',lambda *args:({},{}))
    def finish(*args):
        actual=args[10]
        assert actual['events']==scene['events']
        assert actual['scene_demand']['requested_conditions']==['stable_mask_intervals']
        assert args[12]==source/'visible_native' and args[13]==source/'audio_v0_a0'
        return {'status':'tested_original_delivery_dispatch'}
    monkeypatch.setattr('avengine.timeline.v7_phase2_generation.finish_phase2_delivery',finish)
    result=regenerate_questions(source,output,requested_conditions=['stable_mask_intervals'])
    assert result['new_native_renders']==0
    assert original=={str(p.relative_to(source)):p.read_bytes()for p in source.rglob('*')if p.is_file()}
    with pytest.raises(FileExistsError,match='fresh attempt root'):
        regenerate_questions(source,output)


def test_actual_delivery_entry_has_its_original_color_revision_before_stage(tmp_path,monkeypatch):
    import avengine.timeline.v7_phase2_generation as generation
    from avengine.assets.v7_visible_color import COLOR_EVIDENCE_REVISION
    class StopAtWindows(Exception):pass
    monkeypatch.setattr(generation,'native_checks',lambda *args:{'status':'pass'})
    monkeypatch.setattr(generation,'acceptance_crop_starts',lambda *args:[0.,5.,10.,15.,20.])
    def stage(root,name,recipe,callback):
        assert name=='windows_v0_a0'and recipe['color_evidence_revision']==COLOR_EVIDENCE_REVISION
        raise StopAtWindows()
    monkeypatch.setattr(generation,'stage',stage)
    with pytest.raises(StopAtWindows):
        generation.finish_phase2_delivery({}, {}, {}, [], {}, None, tmp_path, 0.,0.,{}, {}, {},tmp_path,tmp_path,{}, {},{},tmp_path/'feet.json')
