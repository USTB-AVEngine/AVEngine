import json
import pytest
from avengine.timeline.v7_admitted_pool_reselection import parent_entries
pytestmark=pytest.mark.fast_unit

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_rejects_candidate_supplement_when_original_full_pool_replay_is_not_complete(tmp_path,monkeypatch):
    # This fixture isolates the original full-pool replay check. Native scene
    # delivery policy has independent measured-camera/layout/speed tests.
    monkeypatch.setattr("avengine.timeline.v7_export_scene_policy.cached_parent_policy",lambda path,cache:{"status":"pass"})
    root=tmp_path/'native/parents/p';pool=root/'pool/w';seal=root/'seal'
    row={'scene_id':'p','members':['v0_a0']}
    pointer={'attempt':'seal','result':{'status':'integration_pilot_admitted','windows':[{'window_id':'w','member':'v0_a0'}]},
        'recipe':{'input_recipe':{'row':row}}}
    put(root/'integration_seal_completed.json',pointer)
    put(root/'selected_windows_v0_a0_completed.json',{'recipe':{'pool_directory':str(root/'pool')}})
    put(seal/'private/w_answers.json',[])
    put(pool/'questions.json',[{'question_id':'unreplayed'}])
    put(pool/'independent_v6_truth.json',{'status':'pass','checked_forms':0,'mismatches':[]})
    t=tmp_path/'export/parent_rows/000.jsonl';t.parent.mkdir(parents=True);t.write_text('')
    with pytest.raises(ValueError,match='original full-pool replay is incomplete'):
        parent_entries({'slot':0,'export':{'source_seal':str(root/'integration_seal_completed.json')}},tmp_path)


def test_never_reads_evidence_outside_authorized_output_roots(tmp_path):
    with pytest.raises(ValueError,match='escaped authorized'):
        parent_entries({'slot':0,'export':{'source_seal':'/unrelated/other-task/seal.json'}},tmp_path)


def test_actual_positive_taxonomy_and_speaker_feedback_is_replaced_not_accumulated(tmp_path):
    from avengine.timeline.v7_admitted_pool_reselection import record_batch_selection
    p=tmp_path/'state.json'
    def row(qid,family,answer):return {'question_id':qid,'independent_parent_scene_id':'p','qa_id':family,'canonical_answer':answer,'modality':'audio'}
    rows=[row('a','V7-CLASS34','dog_bark'),row('b','STARSS-SPEAKERS',3)]
    record_batch_selection(p,rows);record_batch_selection(p,rows)
    state=json.loads(p.read_text());assert state['class34_positive_counts']=={'dog_bark':1}
    assert state['live_speaker_count_counts']=={'3':1}
    record_batch_selection(p,[row('c','V7-CLASS34','cat_meow')])
    assert json.loads(p.read_text())['class34_positive_counts']=={'cat_meow':1}
