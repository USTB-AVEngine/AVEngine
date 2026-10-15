import copy,json
import pytest
from avengine.timeline.v7_full_question_delivery import advance_independent_feedback,current_selection_snapshot
from avengine.timeline.v7_demand import SELECTION_FEEDBACK_ENV,default_targets
pytestmark=pytest.mark.fast_unit


def item(name,interval,answer='yes',actor='voice1'):
    q={'question_id':name,'logical_question_id':name,'qa_id':'QA-06',
       'question_en':f'During {interval}, did the woman move?'}
    g={'question_id':name,'canonical_answer':answer,'evidence':{'target_actor_id':actor}}
    return q,g


def test_overlapping_crops_consume_only_one_independent_deficit():
    counts={'QA-06':-1.4};seen=set();q,g=item('w0','[0, 10)')
    advance_independent_feedback('p',{'v0_a0':[q]},{'v0_a0':[g]},counts,seen)
    q,g=item('w1','[5, 15)')
    assert advance_independent_feedback('p',{'v0_a0':[q]},{'v0_a0':[g]},counts,seen)=={}
    assert counts['QA-06']==pytest.approx(-.4)


def test_changed_answer_and_different_physical_actor_still_consume_a_deficit():
    counts={};seen=set()
    for i,(answer,actor)in enumerate([('yes','voice1'),('no','voice1'),('yes','voice2')]):
        q,g=item(str(i),'[0, 10)',answer,actor)
        advance_independent_feedback('p',{'v0_a0':[q]},{'v0_a0':[g]},counts,seen)
    assert counts=={'QA-06':3}


def test_same_target_and_answer_in_four_members_does_not_count_four_times():
    chosen={};gold={}
    for member in ['v0_a0','v0_a1','v1_a0','v1_a1']:
        q,g=item(member,'[0, 10)');chosen[member]=[q];gold[member]=[g]
    before=copy.deepcopy((chosen,gold));counts={}
    assert advance_independent_feedback('p',chosen,gold,counts,set())=={'QA-06':1}
    assert (chosen,gold)==before


def test_selection_reads_latest_admitted_feedback_after_rendering(tmp_path,monkeypatch):
    path=tmp_path/'state.json';path.write_text(json.dumps({'logical_questions':100,'family_counts':{'QA-06':100}}))
    monkeypatch.setenv(SELECTION_FEEDBACK_ENV,str(path))
    demand={'observed_logical_questions':0,'observed_family_counts':{},'target_shares':default_targets()}
    snapshot=current_selection_snapshot(demand,next_questions=14)
    assert snapshot['observed_logical_questions']==100
    assert snapshot['observed_family_counts']=={'QA-06':100}
    assert snapshot['family_deficits']['QA-06']==0


def test_live_feedback_does_not_require_stale_constructor_snapshot_fields(tmp_path,monkeypatch):
    path=tmp_path/'actual_state.json';path.write_text(json.dumps({'logical_questions':37,'family_counts':{'QA-15':37}}))
    monkeypatch.setenv(SELECTION_FEEDBACK_ENV,str(path))
    snapshot=current_selection_snapshot({'required_independent_family_counts':{'QA-15':3}},next_questions=70)
    assert snapshot['observed_logical_questions']==37
    assert snapshot['observed_family_counts']=={'QA-15':37}
    assert snapshot['target_shares']==default_targets()
    path.write_text(json.dumps({'logical_questions':1,'family_counts':{'QA-15':37}}))
    with pytest.raises(ValueError,match='count actual selected logical items'):
        current_selection_snapshot({},next_questions=70)
