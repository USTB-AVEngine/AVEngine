from collections import Counter
import pytest
from avengine.timeline.v7_answer_branch_demand import summarize,construction_request,selection_priority
from avengine.timeline.v7_source_usage import next_joint_requests
pytestmark=pytest.mark.fast_unit


def test_branch_ratios_use_actual_independent_answers_and_do_not_conflate_occlusion():
    rows=[{'question_id':str(i),'qa_id':'QA-08','canonical_answer':v}for i,v in enumerate(['out_of_view','visible','fully_occluded','visible'])]
    result=summarize(rows)['families']['QA-08']
    assert result['positive_fraction']==.25 and result['target_met']
    with pytest.raises(ValueError,match='unique'):summarize([rows[0],rows[0]])
    request=construction_request({'answer_branch_distribution':summarize(rows)})
    assert 1<=request['required_offscreen_sounding_people']<=2 and request['offscreen_entities_must_be_real_registered_bodies']


def test_selection_favours_missing_positive_without_changing_an_answer():
    counts=Counter({'QA-08':10});branches=Counter({('QA-08','out_of_view'):0});out={'qa_id':'QA-08','canonical_answer':'out_of_view'};inside={'qa_id':'QA-08','canonical_answer':'visible'}
    assert selection_priority(out,counts,branches)<selection_priority(inside,counts,branches)
    assert out['canonical_answer']=='out_of_view'


def test_joint_inventory_rotation_prefers_unused_record_and_entity():
    inventory={'recordings':{'used':{'sound_class':'dog_bark'},'new':{'sound_class':'dog_bark'}},'sounding_entities':{'used_dog':{'kind':'animal','allowed_classes':['dog_bark']},'new_dog':{'kind':'animal','allowed_classes':['dog_bark']}}}
    result=next_joint_requests(inventory,{'recording_counts':{'used':1},'sounding_entity_counts':{'used_dog':1}})
    assert result[0]['asset_id']=='new_dog'and result[0]['sound_asset_id']=='new'and result[0]['planned_is_not_used']


def test_route_retry_evidence_retains_both_attempts(tmp_path):
    import json
    from avengine.assets.roster_motion.diagnostics import write_route_summary
    p=tmp_path/'motion_route_attempt_summary.json'
    write_route_summary(p,{'attempt':1});write_route_summary(p,{'attempt':2})
    assert json.loads(p.read_text())=={'attempt':2}
    evidence=list(tmp_path.glob('motion_route_attempt_summary_*.json'))
    assert {json.loads(path.read_text())['attempt']for path in evidence}=={1,2}


def test_branch_balance_uses_real_candidates_and_reports_missing_positive():
    from avengine.timeline.v7_answer_branch_demand import balance_available_branches
    entries=[{'row':{'qa_id':'QA-20','canonical_answer':'none_of_visible'if i==0 else str(i)}}for i in range(21)]
    selected,report=balance_available_branches(entries)
    assert .20<=sum(x['row']['canonical_answer']=='none_of_visible'for x in selected)/len(selected)<=.30
    assert all(x in entries for x in selected)and len(selected)>=3
    only_inside=entries[1:];selected,report=balance_available_branches(only_inside)
    assert selected==only_inside and not report['QA-20']['range_feasible']
