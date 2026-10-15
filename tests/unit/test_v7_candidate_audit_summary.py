import json
import pytest
from avengine.timeline.v7_candidate_audit_summary import candidate_window_summary
pytestmark=pytest.mark.fast_unit


def test_cached_stats_are_invalidated_by_actual_native_audit_changes(tmp_path):
    folder=tmp_path/'native';folder.mkdir();(folder/'questions.json').write_text(json.dumps([{'logical_question_id':'one','qa_id':'QA-15'}]))
    audit=folder/'candidate_audit.json';audit.write_text(json.dumps({'deferred':[{'qa_id':'QA-06','code':'distractors_equal_gold'}]}))
    first=candidate_window_summary(folder,tmp_path/'cache')
    assert first['legal_family_counts']=={'QA-15':1}
    audit.write_text(json.dumps({'deferred':[{'qa_id':'QA-06','code':'now_different'}]}))
    second=candidate_window_summary(folder,tmp_path/'cache')
    assert second['candidate_audit_sha256']!=first['candidate_audit_sha256']
    assert second['deferred_codes']['QA-06']==['now_different']
    assert len(list((tmp_path/'cache').glob('*.json')))==2


def test_multiple_forms_of_one_logical_candidate_count_once(tmp_path):
    folder=tmp_path/'native';folder.mkdir();(folder/'candidate_audit.json').write_text('{}')
    (folder/'questions.json').write_text(json.dumps([{'logical_question_id':'same','qa_id':'QA-06'},{'logical_question_id':'same','qa_id':'QA-06'}]))
    assert candidate_window_summary(folder,tmp_path/'cache')['legal_family_counts']=={'QA-06':1}
