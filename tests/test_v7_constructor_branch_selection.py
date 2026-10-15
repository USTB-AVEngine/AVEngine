import copy
import pytest
from avengine.timeline.v7_constructor_branch_selection import restrict_constructed_branch
pytestmark=pytest.mark.fast_unit

def test_constructed_outside_branch_retains_original_open_and_mcq_pair_and_gold():
    public=[{'question_id':'v','logical_question_id':'visible','qa_id':'QA-08','form':'open'},
        {'question_id':'o','logical_question_id':'outside','qa_id':'QA-08','form':'open'},
        {'question_id':'m','logical_question_id':'outside','qa_id':'QA-08','form':'mcq'},
        {'question_id':'other','logical_question_id':'other','qa_id':'QA-20','form':'open'}]
    gold=[{'question_id':'v','canonical_answer':'visible_clear'},{'question_id':'o','canonical_answer':'out_of_view'},
        {'question_id':'m','canonical_answer':'out_of_view'},{'question_id':'other','canonical_answer':'1'}]
    original=copy.deepcopy(gold)
    out,audit=restrict_constructed_branch(public,gold,{'target_family':'QA-08','answer_branch_targets':{'QA-08':{'requested_branch':'outside'}}})
    assert {q['question_id']for q in out}=={'o','m','other'}
    assert audit['answers_changed']is False and gold==original

def test_missing_legal_branch_is_not_synthesized_or_counted_as_fulfilled():
    public=[{'question_id':'v','logical_question_id':'visible','qa_id':'QA-08','form':'open'}]
    out,audit=restrict_constructed_branch(public,[{'question_id':'v','canonical_answer':'visible_clear'}],
        {'target_family':'QA-08','answer_branch_targets':{'QA-08':{'requested_branch':'outside'}}})
    assert out==public and audit['applied']is False
