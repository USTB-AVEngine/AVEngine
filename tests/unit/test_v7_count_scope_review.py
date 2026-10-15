import pytest
from avengine.timeline.v7_count_scope_review import review_qa22_scope
pytestmark=pytest.mark.fast_unit


def test_original_answer_is_not_relabelled_when_offscreen_sound_changes_owner_count():
    row={'qa_id':'QA-22','canonical_answer':[3,2]}
    facts={'events':[{'actor_id':'visible_a'},{'actor_id':'visible_b'},{'actor_id':'offscreen_c'}]}
    result=review_qa22_scope(row,facts)
    assert result['status']=='answer_scope_requires_reconstruction'
    assert row['canonical_answer']==[3,2] and result['owner_requested_sounding_count']==3


def test_all_sounding_entities_visible_is_unambiguous_and_repeated_events_count_one_person():
    assert review_qa22_scope({'qa_id':'QA-22','canonical_answer':[3,2]},
        {'events':[{'actor_id':'a'},{'actor_id':'a'},{'actor_id':'b'}]})['status']=='pass'


def test_visual_single_count_and_other_families_retain_their_original_domain():
    assert review_qa22_scope({'qa_id':'QA-22','canonical_answer':3},{})['status']=='not_applicable_single_visual_count'
    assert review_qa22_scope({'qa_id':'STARSS-SPEAKERS','canonical_answer':4},{})['status']=='not_applicable'
