import pytest
from avengine.timeline.v7_alpha_delivery import readable_training_answer
pytestmark=pytest.mark.fast_unit


def test_bird_class_reuses_admitted_english_alias_at_training_pack_boundary():
    item={'question_id':'q','qa_id':'V7-CLASS34','status':'pass','forms':{'open':{'answer_type':'closed_set',
        'closed_set_policy':'token_negation_v2','classes':{'pet_bird_canary':['pet_bird_canary'],'alarm_beep':['alarm beep']},'truth':'pet_bird_canary'}}}
    question={'question_id':'q','qa_id':'V7-CLASS34','form':'open'}
    gold={'canonical_answer':'pet_bird_canary','authority_question_id':'q'}
    answer,proof=readable_training_answer(question,gold,{'q':item})
    assert answer=='canary calls'and proof['result']['score']==1.
    assert gold['canonical_answer']=='pet_bird_canary'and item['forms']['open']['truth']=='pet_bird_canary'
