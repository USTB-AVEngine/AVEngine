import pytest
from avengine.qa.v7_english_surface import reissue_item_english,CLASS34_DISPLAY_ALIASES
from avengine.qa.unified_scoring import score_unified_item
pytestmark=pytest.mark.fast_unit


@pytest.mark.parametrize('canonical',list(CLASS34_DISPLAY_ALIASES))
def test_all_broader_panel_labels_are_natural_english_without_changing_truth(canonical):
    item={'question_id':'q','qa_id':'V7-CLASS34','status':'pass','forms':{'open':{'answer_type':'closed_set',
        'closed_set_policy':'token_negation_v2','classes':{canonical:[canonical],'alarm_beep':['alarm beep']},'truth':canonical}}}
    result=reissue_item_english(item,{})
    assert result['forms']['open']['truth']==canonical
    assert set(result['forms']['open']['classes'])==set(item['forms']['open']['classes'])
    label=CLASS34_DISPLAY_ALIASES[canonical]
    assert '_'not in label
    assert score_unified_item(result,label,form='open')['score']==1
    assert score_unified_item(result,'alarm beep',form='open')['score']==0
