import pytest
from avengine.timeline.v7_native_color_references import bind_native_color_references
from avengine.qa.kujiale_current_color import filter_current_scene_colors
pytestmark=pytest.mark.fast_unit

def test_chinese_question_uses_existing_authority_actor_id():
    q=[{'question_id':'q','logical_question_id':'l','modality':'audiovisual','question_zh':'蓝色上衣的人说了什么？','question_en':'What did the person in blue say?'}];gold=[{'question_id':'q','canonical_answer':'meaning1','evidence':{'target_actor_id':'1'}}]
    raw={'actors':{'1':{'display_label':'the woman in the blue shirt'}},'appearance_review':{'actors':{'1':{'frame_refs':[0,15]}}}}
    bind_native_color_references(q,gold,raw);retained,answers,audit=filter_current_scene_colors(q,gold,{'native_bundle':raw},{},0)
    assert len(retained)==1 and answers==gold and q[0]['color_actor_ids']==['1']
    raw['appearance_review']['actors']['1']['frame_refs']=[]
    assert not filter_current_scene_colors(q,gold,{'native_bundle':raw},{},0)[0]

def test_absent_native_identity_is_not_invented():
    q=[{'question_id':'q'}];gold=[{'question_id':'q','evidence':{'target_actor_id':'missing'}}]
    bind_native_color_references(q,gold,{'actors':{'1':{}}});assert 'color_actor_ids'not in q[0]
