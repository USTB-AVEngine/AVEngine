import pytest
from avengine.qa.v7_english_surface import appearance_description_en,protocol_question_en
from avengine.timeline.v7_english_export import validate_english_model_text,english_public_question


def test_natural_entity_names_use_metadata_and_never_roster_numbers():
    assert appearance_description_en({'kind':'human','visual_gender':'male','name_zh':'成年男性 14'},'black')=='the man in the black shirt'
    assert appearance_description_en({'kind':'object','object_type':'floorstanding_speaker'},'white')=='the white floor-standing speaker'


def test_protocol_questions_have_complete_english_surface():
    families=['QA-23','QA-04','QA-25','QA-21','STARSS-SPEAKERS','STARSS-CLASS13','STARSS-FRUSTUM']
    for family in families:
        q={'question_en':protocol_question_en(family,{'anchor_s':.4,'query_s':1})}
        assert validate_english_model_text(q)['status']=='pass'


@pytest.mark.parametrize('field,value',[('question_en','Did 蓝色的人 move?'),('model_answer','黑色的人'),('choices',[{'id':'A','text':'adult male 14'}]),('model_answer','v7_male_adult_14_black_v1')])
def test_export_rejects_cjk_and_registration_names(field,value):
    row={'question_en':'Which person is speaking?',field:value}
    with pytest.raises(ValueError):validate_english_model_text(row)


def test_internal_chinese_stem_is_not_model_visible():
    q={'question_en':'Did the the man in the black shirt move?','question_zh':'中文内部题面','choices':[{'id':'A','text':'Yes'}]}
    public=english_public_question(q)
    assert public['question_en']=='Did the man in the black shirt move?'
    assert 'question_zh' not in public


def test_class34_form_truth_is_reissued_without_fabricating_a_top_level_truth():
    from copy import deepcopy
    from avengine.qa.v7_english_surface import reissue_item_english
    from avengine.qa.unified_scoring import score_unified_item
    original={'status':'pass','qa_id':'V7-CLASS34','authority_kind':'v7_admitted_recording_class_100ms_v1',
              'forms':{'open':{'answer_type':'closed_set','closed_set_policy':'token_negation_v2',
                              'classes':{'laughter':['laughter'],'alarm_beep':['alarm beep']},'truth':'laughter'}}}
    saved=deepcopy(original);result=reissue_item_english(original,{})
    assert original==saved and 'truth'not in result
    assert result['forms']['open']['truth']=='laughter'
    for answer,expected in [('laughter',1.),('__invalid_control__',0.)]:
        assert score_unified_item(original,answer,form='open')['score']==expected
        assert score_unified_item(result,answer,form='open')['score']==expected


def test_class34_model_answer_uses_a_natural_english_label_accepted_by_original_scorer():
    from avengine.timeline.v7_alpha_delivery import readable_training_answer
    item={'status':'pass','qa_id':'V7-CLASS34','forms':{'open':{'answer_type':'closed_set','closed_set_policy':'token_negation_v2',
        'classes':{'dog_bark':['dog_bark'],'cat_meow':['cat_meow']},'truth':'dog_bark'}}}
    q={'qa_id':'V7-CLASS34','form':'open'};g={'authority_question_id':'authority','canonical_answer':'dog_bark'}
    answer,proof=readable_training_answer(q,g,{'authority':item})
    assert answer=='dog bark'and proof['result']['score']==1.
    assert g['canonical_answer']=='dog_bark'
