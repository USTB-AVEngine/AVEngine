"""Check unprompted language normalization and the all-source adoption condition."""
import math
import pytest
from avengine.qa.v7_dialogue_voice_audit import baseline_language_status, language_distribution
pytestmark=pytest.mark.fast_unit


def test_language_distribution_ignores_non_language_tokens():
    probs=language_distribution([10000.0,math.log(9),0.0,-1000],{'<|en|>':1,'<|fr|>':2,'<|zh|>':3})
    assert probs['en']==pytest.approx(0.9)
    assert probs['fr']==pytest.approx(0.1)
    assert sum(probs.values())==pytest.approx(1.0)


def test_language_distribution_rejects_missing_or_duplicate_language_ids():
    with pytest.raises(ValueError):
        language_distribution([0,1],{'<|fr|>':1})
    with pytest.raises(ValueError):
        language_distribution([0,1],{'<|en|>':1,'<|fr|>':1})


def row(sid,p=0.9):
    return {'sound_asset_id':sid,'status':'pass','english_probability':p,'audio_sha256_matches':True,'language_or_text_prompt_used':False}


def test_one_low_probability_or_missing_clip_blocks_adoption():
    assert baseline_language_status([row('a'),row('b')],2)['threshold_adoptable']
    assert not baseline_language_status([row('a'),row('b',0.899999)],2)['threshold_adoptable']
    assert not baseline_language_status([row('a')],2)['threshold_adoptable']


@pytest.mark.parametrize('field,value',[('audio_sha256_matches',False),('language_or_text_prompt_used',True),('status','blocked')])
def test_receipt_cannot_pass_with_source_mismatch_or_language_prompt(field,value):
    receipt=row('a');receipt[field]=value
    assert baseline_language_status([receipt],1)['status']=='blocked'


def test_duplicate_source_identity_cannot_satisfy_full_baseline():
    with pytest.raises(ValueError):
        baseline_language_status([row('a'),row('a')],2)
