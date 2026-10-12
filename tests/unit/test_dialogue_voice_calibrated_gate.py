"""The calibrated English gate must preserve semantics and reject Chinese speech."""
import pytest
from avengine.qa.v7_dialogue_voice_audit import contains_han, english_language_gate
from avengine.qa.v7_recorded_audio import semantic_asr_comparison


def proof(wer=0.0, critical=True):
    return {'status': 'pass', 'wer': wer, 'all_critical_slots_correct': critical}


def test_low_confidence_english_is_accepted_when_all_three_rules_pass():
    result = english_language_gate(proof(.1), 'en', 'Remember to hang up the coats.')
    assert result['status'] == 'pass'
    assert result['english_probability_is_diagnostic_only'] is True


def test_non_english_top_language_rejects_perfect_forced_english_asr():
    result = english_language_gate(proof(), 'cy', 'Remember to fold the sofa blanket.')
    assert result['status'] == 'blocked'
    assert result['reasons'] == ['language_top1_not_en']


@pytest.mark.parametrize('text', ['我下午要去買東西', 'Hello 中文', '\u3401', '\uf900', '\U00020000'])
def test_han_rejects_even_when_top_language_and_forced_asr_are_english(text):
    assert contains_han(text)
    assert 'free_transcription_contains_Han' in english_language_gate(proof(), 'en', text)['reasons']


@pytest.mark.parametrize('text', ['English!', 'café', '한글', 'かな', ''])
def test_han_detection_does_not_treat_other_scripts_as_chinese(text):
    assert not contains_han(text)


@pytest.mark.parametrize('wer', [.10001, -1, float('nan'), float('inf'), None, True])
def test_invalid_or_over_threshold_wer_fails_closed(wer):
    assert 'forced_English_ASR_failed' in english_language_gate(proof(wer), 'en', 'English')['reasons']


def test_missing_keyword_still_blocks_low_wer():
    assert english_language_gate(proof(0, False), 'en', 'English')['status'] == 'blocked'


def test_chinese_negative_is_rejected_by_both_language_observations():
    result = english_language_gate(proof(), 'zh', '記得把沙發毯摺好。')
    assert set(result['reasons']) == {'language_top1_not_en', 'free_transcription_contains_Han'}


def test_round5_negation_requirement_survives_new_language_rule():
    asr = semantic_asr_comparison('Do not put the mug on the desk.', 'Do put the mug on the desk.', ['the mug'])
    assert asr['negation_counts_match'] is False
    assert english_language_gate(asr, 'en', asr.get('blind_decoded_text', 'Do put the mug on the desk.'))['status'] == 'blocked'
