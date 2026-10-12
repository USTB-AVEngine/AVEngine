"""Audit Whisper language confidence without a transcript or language prompt."""
from __future__ import annotations

import math


def language_distribution(logits, language_token_ids):
    """Normalize SOT next-token logits over *all* language tokens, as Whisper does."""
    if '<|en|>' not in language_token_ids or len(set(language_token_ids.values())) != len(language_token_ids):
        raise ValueError('complete, unique language-token mapping including English required')
    values = {key[2:-2]: float(logits[token]) for key, token in language_token_ids.items()}
    if not values or not all(math.isfinite(value) for value in values.values()):
        raise ValueError('language logits must be finite')
    maximum = max(values.values())
    weights = {key: math.exp(value - maximum) for key, value in values.items()}
    total = math.fsum(weights.values())
    return {key: value / total for key, value in weights.items()}


def baseline_language_status(records, expected_count):
    """Require every retained source to have a verified, unprompted >=0.9 receipt."""
    ids = [row['sound_asset_id'] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source audio identity')
    passing = sum(row.get('status') == 'pass'
                  and row.get('english_probability', -1) >= 0.9
                  and row.get('audio_sha256_matches') is True
                  and row.get('language_or_text_prompt_used') is False
                  for row in records)
    complete = len(records) == expected_count
    return {'status': 'pass' if complete and passing == expected_count else 'blocked',
            'expected_count': expected_count, 'checked_count': len(records),
            'passing_count': passing, 'nonpassing_count': len(records) - passing,
            'complete': complete, 'threshold': 0.9,
            'threshold_adoptable': complete and passing == expected_count}


def contains_han(text):
    """Cover BMP ideographs, compatibility ideographs and supplementary CJK."""
    return any("\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"
               or "\U00020000" <= ch <= "\U000323af" for ch in text)


def english_language_gate(asr, top_language, free_transcription):
    """Apply the owner-calibrated gate; probability magnitude is diagnostic."""
    reasons = []
    if (asr.get("status") != "pass" or not isinstance(asr.get("wer"), (float, int))
            or isinstance(asr.get("wer"), bool) or not math.isfinite(asr["wer"])
            or not 0 <= asr["wer"] <= .1
            or asr.get("all_critical_slots_correct") is not True):
        reasons.append("forced_English_ASR_failed")
    if top_language != "en":
        reasons.append("language_top1_not_en")
    if contains_han(free_transcription):
        reasons.append("free_transcription_contains_Han")
    return {"status": "pass" if not reasons else "blocked", "reasons": reasons,
            "top_language": top_language,
            "free_transcription_contains_han": contains_han(free_transcription),
            "english_probability_is_diagnostic_only": True}
