"""Recorded-input provenance and natural authored-speech admission for v7.

Recorded mono VCTK/FSD audio is permitted as simulation input. This module does
not admit real spatial evaluation recordings, synthetic effects or eSpeak.
Licensing, corpus split, acoustic label and native capability are separate.
"""
from __future__ import annotations
import math
import hashlib
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

NATURAL_SPEECH_POLICY = 'v7_natural_authored_speech_v1'
REAL_RECORDING_DATASETS = {'VCTK-0.92', 'FSD50K', 'Freesound'}
FORBIDDEN_SOURCE_TOKENS = ('starss', 'dcase', 'tau-srir', 'tau_srir', 'tau_rir', 'tau_test_rir')
FORMANT_TOKENS = ('espeak', 'e-speak', 'formantspeech', 'formant_speech', 'human_live_formant')


def source_policy_blockers(record):
    """Apply owner input policy; a corpus licence alone does not admit a clip."""
    reasons = []
    origin = ' '.join(str(record.get(k, '')) for k in
                      ('source_dataset', 'source_origin', 'source_uri', 'source_pcm_path')).lower()
    engine = str((record.get('synthesis_readback') or {}).get('engine', record.get('engine', ''))).lower()
    mechanism = str(record.get('emission_mechanism', '')).lower()
    if any(t in origin for t in FORBIDDEN_SOURCE_TOKENS) or re.search(r'(?:^|[/_\s-])tau(?:$|[/_\s-])', origin):
        reasons.append('prohibited_real_spatial_or_evaluation_source')
    if any(t in engine or t in mechanism or t in origin for t in FORMANT_TOKENS):
        reasons.append('espeak_formant_not_permitted')
    if record.get('is_program_synthesized') is True and record.get('content_class') != 'speech':
        reasons.append('non_speech_must_be_recorded')
    if record.get('split') in {'eval', 'test', 'evaluation'}:
        reasons.append('heldout_source_split')
    if record.get('evaluation_source_overlap') is True:
        reasons.append('evaluation_original_identity_overlap')
    return reasons


def normalize_transcript_tokens(text):
    """Fixed lexical equivalences; never insert or repair a semantic answer."""
    text = text.lower().replace("’", "'")
    contractions = {"we're":"we are", "i'm":"i am", "you're":"you are",
                    "they're":"they are", "it's":"it is", "that's":"that is",
                    "i'll":"i will", "we'll":"we will", "don't":"do not",
                    "can't":"cannot", "won't":"will not"}
    for source, expanded in contractions.items():
        text = re.sub(r"\b"+re.escape(source)+r"\b", expanded, text)
    return re.findall(r"[a-z0-9]+", text.replace("-", " "))


def semantic_asr_comparison(reference, blind_hypothesis, critical_slots):
    """WER with declared contraction equivalences and exact meaning slots."""
    a, b = normalize_transcript_tokens(reference), normalize_transcript_tokens(blind_hypothesis)
    previous = list(range(len(b)+1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, observed in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1,
                               previous[j-1]+int(word!=observed)))
        previous = current
    wer = previous[-1]/max(1,len(a))
    slots = [{'slot':slot,'present':' '.join(normalize_transcript_tokens(slot)) in ' '.join(b)}
             for slot in critical_slots]
    all_correct = bool(slots) and all(row['present'] for row in slots)
    return {'status':'pass' if wer<=.1 and all_correct else 'blocked',
            'wer':wer, 'all_critical_slots_correct':all_correct,'critical_slots':slots,
            'normalized_reference_tokens':a,'normalized_blind_hypothesis_tokens':b,
            'normalization':'case/punctuation/hyphen plus fixed lexical contraction expansions',
            'maximum_wer':.1}

def natural_speech_blockers(sound):
    """Validate actual natural TTS + VCTK reference + automatic semantic readback."""
    reasons = source_policy_blockers(sound)
    proof = sound.get('synthesis_readback') or {}
    engine = proof.get('engine') or {}
    if (sound.get('speech_policy') != NATURAL_SPEECH_POLICY
            or not isinstance(engine, dict)
            or engine.get('natural_neural_tts') is not True
            or engine.get('name') != 'CosyVoice2'
            or not engine.get('model_id') or not engine.get('model_license')
            or not engine.get('code_revision')):
        reasons.append('natural_TTS_model_and_rights_receipt_missing')
    reference = sound.get('voice_reference') or {}
    if (reference.get('dataset') != 'VCTK-0.92'
            or reference.get('license_spdx') != 'CC-BY-4.0'
            or reference.get('license_chain_status') != 'pass'
            or not reference.get('speaker_id') or not reference.get('utterance_id')
            or not reference.get('source_waveform') or not reference.get('source_transcript')):
        reasons.append('licensed_VCTK_reference_missing')
    voice_gender = sound.get('voice_gender')
    visual_gender = (sound.get('native_placement_readback') or {}).get('visual_gender')
    if voice_gender not in {'male', 'female'} or voice_gender != visual_gender:
        reasons.append('voice_and_visual_gender_binding_missing_or_conflicting')
    if reference.get('gender') != voice_gender:
        reasons.append('reference_and_voice_gender_conflicting')
    if reference.get('split') in {'eval', 'test', 'evaluation'}:
        reasons.append('heldout_voice_reference')
    asr = sound.get('automatic_content_readback') or {}
    if (asr.get('status') != 'pass' or not asr.get('model_id') or not asr.get('source')
            or asr.get('reference_text') != sound.get('transcript')
            or not isinstance(asr.get('wer'), (int, float))
            or isinstance(asr.get('wer'), bool) or not math.isfinite(asr['wer'])
            or not 0 <= asr['wer'] <= .1 or asr.get('all_critical_slots_correct') is not True
            or asr.get('human_listening_used') is not False):
        reasons.append('automatic_speech_content_readback_missing_or_failed')
    path=Path(proof.get('path',''))
    if (not path.is_file() or not asr.get('audio_sha256')
            or hashlib.sha256(path.read_bytes()).hexdigest()!=asr.get('audio_sha256')):
        reasons.append('automatic_content_readback_not_bound_to_actual_waveform')
    return sorted(set(reasons))


def vctk_original_identity(record):
    """Resolve a stable corpus sentence ID, independently of prepared clip slots."""
    values = [str(record.get(k, '')) for k in ('source_pcm_path', 'source_origin', 'source_id')]
    normal = (record.get('source_normalization') or {}).get('source') or {}
    values += [str(normal.get('source_pcm_path', '')), str(normal.get('source_relative', ''))]
    matches = {m.groups() for value in values for m in
               re.finditer(r'(?:vctk_)?((?:p|s)\d+)_(\d+)(?:_mic[12])?', value)}
    if len(matches) != 1:
        raise ValueError('VCTK original sentence ID missing or ambiguous')
    speaker, sentence = matches.pop()
    return speaker, sentence


def waveform_lineage(child, parent):
    """Find one contiguous recording segment despite resampler and constant gain.

    Returns a numerical receipt. Caller binds actual corpus IDs, paths, crop,
    sampling clock and microphone. This does not identify the spoken semantics.
    """
    x, y = np.asarray(child, dtype=float), np.asarray(parent, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or not len(x) or len(x) > len(y):
        raise ValueError('lineage needs a nonempty mono child within the parent')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('lineage audio is not finite')
    x = x - x.mean()
    energy = float(x @ x)
    if energy <= 1e-12:
        raise ValueError('silent audio cannot prove recording lineage')
    cross = signal.fftconvolve(y, x[::-1], mode='valid')
    prefix = np.r_[0., np.cumsum(y)]
    prefix_sq = np.r_[0., np.cumsum(y * y)]
    local_sum = prefix[len(x):] - prefix[:-len(x)]
    local_energy = (prefix_sq[len(x):] - prefix_sq[:-len(x)]) - local_sum**2 / len(x)
    correlation = cross / np.sqrt(np.maximum(local_energy * energy, 1e-30))
    offset = int(np.argmax(correlation))
    segment = y[offset:offset + len(x)]
    centered = segment - segment.mean()
    gain = float(centered @ x / max(float(centered @ centered), 1e-30))
    residual = float(np.sqrt(np.mean((x - gain * centered)**2)))
    normalized_residual = residual / max(float(np.sqrt(np.mean(x*x))), 1e-30)
    return {'status': 'pass' if correlation[offset] >= .995 and normalized_residual <= .1 else 'blocked',
            'normalized_correlation': float(correlation[offset]), 'offset_samples': offset,
            'sample_count': len(x), 'linear_gain': gain,
            'normalized_rms_residual': normalized_residual,
            'method': 'one_contiguous_segment_normalized_cross_correlation',
            'pass_minimum_correlation': .995, 'pass_maximum_relative_rms': .1}


def vctk_license_lineage(record, corpus_root):
    """Verify local copied recording against VCTK 0.92 originals and licence."""
    root = Path(corpus_root)
    licence = root / 'license_text.txt'
    readme = root / 'README.txt'
    if ('Attribution 4.0 International' not in licence.read_text()
            or 'Version 0.92' not in readme.read_text()):
        raise ValueError('local VCTK corpus/licence association is not established')
    speaker, sentence = vctk_original_identity(record)
    info = {}
    for line in (root / 'speaker-info.txt').read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 3 and fields[2] in {'M', 'F'}:
            info[fields[0]] = fields[2]
    if speaker not in info:
        raise ValueError('original VCTK speaker metadata missing')
    raw_source = (record.get('source_normalization') or {}).get('source') or {}
    copied_path = Path(record.get('source_pcm_path') or raw_source.get('source_pcm_path') or record['path'])
    child, child_rate = sf.read(copied_path, always_2d=True)
    if child.shape[1] != 1:
        raise ValueError('local VCTK dry copy is not mono')
    candidates = []
    for microphone in ('mic1', 'mic2'):
        original = root / 'wav48_silence_trimmed' / speaker / f'{speaker}_{sentence}_{microphone}.flac'
        if not original.exists():
            continue
        parent, rate = sf.read(original, always_2d=True)
        if parent.shape[1] != 1:
            continue
        divisor = math.gcd(int(rate), int(child_rate))
        resampled = signal.resample_poly(parent[:, 0], child_rate // divisor, rate // divisor)
        # Old raw copies occasionally have a final resampler guard sample.
        if len(child) > len(resampled):
            resampled = np.pad(resampled, (0, len(child)-len(resampled)))
        receipt = waveform_lineage(child[:, 0], resampled)
        candidates.append({**receipt, 'microphone': microphone, 'source_waveform': str(original),
                           'original_sample_rate_hz': int(rate)})
    if not candidates:
        raise ValueError('VCTK source waveform unavailable')
    best = max(candidates, key=lambda r: r['normalized_correlation'])
    transcript_path = root / 'txt' / speaker / f'{speaker}_{sentence}.txt'
    if not transcript_path.is_file():
        raise ValueError('VCTK original transcript unavailable')
    prepared_path=Path(record['path'])
    prepared,prepared_rate=sf.read(prepared_path,always_2d=True)
    original,parent_rate=sf.read(best['source_waveform'],always_2d=True)
    divisor=math.gcd(int(parent_rate),int(prepared_rate))
    parent=signal.resample_poly(original[:,0],int(prepared_rate)//divisor,int(parent_rate)//divisor)
    if len(prepared)>len(parent) and len(prepared)-len(parent)<=8:
        parent=np.pad(parent,(0,len(prepared)-len(parent)))
    try:
        if prepared.shape[1]!=1:raise ValueError('prepared input not mono')
        prepared_lineage=waveform_lineage(prepared[:,0],parent)
    except ValueError as error:
        prepared_lineage={'status':'blocked','reason':str(error)}
    prepared_lineage.update(path=str(prepared_path),sample_rate_hz=int(prepared_rate),
       sample_count=len(prepared),original_waveform=best['source_waveform'],
       independent_of_raw_copy_license_proof=True)
    return {'dataset': 'VCTK-0.92', 'original_id': f'{speaker}_{sentence}',
            'speaker_id': speaker, 'utterance_id': sentence,
            'gender': 'male' if info[speaker] == 'M' else 'female',
            'source_transcript': transcript_path.read_text().strip(),
            'source_transcript_path': str(transcript_path),
            'local_copied_waveform': str(copied_path), 'copy_sample_rate_hz': int(child_rate),
            'license_spdx': 'CC-BY-4.0', 'license_text': str(licence),
            'corpus_readme': str(readme), 'source_uri': 'https://datashare.ed.ac.uk/handle/10283/3443',
            'attribution': 'Yamagishi, Veaux, MacDonald (2019), CSTR VCTK Corpus 0.92, University of Edinburgh',
            'license_chain_status': best['status'], 'recording_lineage': best,
            'prepared_audio_lineage':prepared_lineage,
            'source_waveform': best['source_waveform'], 'microphone': best['microphone'],
            'split': record.get('split'), 'language': 'en'}
