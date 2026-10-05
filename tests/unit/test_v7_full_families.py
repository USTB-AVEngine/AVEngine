from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from avengine.qa import unified_catalog as u
from avengine.qa.v7_families import collect_questions, select_questions, coverage_projection
from avengine.qa.v7_protocol import generate_questions

pytestmark = pytest.mark.fast_unit


def fixture():
    spec = importlib.util.spec_from_file_location('v6_authority_fixture', Path(__file__).with_name('test_qa_unified_catalog.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = module._fixture()
    raw['audio_readback']['source_activity_intervals_samples'] = [
        {'event_id': row['event_id'], 'start_sample': row['start_sample'],
         'end_sample_exclusive': row['end_sample_exclusive']}
        for row in raw['audio_program']['events']]
    return raw


def scene(events):
    ids = {e['source_id'] for e in events}
    return {'events': events, 'listener_m': [0., 1.5, 0.], 'camera_forward': [0., 0., -1.],
            'sources': {f'actor{source}': {'actor_id': source, 'mechanism': 'human_live',
                        'trajectory': [{'time_s': 0., 'position_m': [0., 1.5, -3.]},
                                       {'time_s': 30., 'position_m': [0., 1.5, -3.]}]}
                        for source in ids}}


def event(source, lo=0, hi=40, cls=1):
    return {'source_id': source, 'class_id': cls, 'endpoint_id': f'actor{source}',
            'start_tick': lo, 'end_tick': hi}


def answers(public, private, qa_id):
    by_id = {r['question_id']: r['canonical_answer'] for r in private}
    return [by_id[r['question_id']] for r in public if r['qa_id'] == qa_id and r['form'] == 'open']


def test_all_v6_truth_and_wording_are_preserved_by_collection():
    raw = fixture()
    expected = u.generate_unified_questions(raw, items_per_type=5, seed='v7-test', include_angle_followups=False)
    _, _, actual = collect_questions(raw, 'test', {}, items_per_family=5, seed='v7-test')
    assert actual['v6_generation'] == expected
    assert {r['qa_id'] for r in actual['deferred'] if r['code'] == 'authored_semantic_manifest_missing'} == {'QA-26', 'QA-27', 'QA-28'}


@pytest.mark.parametrize('n', range(10))
def test_speakers_full_domain_zero_to_eight_and_open_overflow(n):
    p, g, _ = generate_questions(scene([event(i + 1, i, 80) for i in range(n)]), 0, 'n', {})
    assert answers(p, g, 'STARSS-SPEAKERS') == [str(n)]
    rows = [r for r in p if r['qa_id'] == 'STARSS-SPEAKERS']
    if n <= 8:
        assert rows[0]['answer_domain'] == [str(i) for i in range(9)]
        assert len(rows) == 2
    else:
        assert len(rows) == 1 and rows[0]['form'] == 'open'


def test_same_actor_different_speech_class_at_different_times_counts_once():
    p, g, _ = generate_questions(scene([event(1, 0, 10, 0), event(1, 20, 30, 1)]), 0, 'same', {})
    assert answers(p, g, 'STARSS-SPEAKERS') == ['1']
    assert answers(p, g, 'QA-23') == ['2']


def test_half_open_carry_in_gap_and_no_right_boundary_activity():
    events = [event(1, 95, 105), event(1, 106, 110), event(2, 200, 205)]
    p, g, audit = generate_questions(scene(events), 10, 'half', {})
    assert answers(p, g, 'QA-23') == ['2']
    assert answers(p, g, 'STARSS-SPEAKERS') == ['1']
    assert audit['segments'][0]['carry_in'] is True
    assert audit['segments'][1]['carry_in'] is False
    assert any('片段开始时已在响' in r['question_zh'] for r in p if r['qa_id'] == 'QA-04')


def test_first_three_ties_are_checked_against_all_segments_and_not_replaced():
    # The fourth onset ties the third one; inspecting only the first three is unsafe.
    p, _, _ = generate_questions(scene([event(1, 0, 10), event(2, 10, 20), event(3, 20, 30), event(4, 20, 30)]), 0, 'tie', {})
    targets = [r['logical_question_id'].rsplit('__', 1)[-1] for r in p if r['qa_id'] == 'STARSS-FRUSTUM' and r['form'] == 'open']
    assert targets == ['target1', 'target2']


def test_duplicate_track_excludes_count_and_targets_and_speech_identity_conflict():
    p, _, audit = generate_questions(scene([event(1, 0, 20), event(1, 0, 20)]), 0, 'dup', {})
    assert audit['flags']['duplicate_track']
    assert not p
    p, _, audit = generate_questions(scene([event(1, 0, 20, 0), event(1, 0, 20, 1)]), 0, 'conflict', {})
    assert audit['flags']['speech_class_conflict']
    assert not any(r['qa_id'] == 'STARSS-SPEAKERS' for r in p)


def test_legacy_class_domain_full_and_class13_separate():
    p, g, _ = generate_questions(scene([event(1, 0, 30, 8)]), 0, 'music', {})
    legacy = next(r for r in p if r['qa_id'] == 'QA-21')
    assert len(legacy['answer_domain']) == 34
    assert 'any_audioset_class_playback' not in legacy['answer_domain']
    assert answers(p, g, 'QA-21') == ['music_playback']
    assert answers(p, g, 'STARSS-CLASS13') == ['music']
    assert not any(r['qa_id'] == 'QA-25' and r['form'] == 'mcq' for r in p)


def test_moving_unobserved_subframe_is_excluded_instead_of_interpolated():
    s = scene([event(1, 1, 30)])
    s['sources']['actor1']['trajectory'] = [
        {'time_s': 0., 'position_m': [0., 1.5, -3.]},
        {'time_s': 2 / 15, 'position_m': [1., 1.5, -3.]},
        {'time_s': 30., 'position_m': [1., 1.5, -3.]}]
    p, _, audit = generate_questions(s, 0, 'move', {})
    assert not any(r['qa_id'] in {'QA-04', 'STARSS-FRUSTUM'} for r in p)
    assert any(r['reason'] == 'moving_geometry_at_unobserved_subframe' for r in audit['exclusions'])


def test_no_teacher_or_source_identity_in_public_requests():
    p, _, _ = generate_questions(scene([event(1), event(2, 50, 70)]), 0, 'public', {})
    def check(value):
        if isinstance(value, dict):
            assert not set(value) & {'gold', 'answer', 'canonical_answer', 'evidence', 'actor_id', 'source_id'}
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(p)


def test_budget_never_fills_empty_modality_with_another_modality():
    p, g, _ = generate_questions(scene([event(1)]), 0, 'budget', {})
    with pytest.raises(ValueError, match='vision: 0 legal candidates'):
        select_questions(p, g, {'vision': 4, 'audio': 4, 'audiovisual': 6})


def test_playback_cannot_be_counted_as_live_speech():
    row = event(1)
    row['mechanism'] = 'loudspeaker_playback'
    with pytest.raises(ValueError, match='playback'):
        generate_questions(scene([row]), 0, 'playback', {})


def test_semantic_assets_require_native_placement_and_roles(tmp_path):
    import soundfile as sf
    import numpy as np
    from avengine.qa.v7_families import _semantic_manifest
    wav = tmp_path / 'complete.wav'
    sf.write(wav, np.ones(1600) * .1, 16000, subtype='FLOAT')
    sound = {'sound_asset_id': 's', 'scenario_id': 'request', 'transcript': 'Please bring me a cup.',
             'script_author': 'project_author',
             'synthesis_readback': {'path': str(wav), 'sample_rate_hz': 16000, 'sample_count': 1600,
                                   'transcript': 'Please bring me a cup.', 'content_complete': True,
                                   'trimmed_or_repeated': False}}
    facts = {'events': [{'event_id': 'e', 'actor_id': '1', 'sound_asset_id': 's', 'transcript': sound['transcript']}]}
    filtered, deferred = _semantic_manifest(facts, {'sounds': [sound]})
    assert not filtered['sounds']
    assert 'complete_native_utterance_placement_missing' in {x['code'] for x in deferred}
    assert 'native_dialogue_role_evidence_missing' in {x['code'] for x in deferred}
    sound['native_placement_readback'] = {'status': 'pass', 'event_id': 'e', 'actor_id': '1',
                                        'full_content_preserved': True, 'source': 'retained-placement.json', 'visual_gender':'female'}
    sound['dialogue_role_evidence'] = {'role': 'requester', 'actor_id': '1', 'scenario_id': 'request',
                                     'source': 'retained-role.json'}
    # The updated owner policy requires a natural model + real licensed reference
    # and machine ASR. These are explicit unit receipts, not native qualification.
    sound['speech_policy']='v7_natural_authored_speech_v1'
    sound['voice_gender']='female'
    sound['synthesis_readback']['engine']={'name':'CosyVoice2','natural_neural_tts':True,
       'model_id':'test-natural-model','model_license':'Apache-2.0','code_revision':'unit-fixture'}
    sound['voice_reference']={'dataset':'VCTK-0.92','license_spdx':'CC-BY-4.0',
       'license_chain_status':'pass','speaker_id':'unit-speaker','utterance_id':'unit-sentence',
       'source_waveform':'unit-original.flac','source_transcript':'unit prompt','gender':'female','split':'train'}
    sound['automatic_content_readback']={'status':'pass','model_id':'unit-ASR','source':'unit-asr.json',
       'reference_text':sound['transcript'],'audio_sha256':__import__('hashlib').sha256(wav.read_bytes()).hexdigest(),'wer':0.0,'all_critical_slots_correct':True,'human_listening_used':False}
    filtered, deferred = _semantic_manifest(facts, {'sounds': [sound]})
    assert filtered['sounds'] == [sound] and not deferred
    sound['synthesis_readback']['engine']='eSpeak-ng'
    filtered, deferred = _semantic_manifest(facts, {'sounds': [sound]})
    assert not filtered['sounds']
    assert 'espeak_formant_not_permitted' in {x['code'] for x in deferred}


def test_v7_fixed_domain_does_not_change_other_v6_family_results():
    from avengine.timeline.continuous_scene import QA21_CLASSES
    raw = fixture()
    raw.setdefault('sampling', {})['v7_sound_class_options'] = list(QA21_CLASSES)
    expected = u.generate_unified_questions(raw, items_per_type=5, seed='fixed', include_angle_followups=False)
    _, _, audit = collect_questions(raw, 'fixed', {}, seed='fixed')
    assert [x for x in audit['v6_generation']['items'] if x['qa_id'] != 'QA-21'] == [
        x for x in expected['items'] if x['qa_id'] != 'QA-21']
    assert list(audit['family_fact_overrides']['QA-21']['sampling']['acceptance_policy']['sound_class_options']) == list(QA21_CLASSES)


def test_independent_v6_replay_detects_poisoned_exported_answer():
    from avengine.qa.v7_validation import validate_v6_items
    p, g, audit = collect_questions(fixture(), 'replay', {}, seed='verify')
    import json
    audit = json.loads(json.dumps(audit))
    p, g = json.loads(json.dumps(p)), json.loads(json.dumps(g))
    good = validate_v6_items(p, g, audit, seed='verify')
    assert good['checked_forms'] == len(p)
    assert good['passed_forms'] == len(p)
    poisoned = deepcopy(g)
    poisoned[0]['canonical_answer'] = 'incorrect'
    bad = validate_v6_items(p, poisoned, audit, seed='verify')
    assert len(bad['mismatches']) == 1


def test_v02_wording_interchange_keeps_truth_and_complete_domains():
    from avengine.qa.v7_protocol import generate_questions
    for n in (0, 1, 3, 4, 5, 8):
        raw = scene([event(i + 1, i * 2, 70) for i in range(n)])
        p, g, _ = generate_questions(raw, 0, 'opaque', {}, wording_variant='standard')
        q, h, _ = generate_questions(raw, 0, 'opaque', {}, wording_variant='alternate')
        assert g == h
        assert [(x['qa_id'], x.get('choices'), x['answer_domain']) for x in p] == [
            (x['qa_id'], x.get('choices'), x['answer_domain']) for x in q]
        assert any(a['question_zh'] != b['question_zh'] for a, b in zip(p, q))
    p0, _, _ = generate_questions(scene([]), 0, 'opaque', {})
    p5, _, _ = generate_questions(scene([event(i + 1, i * 2, 70) for i in range(5)]), 0, 'opaque', {})
    assert [q for q in p0 if q['qa_id'] == 'STARSS-SPEAKERS'] == [
        q for q in p5 if q['qa_id'] == 'STARSS-SPEAKERS']
