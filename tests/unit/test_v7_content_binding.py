from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from avengine.qa.v7_content_binding import (
    content_truth, question_text, admission_blockers, four_state_review, CAPABILITIES)
from avengine.qa.v7_authored_scenarios import build_catalog, validate_catalog
from avengine.qa.v7_recorded_audio import natural_speech_blockers, source_policy_blockers, waveform_lineage

def actor(key, kind='guitar_player', visible=True, descriptor=None, subtype='acoustic'):
    return {'actor_id':key,'descriptor_zh':descriptor or ('红衣人物' if key=='a' else '蓝衣人物'),
            'visible':visible,'capability':{'asset_kind':kind,'instrument_subtype':subtype}}

def emission(key='a', content='guitar_live', interval=(0,2), eid='e'):
    points={'guitar_live':'guitar_soundboard','violin_live':'violin_body','speech':'mouth'}
    return {'emission_id':eid,'actor_id':key,'mechanism':'actor_live','content_class':content,
            'instrument_subtype':'acoustic','emitter_point_kind':points[content],
            'measured_activity_s':[list(interval)]}

def world(actors=None, emissions=None):
    return {'window_s':[0,5],'actors':actors or [actor('a'),actor('b')],
            'emissions':emissions if emissions is not None else [emission()]}

def query(kind='same_class_actor', **kw):
    return {'kind':kind,'time_s':1,'content_class':'guitar_live',**kw}

def test_same_class_counts_actual_actor_once_and_half_open_end():
    s=world(emissions=[emission(),emission(eid='e2')])
    assert content_truth(s,query())['actor_id']=='a'
    with pytest.raises(ValueError,match='exactly one'):
        content_truth(s,query(time_s=2))
    s['emissions'].append(emission('b',eid='b'))
    with pytest.raises(ValueError,match='exactly one'):
        content_truth(s,query())

def test_impossible_guitar_flute_pair_is_rejected():
    s=world();s['emissions'][0]['content_class']='flute_live'
    with pytest.raises(ValueError,match='impossible'):
        content_truth(s,query(content_class='flute_live'))

def test_acoustic_recording_cannot_be_assigned_to_electric_instrument():
    s=world();s['actors'][0]['capability']['instrument_subtype']='electric'
    with pytest.raises(ValueError,match='subtype'):
        content_truth(s,query())

def test_visible_silent_holder_is_not_counted_as_sound_source():
    s=world([actor('a'),actor('b','violin_player')],[emission('b','violin_live')])
    assert content_truth(s,query('mixed_class_actor',content_class='violin_live'))['answer']=='蓝衣人物'

def test_offscreen_source_and_named_instrument_leakage_are_explicit():
    s=world([actor('a'),actor('b','violin_player',False)],[emission('b','violin_live')])
    q=query('offscreen_origin',content_class='violin_live',subject_actor_id='a',
            subject_descriptor_zh='红衣人物',names_sound_class_in_question=True)
    assert content_truth(s,q)['answer']=='否'
    assert 'named_incompatible_instrument_allows_visual_only_elimination' in admission_blockers(s,q)

def test_actual_loudspeaker_cannot_masquerade_as_a_person():
    s=world();p={'emission_id':'p','device_id':'speaker1','mechanism':'loudspeaker_playback',
      'content_class':'guitar_live','emitter_point_kind':'loudspeaker_driver','measured_activity_s':[[0,2]]}
    s['emissions']=[p]
    assert content_truth(s,query('live_or_playback'))['answer']=='音箱播放'
    p['actor_id']='a'
    with pytest.raises(ValueError,match='masquerade'):
        content_truth(s,query('live_or_playback'))

def test_phone_remote_voice_is_device_playback_not_onsite_caller():
    s=world([actor('a','phone_call'),actor('b','phone_call')],[emission('a','speech')])
    s['emissions'][0]['emitter_point_kind']='phone_driver'
    with pytest.raises(ValueError,match='onsite phone'):
        content_truth(s,query('mixed_class_actor',content_class='speech'))

def test_megaphone_uses_driver_while_actor_identity_stays_with_announcer():
    s=world([actor('a','megaphone_speaker'),actor('b','megaphone_speaker')],[emission('a','speech')])
    s['emissions'][0]['emitter_point_kind']='megaphone_driver'
    assert content_truth(s,query(content_class='speech'))['actor_id']=='a'
    s['emissions'][0]['emitter_point_kind']='mouth'
    with pytest.raises(ValueError,match='megaphone'):
        content_truth(s,query(content_class='speech'))

def test_order_uses_distinct_actor_first_clipped_anchor_and_rejects_ties():
    s=world(emissions=[emission('a',interval=(0,2)),emission('b',interval=(3,4),eid='b'),
                       emission('a',interval=(4,5),eid='again')])
    q=query('first_actor',content_class=None)
    assert content_truth(s,q)['distinct_actor_onsets']=={'a':0.,'b':3.}
    s['window_s']=[1,5];s['emissions'][1]['measured_activity_s']=[[0,3]]
    with pytest.raises(ValueError,match='tied'):
        content_truth(s,q)

def test_design_quadruplet_has_balanced_answers_but_no_native_admission():
    members=[]
    for active in (0,1):
        for permutation in (0,1):
            s=world()
            if permutation:
                s['actors'][0]['descriptor_zh'],s['actors'][1]['descriptor_zh']=s['actors'][1]['descriptor_zh'],s['actors'][0]['descriptor_zh']
            s['emissions']=[emission(('a','b')[active])]
            members.append({'active_choice':active,'appearance_permutation':permutation,
                            'base_world_id':'w','pairing_id':'p','scene':s,'query':query()})
    result=four_state_review(members)
    assert result['answers']==['红衣人物','蓝衣人物','蓝衣人物','红衣人物']
    assert not result['native_four_member_binding_eligible']
    assert result['status']=='blocked'

@pytest.mark.parametrize('kind', ['same_class_actor','mixed_class_actor','offscreen_origin','live_or_playback','first_actor'])
def test_wording_interchange_is_not_selected_by_answer(kind):
    q=query(kind,subject_descriptor_zh='指定人物')
    a,b=question_text(q),question_text(q,'alternate')
    assert a!=b
    assert '吉他' not in a+b and '红衣' not in a+b
    with pytest.raises(ValueError):
        question_text(q,'answer_conditioned')

def test_daily_catalog_180_author_scenarios_with_540_unique_role_assignments():
    c=build_catalog();a=validate_catalog(c)
    assert a['status']=='pass' and a['scenario_count']==180
    assert a['scenarios_by_role_count']=={2:60,3:60,4:60}
    assert a['assigned_role_utterances']==540 and a['author_options']==720
    assert a['actual_synthesized_utterances']==0
    assert c['language_policy']['english_fraction']==1.0
    assert sum(s['conversational_function']=='phone_message' for s in c['scenarios'])==12
    assert sum(s['conversational_function']=='megaphone_notice' for s in c['scenarios'])==12

def test_duplicate_meanings_and_role_transcript_poisoning_are_detected():
    c=build_catalog();c['scenarios'][0]['roles'][1]['meaning_answer']=c['scenarios'][0]['roles'][0]['meaning_answer']
    c['scenarios'][0]['roles'][0]['transcript']='wrong message'
    a=validate_catalog(c)
    assert a['status']=='blocked'
    assert any('roles_not_distinct' in e for e in a['errors'])
    assert any('role_script_mismatch' in e for e in a['errors'])

def test_recorded_vctk_licence_does_not_allow_eval_or_formant_tts():
    assert source_policy_blockers({'source_dataset':'VCTK-0.92','split':'train'})==[]
    assert 'heldout_source_split' in source_policy_blockers({'source_dataset':'VCTK-0.92','split':'eval'})
    assert 'espeak_formant_not_permitted' in source_policy_blockers({'engine':'eSpeak-ng'})
    assert 'non_speech_must_be_recorded' in source_policy_blockers({'is_program_synthesized':True,'content_class':'guitar_live'})
    assert 'prohibited_real_spatial_or_evaluation_source' in source_policy_blockers({'source_dataset':'STARSS23'})
    assert source_policy_blockers({'synthesis_readback':None})==[]

def test_natural_speech_needs_actual_engine_reference_gender_and_ASR():
    s={'synthesis_readback':{'engine':{'name':'CosyVoice2','natural_neural_tts':True}},
       'voice_gender':'male','native_placement_readback':{'visual_gender':'female'}}
    reasons=natural_speech_blockers(s)
    assert 'natural_TTS_model_and_rights_receipt_missing' in reasons
    assert 'voice_and_visual_gender_binding_missing_or_conflicting' in reasons
    assert 'automatic_speech_content_readback_missing_or_failed' in reasons

def test_vctk_recording_lineage_accepts_gain_crop_and_rejects_wrong_source():
    rng=np.random.default_rng(31);parent=rng.normal(size=8000)
    child=parent[700:3200]*.17
    a=waveform_lineage(child,parent)
    assert a['status']=='pass' and a['offset_samples']==700
    b=waveform_lineage(rng.normal(size=2500),parent)
    assert b['status']=='blocked'
    with pytest.raises(ValueError,match='silent'):
        waveform_lineage(np.zeros(400),parent)


def test_audio_parent_labels_are_pruned_but_guitar_remains_a_violin_negative():
    from avengine.qa.v7_sound_extensions import specific_source_mids,score_and_rival,ontology_relatives
    ontology=[{'id':'music','child_ids':['instrument']},{'id':'instrument','child_ids':['guitar','violin']},
              {'id':'guitar','child_ids':['acoustic']},{'id':'acoustic','child_ids':[]},{'id':'violin','child_ids':[]}]
    assert specific_source_mids(['music','instrument','guitar'],ontology)=={'guitar'}
    relatives=ontology_relatives(ontology)
    assert 'violin' not in relatives['guitar']
    s,r,mid=score_and_rival({'scores_by_mid':{'music':.99,'instrument':.98,'guitar':.8,'violin':.1}},['guitar'],relatives)
    assert s==.8 and r==.1 and mid=='violin'

def test_specific_instrument_metadata_blocks_wrong_geometry_subtype():
    from avengine.qa.v7_sound_extensions import recording_metadata_blockers
    assert 'flute_recording_is_different_instrument_subtype' in recording_metadata_blockers(
        {'sound_class':'flute_live','source_metadata':{'title':'Real pan flute'}})
    assert 'piano_recording_not_confirmed_acoustic_piano' in recording_metadata_blockers(
        {'sound_class':'piano_live','source_metadata':{'title':'Rhodes electric piano'}})
    assert 'original_bird_species_metadata_missing' in recording_metadata_blockers(
        {'sound_class':'pet_bird_parrot','source_metadata':{'title':'generic birdsong'}})
    assert 'synthetic_spatial_or_prohibited_source_metadata' in recording_metadata_blockers(
        {'sound_class':'guitar_live','source_metadata':{'description':'made using MIDI and a VST'}})

def test_modality_proof_checkbox_does_not_admit_a_fake_native_pair():
    s=world();q=query()
    s['native_readback']={key:{'status':'pass','source':'test-only'}
      for key in ('video','audio','pixel_actor_identity','measured_source_activity',
                  'asset_license_and_capability','physical_emitter_attachment','audible_emitter_uniqueness')}
    witness={key:{'status':'pass','source':'test-only'}
      for key in ('same_V_different_A_answer_flip','same_A_different_V_answer_flip')}
    reasons=admission_blockers(s,q,binding_witness=witness)
    assert sum('actual_native_pair_failed' in r for r in reasons)==2
    assert sum('actual_receipt_file_missing' in r for r in reasons)==7

def test_natural_ASR_receipt_is_bound_to_the_actual_audio_bytes(tmp_path):
    import soundfile as sf
    import hashlib
    wav=tmp_path/'a.wav'
    sf.write(wav,np.ones(1600)*.1,16000,subtype='FLOAT')
    s={'synthesis_readback':{'path':str(wav)},'automatic_content_readback':{'audio_sha256':hashlib.sha256(wav.read_bytes()).hexdigest()}}
    assert 'automatic_content_readback_not_bound_to_actual_waveform' not in natural_speech_blockers(s)
    sf.write(wav,np.ones(1600)*.2,16000,subtype='FLOAT')
    assert 'automatic_content_readback_not_bound_to_actual_waveform' in natural_speech_blockers(s)

def test_natural_canary_cannot_expand_into_large_or_missing_model_batch(tmp_path):
    from avengine.qa.v7_semantic_assets import synthesize
    with pytest.raises(ValueError,match='one to eight'):
        synthesize(tmp_path/'missing.json',tmp_path/'output',model_dir=tmp_path,
                   code_receipt=tmp_path/'no.json',maximum_utterances=540)

def test_generic_who_question_cannot_hide_another_sounding_instrument():
    scene=world([actor('a'),actor('b','violin_player')],
                [emission('a'),emission('b','violin_live',eid='violin')])
    generic=query('mixed_class_actor',content_class='guitar_live')
    with pytest.raises(ValueError,match='exactly one'):
        content_truth(scene,generic)
    named=dict(generic,names_sound_class_in_question=True)
    assert content_truth(scene,named)['actor_id']=='a'
    assert '吉他' in question_text(named)
    assert '吉他' in question_text(named,'alternate')

def test_generic_first_actor_does_not_ignore_the_earliest_different_content():
    scene=world([actor('a'),actor('b','violin_player')],
                [emission('a',interval=(2,4)),emission('b','violin_live',interval=(0,1),eid='first')])
    assert content_truth(scene,query('first_actor',content_class='guitar_live'))['actor_id']=='b'

def test_remote_phone_is_a_real_device_but_not_an_audio_loudspeaker_answer():
    scene=world()
    scene['emissions']=[{'emission_id':'remote','mechanism':'loudspeaker_playback',
      'device_id':'phone1','content_class':'speech','emitter_point_kind':'phone_driver',
      'measured_activity_s':[[0,2]]}]
    assert content_truth(scene,query('offscreen_origin',subject_actor_id='a',subject_descriptor_zh='红衣人物'))['answer']=='否'
    with pytest.raises(ValueError,match='phone'):
        content_truth(scene,query('live_or_playback'))

def test_fixed_ASR_contractions_do_not_repair_wrong_semantic_slots():
    from avengine.qa.v7_recorded_audio import semantic_asr_comparison
    good=semantic_asr_comparison("We are ready. I am calling to say I will toast the bread.",
                                "We're ready. I'm calling to say I will toast the bread.",['toast the bread'])
    assert good['status']=='pass' and good['wer']==0
    bad=semantic_asr_comparison("Please hang up the coats.","Please hang of the coats.",['hang up the coats'])
    assert bad['status']=='blocked' and not bad['all_critical_slots_correct']
    assert 'prohibited_real_spatial_or_evaluation_source' in source_policy_blockers({'source_dataset':'TAU'})

def test_time_anchor_is_public_and_relative_to_the_native_window():
    scene=world();scene['window_s']=[1,5]
    with pytest.raises(ValueError,match='public relative'):
        content_truth(scene,query())
    q=query(public_time_s=0)
    assert content_truth(scene,q)['actor_id']=='a'
    assert '第0秒' in question_text(q)
    assert '第0秒' in question_text(q,'alternate')

def test_same_class_private_label_cannot_claim_a_silent_instrument_sound():
    scene=world(emissions=[emission('a','speech')])
    with pytest.raises(ValueError,match='same-class sounding'):
        content_truth(scene,query())

def test_eval_original_identity_is_blocked_across_different_prepared_slots():
    from avengine.qa.v7_extended_admission import source_eval_originals
    pool=[{'sound_asset_id':'heldout_crop','split':'eval','source_origin':'fsd50k_1234'},
          {'sound_asset_id':'train_crop','split':'train','source_origin':'fsd50k_1234'}]
    groups=source_eval_originals(pool,[{'sound_asset_id':'heldout_crop','fsd50k_id':1234}])
    assert 'fsd:1234' in groups

def test_loudness_normalization_preserves_the_complete_activity_clock(tmp_path):
    import soundfile as sf
    from avengine.qa.v7_extended_admission import normalize_recorded
    rate=16000
    # Numerical fixture, never a permitted non-speech training asset.
    x=np.zeros(48000)
    rng=np.random.default_rng(18)
    x[8000:20000]=rng.normal(0,.08,size=12000)
    x[30000:40000]=rng.normal(0,.08,size=10000)
    source=tmp_path/'fixture.wav';sf.write(source,x,rate,subtype='FLOAT')
    before,_=sf.read(source)
    receipt=normalize_recorded({'path':str(source),'sound_class':'fixture','sound_asset_id':'fixture'},tmp_path/'normalized')
    after,sr=sf.read(receipt['path'])
    assert receipt['status']=='pass' and sr==rate
    assert len(before)==len(after)==48000
    assert np.allclose(after,before*10**(receipt['constant_gain_db']/20),atol=1e-6)
    assert np.array_equal(before==0,after==0)
    assert receipt['readback_loudness']['input_tp']<=-3

def test_public_subject_descriptor_must_identify_the_same_physical_actor():
    scene=world()
    with pytest.raises(ValueError,match='specified origin-test'):
        content_truth(scene,query('offscreen_origin',subject_actor_id='a',subject_descriptor_zh='蓝衣人物'))

def test_missing_actor_and_reused_emission_identifiers_are_rejected():
    scene=world();scene['actors'][0]['actor_id']=None
    with pytest.raises(ValueError,match='identity is missing'):
        content_truth(scene,query())
    scene=world(emissions=[emission(),emission()])
    with pytest.raises(ValueError,match='emission identity'):
        content_truth(scene,query())

def test_fsd_program_created_tones_and_synth_tags_are_not_recorded_inputs():
    from avengine.qa.v7_sound_extensions import recording_metadata_blockers
    a={'sound_class':'telephone_dialing_dtmf','source_metadata':{'description':'Dual tone dialing. Made with Audacity.'}}
    assert 'explicit_program_generated_effect_not_permitted' in recording_metadata_blockers(a)
    b={'sound_class':'music_playback','source_metadata':{'tags':['music','synth']}}
    assert 'synthetic_spatial_or_prohibited_source_metadata' in recording_metadata_blockers(b)
    c={'sound_class':'guitar_live','source_metadata':{'description':'I recorded my acoustic guitar, then edited the recording with Audacity.'}}
    assert recording_metadata_blockers(c)==[]

def test_real_pair_witness_cannot_be_reused_for_an_unrelated_candidate(monkeypatch,tmp_path):
    import avengine.qa.v7_content_binding as module
    candidate=world();other=world();other['actors'][0]['descriptor_zh']='黄衣人物'
    q=query()
    source=tmp_path/'receipt.json';source.write_text('{}')
    witness={axis:{'status':'pass','source':str(source),
                  'native_pair':{'scenes':[other,other],'queries':[q,q]}}
             for axis in ['same_V_different_A_answer_flip','same_A_different_V_answer_flip']}
    monkeypatch.setattr(module,'validate_modality_witness',lambda pair,axis:{'status':'pass'})
    reasons=module.admission_blockers(candidate,q,binding_witness=witness)
    assert any('pair_does_not_include_this_candidate' in r for r in reasons)
