import pytest
from avengine.timeline.v7_taxonomy_demand import choose_unseen_class
pytestmark=pytest.mark.fast_unit
def test_new_class_must_have_both_registered_capability_and_admitted_recording():
    asset={'asset_id':'mouth','allowed_classes':['coughing','laughter','unadmitted']}
    records=[{'sound_class':'coughing'},{'sound_class':'laughter'},{'sound_class':'music_playback'}]
    cls,proof=choose_unseen_class(asset,records,{'coughing':3})
    assert cls=='laughter'
    assert proof['eligible_classes']==['coughing','laughter']
    assert choose_unseen_class(asset,records,{'coughing':3,'laughter':4})[0]=='coughing'
def test_requested_unknown_audio_cannot_stand_in_for_missing_source_conditions():
    with pytest.raises(ValueError,match='actual source capability'):
        choose_unseen_class({'asset_id':'speaker','allowed_classes':['music_playback']},
            [{'sound_class':'coughing'}],{})

@pytest.mark.parametrize('asset,cls,mechanisms,eligible',[
    ({'kind':'human'},'coughing',None,False),
    ({'kind':'human'},'laughter',None,True),
    ({'kind':'object','entity_class':'loudspeaker'},'coughing',None,True),
    ({'kind':'human'},'footsteps',{'footsteps':'walking_human_registered_foot'},True),
])
def test_planning_mechanism_eligibility_matches_original_class34_candidates(asset,cls,mechanisms,eligible):
    from avengine.timeline.v7_environmental_sources import collect_admitted_class_questions
    asset={**asset,'asset_id':'registered_source','allowed_classes':[cls]}
    if not eligible:
        with pytest.raises(ValueError,match='unchanged V7-CLASS34 predicate'):
            choose_unseen_class(asset,[{'sound_class':cls}],{},emission_mechanisms=mechanisms)
        mechanism='human_live_recorded_'+cls
    else:
        chosen,proof=choose_unseen_class(asset,[{'sound_class':cls}],{},emission_mechanisms=mechanisms)
        assert chosen==cls
        mechanism=proof['source_mechanisms_by_class'][cls]
    proposal={'window_start_s':0.,'events':{'event':{'sound_class':cls,
        'emission_mechanism':mechanism,'native_endpoint_id':'voice1',
        'activity_intervals_samples_global':[[1600,14400]]}}}
    public,gold=collect_admitted_class_questions({},'unit',{},proposal,{},
        ['coughing','laughter','footsteps','music_playback'])
    assert bool(public)==eligible
    assert [r['canonical_answer']for r in gold]==([cls]if eligible else [])

def test_real_human_rotation_constructs_laughter_instead_of_unanswerable_live_coughing():
    chosen,proof=choose_unseen_class({'asset_id':'actual_human','kind':'human',
        'allowed_classes':['coughing','laughter']},
        [{'sound_class':'coughing'},{'sound_class':'laughter'}],{})
    assert chosen=='laughter'
    assert proof['excluded_by_unchanged_class34_predicate']=={'coughing':'human_live_recorded_coughing'}


def test_registered_loudspeaker_can_emit_actual_recorded_speech_without_relabelling_a_person():
    cls,proof=choose_unseen_class({'asset_id':'real_device','kind':'object','entity_class':'loudspeaker',
        'allowed_classes':['speech_playback']},[{'sound_class':'speech_playback'}],{})
    assert cls=='speech_playback'and proof['source_mechanisms_by_class'][cls]=='loudspeaker_playback'


def test_appearance_class_constructor_keeps_the_original_fixed_answer_domain():
    from avengine.timeline.continuous_scene import QA21_CLASSES
    asset={'asset_id':'registered_alarm','kind':'object','allowed_classes':['alarm_beep','laughter']}
    records=[{'sound_class':'alarm_beep'},{'sound_class':'laughter'}]
    assert choose_unseen_class(asset,records,{'alarm_beep':99})[0]=='laughter'
    cls,proof=choose_unseen_class(asset,records,{'alarm_beep':99},answer_domain=QA21_CLASSES)
    assert cls=='alarm_beep'
    assert set(proof['eligible_classes'])<=set(QA21_CLASSES)
    assert proof['required_answer_domain']==sorted(QA21_CLASSES)


def test_laughter_only_person_cannot_stand_in_for_qa21_fixed_domain_emitter():
    from avengine.timeline.continuous_scene import QA21_CLASSES
    with pytest.raises(ValueError,match='requested answer domain'):
        choose_unseen_class({'asset_id':'person','kind':'human','allowed_classes':['laughter']},
                            [{'sound_class':'laughter'}],{},answer_domain=QA21_CLASSES)
