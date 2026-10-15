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
