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
