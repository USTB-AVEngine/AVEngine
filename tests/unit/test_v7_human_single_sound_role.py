import pytest
from avengine.timeline.v7_phase2_sources import footstep_walkers_for_row
pytestmark=pytest.mark.fast_unit
@pytest.mark.parametrize('classes,walk,expected',[(['speech_playback'],True,[]),([],True,[]),(['footsteps'],True,['person'])])
def test_sound_role_is_independent_from_motion(classes,walk,expected):
 assert footstep_walkers_for_row({'source_classes':{'person':classes},'motion_requests':{'person':{'action':'walk'}}if walk else{}})==expected
@pytest.mark.parametrize('row',[{'source_classes':{'person':['footsteps','speech_playback']},'motion_requests':{'person':{'action':'walk'}}},{'source_classes':{'person':['footsteps']},'motion_requests':{}},{'source_classes':{'person':['speech_playback']},'motion_requests':{'person':{'action':'walk'}},'phase2_include_footsteps_asset_ids':['person']}])
def test_conflicting_sound_role_cannot_enter_native_generation(row):
 with pytest.raises(ValueError):footstep_walkers_for_row(row)
