import pytest
from avengine.timeline.v7_motion_source_roles import select_source_program,choose_motion_source_type,classify_source_type,SOURCE_TYPES
pytestmark=pytest.mark.fast_unit

def sound(name,seconds,spans,cls='cat_meow'):
    return {'sound_asset_id':name,'sound_class':cls,'duration_s':seconds,'header':{'sample_rate_hz':16000,'sample_count':round(seconds*16000)},'screens':{'automatic_activity':{'activity_intervals_samples':spans}}}

def test_longest_complete_call_is_selected_and_other_original_calls_are_retained():
    record=sound('ordinary reviewed recording',6.,[[1600,4800],[32000,56000],[80000,83000]])
    selected=select_source_program([record],{'cat_meow'},(.6,.8))
    assert selected['program']['selected_activity_index']==1
    assert selected['program']['required_activity_intervals_samples']==[[32000,56000]]
    assert selected['record']['screens']==record['screens']
    assert selected['minimum_designed_radial_change_m']>=.4

def test_a_whole_short_laugh_can_drive_a_walking_human_without_footsteps():
    selected=select_source_program([sound('complete laugh',2.1,[[800,33000]],'laughter')],{'laughter','coughing'},(.8,1.2))
    assert selected['record']['sound_class']=='laughter'
    assert selected['program']is None

def test_short_barks_cannot_be_declared_long_enough_for_the_requested_margin():
    with pytest.raises(ValueError,match='no complete admitted'):
        select_source_program([sound('short bark',.3,[[0,4800]],'dog_bark')],{'dog_bark'},(.9,1.2))

def test_planner_rotates_all_supported_movable_roles():
    state={'parents':{}}
    for kind in SOURCE_TYPES:
        assert choose_motion_source_type(state)==kind
        state['parents'][kind]={'motion_source_type':kind}

def test_question_source_classification_distinguishes_other_voice_feet_and_slide():
    assert classify_source_type({'species_id':'human','sound_class':'laughter'})=='human_other'
    assert classify_source_type({'species_id':'human','sound_class':'footsteps'})=='human_footsteps'
    assert classify_source_type({'species_id':'cat','motion_metadata':{'mode':'slide_no_animation'}})=='animal_slide'
