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

def test_construction_distance_keeps_half_metre_margin_and_original_reversal_tolerance():
    from avengine.assets.roster_motion.construction_distance import accepts_distance_trend
    assert accepts_distance_trend([3.,2.7,2.74,2.49],-1)
    assert not accepts_distance_trend([3.,2.7,2.76,2.4],-1)
    assert not accepts_distance_trend([3.,2.6],-1)

def test_native_route_pool_exceeds_twenty_four_and_ranks_clearance(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from avengine.assets.roster_motion import straight_window_routes as routes
    proposed=[SimpleNamespace(positions=np.array([[i,0.,0.],[i,.0,.1]]),receipt={'proposal':i+1})for i in range(32)]
    monkeypatch.setattr(routes,'_unsorted_straight_window_routes',lambda *a:iter(proposed))
    video=SimpleNamespace(pathfinder=SimpleNamespace(distance_to_closest_obstacle=lambda point,limit:.35+point[0]*.01))
    actual=list(routes.straight_window_routes(video,None,None,None,None,{}))
    assert len(actual)==32 and actual[0].receipt['proposal']==32
    assert all(r.receipt['prequalified_candidate_count']==32 for r in actual)
