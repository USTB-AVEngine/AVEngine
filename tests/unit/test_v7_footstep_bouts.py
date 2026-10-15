from copy import deepcopy
import pytest
from avengine.timeline.v7_footstep_bouts import group_footstep_bouts,footstep_bout_question_inputs
pytestmark=pytest.mark.fast_unit

def fixture(ticks=(10,16,23),owners=('person','person','person')):
    scene={'sources':{},'events':[]};proof={'events':{},'protocol_scene':{'sources':{},'events':[]}}
    for index,(tick,owner)in enumerate(zip(ticks,owners)):
        alias=owner+'_foot_'+('L'if index%2 else'R');scene['sources'][owner]={'actor_id':1 if owner=='person'else 2}
        scene['sources'][alias]={'actor_id':scene['sources'][owner]['actor_id'],'owner_endpoint_id':owner}
        event={'event_id':str(index),'endpoint_id':alias,'source_id':scene['sources'][owner]['actor_id'],'sound_class':'footsteps','class_id':2,'start_tick':tick,'end_tick':tick+1,'mechanism':'walking_human_registered_foot'};scene['events'].append(event)
        proof['events'][str(index)]={'sound_asset_id':'registered real footstep','activity_intervals_samples_global':[[tick*1600,(tick+1)*1600]],'wet_intervals_global_s':[[tick/10,(tick+1)/10+.3]]}
        proof['protocol_scene']['events'].append(deepcopy(event))
    return scene,proof

def test_onset_threshold_is_inclusive_and_gap_over_six_tenths_splits():
    scene,_=fixture();assert [len(steps)for _,steps in group_footstep_bouts(scene)]==[2,1]

def test_person_identity_never_merges_across_owners():
    scene,_=fixture((10,11,12),('person','other','person'))
    assert sorted(len(steps)for _,steps in group_footstep_bouts(scene))==[1,2]

def test_question_and_protocol_counts_are_bouts_and_raw_steps_are_preserved():
    scene,proof=fixture();original=deepcopy((scene,proof));result,evidence=footstep_bout_question_inputs(scene,proof)
    assert len(result['events'])==len(evidence['events'])==len(evidence['protocol_scene']['events'])==2
    assert evidence['footstep_bouts']['raw_step_count']==3
    assert len(evidence['footstep_bouts']['raw_step_evidence'])==3
    assert result['events'][0]['end_tick']==20
    assert result['events'][0]['emission_end_s']==1.7
    assert (scene,proof)==original
    assert evidence['events'][result['events'][0]['event_id']]['activity_intervals_samples_global']==[[16000,17600],[25600,27200]]

def test_missing_measured_tail_cannot_certify_bout_stop():
    scene,proof=fixture();proof['events']['0'].pop('wet_intervals_global_s')
    with pytest.raises(ValueError,match='actual measured wet'):footstep_bout_question_inputs(scene,proof)


def test_bout_geometry_retains_a_real_native_attached_foot_instead_of_the_mouth():
    scene,proof=fixture((10,15),('person','person'))
    result,evidence=footstep_bout_question_inputs(scene,proof)
    event=result['events'][0]
    assert event['endpoint_id']==scene['events'][0]['endpoint_id']
    assert event['endpoint_id']!='person'
    assert evidence['events'][event['event_id']]['native_endpoint_id']==event['endpoint_id']
    assert scene['sources']['person']==result['sources']['person']
