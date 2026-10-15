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

def test_native_contact_onsets_emit_every_real_step_once_without_a_six_event_quota():
    from avengine.timeline.v7_phase2_sources import resolve_footstep_events
    frames=[]
    for i in range(91):
        feet=[]
        for side,spans in [('L',[(9,21),(39,51)]),('R',[(24,36),(54,66)])]:
            feet.append({'endpoint_id':'person_foot_'+side,'in_contact':any(lo<=i<=hi for lo,hi in spans),'floor_delta_m':0.})
        frames.append({'moving':True,'attached_emitter_readbacks':feet})
    scene={'planned_footsteps':[{'endpoint_id':'person','sound_asset_id':'real registered step','requested_events':6}],'sources':{'person':{'actor_id':1},'person_foot_L':{'actor_id':1,'class_id':2,'mechanism':'walking_human_registered_foot','owner_endpoint_id':'person'},'person_foot_R':{'actor_id':1,'class_id':2,'mechanism':'walking_human_registered_foot','owner_endpoint_id':'person'}},'events':[]}
    program={'records':[{'sound_asset_id':'real registered step','screens':{'automatic_activity':{'activity_intervals_samples':[[0,1600]]}},'header':{'sample_count':16000}}],'events':[]}
    result=resolve_footstep_events(scene,program,{'native_motion_admissions':{'person':{'native_30Hz_readbacks':frames}}})
    assert [r['actual_contact_onset_frame_index']for r in result['events']]==[9,24,39,54]
    assert len(scene['events'])==4
    assert [len(steps)for _,steps in group_footstep_bouts(scene)]==[4]
