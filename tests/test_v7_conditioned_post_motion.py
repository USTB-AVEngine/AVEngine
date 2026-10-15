"""The v7 adapter consumes the shipped compiler/solver without new predicates."""
import numpy as np
import pytest
from avengine.timeline.v7_conditioned_post_motion import plan_post_sound_motion,solve_post_sound_window
pytestmark=pytest.mark.fast_unit
ASSETS=[{'asset_id':a,'kind':'human','entity_class':'articulated_human'} for a in ('target','companion')]
REGISTRY={'assets':[{**a,'timeline':{'idle_action_id':'stand','walking_action_id':'walk',
    'body_plan_id':'human','walk_phase_period_frames':30}}for a in ASSETS]}
def record(name):
    return {'sound_asset_id':name,'header':{'sample_rate_hz':16000,'sample_count':32000},
        'path':'/test/'+name+'.wav','screens':{'automatic_activity':{'activity_intervals_samples':[[160,31840]]}}}
def row():
    return {'camera_yaw_knots':[{'time_s':0.,'yaw_deg':-90.},{'time_s':30.,'yaw_deg':-90.}],
        'source_classes':{'target':['speech_playback'],'companion':['speech_playback']},
        'scene_demand':{'distance_trend':'approaching'}}
@pytest.mark.parametrize('qa_id',['QA-13','QA-16','QA-17'])
def test_real_compiler_owns_three_epoch_post_sound_intervals(qa_id):
    records={'target':[record('target_'+str(i))for i in range(3)],
        'companion':[record('companion')]*3}
    result=plan_post_sound_motion(row(),qa_id=qa_id,target='target',companion='companion',
        assets=ASSETS,registry=REGISTRY,records_by_asset=records)
    assert result['acceptance_window_starts_s']==[0.,5.,10.,15.,20.]
    assert result['phase2_recorded_speech_schedule_s']['target']==[2.5,12.5,22.5]
    assert result['phase2_recorded_sound_ids']['target']==['target_0','target_1','target_2']
    intervals=result['motion_requests']['target']['intervals_s']
    for i,(lo,hi)in enumerate(intervals):
        assert lo>=10*i+4.5 and hi-lo>=2.-1e-8
        proof=result['conditioned_post_motion']['epochs'][i]
        assert proof['movement_solution']['status']=='solved'
        assert proof['requires_native_wet_tail_requery'] is True
    assert result['source_classes']['target']==['speech_playback']
    assert 'companion'not in result['motion_requests']
def test_adapter_rechecks_existing_solver_with_measured_tail():
    _,_,_,solution=solve_post_sound_window('QA-16',target='target',companion='companion',
        assets=ASSETS,registry=REGISTRY,records={'target':record('target'),'companion':record('companion')},
        measured_wet_tail_s=7.1)
    assert solution.solved
    assert solution.query_window['earliest_query_frame']>7.1*15
def test_static_camera_and_one_human_sound_source_are_preserved():
    r=row();r['camera_yaw_knots'][1]['yaw_deg']=0.
    with pytest.raises(ValueError,match='static'):
        plan_post_sound_motion(r,qa_id='QA-17',target='target',companion='companion',
            assets=ASSETS,registry=REGISTRY,records_by_asset={'target':[record('t')]*3,'companion':[record('c')]*3})
    r=row();r['source_classes']['target']=['speech_playback','footsteps']
    with pytest.raises(ValueError,match='one speaking source'):
        plan_post_sound_motion(r,qa_id='QA-17',target='target',companion='companion',
            assets=ASSETS,registry=REGISTRY,records_by_asset={'target':[record('t')]*3,'companion':[record('c')]*3})


def test_authored_companion_uses_all_three_solver_epochs_without_changing_full_words():
    from avengine.timeline.v7_conditioned_post_motion import apply_authored_post_schedule
    source={'sound_asset_id':'authored','source_type':'author_tts',
        'header':{'sample_rate_hz':16000,'sample_count':32000},'transcript':'Keep these complete words.'}
    scene={'events':[{'event_id':'a','endpoint_id':'voice2','start_tick':0,'end_tick':20},
                     {'event_id':'b','endpoint_id':'voice2','start_tick':150,'end_tick':170},
                     {'event_id':'other','endpoint_id':'voice1','start_tick':23,'end_tick':45}]}
    programme={'records':[source],'events':[{'event_id':x,'sound_asset_id':'authored','crop_start_sample':0} for x in ('a','b')]+[{'event_id':'other','sound_asset_id':'recorded'}]}
    result,bound=apply_authored_post_schedule({'phase2_authored_speech_schedule_s':{'companion':[0.,10.,20.]}},
        scene,programme,[{'asset_id':'companion','endpoint_id':'voice2'}])
    assert [e['start_tick'] for e in result['events'] if e['endpoint_id']=='voice2']==[0,100,200]
    assert next(e for e in result['events'] if e['event_id']=='other')==scene['events'][2]
    assert bound['records']==programme['records']
    assert bound['records'][0]['transcript']=='Keep these complete words.'
    assert all(b['crop_start_sample']==0 for b in bound['events'] if b['sound_asset_id']=='authored')
    assert [e['start_tick'] for e in scene['events'] if e['endpoint_id']=='voice2']==[0,150]
