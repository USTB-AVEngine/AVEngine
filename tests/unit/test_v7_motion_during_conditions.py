import pytest
from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row,verify_motion_during_native
pytestmark=pytest.mark.fast_unit

def fixture(qa='QA-15',branch='nearer'):
    assets=[{'asset_id':k,'kind':'human','walk_allowed':True,'source_registry_sha256':'admitted-original'}for k in ['target','companion']]
    records=[{'sound_asset_id':k,'path':'/read-only/'+k+'.wav','header':{'sample_count':n,'sample_rate_hz':16000},'screens':{'automatic_activity':{'activity_intervals_samples':[[1600,n-1600]]}}}for k,n in [('speech',80000),('other',16000)]]
    row={'asset_ids':['target','companion'],'motion_requests':{'target':{'action':'walk'}},'proposed_sound_asset_ids':{'target':'speech','companion':'other'},'phase2_recorded_speech_schedule_s':{'target':[.3],'companion':[0.]},'camera_yaw_deg':0.,'camera_yaw_knots':[{'time_s':0.,'yaw_deg':0.},{'time_s':10.,'yaw_deg':80.}],
        'motion_during_condition_requests':[{'qa_id':qa,'branch':branch,'target_asset_id':'target','speed_m_s':.4,'minimum_distance_change_m':1.5}]}
    return row,assets,records

@pytest.mark.parametrize('qa,branch',[('QA-06','moving'),('QA-15','nearer'),('QA-15','farther')])
def test_original_solver_drives_native_clock_intervals(qa,branch):
    row,assets,records=fixture(qa,branch);result=prepare_motion_during_row(row,assets,records)
    p=result['motion_during_condition_plan']['plans'][0];assert p['solution']['status']=='solved'
    lo,hi=p['solution']['requirements'][0]['moving_frames'];a,b=result['motion_requests']['target']['intervals_s'][0]
    assert a<=lo/15. and b>=hi/15.
    assert result['camera_yaw_knots']==[{'time_s':0.,'yaw_deg':0.},{'time_s':30.,'yaw_deg':0.}]
    assert row['camera_yaw_knots'][-1]['yaw_deg']==80.

def test_original_solver_verifier_checks_actual_native_trajectory():
    row,assets,records=fixture();result=prepare_motion_during_row(row,assets,records)
    scene={'listener_m':[0.,0.,0.],'sources':{'t':{'asset_id':'target','trajectory':[{'time_s':0.,'position_m':[0.,0.,4.]},{'time_s':6.,'position_m':[0.,0.,1.6]},{'time_s':30.,'position_m':[0.,0.,1.6]}]},'c':{'asset_id':'companion','trajectory':[{'time_s':0.,'position_m':[3.,0.,3.]},{'time_s':30.,'position_m':[3.,0.,3.]}]}}}
    report=verify_motion_during_native(result,scene,{})
    assert report['status']=='pass',report
    scene['sources']['t']['trajectory']=[{'time_s':0.,'position_m':[0.,0.,4.]},{'time_s':30.,'position_m':[0.,0.,4.]}]
    assert verify_motion_during_native(result,scene,{})['status']=='fail'


def test_whole_animal_recording_uses_measured_v3_sound_clock():
    row,assets,records=fixture('QA-15','farther')
    assets[0].update(kind='animal')
    row['phase2_recorded_speech_schedule_s'].pop('target')
    row['phase2_whole_animal_recordings']={'target':[.7]}
    out=prepare_motion_during_row(row,assets,records)
    assert out['motion_during_condition_plan']['status']=='solved'
    assert out['motion_during_condition_plan']['plans'][0]['config']['event_start_s']['target']==.7


def test_whole_animal_clip_retains_length_and_does_not_add_a_human_sound():
    from avengine.timeline.v7_visible_sources import recorded_event_layout
    row={'motion_requests':{'dog':{}},'phase2_whole_animal_recordings':{'dog':[.3]}}
    record={'duration_s':3.14,'source_type':'recorded'}
    assert recorded_event_layout(row,{'asset_id':'dog','kind':'animal'},record,False,[.2,5.2])==([.3],True)
    assert recorded_event_layout(row,{'asset_id':'human','kind':'human'},record,False,[.2,5.2])==([.2,5.2],False)
    row['phase2_whole_animal_recordings']['dog']=[28.]
    with pytest.raises(ValueError,match='does not fit'):recorded_event_layout(row,{'asset_id':'dog','kind':'animal'},record,False,[.2])


def test_native_verifier_uses_body_root_speed_and_actual_mouth_distance():
    import numpy as np
    row,assets,records=fixture();out=prepare_motion_during_row(row,assets,records)
    ts=np.arange(901)/30.
    roots=np.column_stack((np.zeros(901),np.zeros(901),4.-.4*ts))
    mouth=roots+np.column_stack((np.zeros(901),np.zeros(901),.01*np.sin(20*ts)))
    scene={'listener_m':[0.,0.,0.],'sources':{
        'target_ep':{'asset_id':'target','trajectory':[{'time_s':float(t),'position_m':p.tolist()}for t,p in zip(ts,mouth)]},
        'companion_ep':{'asset_id':'companion','trajectory':[{'time_s':0.,'position_m':[3.,0.,3.]},{'time_s':30.,'position_m':[3.,0.,3.]}]}}}
    assert verify_motion_during_native(out,scene,{})['status']=='fail'
    visual={'native_motion_admissions':{'target_ep':{'world_roots_m':roots.tolist()}}}
    report=verify_motion_during_native(out,scene,visual)
    assert report['status']=='pass',report


def test_repaired_native_verification_keeps_original_failed_evidence(tmp_path):
    import json
    from avengine.timeline.v7_motion_during_conditions import write_native_motion_condition_report
    old=tmp_path/'conditioned_motion_native_verification.json';old.write_text('old failed native result')
    saved=write_native_motion_condition_report(tmp_path,{'status':'pass','checks':['body root and source distance']})
    assert saved!=old and old.read_text()=='old failed native result'
    assert json.loads(saved.read_text())['status']=='pass'
    another=write_native_motion_condition_report(tmp_path,{'checks':['body root and source distance'],'status':'pass'})
    assert another!=saved and another.parent!=saved.parent
    assert another.read_bytes()==saved.read_bytes()


@pytest.mark.parametrize('qa,branch',[('QA-16',None),('QA-17','yes')])
def test_original_post_sound_solver_places_a_complete_two_second_straight_query(qa,branch):
    row,assets,records=fixture(qa,branch)
    records[0]['header']['sample_count']=38400
    records[0]['screens']['automatic_activity']['activity_intervals_samples']=[[1600,36800]]
    row['motion_during_condition_requests'][0].update(speed_m_s=.9,speed_range_m_s=[.8,1.4],minimum_motion_s=2.,measured_wet_tail_s=4.1)
    out=prepare_motion_during_row(row,assets,records)
    windows=out['motion_requests']['target']['straight_motion_windows_s']
    assert all(hi-lo>=2. for lo,hi in windows)
    assert min(lo for lo,_ in windows)>.3+2.3+4.1
    assert out['motion_during_condition_plan']['plans'][0]['solution']['status']=='solved'
