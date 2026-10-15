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
