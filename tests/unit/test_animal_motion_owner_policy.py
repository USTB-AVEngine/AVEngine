"""Owner-selected motion checks keep the strict policy as the default."""
import copy
import numpy as np
from avengine.assets.animal_motion.checks import direction_check, frame_checks, direction_negative_controls
from avengine.assets.quadruped_penetration import visible_penetration_gate
RELAXED='owner_relaxed_20261014'
def trajectory(angle):
    t=np.arange(6)/30
    roots=np.c_[np.zeros(6),np.zeros(6),-.03*t]
    a=np.radians(angle);axis=np.array([np.sin(a),0,-np.cos(a)])
    return t,roots,roots+axis*.3+[0,.2,0],roots-axis*.3+[0,.2,0]
def test_owner_heading_policy_accepts17_degrees_and_still_rejects_sideways_and_backwards():
    assert direction_check(*trajectory(17))['status']=='fail'
    assert direction_check(*trajectory(17),policy=RELAXED)['status']=='pass'
    for angle in (21,90,180):
        assert direction_check(*trajectory(angle),policy=RELAXED)['status']=='fail'
def contact_frames():
    t,r,h,q=trajectory(0);rows=[]
    for i in range(6):
        feet={}
        for side in ('FL','FR','HL','HR'):
            contact=(side in ('FL','FR'))==(i<3);phase=i if i<3 else i-3
            feet[side]={'floor_delta_m':0.,'in_contact':contact,'support_segment':0 if i<3 else 1,
                        'support_world_m':[.005*phase,0,0]}
        rows.append({'time_s':float(t[i]),'feet':feet,'root_world_m':r[i].tolist(),
                     'head_world_m':h[i].tolist(),'tail_world_m':q[i].tolist(),
                     'body_floor_min_m':0.,'body_head_tail_pitch_deg':0.})
    return rows
def test_cumulative_step_sliding_is15mm_while_per_frame_speed_is_diagnostic():
    f=contact_frames()
    old=frame_checks(f);new=frame_checks(f,policy=RELAXED)
    assert old['status']=='fail' and any('stance_drift' in x for x in old['failures'])
    assert new['status']=='pass' and abs(new['max_stance_drift_m']-.010)<1e-12
    assert new['max_stance_speed_m_s']>.03
    f[2]['feet']['FL']['support_world_m'][0]=.016
    assert any('stance_drift' in x for x in frame_checks(f,policy=RELAXED)['failures'])
def test_floor_and_real_moving_direction_remain_required_under_owner_policy():
    f=contact_frames()
    assert direction_negative_controls(f,policy=RELAXED)['status']=='pass'
    f[0]['body_floor_min_m']=-.013
    assert '0:full_skin_penetration' in frame_checks(f,policy=RELAXED)['failures']
def test_visible_large_crossings_are_rejected_but_intraregion_crossings_are_ignored():
    rows=[{'region_pair':['HL','HR'],'new_area_m2':200e-6,'new_maximum_depth_m':.010},
          {'region_pair':['FL','Body'],'new_area_m2':200e-6,'new_maximum_depth_m':.001},
          {'region_pair':['Body','Body'],'new_area_m2':.1,'new_maximum_depth_m':.05}]
    kwargs=dict(cross_region_area_limit_m2=400e-6,depth_limit_m=.010,check_intraregion=False)
    assert visible_penetration_gate(rows)
    assert visible_penetration_gate(rows,**kwargs)==[]
    bad=copy.deepcopy(rows);bad[1]['new_area_m2']+=.1e-6
    assert visible_penetration_gate(bad,**kwargs)
    bad=copy.deepcopy(rows);bad[0]['new_maximum_depth_m']=.0101
    assert visible_penetration_gate(bad,**kwargs)
