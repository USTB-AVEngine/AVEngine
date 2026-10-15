import copy
import numpy as np
import pytest
from avengine.timeline.v7_fixed_camera_layout import natural_person_yaws,require_fixed_camera_plan,assert_fixed_native_camera
pytestmark=pytest.mark.fast_unit


def test_complete_native_camera_cannot_hide_a_late_pan_or_translation():
    visual={'camera_frame_readbacks':[{'frame_index':i,'yaw_deg':23.,'position_m':[1.,1.5,2.]}for i in range(450)]}
    assert assert_fixed_native_camera(visual)['yaw_range_deg']==0.
    for field,value in [('yaw_deg',23.6),('position_m',[1.02,1.5,2.])]:
        bad=copy.deepcopy(visual);bad['camera_frame_readbacks'][-1][field]=value
        with pytest.raises(ValueError,match='fixed camera failed'):assert_fixed_native_camera(bad)
    with pytest.raises(ValueError):require_fixed_camera_plan({'listener_m':[0.,1.5,0.],'camera_yaw_knots':[{'yaw_deg':0.},{'yaw_deg':90.}]})


def test_conversational_pair_and_bystander_do_not_all_face_the_camera():
    points=np.array([[-1.,1.3,-3.],[1.,1.3,-3.],[0.,1.3,-1.5]]);listener=np.array([0.,1.5,0.]);yaws=natural_person_yaws(points,listener);toward=np.degrees(np.arctan2((listener-points)[:,0],-(listener-points)[:,2]));offset=np.abs((np.asarray(yaws)-toward+180)%360-180)
    assert all(offset[:2]>25)and offset[2]>90


def test_explicit_legacy_review_profile_now_stays_at_one_selected_pose():
    from avengine.timeline.v7_camera_evidence import configure_camera_evidence,recenter_on_actual_roots,PROFILE
    scene={'listener_m':[0.,1.3,0.],'duration_s':30.,'camera_forward':[0.,0.,-1.],'camera_yaw_knots':[{'time_s':0.,'yaw_deg':0.},{'time_s':30.,'yaw_deg':360.}],'window_starts_s':[0.,5.,10.,15.,20.]};row={'members':['v0_a0'],'production_scenario':{'camera_policy':'review_sweep_360'}}
    configure_camera_evidence(scene,row,{'profile':PROFILE});before=copy.deepcopy(scene['camera_yaw_knots']);recenter_on_actual_roots(scene,[])
    assert scene['camera_yaw_knots']==before and {k['yaw_deg']for k in before}=={0.}
