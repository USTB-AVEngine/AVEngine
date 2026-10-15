from copy import deepcopy
import pytest
from avengine.timeline.v7_camera_evidence import configure_camera_evidence, set_static_camera, assert_static_camera, PROFILE
from avengine.timeline.v7_alpha_sampling import camera_knots
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize('members',[['base'],['v0_a0','v0_a1','v1_a0','v1_a1']])
@pytest.mark.parametrize('policy',[None,{'profile':PROFILE,'sweep_peak_deg':120.}])
def test_old_sweep_requests_produce_one_camera_pose(members,policy):
    scene={'listener_m':[1.,1.2,3.],'camera_forward':[0.,0.,-1.], 'duration_s':30.}
    row={'members':members,'production_scenario':{'camera_policy':'review_sweep_360'}}
    configure_camera_evidence(scene,row,policy)
    assert scene['camera_yaw_knots']==[{'time_s':0.,'yaw_deg':0.},{'time_s':30.,'yaw_deg':0.}]
    assert scene['listener_m']==[1.,1.2,3.]
    assert_static_camera(scene)

def test_capture_refuses_later_camera_mutation():
    scene={'listener_m':[0.,1.2,0.],'camera_forward':[0.,0.,-1.],'duration_s':30.}
    set_static_camera(scene)
    bad=deepcopy(scene);bad['camera_yaw_knots'][1]['yaw_deg']=.6
    with pytest.raises(ValueError,match='0.5 degrees'):assert_static_camera(bad)
    bad=deepcopy(scene);bad['listener_m'][0]=.02
    with pytest.raises(ValueError,match='1 cm'):assert_static_camera(bad)

def test_alpha_windows_do_not_generate_entry_exit_pans():
    assert camera_knots([0.,5.,10.,15.,20.],42.)==[{'time_s':0.,'yaw_deg':42.},{'time_s':30.,'yaw_deg':42.}]
