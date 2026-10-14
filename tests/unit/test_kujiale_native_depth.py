import numpy as np
from avengine.qa.kujiale_native_depth import world_pixels,visibility

def test_radial_meters_recover_off_axis_world_distance():
    view={'position_m':[0.,1.,0.],'yaw_deg':90.,'pitch_deg':0.,'hfov_deg':90.}
    # A center-column pixel 1/3 below center has an unnormalized ray
    # (0,-1/3,1); native radial range sqrt(10) recovers (0,0,3).
    depth=np.full((7,9),65504.,dtype=float);depth[5,4]=np.sqrt(10.)
    mask=np.zeros_like(depth,dtype=bool);mask[5,4]=True
    actual=world_pixels(depth,view,mask)[0]
    # For W=9, v=(3.5-5.5)/4.5=-4/9; use the analytic ray, not z-depth.
    expected=np.array([0.,1.,0.])+np.sqrt(10.)*np.array([0.,-4/9,1.])/np.sqrt(1+(4/9)**2)
    assert np.allclose(actual,expected)
    assert abs(np.linalg.norm(actual-np.array(view['position_m']))-np.sqrt(10.))<1e-12

def test_nearby_occluder_and_finite_background_do_not_count_as_visible():
    isolated=np.array([[3.,3.,65504.]])
    scene=np.array([[3.,2.8,65504.]])
    mask,visible=visibility(scene,isolated,4.)
    assert mask.tolist()==[[True,True,False]]
    assert visible.tolist()==[[True,False,False]]
